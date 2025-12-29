import sys
import os
# CHANGED: 在导入其他包前，将当前目录加入系统路径，确保能正确导入本地模块（如convert, inference）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import threading
import queue
import time
import random
from io import TextIOBase
import datetime
import subprocess

# CHANGED: 移除 spaces 和 zero 的导入，因为它们与本地运行无关
# import spaces  # 不再需要
# import zero    # 不再需要

from inference import postprocess_inst_names
from inference import inference_patch
from convert import abc2xml, xml2, pdf2img


title_html = """
<div class="title-container">
    <h1 class="title-text">NotaGen</h1> &nbsp;
        <!-- ArXiv -->
        <a href="https://arxiv.org/abs/2502.18008">
            <img src="https://img.shields.io/badge/NotaGen_Paper-ArXiv-%23B31B1B?logo=arxiv&logoColor=white" alt="Paper">
        </a>
        &nbsp;
        <!-- GitHub -->
        <a href="https://github.com/ElectricAlexis/NotaGen">
            <img src="https://img.shields.io/badge/NotaGen_Code-GitHub-%23181717?logo=github&logoColor=white" alt="GitHub">
        </a>
        &nbsp;
        <!-- HuggingFace -->
        <a href="https://huggingface.co/ElectricAlexis/NotaGen">
            <img src="https://img.shields.io/badge/NotaGen_Weights-HuggingFace-%23FFD21F?logo=huggingface&logoColor=white" alt="Weights">
        </a>
        &nbsp;
        <!-- Web Demo -->
        <a href="https://electricalexis.github.io/notagen-demo/">
            <img src="https://img.shields.io/badge/NotaGen_Demo-Web-%23007ACC?logo=google-chrome&logoColor=white" alt="Demo">
        </a>
</div>
<bp>
<p style="font-size: 1.2em;">NotaGen is a model for generating sheet music in ABC notation format. Select a period, composer, and instrumentation to generate classical-style music!</p>
"""

# Read prompt combinations
with open('prompts.txt', 'r') as f:
    prompts = f.readlines()

valid_combinations = set()
for prompt in prompts:
    prompt = prompt.strip()
    parts = prompt.split('_')
    valid_combinations.add((parts[0], parts[1], parts[2]))

# Prepare dropdown options
periods = sorted({p for p, _, _ in valid_combinations})
composers = sorted({c for _, c, _ in valid_combinations})
instruments = sorted({i for _, _, i in valid_combinations})

# Dynamically update composer and instrument dropdown options
def update_components(period, composer):
    if not period:
        return [
            gr.update(choices=[], value=None, interactive=False),
            gr.update(choices=[], value=None, interactive=False)
        ]

    valid_composers = sorted({c for p, c, _ in valid_combinations if p == period})
    valid_instruments = sorted({i for p, c, i in valid_combinations if p == period and c == composer}) if composer else []

    return [
        gr.update(
            choices=valid_composers,
            value=composer if composer in valid_composers else None,
            interactive=True
        ),
        gr.update(
            choices=valid_instruments,
            value=None,
            interactive=bool(valid_instruments)
        )
    ]

# Custom realtime stream for outputting model inference process to frontend
class RealtimeStream(TextIOBase):
    def __init__(self, queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)
        return len(text)

def convert_files(abc_content, period, composer, instrumentation):
    if not all([period, composer, instrumentation]):
        raise gr.Error("Please complete a valid generation first before saving")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_str = f"{period}_{composer}_{instrumentation}"
    filename_base = f"{timestamp}_{prompt_str}"

    abc_filename = f"{filename_base}.abc"
    with open(abc_filename, "w", encoding="utf-8") as f:
        f.write(abc_content)

    # instrumentation replacement
    postprocessed_inst_abc = postprocess_inst_names(abc_content)
    filename_base_postinst = f"{filename_base}_postinst"
    with open(filename_base_postinst + ".abc", "w", encoding="utf-8") as f:
        f.write(postprocessed_inst_abc)

    # Convert files
    file_paths = {'abc': abc_filename}
    try:
        # abc2xml
        abc2xml(filename_base)
        abc2xml(filename_base_postinst)

        # xml2pdf
        xml2(filename_base, 'pdf')

        # xml2mid
        xml2(filename_base, 'mid')
        xml2(filename_base_postinst, 'mid')

        # xml2mp3
        xml2(filename_base, 'mp3')
        xml2(filename_base_postinst, 'mp3')

        # 将PDF转为图片
        images = pdf2img(filename_base)
        for i, image in enumerate(images):
            image.save(f"{filename_base}_page_{i+1}.png", "PNG")

        file_paths.update({
            'xml': f"{filename_base_postinst}.xml",
            'pdf': f"{filename_base}.pdf",
            'mid': f"{filename_base_postinst}.mid",
            'mp3': f"{filename_base_postinst}.mp3",
            'pages': len(images),
            'current_page': 0,
            'base': filename_base
        })

    except Exception as e:
        raise gr.Error(f"File processing failed: {str(e)}")

    return file_paths


# Page navigation control function
def update_page(direction, data):
    """
    data contains three key pieces of information: 'pages', 'current_page', and 'base'
    """
    if not data:
        return None, gr.update(interactive=False), gr.update(interactive=False), data

    if direction == "prev" and data['current_page'] > 0:
        data['current_page'] -= 1
    elif direction == "next" and data['current_page'] < data['pages'] - 1:
        data['current_page'] += 1

    current_page_index = data['current_page']
    # Update image path
    new_image = f"{data['base']}_page_{current_page_index+1}.png"
    # When current_page==0, prev_btn is disabled; when current_page==pages-1, next_btn is disabled
    prev_btn_state = gr.update(interactive=(current_page_index > 0))
    next_btn_state = gr.update(interactive=(current_page_index < data['pages'] - 1))

    return new_image, prev_btn_state, next_btn_state, data


# CHANGED: 完全移除 @spaces.GPU 装饰器，直接使用本地GPU
def generate_music(period, composer, instrumentation):
    """
    Must ensure each yield returns the same number of values.
    We're preparing to return 5 values, corresponding to:
    1) process_output (intermediate inference information)
    2) final_output (final ABC)
    3) pdf_image (path to the PNG of the first page of the PDF)
    4) audio_player (mp3 path)
    5) pdf_state (state for page navigation)
    """
    # Set a different random seed each time based on current timestamp
    random_seed = int(time.time()) % 10000
    random.seed(random_seed)
    
    # For numpy if you're using it
    try:
        import numpy as np
        np.random.seed(random_seed)
    except ImportError:
        pass
        
    # For torch if you're using it
    try:
        import torch
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            # CHANGED: 明确打印使用的设备信息
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Warning: CUDA not available, using CPU instead")
    except ImportError:
        print("Warning: PyTorch not installed, using CPU mode")

    if (period, composer, instrumentation) not in valid_combinations:
        # If the combination is invalid, raise an error
        raise gr.Error("Invalid prompt combination! Please re-select from the period options")

    output_queue = queue.Queue()
    original_stdout = sys.stdout
    sys.stdout = RealtimeStream(output_queue)

    result_container = []

    def run_inference():
        try:
            # Use downloaded model weights path for inference
            result = inference_patch(period, composer, instrumentation)
            result_container.append(result)
        finally:
            sys.stdout = original_stdout

    thread = threading.Thread(target=run_inference)
    thread.start()

    process_output = ""
    final_output_abc = ""
    pdf_image = None
    audio_file = None
    pdf_state = None

    # First continuously read intermediate output
    while thread.is_alive():
        try:
            text = output_queue.get(timeout=0.1)
            process_output += text
            # No final ABC yet, files not yet converted
            yield process_output, final_output_abc, pdf_image, audio_file, pdf_state, gr.update(value=None, visible=False)
        except queue.Empty:
            continue

    # After thread ends, get all remaining items from the queue
    while not output_queue.empty():
        text = output_queue.get()
        process_output += text

    # Final inference result
    final_result = result_container[0] if result_container else ""
    
    # Display file conversion prompt
    final_output_abc = "Converting files..."
    yield process_output, final_output_abc, pdf_image, audio_file, pdf_state, gr.update(value=None, visible=False)


    # Convert files
    try:
        file_paths = convert_files(final_result, period, composer, instrumentation)
        final_output_abc = final_result
        # Get the first image and mp3 file
        if file_paths['pages'] > 0:
            pdf_image = f"{file_paths['base']}_page_1.png"
        audio_file = file_paths['mp3']
        pdf_state = file_paths  # Directly use the converted information dictionary as state
        
        # Prepare download file list
        download_list = []
        if 'abc' in file_paths and os.path.exists(file_paths['abc']):
            download_list.append(file_paths['abc'])
        if 'xml' in file_paths and os.path.exists(file_paths['xml']):
            download_list.append(file_paths['xml'])
        if 'pdf' in file_paths and os.path.exists(file_paths['pdf']):
            download_list.append(file_paths['pdf'])
        if 'mid' in file_paths and os.path.exists(file_paths['mid']):
            download_list.append(file_paths['mid'])
        if 'mp3' in file_paths and os.path.exists(file_paths['mp3']):
            download_list.append(file_paths['mp3'])
    except Exception as e:
        # If conversion fails, return error message to output box
        yield process_output, f"Error converting files: {str(e)}", None, None, None, gr.update(value=None, visible=False)
        return

   # Final yield with all information - modify here to make component visible
    yield process_output, final_output_abc, pdf_image, audio_file, pdf_state, gr.update(value=download_list, visible=True)


def get_file(file_type, period, composer, instrumentation):
    """
    Returns the local file of specified type for Gradio download
    """
    # Here you actually need to return based on specific file paths saved earlier, simplified for demo
    # If matching by timestamp, you can store all converted files in a directory and get the latest
    # This is just an example:
    possible_files = [f for f in os.listdir('.') if f.endswith(f'.{file_type}')]
    if not possible_files:
        return None
    # Simply return the latest
    possible_files.sort(key=os.path.getmtime)
    return possible_files[-1]


css = """
/* Compact button style */
button[size="sm"] {
    padding: 4px 8px !important;
    margin: 2px !important;
    min-width: 60px;
}

/* PDF preview area */
#pdf-preview {
    border-radius: 8px;  /* Rounded corners */
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);  /* Shadow */
}

.page-btn {
    padding: 12px !important;  /* Increase clickable area */
    margin: auto !important;   /* Vertical center */
}

/* Button hover effect */
.page-btn:hover {
    background: #f0f0f0 !important;
    transform: scale(1.05);
}

/* Layout adjustment */
.gr-row {
    gap: 10px !important;  /* Element spacing */
}

/* Audio player */
.audio-panel {
    margin-top: 15px !important;
    max-width: 400px;
}

#audio-preview audio {
    height: 200px !important;
}

/* Save functionality area */
.save-as-row {
    margin-top: 15px;
    padding: 10px;
    border-top: 1px solid #eee;
}

/* Download files styling */
.download-files {
    margin-top: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* Social icons styling */
.title-container {
    display: flex;
    align-items: center;
    gap: 15px;
    margin-bottom: 10px;
}

.title-text {
    margin: 0;
    font-size: 1.8em;
}

.social-icons {
    display: flex;
    gap: 10px;
}

.social-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background-color: #f5f5f5;
    text-decoration: none;
    transition: transform 0.2s, background-color 0.2s;
}

.social-icon:hover {
    transform: scale(1.1);
    background-color: #e0e0e0;
}

.social-icon img {
    width: 20px;
    height: 20px;
}

"""

# CHANGED: 修复Gradio 6.0+兼容性问题 - 将css参数从Blocks()移到launch()
with gr.Blocks() as demo:  # 移除了css=css参数
    gr.HTML(title_html)

    # For storing PDF page count, current page and other information
    pdf_state = gr.State()

    with gr.Column():
        with gr.Row():
            # Left sidebar
            with gr.Column():
                with gr.Row():
                    period_dd = gr.Dropdown(
                        choices=periods,
                        value=None,
                        label="Period",
                        interactive=True
                    )
                    composer_dd = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Composer",
                        interactive=False
                    )
                    instrument_dd = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Instrumentation",
                        interactive=False
                    )

                generate_btn = gr.Button("Generate!", variant="primary")

                process_output = gr.Textbox(
                    label="Generation process",
                    interactive=False,
                    lines=2,
                    max_lines=2,
                    placeholder="Generation progress will be shown here..."
                )

                final_output = gr.Textbox(
                    label="Post-processed ABC notation scores",
                    interactive=True,
                    lines=8,
                    max_lines=8,
                    placeholder="Post-processed ABC scores will be shown here..."
                )

                # Audio playback
                audio_player = gr.Audio(
                    label="Audio Preview",
                    format="mp3",
                    interactive=False,
                )

            # Right sidebar
            with gr.Column():
                # Image container
                # CHANGED: 移除Gradio 6.x中不支持的show_download_button参数
                pdf_image = gr.Image(
                    label="Sheet Music Preview",
                    show_label=False,
                    height=650,
                    type="filepath",
                    elem_id="pdf-preview",
                    interactive=False
                    # 移除了 show_download_button=False
                )

                # Page navigation buttons
                with gr.Row():
                    prev_btn = gr.Button(
                        "⬅️ Last Page",
                        variant="secondary",
                        size="sm",
                        elem_classes="page-btn"
                    )
                    next_btn = gr.Button(
                        "Next Page ➡️",
                        variant="secondary",
                        size="sm",
                        elem_classes="page-btn"
                    )

        with gr.Column():
            gr.Markdown("**Download Files:**")
            download_files = gr.Files(
                label="Generated Files", 
                visible=False,
                elem_classes="download-files",
                type="filepath"  # Make sure this is set to filepath
            )

    # Dropdown linking
    period_dd.change(
        update_components,
        inputs=[period_dd, composer_dd],
        outputs=[composer_dd, instrument_dd]
    )
    composer_dd.change(
        update_components,
        inputs=[period_dd, composer_dd],
        outputs=[composer_dd, instrument_dd]
    )

    # Click generate button, note outputs must match each yield in generate_music
    generate_btn.click(
        generate_music,
        inputs=[period_dd, composer_dd, instrument_dd],
        outputs=[process_output, final_output, pdf_image, audio_player, pdf_state, download_files]
    )

    # Page navigation
    prev_signal = gr.Textbox(value="prev", visible=False)
    next_signal = gr.Textbox(value="next", visible=False)

    prev_btn.click(
        update_page,
        inputs=[prev_signal, pdf_state],  # ✅ Use component
        outputs=[pdf_image, prev_btn, next_btn, pdf_state]
    )

    next_btn.click(
        update_page,
        inputs=[next_signal, pdf_state],  # ✅ Use component
        outputs=[pdf_image, prev_btn, next_btn, pdf_state]
    )


if __name__ == "__main__":
    # CHANGED: 简化启动配置，直接本地运行
    print("Starting NotaGen server locally...")
    print(f"Access the application at: http://localhost:7860")
    # CHANGED: 将css参数添加到launch()方法中
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # 本地运行无需分享链接
        css=css       # 将CSS样式移到这里
    )
