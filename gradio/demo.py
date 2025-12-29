import gradio as gr
import sys
import threading
import queue
from io import TextIOBase
from inference import inference_patch
import datetime
import subprocess
import os
import torch  # 添加这一行

# Predefined valid combinations set
with open('prompts.txt', 'r') as f:
    prompts = f.readlines()
valid_combinations = set()
for prompt in prompts:
    prompt = prompt.strip()
    parts = prompt.split('_')
    valid_combinations.add((parts[0], parts[1], parts[2]))

# Generate available options
periods = sorted({p for p, _, _ in valid_combinations})
composers = sorted({c for _, c, _ in valid_combinations})
instruments = sorted({i for _, _, i in valid_combinations})

# Dynamic component updates
def update_components(period, composer):
    if not period:
        return [
            gr.Dropdown(choices=[], value=None, interactive=False),
            gr.Dropdown(choices=[], value=None, interactive=False)
        ]
    
    valid_composers = sorted({c for p, c, _ in valid_combinations if p == period})
    valid_instruments = sorted({i for p, c, i in valid_combinations if p == period and c == composer}) if composer else []
    
    return [
        gr.Dropdown(
            choices=valid_composers,
            value=composer if composer in valid_composers else None,
            interactive=True  
        ),
        gr.Dropdown(
            choices=valid_instruments,
            value=None,
            interactive=bool(valid_instruments)  
        )
    ]


class RealtimeStream(TextIOBase):
    def __init__(self, queue):
        self.queue = queue
    
    def write(self, text):
        self.queue.put(text)
        return len(text)


def save_and_convert(abc_content, period, composer, instrumentation):
    if not all([period, composer, instrumentation]):
        raise gr.Error("Please complete a valid generation first before saving")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_str = f"{period}_{composer}_{instrumentation}"
    filename_base = f"{timestamp}_{prompt_str}"
    
    abc_filename = f"{filename_base}.abc"
    with open(abc_filename, "w", encoding="utf-8") as f:
        f.write(abc_content)

    xml_filename = f"{filename_base}.xml"
    try:
        subprocess.run(
            ["python", "abc2xml.py", '-o', '.', abc_filename, ],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        error_msg = f"Conversion failed: {e.stderr}" if e.stderr else "Unknown error"
        raise gr.Error(f"ABC to XML conversion failed: {error_msg}. Please try to generate another composition.")
    
    return f"Saved successfully: {abc_filename} -> {xml_filename}"



def generate_music(period, composer, instrumentation):
    if (period, composer, instrumentation) not in valid_combinations:
        raise gr.Error("Invalid prompt combination! Please re-select from the period options")
    
    output_queue = queue.Queue()
    original_stdout = sys.stdout
    sys.stdout = RealtimeStream(output_queue)
    
    result_container = []
    def run_inference():
        try:
            result_container.append(inference_patch(period, composer, instrumentation))
        finally:
            sys.stdout = original_stdout
    
    thread = threading.Thread(target=run_inference)
    thread.start()
    
    process_output = ""
    while thread.is_alive():
        try:
            text = output_queue.get(timeout=0.1)
            process_output += text
            yield process_output, None  
        except queue.Empty:
            continue
    
    while not output_queue.empty():
        text = output_queue.get()
        process_output += text
        yield process_output, None
    
    final_result = result_container[0] if result_container else ""
    yield process_output, final_result

with gr.Blocks() as demo:
    gr.Markdown("## NotaGen")
    
    with gr.Row():
        # 左侧栏
        with gr.Column():
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
                lines=15,
                max_lines=15,
                placeholder="Generation progress will be shown here...",
                elem_classes="process-output"
            )

        # 右侧栏
        with gr.Column():
            final_output = gr.Textbox(
                label="Post-processed ABC notation scores",
                interactive=True,
                lines=23,
                placeholder="Post-processed ABC scores will be shown here...",
                elem_classes="final-output"
            )
            
            with gr.Row():
                save_btn = gr.Button("💾 Save as ABC & XML files", variant="secondary")
            
            save_status = gr.Textbox(
                label="Save Status",
                interactive=False,
                visible=True,
                max_lines=2
            )
    
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
    
    generate_btn.click(
        generate_music,
        inputs=[period_dd, composer_dd, instrument_dd],
        outputs=[process_output, final_output]
    )
    
    save_btn.click(
        save_and_convert,
        inputs=[final_output, period_dd, composer_dd, instrument_dd],
        outputs=[save_status]
    )


css = """
.process-output {
    background-color: #f0f0f0;
    font-family: monospace;
    padding: 10px;
    border-radius: 5px;
}
.final-output {
    background-color: #ffffff;
    font-family: sans-serif;
    padding: 10px;
    border-radius: 5px;
}

.process-output textarea {
    max-height: 500px !important;
    overflow-y: auto !important;
    white-space: pre-wrap;
}

"""
css += """
button#💾-save-convert:hover {
    background-color: #ffe6e6;
}
"""

demo.css = css

if __name__ == "__main__":
    # === 添加启动诊断 ===
    print("=" * 60)
    print("🚀 NotaGen 启动中...")
    print(f"📁 工作目录: {os.getcwd()}")
    
    # 检查 torch 是否可用
    try:
        print(f"🔧 PyTorch 版本: {torch.__version__}")
        print(f"🔧 CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🔧 GPU 设备: {torch.cuda.get_device_name(0)}")
            # 检查模型是否在GPU上
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🔧 GPU 总内存: {gpu_memory:.1f} GB")
    except Exception as e:
        print(f"⚠️  PyTorch 检查失败: {e}")
    
    print("=" * 60)
    
    # === 启动 Gradio 界面 ===
    try:
        demo.launch(
            server_name="127.0.0.1",  # 改为127.0.0.1
            server_port=7861,
            share=False,
            show_error=True,
            quiet=False,
            # 添加这些参数以解决Windows文件问题
            prevent_thread_lock=False,
            enable_queue=True,
            # 设置最大文件大小（如果需要上传文件的话）
            max_file_size="10mb"
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("尝试使用备用配置...")
        # 备用方案：最简配置
        demo.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            debug=False
        )
