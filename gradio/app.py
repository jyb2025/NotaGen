import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import threading
import queue
import time
import random
from io import TextIOBase
import datetime
import subprocess


# === 明确导入所需函数，避免 * 导致命名冲突或循环 ===
from utils import *
from inference import *
from convert import *

# === 新版标签库定义 ===
GENRE_LABELS = [
    'classical',    # 古典（含巴洛克、文艺复兴、中世纪、浪漫主义）
    'jazz',         # 爵士
    'pop',          # 流行
    'folk',         # 民谣（含乡村、蓝草、凯尔特）
    'electronic',   # 电子（含 EDM、New Age、迪斯科）
    'blues',        # 蓝调（含 Soul、R&B）
    'rock',         # 摇滚（含金属）
    'hiphop',       # 嘻哈（含说唱）
    'latin',        # 拉丁（含探戈、桑巴、弗拉门戈、雷鬼）
    'christian',    # 基督教音乐
    'children',     # 儿童音乐
    'epic',         # 史诗音乐
    'other',        # 其他（无法归类的，含 20th_century、contemporary、worldmusic 等）
]

INSTRUMENT_LABELS = [
    'piano',        # 钢琴（含键盘、大键琴）
    'guitar',       # 吉他（含原声、电吉他）
    'voice',        # 人声（含声乐、各声部）
    'strings',      # 弦乐（含小提琴、中提琴、大提琴、低音提琴）
    'woodwinds',    # 木管（含长笛、单簧管、双簧管、巴松管、萨克斯）
    'brass',        # 铜管（含小号、圆号、长号、大号）
    'percussion',   # 打击乐（含鼓）
    'synth',        # 合成器
    'ensemble',     # 合奏（含二重奏、三重奏、四重奏、管弦乐队、室内乐）
]

EMOTION_LABELS = [
    'happy',        # 快乐（含嬉戏、欢乐）
    'sad',          # 悲伤（含忧郁、怀旧、哀悼）
    'calm',         # 平静（含宁静、温柔、优雅、安详）
    'energetic',    # 充满活力（含激进、强烈、热情）
    'dramatic',     # 戏剧性（含紧张、庄重、恐怖）
    'mysterious',   # 神秘（含梦幻）
    'romantic',     # 浪漫（含爱意）
    'heroic',       # 英雄（含史诗、胜利）
    'neutral',      # 中性（占位符，含适中）
]

TEMPO_LABELS = [
    'very_slow',    # 极慢（Largo、Grave，BPM < 60）
    'slow',         # 慢（Adagio、Andante，BPM 60-80）
    'medium',       # 中速（Moderato，BPM 80-120）
    'fast',         # 快（Allegro、Vivace，BPM 120-160）
    'very_fast',    # 极快（Presto，BPM > 160）
]

# === 新版标签分类定义（带详细说明）===
TAG_CATEGORIES = {
    "🎵 音乐流派 (Genre)": {
        'classical': '古典 (含巴洛克、文艺复兴、中世纪、浪漫主义)',
        'jazz': '爵士',
        'pop': '流行',
        'folk': '民谣 (含乡村、蓝草、凯尔特)',
        'electronic': '电子 (含 EDM、新世纪、迪斯科)',
        'blues': '蓝调 (含灵魂乐、节奏布鲁斯)',
        'rock': '摇滚 (含金属)',
        'hiphop': '嘻哈 (含说唱)',
        'latin': '拉丁 (含探戈、桑巴、弗拉门戈、雷鬼)',
        'christian': '基督教音乐',
        'children': '儿童音乐',
        'epic': '史诗音乐',
        'other': '其他 (20世纪、当代、世界音乐等)',
    },
    
    "🎻 乐器编制 (Instrument)": {
        'piano': '钢琴 (含键盘、大键琴)',
        'guitar': '吉他 (含原声、电吉他)',
        'voice': '人声 (含声乐、各声部)',
        'strings': '弦乐 (含小提琴、中提琴、大提琴、低音提琴)',
        'woodwinds': '木管 (含长笛、单簧管、双簧管、巴松管、萨克斯)',
        'brass': '铜管 (含小号、圆号、长号、大号)',
        'percussion': '打击乐 (含鼓)',
        'synth': '合成器',
        'ensemble': '合奏 (含二重奏、三重奏、四重奏、管弦乐队、室内乐)',
    },

    "😊 情绪情感 (Emotion)": {
        'happy': '快乐 (含嬉戏、欢乐)',
        'sad': '悲伤 (含忧郁、怀旧、哀悼)',
        'calm': '平静 (含宁静、温柔、优雅、安详)',
        'energetic': '充满活力 (含激进、强烈、热情)',
        'dramatic': '戏剧性 (含紧张、庄重、恐怖)',
        'mysterious': '神秘 (含梦幻)',
        'romantic': '浪漫 (含爱意)',
        'heroic': '英雄 (含史诗、胜利)',
        'neutral': '中性 (含适中)',
    },

    "⏱️ 速度标记 (Tempo)": {
        'very_slow': '极慢 (Largo/Grave, BPM < 60)',
        'slow': '慢 (Adagio/Andante, BPM 60-80)',
        'medium': '中速 (Moderato, BPM 80-120)',
        'fast': '快 (Allegro/Vivace, BPM 120-160)',
        'very_fast': '极快 (Presto, BPM > 160)',
    },
}

# 构建平铺翻译字典（用于显示）
TAG_TRANSLATIONS = {}
for category, tags in TAG_CATEGORIES.items():
    for tag_en, tag_cn in tags.items():
        TAG_TRANSLATIONS[tag_en] = tag_cn

# 所有标签列表 —— 严格按照 UI 显示顺序构建
ALL_TAGS = []
for category, tags in TAG_CATEGORIES.items():
    for tag in tags.keys():
        ALL_TAGS.append(tag)

# 验证无重复
from collections import Counter
counts = Counter(ALL_TAGS)
duplicates = [tag for tag, cnt in counts.items() if cnt > 1]
if duplicates:
    raise ValueError(f"Duplicate tag keys found: {duplicates}")

# 有效标签集合（用于验证）
VALID_TAGS = set(ALL_TAGS)

title_html = """
<div class="title-container">
    <h1 class="title-text">NotaGen - 标签条件生成 (新版标签库)</h1> &nbsp;
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
</div>
<p style="font-size: 1.2em;">选择最多4个音乐标签，模型将根据这些标签生成对应的乐谱！</p>
<p style="font-size: 1em; color: #666;">🎯 新版标签库：简化分类，更直观易用（共36个标签）</p>
"""

class RealtimeStream(TextIOBase):
    def __init__(self, queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)
        return len(text)

def convert_files_from_tags(abc_content, tags):
    """基于标签列表保存文件"""
    if not tags:
        raise gr.Error("Please provide valid tags for generation")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_str = "_".join(tags[:3]) if tags else "no_tags"
    filename_base = f"{timestamp}_{tag_str}"

    abc_filename = f"{filename_base}.abc"
    with open(abc_filename, "w", encoding="utf-8") as f:
        f.write(abc_content)

    postprocessed_inst_abc = postprocess_inst_names(abc_content)
    filename_base_postinst = f"{filename_base}_postinst"
    with open(filename_base_postinst + ".abc", "w", encoding="utf-8") as f:
        f.write(postprocessed_inst_abc)

    file_paths = {'abc': abc_filename}
    try:
        abc2xml(filename_base)
        abc2xml(filename_base_postinst)
        xml2(filename_base, 'pdf')
        xml2(filename_base, 'mid')
        xml2(filename_base_postinst, 'mid')
        xml2(filename_base, 'mp3')
        xml2(filename_base_postinst, 'mp3')
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

def update_selected_tags(*selected_checkboxes):
    """更新选中的标签显示"""
    selected_tags = []
    for i, tag_en in enumerate(ALL_TAGS):
        if i < len(selected_checkboxes) and selected_checkboxes[i]:
            selected_tags.append(tag_en)
    
    if len(selected_tags) > 12:
        selected_tags = selected_tags[:12]
    
    html_parts = []
    for tag in selected_tags:
        cn_text = TAG_TRANSLATIONS.get(tag, tag)
        # 只显示中文描述的主要部分（括号前的内容）
        main_cn = cn_text.split(' (')[0] if ' (' in cn_text else cn_text
        html_parts.append(f'<span class="tag"><span class="tag-en">{tag}</span><span class="tag-cn">({main_cn})</span></span>')
    
    display_html = f"""
    <div id='tag-display' style='min-height: 60px; padding: 10px; background: #f9f9f9; border-radius: 8px;'>
        {''.join(html_parts)}
    </div>
    """
    
    tag_string = " ".join(selected_tags)
    return display_html, tag_string

def update_tag_display(tag_text):
    """从文本框更新标签显示"""
    if not tag_text:
        return "<div id='tag-display' style='min-height: 60px; padding: 10px; background: #f9f9f9; border-radius: 8px;'></div>"
    
    tags = tag_text.strip().split()[:12]
    html_parts = []
    for tag in tags:
        if tag in VALID_TAGS:
            cn_text = TAG_TRANSLATIONS.get(tag, tag)
            main_cn = cn_text.split(' (')[0] if ' (' in cn_text else cn_text
            html_parts.append(f'<span class="tag"><span class="tag-en">{tag}</span><span class="tag-cn">({main_cn})</span></span>')
        else:
            html_parts.append(f'<span class="tag" style="background:#ffebee; color:#c62828;"><span class="tag-en">{tag}</span><span class="tag-cn">(无效标签)</span></span>')
    
    return f"""
    <div id='tag-display' style='min-height: 60px; padding: 10px; background: #f9f9f9; border-radius: 8px;'>
        {''.join(html_parts)}
    </div>
    """

def update_page(direction, data):
    if not data:
        return None, gr.update(interactive=False), gr.update(interactive=False), data

    if direction == "prev" and data['current_page'] > 0:
        data['current_page'] -= 1
    elif direction == "next" and data['current_page'] < data['pages'] - 1:
        data['current_page'] += 1

    current_page_index = data['current_page']
    new_image = f"{data['base']}_page_{current_page_index+1}.png"
    prev_btn_state = gr.update(interactive=(current_page_index > 0))
    next_btn_state = gr.update(interactive=(current_page_index < data['pages'] - 1))

    return new_image, prev_btn_state, next_btn_state, data

def generate_music_from_tags(tag_input_text):
    if not tag_input_text.strip():
        raise gr.Error("Please select at least one tag!")
    
    tags = tag_input_text.strip().split()[:12]
    valid_tags = []
    invalid_tags = []
    
    for tag in tags:
        if tag in VALID_TAGS:
            valid_tags.append(tag)
        else:
            invalid_tags.append(tag)
    
    if invalid_tags:
        raise gr.Error(f"Invalid tags found: {', '.join(invalid_tags)}. Please only use tags from the provided list.")
    
    if not valid_tags:
        raise gr.Error("No valid tags found! Please use tags from the provided list.")
    
    random_seed = int(time.time()) % 10000
    random.seed(random_seed)
    try:
        import numpy as np
        np.random.seed(random_seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
    except ImportError:
        pass

    output_queue = queue.Queue()
    original_stdout = sys.stdout
    sys.stdout = RealtimeStream(output_queue)

    result_container = []

    def run_inference():
        try:
            result = inference_from_tags(valid_tags)
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

    while thread.is_alive():
        try:
            text = output_queue.get(timeout=0.1)
            process_output += text
            yield process_output, final_output_abc, pdf_image, audio_file, pdf_state, gr.update(value=None, visible=False)
        except queue.Empty:
            continue

    while not output_queue.empty():
        text = output_queue.get()
        process_output += text

    final_result = result_container[0] if result_container else ""
    final_output_abc = "Converting files..."
    yield process_output, final_output_abc, pdf_image, audio_file, pdf_state, gr.update(value=None, visible=False)

    try:
        file_paths = convert_files_from_tags(final_result, valid_tags)
        final_output_abc = final_result
        if file_paths['pages'] > 0:
            pdf_image = f"{file_paths['base']}_page_1.png"
        audio_file = file_paths['mp3']
        pdf_state = file_paths
        
        download_list = []
        for ext in ['abc', 'xml', 'pdf', 'mid', 'mp3']:
            if ext in file_paths and os.path.exists(file_paths[ext]):
                download_list.append(file_paths[ext])
    except Exception as e:
        yield process_output, f"Error converting files: {str(e)}", None, None, None, gr.update(value=None, visible=False)
        return

    yield process_output, final_output_abc, pdf_image, audio_file, pdf_state, gr.update(value=download_list, visible=True)

css = """
#tag-input {
    font-size: 16px !important;
    padding: 12px !important;
}

#tag-display .tag {
    display: inline-block;
    margin: 4px;
    padding: 6px 12px;
    background: #e3f2fd;
    border-radius: 16px;
    font-size: 14px;
    color: #1976d2;
}

#tag-display .tag-en {
    font-weight: bold;
    margin-right: 8px;
}

#tag-display .tag-cn {
    color: #666;
    font-size: 13px;
}

.tag-limit-hint {
    background: #fff3cd;
    border: 1px solid #ffecb5;
    color: #856404;
    padding: 8px 12px;
    border-radius: 6px;
    margin: 10px 0;
    font-size: 13px;
}

#pdf-preview {
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.page-btn {
    padding: 12px !important;
    margin: auto !important;
}

.page-btn:hover {
    background: #f0f0f0 !important;
    transform: scale(1.05);
}

.gr-row {
    gap: 10px !important;
}

.download-files {
    margin-top: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

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

.clear-tags-btn {
    margin-left: 10px;
    padding: 6px 12px !important;
    font-size: 13px !important;
}

.tag-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 10px;
    margin: 10px 0;
}

.tag-item {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 8px 12px;
    transition: all 0.2s;
}

.tag-item:hover {
    border-color: #1976d2;
    box-shadow: 0 2px 8px rgba(25, 118, 210, 0.1);
}

.tag-checkbox {
    margin: 0 !important;
}

.tag-checkbox label {
    font-size: 14px !important;
    color: #2c3e50 !important;
}

.tag-checkbox label span {
    font-weight: normal !important;
}

.category-header {
    margin-top: 20px !important;
    margin-bottom: 10px !important;
    color: #1976d2 !important;
    font-size: 1.3em !important;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 5px;
}

.stats-badge {
    background: #e3f2fd;
    color: #1976d2;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    margin-left: 10px;
}
"""

def create_checkbox_for_tag(tag_en, tag_cn):
    """为标签创建复选框组件"""
    # 提取主要中文描述（括号前的内容）
    main_cn = tag_cn.split(' (')[0] if ' (' in tag_cn else tag_cn
    return gr.Checkbox(
        label=f"**{tag_en}** — {main_cn}",
        value=False,
        elem_classes=["tag-checkbox"],
        elem_id=f"checkbox_{tag_en}"
    )

with gr.Blocks(css=css, title="NotaGen - 标签条件生成") as demo:
    gr.HTML(title_html)
    pdf_state = gr.State()
    
    tag_input_hidden = gr.Textbox(value="", visible=False, elem_id="tag-input-hidden")

    with gr.Column():
        gr.Markdown("### 🏷️ 选择标签（最多选择4个）")
        
        gr.Markdown("""
        <div class="tag-limit-hint">
        ⚠️ 最多可以选择4个标签。
        </div>
        """)
        
        # 添加新版标签库说明和统计
        with gr.Row():
            gr.Markdown(f"""
            <div style="background: #f0f7ff; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <p style="margin: 0 0 10px 0; color: #1976d2; font-weight: bold;">✨ 新版标签库（共36个标签）</p>
                <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                    <span class="stats-badge">🎵 流派: 13个</span>
                    <span class="stats-badge">🎻 乐器: 9个</span>
                    <span class="stats-badge">😊 情绪: 9个</span>
                    <span class="stats-badge">⏱️ 速度: 5个</span>
                </div>
            </div>
            """)
        
        checkbox_components = []
        
        # 遍历所有类别生成复选框
        for category, tags in TAG_CATEGORIES.items():
            gr.Markdown(f"#### {category}", elem_classes=["category-header"])
            
            # 使用网格布局显示标签
            with gr.Column(elem_classes=["tag-grid"]):
                for tag_en, tag_cn in tags.items():
                    with gr.Column(elem_classes=["tag-item"]):
                        checkbox = create_checkbox_for_tag(tag_en, tag_cn)
                        checkbox_components.append(checkbox)
        
        # 验证复选框数量与标签数量一致
        assert len(checkbox_components) == len(ALL_TAGS), f"Length mismatch: {len(checkbox_components)} vs {len(ALL_TAGS)}"
        
        tag_display = gr.HTML(
            value="<div id='tag-display' style='min-height: 60px; padding: 10px; background: #f9f9f9; border-radius: 8px;'></div>",
            elem_id="tag-display"
        )
        
        with gr.Row():
            generate_btn = gr.Button("🎵 Generate Music", variant="primary", size="lg", scale=2)
            clear_btn = gr.Button("🗑️ Clear Selection", variant="secondary", elem_classes="clear-tags-btn", scale=1)
        
        process_output = gr.Textbox(
            label="Generation process",
            interactive=False,
            lines=2,
            max_lines=2,
            placeholder="Generation progress will be shown here..."
        )

        final_output = gr.Textbox(
            label="Generated ABC notation scores",
            interactive=True,
            lines=8,
            max_lines=8,
            placeholder="Generated ABC scores will be shown here..."
        )

        audio_player = gr.Audio(
            label="Audio Preview",
            format="mp3",
            interactive=False,
        )

    with gr.Column():
        pdf_image = gr.Image(
            label="Sheet Music Preview",
            show_label=False,
            height=650,
            type="filepath",
            elem_id="pdf-preview",
            interactive=False
        )

        with gr.Row():
            prev_btn = gr.Button("⬅️ Last Page", variant="secondary", size="sm", elem_classes="page-btn")
            next_btn = gr.Button("Next Page ➡️", variant="secondary", size="sm", elem_classes="page-btn")

    with gr.Column():
        gr.Markdown("**Download Files:**")
        download_files = gr.Files(
            label="Generated Files", 
            visible=False,
            elem_classes="download-files",
            type="filepath"
        )

    def clear_all_checkboxes():
        return [False] * len(checkbox_components), "", "<div id='tag-display' style='min-height: 60px; padding: 10px; background: #f9f9f9; border-radius: 8px;'></div>"
    
    for checkbox in checkbox_components:
        checkbox.change(
            update_selected_tags,
            inputs=checkbox_components,
            outputs=[tag_display, tag_input_hidden]
        )
    
    clear_btn.click(
        clear_all_checkboxes,
        outputs=checkbox_components + [tag_input_hidden, tag_display]
    )
    
    generate_btn.click(
        generate_music_from_tags,
        inputs=[tag_input_hidden],
        outputs=[process_output, final_output, pdf_image, audio_player, pdf_state, download_files]
    )

    prev_signal = gr.Textbox(value="prev", visible=False)
    next_signal = gr.Textbox(value="next", visible=False)

    prev_btn.click(update_page, inputs=[prev_signal, pdf_state], outputs=[pdf_image, prev_btn, next_btn, pdf_state])
    next_btn.click(update_page, inputs=[next_signal, pdf_state], outputs=[pdf_image, prev_btn, next_btn, pdf_state])

if __name__ == "__main__":
    print("=" * 60)
    print("Starting NotaGen tag-based generation server...")
    print("=" * 60)
    print(f"Access the application at: http://localhost:7860")
    print(f"\n📊 New Tag Library Statistics:")
    print(f"   ├─ 🎵 Genre: {len(GENRE_LABELS)} tags")
    print(f"   ├─ 🎻 Instrument: {len(INSTRUMENT_LABELS)} tags")
    print(f"   ├─ 😊 Emotion: {len(EMOTION_LABELS)} tags")
    print(f"   └─ ⏱️ Tempo: {len(TEMPO_LABELS)} tags")
    print(f"\n📌 Total: {len(ALL_TAGS)} tags available for generation")
    print("=" * 60)
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        css=css
    )
