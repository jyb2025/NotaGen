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

from inference import inference_from_tags, postprocess_inst_names
from convert import abc2xml, xml2, pdf2img

# === 分类标签定义（中英文双语）===
TAG_CATEGORIES = {
    "🎼 主要流派": {
        'classical': '古典',
        'jazz': '爵士',
        'rock': '摇滚',
        'pop': '流行',
        'folk': '民谣',
        'reggae': '雷鬼',
        'rap': '说唱',
        'country': '乡村',
        'blues': '蓝调',
        'electronic': '电子',
        'hiphop': '嘻哈',
        'metal': '金属', 
        'edm': '电子舞曲',
        'r&b': '节奏布鲁斯',
        'world': '世界音乐',
        'christian': '基督教音乐',
        'children': '儿童音乐',
        'disco': '迪斯科',
        'soul': '灵魂',     
        'experimental': '实验音乐',
        'latin': '拉丁', 
        'newage': '新世纪'
    },
    
    "⚙️ 技术特征": {
        'very_simple': '极简',
        'simple': '简单',
        'medium': '中等',
        'complex': '复杂',
        'very_complex': '极复杂',
        'very_slow': '极慢',
        'slow': '慢',
        'fast': '快',
        'very_fast': '极快',
        'very_soft': '极弱',
        'soft': '弱',
        'loud': '强',
        'very_loud': '极强',
        'legato': '连奏',
        'staccato': '断奏',
        'mixed': '混合',
        'syncopated': '切分',
        'irregular': '不规则',
        'diatonic': '自然音阶',
        'chromatic': '半音阶',
        'modal': '调式',
        'atonal': '无调性',
        'jazz_harmony': '爵士和声',
        'monophonic': '单声部',
        'homophonic': '主调',
        'polyphonic': '复调',
        'heterophonic': '异音同奏',
        'binary': '二部',
        'ternary': '三部',
        'rondo': '回旋',
        'theme_variations': '主题变奏',
        'through_composed': '通谱'
    },
    
    "🎹 乐器相关": {
        'solo': '独奏',
        'duet': '二重奏',
        'trio': '三重奏',
        'quartet': '四重奏',
        'small_ensemble': '小编制',
        'large_ensemble': '大编制',
        'orchestra': '管弦乐队',
        'strings': '弦乐',
        'woodwinds': '木管',
        'brass': '铜管',
        'percussion': '打击乐',
        'keyboard': '键盘',
        'voice': '人声',
        'piano': '钢琴',
        'guitar': '吉他',
        'ukulele': '尤克里里',
        'violin': '小提琴',
        'viola': '中提琴',
        'cello': '大提琴',
        'flute': '长笛',
        'clarinet': '单簧管',      
        'oboe': '双簧管',
        'trumpet': '小号',
        'saxophone': '萨克斯',
        'drums': '鼓',
        'bass': '贝斯',
        'organ': '管风琴',
        'harp': '竖琴',
        'dizi': '笛子',
        'accordion': '手风琴',
        'mandolin': '曼陀林',
        'banjo': '班卓琴',
        'harmonica': '口琴'    
    },
    
    "😊 情绪情感": {
        'happy': '快乐',
        'sad': '悲伤',
        'angry': '愤怒',
        'peaceful': '宁静',
        'energetic': '充满活力',
        'melancholic': '忧郁',
        'romantic': '浪漫',
        'dramatic': '戏剧性',
        'gentle': '绅士',
        'calm': '平静',
        'moderate': '适中',
        'intense': '强烈',
        'passionate': '热情',
        'tense': '紧张',
        'playful': '嬉戏',
        'solemn': '庄重',
        'mysterious': '神秘',
        'heroic': '英雄',
        'nostalgic': '怀旧',
        'dreamy': '梦幻',
        'aggressive': '激进',
        'graceful': '优雅',
        'horrifying': '震惊'
    },
    
    "🌍 文化地域": {
        'europe': '欧洲',
        'north_america': '北美',
        'south_america': '南美',
        'asia': '亚洲',
        'africa': '非洲',
        'middle_east': '中东',
        'oceania': '大洋洲',
        'medieval': '中世纪',
        'renaissance': '文艺复兴',
        'baroque': '巴洛克',
        'classical': '古典',
        'romantic': '浪漫',
        '20th_century': '20世纪',
        'contemporary': '当代',
        'celtic': '凯尔特',
        'flamenco': '弗拉门戈',
        'tango': '探戈',
        'samba': '桑巴',        
        'bluegrass': '蓝草',
        'klezmer': '克莱兹默',
        'gamelan': '甘美兰'
    },
    
    "🎯 功能用途": {
        'etude': '练习曲',
        'scale_exercise': '音阶练习',
        'recital': '独奏会',
        'competition': '比赛',
        'audition': '试音',
        'worship': '崇拜',
        'ceremonial': '典礼',
        'dance_accompaniment': '舞蹈伴奏',
        'background': '背景音乐',
        'focus': '专注',
        'relaxation': '放松',
        'meditation': '冥想',
        'workout': '健身',
        'party': '派对'
    }
}

# 从分类字典构建平铺的翻译字典（用于验证）
TAG_TRANSLATIONS = {}
for category, tags in TAG_CATEGORIES.items():
    TAG_TRANSLATIONS.update(tags)

# 构建所有标签的列表
ALL_TAGS = list(TAG_TRANSLATIONS.keys())

title_html = """
<div class="title-container">
    <h1 class="title-text">NotaGen - 标签条件生成</h1> &nbsp;
        <!-- ArXiv -->
        <a href="https://arxiv.org/abs/2502.18008   ">
            <img src="https://img.shields.io/badge/NotaGen_Paper-ArXiv-%23B31B1B?logo=arxiv&logoColor=white   " alt="Paper">
        </a>
        &nbsp;
        <!-- GitHub -->
        <a href="https://github.com/ElectricAlexis/NotaGen   ">
            <img src="https://img.shields.io/badge/NotaGen_Code-GitHub-%23181717?logo=github&logoColor=white   " alt="GitHub">
        </a>
        &nbsp;
        <!-- HuggingFace -->
        <a href="https://huggingface.co/ElectricAlexis/NotaGen   ">
            <img src="https://img.shields.io/badge/NotaGen_Weights-HuggingFace-%23FFD21F?logo=huggingface&logoColor=white   " alt="Weights">
        </a>
</div>
<p style="font-size: 1.2em;">选择最多 12 个音乐标签，模型将根据这些标签生成对应的乐谱！</p>
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
    # selected_checkboxes 是一个包含所有复选框状态的元组
    
    # 过滤出选中的复选框
    selected_tags = []
    for i, tag_en in enumerate(ALL_TAGS):
        if i < len(selected_checkboxes) and selected_checkboxes[i]:  # 如果标签被选中
            selected_tags.append(tag_en)
    
    # 限制最多12个标签
    if len(selected_tags) > 12:
        selected_tags = selected_tags[:12]
        # 这里可以添加一个警告提示，但为了简单起见，我们只截取前12个
    
    # 生成标签显示HTML
    html_parts = []
    for tag in selected_tags:
        cn_text = TAG_TRANSLATIONS.get(tag, tag)
        html_parts.append(f'<span class="tag"><span class="tag-en">{tag}</span><span class="tag-cn">({cn_text})</span></span>')
    
    display_html = f"""
    <div id='tag-display' style='min-height: 60px; padding: 10px; background: #f9f9f9; border-radius: 8px;'>
        {''.join(html_parts)}
    </div>
    """
    
    # 将选中的标签转换为空格分隔的字符串
    tag_string = " ".join(selected_tags)
    
    return display_html, tag_string

def update_tag_display(tag_text):
    """兼容性函数：从文本框更新标签显示"""
    if not tag_text:
        return "<div id='tag-display' style='min-height: 60px; padding: 10px; background: #f9f9f9; border-radius: 8px;'></div>"
    
    tags = tag_text.strip().split()[:12]
    html_parts = []
    for tag in tags:
        tag_lower = tag.lower()
        cn_text = TAG_TRANSLATIONS.get(tag_lower, tag)
        html_parts.append(f'<span class="tag"><span class="tag-en">{tag}</span><span class="tag-cn">({cn_text})</span></span>')
    
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
    for tag in tags:
        if tag.lower() in TAG_TRANSLATIONS:
            valid_tags.append(tag.lower())
        else:
            print(f"Warning: Unknown tag '{tag}'")
    
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

/* 标签参考区域样式 */
.tag-reference {
    font-size: 11px;
    line-height: 1.4;
    max-height: 400px; /* 增加高度以容纳更多内容 */
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 6px;
    background: #fafafa;
    margin-bottom: 15px;
}

.tag-reference h3 {
    margin-top: 15px;
    margin-bottom: 8px;
    color: #2c3e50;
    font-size: 16px;
    border-bottom: 1px solid #eee;
    padding-bottom: 4px;
}

.tag-category {
    margin-bottom: 15px;
    display: flex;
    flex-wrap: wrap;
}

.tag-checkbox-container {
    display: flex;
    align-items: center;
    margin: 4px 8px 4px 0;
    padding: 4px 8px;
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s;
}

.tag-checkbox-container:hover {
    background: #f5f5f5;
    border-color: #1976d2;
}

.tag-checkbox-container.selected {
    background: #e3f2fd;
    border-color: #1976d2;
}

.tag-checkbox {
    margin-right: 6px !important;
    cursor: pointer;
}

.tag-text {
    font-family: monospace;
    color: #d35400;
    font-weight: bold;
    margin-right: 4px;
}

.tag-translation {
    color: #666;
    font-size: 11px;
}

/* 选中的标签限制提示 */
.tag-limit-hint {
    background: #fff3cd;
    border: 1px solid #ffecb5;
    color: #856404;
    padding: 8px 12px;
    border-radius: 6px;
    margin: 10px 0;
    font-size: 13px;
}

.tag-search-box {
    margin-bottom: 15px;
}

button[size="sm"] {
    padding: 4px 8px !important;
    margin: 2px !important;
    min-width: 60px;
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

.audio-panel {
    margin-top: 15px !important;
    max-width: 400px;
}

#audio-preview audio {
    height: 200px !important;
}

.save-as-row {
    margin-top: 15px;
    padding: 10px;
    border-top: 1px solid #eee;
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
"""

# 注意：将 css 参数移到 launch() 方法中
with gr.Blocks() as demo:
    gr.HTML(title_html)
    pdf_state = gr.State()
    
    # 隐藏的文本框，用于存储选中的标签字符串
    tag_input_hidden = gr.Textbox(
        value="",
        visible=False,
        elem_id="tag-input-hidden"
    )

    with gr.Column():
        # === 修改后的标签参考区域 ===
        gr.Markdown("### 🏷️ 选择标签（最多选择12个）")
        
        # 标签限制提示
        gr.Markdown("""
        <div class="tag-limit-hint">
        ⚠️ 最多可以选择12个标签。超过12个时，将自动选择前12个。
        </div>
        """)
        
        # 创建所有复选框
        checkbox_components = []
        
        # 按类别创建复选框
        for category, tags in TAG_CATEGORIES.items():
            with gr.Group():
                gr.Markdown(f"#### {category}")
                with gr.Row():
                    for i, (tag_en, tag_cn) in enumerate(tags.items()):
                        if i % 6 == 0 and i > 0:
                            gr.Markdown("", visible=False)  # 换行占位符
                        checkbox = gr.Checkbox(
                            label=f"**{tag_en}** ({tag_cn})",
                            value=False,
                            elem_classes=["tag-checkbox"],
                            elem_id=f"checkbox_{tag_en}"
                        )
                        checkbox_components.append(checkbox)
        
        # 选中的标签显示区域
        tag_display = gr.HTML(
            value="<div id='tag-display' style='min-height: 60px; padding: 10px; background: #f9f9f9; border-radius: 8px;'></div>",
            elem_id="tag-display"
        )
        
        # 生成按钮
        generate_btn = gr.Button("Generate Music", variant="primary", size="lg")
        
        # 清空选择按钮
        with gr.Row():
            clear_btn = gr.Button("Clear Selection", variant="secondary", elem_classes="clear-tags-btn")
        
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
            type="filepath"
        )

    # 定义清空选择按钮的功能
    def clear_all_checkboxes():
        # 返回所有复选框的False值
        return [False] * len(checkbox_components), "", "<div id='tag-display' style='min-height: 60px; padding: 10px; background: #f9f9f9; border-radius: 8px;'></div>"
    
    # 复选框变化时更新显示
    for checkbox in checkbox_components:
        checkbox.change(
            update_selected_tags,
            inputs=checkbox_components,  # 传入所有复选框的状态
            outputs=[tag_display, tag_input_hidden]
        )
    
    # 清空按钮点击事件
    clear_btn.click(
        clear_all_checkboxes,
        outputs=checkbox_components + [tag_input_hidden, tag_display]
    )
    
    # 生成按钮点击事件（使用隐藏的标签输入）
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
    print("Starting NotaGen tag-based generation server locally...")
    print(f"Access the application at: http://localhost:7860")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        css=css  # 将 css 参数移到这里
    )
