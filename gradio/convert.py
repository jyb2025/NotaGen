import os
# ========== 关键修复：在此处设置Qt插件路径 ==========
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = r"C:\Program Files\MuseScore 4\bin\platforms"
# ===================================================

import subprocess
import fitz
from PIL import Image

# ========== 核心配置：MuseScore 可执行文件路径 ==========
MSCORE = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
# ========================================================

def abc2xml(filename_base):
    """将ABC记谱文件转换为MusicXML文件"""
    abc_filename = f"{filename_base}.abc"
    subprocess.run(
        ["python", "abc2xml.py", '-o', '.', abc_filename, ],
        check=True,
        capture_output=True,
        text=True
    )


def xml2(filename_base, target_fmt):
    """将XML文件转换为其他格式（PDF、MIDI、MP3）"""
    xml_file = filename_base + '.xml'
    if not "." in target_fmt:
        target_fmt = "." + target_fmt

    target_file = filename_base + target_fmt
    command = [MSCORE, "-o", target_file, xml_file]

    # ===== 核心修复：添加Windows隐藏窗口逻辑 =====
    startupinfo = None
    if os.name == 'nt':  # 判断是否为Windows系统
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
    # ===== 核心修复结束 =====

    # 关键：执行命令并返回结果
    result = subprocess.run(command, env=os.environ, startupinfo=startupinfo)
    
    # 添加简单调试信息
    print(f"[xml2] 已尝试转换: {xml_file} -> {target_file}")
    
    return target_file


def pdf2img(filename_base, dpi=300):
    """将PDF文件转换为图片列表"""
    pdf_path = f"{filename_base}.pdf"
    
    # 安全验证：确保文件存在
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    img_list = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # 创建高分辨率矩阵
        matrix = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=matrix)
        
        # 转换为PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_list.append(img)

    return img_list


# if __name__ == '__main__':
#     # 此处可用于独立测试
#     pass
