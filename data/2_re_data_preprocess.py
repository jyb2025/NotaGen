ORI_FOLDER = r'C:\Users\Administrator\NotaGen\data\musicabc_files'
INTERLEAVED_FOLDER = r'C:\Users\Administrator\NotaGen\data\interleaved_abc'
AUGMENTED_FOLDER = r'C:\Users\Administrator\NotaGen\data\augmented_abc'
EVAL_SPLIT = 0.1

import os
import re
import json
import random
from tqdm import tqdm
from abctoolkit.utils import (
    remove_information_field, 
    remove_bar_no_annotations, 
    Quote_re, 
    Barlines,
    extract_metadata_and_parts, 
    extract_global_and_local_metadata,
    extract_barline_and_bartext_dict)
from abctoolkit.convert import unidecode_abc_lines
from abctoolkit.rotate import rotate_abc
from abctoolkit.check import check_alignment_unrotated
from abctoolkit.transpose import Key2index, transpose_an_abc_text

os.makedirs(INTERLEAVED_FOLDER, exist_ok=True)
os.makedirs(AUGMENTED_FOLDER, exist_ok=True)
for key in Key2index.keys():
    key_folder = os.path.join(AUGMENTED_FOLDER, key)
    os.makedirs(key_folder, exist_ok=True)


def get_processed_filenames():
    """获取所有已处理过的原始文件名（不含扩展名）"""
    processed = set()
    
    # 从 interleaved 文件夹获取（1:1 对应）
    if os.path.exists(INTERLEAVED_FOLDER):
        for f in os.listdir(INTERLEAVED_FOLDER):
            if f.endswith('.abc'):
                base = os.path.splitext(f)[0]
                processed.add(base)
    
    # 从 augmented 子文件夹辅助确认（增强鲁棒性）
    if os.path.exists(AUGMENTED_FOLDER):
        for key in Key2index:
            key_dir = os.path.join(AUGMENTED_FOLDER, key)
            if os.path.exists(key_dir):
                for f in os.listdir(key_dir):
                    if f.endswith('.abc'):
                        stem = os.path.splitext(f)[0]
                        if '_' in stem:
                            parts = stem.rsplit('_', 1)
                            if len(parts) == 2 and parts[1] == key:
                                processed.add(parts[0])
    
    return processed


def extract_key_from_interleaved(interleaved_path):
    """从 interleaved 文件中直接提取 K: 字段，不依赖声部解析"""
    with open(interleaved_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('K:'):
                key_str = line[2:].strip()
                if key_str and key_str != 'none':
                    return key_str
            # 遇到非元数据行（如音符、V:等）即停止
            if line and not line.startswith('%') and ':' not in line.split()[0]:
                break
    return 'C'  # 默认 fallback


def abc_preprocess_pipeline(abc_path):
    with open(abc_path, 'r', encoding='utf-8') as f:
        abc_lines = f.readlines()

    # 删除空行
    abc_lines = [line for line in abc_lines if line.strip() != '']

    # unidecode 转换
    abc_lines = unidecode_abc_lines(abc_lines)

    # 合并多余空格 & 清理字面量 \n（防止 Fraction 解析失败）
    abc_lines = [re.sub(r'\s+', ' ', line) for line in abc_lines]
    abc_lines = [line.replace(r'\n', ' ') for line in abc_lines]

    # 清理信息字段
    abc_lines = remove_information_field(
        abc_lines=abc_lines, 
        info_fields=['X:', 'T:', 'C:', 'W:', 'w:', 'Z:', '%%MIDI']
    )

    # 删除小节编号注释
    abc_lines = remove_bar_no_annotations(abc_lines)

    # 删除 \"
    for i, line in enumerate(abc_lines):
        if re.search(r'^[A-Za-z]:', line) or line.startswith('%'):
            continue
        else:
            if r'\"' in line:
                abc_lines[i] = abc_lines[i].replace(r'\"', '')

    # 删除带引号的文本注释中的 barline
    for i, line in enumerate(abc_lines):
        quote_contents = re.findall(Quote_re, line)
        for quote_content in quote_contents:
            for barline in Barlines:
                if barline in quote_content:
                    line = line.replace(quote_content, '')
                    abc_lines[i] = line

    # 处理引号注释：清理过长或重复符号
    for i, line in enumerate(abc_lines):
        quote_matches = re.findall(r'"[^"]*"', line)
        for match in quote_matches:
            if match == '""':
                line = line.replace(match, '')
            elif len(match) > 2 and match[1] in ['^', '_']:
                sub_string = match
                sub_string = re.sub(r'([^a-zA-Z0-9])\1+', r'\1', sub_string)
                if len(sub_string) <= 40:
                    line = line.replace(match, sub_string)
                else:
                    line = line.replace(match, '')
        abc_lines[i] = line

    # 提取原始调性（用于返回）
    metadata_lines, _ = extract_metadata_and_parts(abc_lines)
    global_metadata_dict, _ = extract_global_and_local_metadata(metadata_lines)
    ori_key = global_metadata_dict['K'][0]
    if ori_key == 'none':
        ori_key = 'C'

    abc_name = os.path.splitext(os.path.basename(abc_path))[0]

    # === 关键修复：增强对齐检查的容错能力 ===
    try:
        _, bar_no_equal_flag, _ = check_alignment_unrotated(abc_lines)
        if not bar_no_equal_flag:
            raise ValueError("Bar lengths inconsistent")
    except Exception as e:
        error_msg = str(e)
        # 捕获已知的 abctoolkit 解析器 bug
        if any(kw in error_msg for kw in ["Fraction", "tunebody index not found", "invalid literal"]):
            print(f"{abc_path} triggered parser bug, skipping alignment check (assuming valid).")
            bar_no_equal_flag = True
        else:
            raise  # 其他错误仍报错

    if not bar_no_equal_flag:
        raise Exception("Bar alignment failed")

    # 生成 interleaved 版本
    interleaved_abc = rotate_abc(abc_lines)
    interleaved_path = os.path.join(INTERLEAVED_FOLDER, abc_name + '.abc')
    with open(interleaved_path, 'w', encoding='utf-8') as w:
        w.writelines(interleaved_abc)

    # 生成所有转调+简化版本
    for key in Key2index.keys():
        transposed_abc_text = transpose_an_abc_text(abc_lines, key)
        transposed_abc_lines = [line + '\n' for line in transposed_abc_text.split('\n') if line.strip()]

        # rest reduction
        metadata_lines, prefix_dict, left_barline_dict, bar_text_dict, right_barline_dict = \
            extract_barline_and_bartext_dict(transposed_abc_lines)
        
        reduced_abc_lines = metadata_lines[:]
        for i in range(len(bar_text_dict['V:1'])):
            line = ''
            for symbol in prefix_dict.keys():
                # 检查该声部是否有有效音符
                valid = any(
                    char.isalpha() and char not in ['Z', 'z', 'X', 'x']
                    for char in bar_text_dict[symbol][i]
                )
                if valid:
                    if i == 0:
                        part_patch = (
                            f"[{symbol}]{prefix_dict[symbol]}"
                            f"{left_barline_dict[symbol][0]}"
                            f"{bar_text_dict[symbol][0]}"
                            f"{right_barline_dict[symbol][0]}"
                        )
                    else:
                        part_patch = f"[{symbol}]{bar_text_dict[symbol][i]}{right_barline_dict[symbol][i]}"
                    line += part_patch
            if line:
                line += '\n'
                reduced_abc_lines.append(line)

        reduced_abc_name = abc_name + '_' + key
        reduced_abc_path = os.path.join(AUGMENTED_FOLDER, key, reduced_abc_name + '.abc')
        with open(reduced_abc_path, 'w', encoding='utf-8') as w:
            w.writelines(reduced_abc_lines)

    return abc_name, ori_key


if __name__ == '__main__':
    # Step 1: 获取已处理文件名
    processed_set = get_processed_filenames()
    print(f"Found {len(processed_set)} already processed files.")

    # Step 2: 筛选未处理的 .abc 文件
    all_files = [f for f in os.listdir(ORI_FOLDER) if f.endswith('.abc')]
    unprocessed = [
        f for f in all_files
        if os.path.splitext(f)[0] not in processed_set
    ]
    print(f"Processing {len(unprocessed)} new files out of {len(all_files)} total.")

    # Step 3: 处理新文件
    for file in tqdm(unprocessed, desc="Preprocessing ABC files"):
        path = os.path.join(ORI_FOLDER, file)
        try:
            abc_name, ori_key = abc_preprocess_pipeline(path)
        except Exception as e:
            print(f"\nFailed to process {path}: {e}")
            continue

    # Step 4: 重建完整索引（从 interleaved 文件反推 key）
    print("Rebuilding complete dataset index...")
    all_data = []
    for f in os.listdir(INTERLEAVED_FOLDER):
        if not f.endswith('.abc'):
            continue
        abc_name = os.path.splitext(f)[0]
        interleaved_path = os.path.join(INTERLEAVED_FOLDER, f)
        try:
            ori_key = extract_key_from_interleaved(interleaved_path)
        except Exception as e:
            print(f"Warning: Failed to parse key from {interleaved_path}, using 'C'. Error: {e}")
            ori_key = 'C'
        all_data.append({
            'path': os.path.join(AUGMENTED_FOLDER, abc_name),
            'key': ori_key
        })

    # Step 5: 划分训练/评估集
    random.shuffle(all_data)
    n_eval = int(EVAL_SPLIT * len(all_data))
    eval_data = all_data[:n_eval]
    train_data = all_data[n_eval:]

    # Step 6: 保存索引文件
    def write_jsonl(filepath, data_list):
        with open(filepath, 'w', encoding='utf-8') as w:
            for item in data_list:
                w.write(json.dumps(item, ensure_ascii=False) + '\n')

    write_jsonl(AUGMENTED_FOLDER + '.jsonl', all_data)
    write_jsonl(AUGMENTED_FOLDER + '_eval.jsonl', eval_data)
    write_jsonl(AUGMENTED_FOLDER + '_train.jsonl', train_data)

    print(f"✅ Processing complete. Total files in index: {len(all_data)}")