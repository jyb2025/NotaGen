'''
输入：一个包含 ABC 文件的文件夹
输出：每个 .abc 文件头部插入 1~12 个具体标签（从 output_with_auto_tags.csv 中按 song_name 模糊匹配）
生成：添加失败的文件后缀名为 N
'''
# python prepend_tags_to_abc.py "C:\Users\jybwo\NotaGen\data\musicabc_files" "C:\Users\jybwo\NotaGen\data\pop_filter_results\reports\output_with_auto_tags.csv"

import os
import sys
import csv
import re
import difflib
import string

def extract_song_name_from_filename(filename):
    """从文件名提取主歌名（优先取首对单引号内内容）"""
    name = filename[:-4]  # 移除 .abc
    match = re.match(r"^'([^']*)'", name)
    if match:
        return match.group(1).strip()
    return name.strip()

def normalize_for_fuzzy_match(text):
    """标准化文本用于模糊匹配：小写 + 去标点 + 空格归一"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # 保留字母、数字、空格；其余替换为空格
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # 多空格合并为单空格，去首尾空格
    text = ' '.join(text.split())
    return text

def load_tag_mapping(index_file):
    """加载原始 song_name → 标签，并预计算标准化 key 列表用于模糊匹配"""
    original_map = {}
    normalized_list = []  # 存 (normalized_name, original_name)

    with open(index_file, 'r', encoding='utf-8') as f:
        sample = f.read(1024)
        f.seek(0)
        delimiter = '\t' if '\t' in sample else ','
        reader = csv.DictReader(f, delimiter=delimiter)

        for row in reader:
            song_name = row.get('song_name', '').strip()
            auto_tags = row.get('auto_tags', '').strip()

            if not song_name or not auto_tags or auto_tags == "None":
                continue

            tags = auto_tags.split()[:12]
            tag_lines = [f"%{tag}\n" for tag in tags]
            tag_lines.append("%end\n")
            original_map[song_name] = tag_lines
            normalized_list.append((normalize_for_fuzzy_match(song_name), song_name))

    print(f"Success: Loaded tags for {len(original_map)} songs by song_name.")
    return original_map, normalized_list

def find_best_match(candidate, original_map, normalized_list, threshold=0.85):
    """
    尝试匹配 candidate：
      1. 精确匹配 original_map
      2. 模糊匹配 normalized_list
    返回 (matched_original_name, is_fuzzy)
    """
    # 1. 精确匹配
    if candidate in original_map:
        return candidate, False

    # 2. 模糊匹配
    norm_candidate = normalize_for_fuzzy_match(candidate)
    best_ratio = 0.0
    best_original = None

    for norm_name, orig_name in normalized_list:
        ratio = difflib.SequenceMatcher(None, norm_candidate, norm_name).ratio()
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_original = orig_name

    if best_original:
        return best_original, True
    else:
        return None, False

def prepend_tags_to_abc_files(base_folder, original_map, normalized_list):
    processed = 0
    skipped = 0
    renamed = 0
    missing_names = []

    for root, _, files in os.walk(base_folder):
        for filename in files:
            if not filename.lower().endswith(".abc"):
                continue

            file_path = os.path.join(root, filename)
            candidate = extract_song_name_from_filename(filename)

            matched_name, is_fuzzy = find_best_match(candidate, original_map, normalized_list)

            if matched_name is None:
                new_path = file_path + "N"
                if not os.path.exists(new_path):
                    os.rename(file_path, new_path)
                    print(f"Renamed (no match): {filename} → {os.path.basename(new_path)}")
                    missing_names.append(candidate)
                    renamed += 1
                else:
                    print(f"Warning: Skip rename (target exists): {filename}")
                continue

            # 获取标签
            tag_lines = original_map[matched_name]

            # 防重复处理
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                first_line = content.lstrip().split('\n')[0] if content.strip() else ""
                if first_line.startswith("%") and first_line != "%end":
                    print(f"Skipped (already processed): {filename}")
                    skipped += 1
                    continue

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(tag_lines)
                    f.write(content)

                match_type = "(fuzzy)" if is_fuzzy else "(exact)"
                tag_count = len(tag_lines) - 1
                print(f"Processed {match_type}: Added {tag_count} tags to {filename} "
                      f"[matched: '{matched_name}']")
                processed += 1

            except UnicodeDecodeError:
                print(f"Error: Encoding issue (skipping): {filename}")
            except Exception as e:
                print(f"Error: Failed to process {filename} - {e}")

    print("\n" + "="*50)
    print(f"Summary: Processed={processed}, Skipped={skipped}, Renamed={renamed}")

    if missing_names:
        output_file = "missing_song_names.txt"
        with open(output_file, "w", encoding='utf-8') as f:
            for name in sorted(set(missing_names)):
                f.write(name + "\n")
        print(f"\nMissing extracted names saved to: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prepend_tags_to_abc.py <abc_folder> <index_csv>")
        sys.exit(1)

    base_folder = sys.argv[1]
    index_file = sys.argv[2]

    if not os.path.isdir(base_folder):
        print(f"Error: Folder not found: {base_folder}")
        sys.exit(1)
    if not os.path.isfile(index_file):
        print(f"Error: Index file not found: {index_file}")
        sys.exit(1)

    print("=== Prepending Tags to ABC Files (by song_name with fuzzy matching) ===")
    original_map, normalized_list = load_tag_mapping(index_file)
    prepend_tags_to_abc_files(base_folder, original_map, normalized_list)