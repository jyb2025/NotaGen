# inference.py
import os
import time
import torch
import re
import difflib
import json
from utils import *
from config import *
from transformers import GPT2Config
from abctoolkit.utils import Exclaim_re, Quote_re, SquareBracket_re, Barline_regexPattern
from abctoolkit.transpose import Note_list, Pitch_sign_list
from abctoolkit.duration import calculate_bartext_duration
import logging


# === 在 inference_from_tags() 函数开头添加 ===
def inference_from_tags(tags: list):
    """从标签列表生成 ABC 音乐（4 标签库版本）"""
    # 标签验证
    VALID_TAGS_4CATEGORIES = {
        'classical', 'jazz', 'pop', 'folk', 'electronic',
        'blues', 'rock', 'hiphop', 'latin', 'christian',
        'children', 'epic', 'other',
        'piano', 'guitar', 'voice', 'strings', 'woodwinds',
        'brass', 'percussion', 'synth', 'ensemble',
        'happy', 'sad', 'calm', 'energetic', 'dramatic',
        'mysterious', 'romantic', 'heroic', 'neutral',
        'very_slow', 'slow', 'medium', 'fast', 'very_fast'
    }
    
    valid_tags = []
    for tag in tags:
        normalized = normalize_tag(tag)
        if normalized in VALID_TAGS_4CATEGORIES:
            valid_tags.append(normalized)
        else:
            logger.warning(f"Invalid tag ignored: {tag}")
    
    if not valid_tags:
        logger.warning("No valid tags provided. Using default 'classical'.")
        valid_tags = ['classical']
    
    # ... 其余代码保持不变

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Extend note list if needed
Note_list = Note_list + ['z', 'x']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize patchilizer and model
patchilizer = Patchilizer()

patch_config = GPT2Config(
    num_hidden_layers=PATCH_NUM_LAYERS,
    max_length=PATCH_LENGTH,
    max_position_embeddings=PATCH_LENGTH,
    n_embd=HIDDEN_SIZE,
    num_attention_heads=HIDDEN_SIZE // 64,
    vocab_size=1
)
byte_config = GPT2Config(
    num_hidden_layers=CHAR_NUM_LAYERS,
    max_length=PATCH_SIZE + 1,
    max_position_embeddings=PATCH_SIZE + 1,
    hidden_size=HIDDEN_SIZE,
    num_attention_heads=HIDDEN_SIZE // 64,
    vocab_size=128
)

model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=byte_config).to(device)


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    """模型量化准备（保持原逻辑）"""
    model = model.to(dtype=torch.float16)
    for param in model.parameters():
        if param.dtype == torch.float32:
            param.requires_grad = False
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


# Load model
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
logger.info(f"Parameter Number: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# === 修改核心：直接加载本地模型权重 ===
model_weights_path = INFERENCE_WEIGHTS_PATH

# 支持相对路径和绝对路径
if not os.path.isabs(model_weights_path):
    model_weights_path = os.path.join(os.getcwd(), model_weights_path)

if not os.path.exists(model_weights_path):
    raise FileNotFoundError(
        f"❌ Model weights not found at: '{model_weights_path}'\n"
        f"Please ensure:\n"
        f"  1. INFERENCE_WEIGHTS_PATH in config.py is correctly set\n"
        f"  2. The file exists at the specified local path\n"
        f"  3. Path uses forward slashes (/) or escaped backslashes (\\\\) on Windows"
    )

logger.info(f"✅ Loading model weights from: {model_weights_path}")
checkpoint = torch.load(model_weights_path, weights_only=True, map_location=device)
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()
# === 修改结束 ===


def postprocess_inst_names(abc_text):
    """标准化乐器名称"""
    try:
        with open('standard_inst_names.txt', 'r', encoding='utf-8') as f:
            standard_instruments_list = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.warning("standard_inst_names.txt not found. Skipping instrument name postprocessing.")
        return abc_text

    try:
        with open('instrument_mapping.json', 'r', encoding='utf-8') as f:
            instrument_mapping = json.load(f)
    except FileNotFoundError:
        logger.warning("instrument_mapping.json not found. Skipping instrument name mapping.")
        return abc_text

    abc_lines = abc_text.split('\n')
    abc_lines = [line + '\n' for line in abc_lines if line.strip()]

    for i, line in enumerate(abc_lines):
        if line.startswith('V:') and 'nm=' in line:
            match = re.search(r'nm="([^"]*)"', line)
            if match:
                inst_name = match.group(1)
                if inst_name in standard_instruments_list:
                    continue
                matching_key = difflib.get_close_matches(inst_name, list(instrument_mapping.keys()), n=1, cutoff=0.6)
                if matching_key:
                    replacement = instrument_mapping[matching_key[0]]
                    abc_lines[i] = line.replace(f'nm="{inst_name}"', f'nm="{replacement}"')

    return ''.join(abc_lines)


def complete_brackets(s):
    """补全未闭合的括号"""
    stack = []
    bracket_map = {'{': '}', '[': ']', '(': ')'}
    for char in s:
        if char in bracket_map:
            stack.append(char)
        elif char in bracket_map.values():
            for key, value in bracket_map.items():
                if value == char:
                    if stack and stack[-1] == key:
                        stack.pop()
                    break
    completion = ''.join(bracket_map[c] for c in reversed(stack))
    return s + completion


def rest_unreduce(abc_lines):
    """恢复被压缩的休止符"""
    tunebody_index = None
    for i in range(len(abc_lines)):
        if abc_lines[i].startswith('%%score'):
            abc_lines[i] = complete_brackets(abc_lines[i])
        if '[V:' in abc_lines[i]:
            tunebody_index = i
            break

    if tunebody_index is None:
        return abc_lines

    metadata_lines = abc_lines[:tunebody_index]
    tunebody_lines = abc_lines[tunebody_index:]

    part_symbol_list = []
    voice_group_list = []
    for line in metadata_lines:
        if line.startswith('%%score'):
            for round_bracket_match in re.findall(r'\((.*?)\)', line):
                voice_group_list.append(round_bracket_match.split())
            existed_voices = [item for sublist in voice_group_list for item in sublist]
        if line.startswith('V:'):
            symbol = line.split()[0]
            part_symbol_list.append(symbol)
            if symbol[2:] not in existed_voices:
                voice_group_list.append([symbol[2:]])

    z_symbol_list = ['V:' + group[0] for group in voice_group_list]
    x_symbol_list = ['V:' + voice for group in voice_group_list for voice in group[1:]]

    part_symbol_list.sort(key=lambda x: int(x[2:]) if x[2:].isdigit() else 0)

    unreduced_tunebody_lines = []
    ref_dur = 1  # default fallback

    for i, line in enumerate(tunebody_lines):
        unreduced_line = ''
        line = re.sub(r'^\[r:[^\]]*\]', '', line)
        pattern = r'\[V:(\d+)\](.*?)(?=\[V:|$)'
        matches = re.findall(pattern, line)
        line_bar_dict = {f'V:{match[0]}': match[1] for match in matches}

        # Calculate reference duration
        dur_dict = {}
        for symbol, bartext in line_bar_dict.items():
            right_barline = ''.join(re.split(Barline_regexPattern, bartext)[-2:])
            bartext_clean = bartext[:-len(right_barline)] if right_barline else bartext
            try:
                bar_dur = calculate_bartext_duration(bartext_clean)
                if bar_dur is not None:
                    dur_dict[bar_dur] = dur_dict.get(bar_dur, 0) + 1
            except:
                pass

        if dur_dict:
            ref_dur = max(dur_dict, key=dur_dict.get)

        prefix_left_barline = line.split('[V:')[0] if i == 0 else ''

        for symbol in part_symbol_list:
            if symbol in line_bar_dict:
                symbol_bartext = line_bar_dict[symbol]
            else:
                rest_char = 'z' if symbol in z_symbol_list else 'x'
                symbol_bartext = prefix_left_barline + rest_char + str(ref_dur) + right_barline
            unreduced_line += '[' + symbol + ']' + symbol_bartext

        unreduced_tunebody_lines.append(unreduced_line + '\n')

    return metadata_lines + unreduced_tunebody_lines


def inference_from_tags(tags: list):
    """
    从标签列表生成 ABC 音乐（支持新扩容的乐器标签）
    """
    # === 关键：标准化并验证标签 ===
    valid_tags = []
    for tag in tags:
        normalized = normalize_tag(tag)
        if normalized in model.tag_to_id:
            valid_tags.append(normalized)
    
    if not valid_tags:
        logger.warning("No valid tags provided. Using default 'classical'.")
        valid_tags = ['classical']
    
    prompt_lines = [f'%{tag}\n' for tag in valid_tags]

    while True:
        failure_flag = False
        bos_patch = [patchilizer.bos_token_id] * (PATCH_SIZE - 1) + [patchilizer.eos_token_id]
        start_time = time.time()

        prompt_patches = patchilizer.patchilize_metadata(prompt_lines)
        byte_list = list(''.join(prompt_lines))
        context_tunebody_byte_list = []
        metadata_byte_list = []

        print(''.join(byte_list), end='', flush=True)

        prompt_patches = [
            [ord(c) for c in patch] + [patchilizer.special_token_id] * (PATCH_SIZE - len(patch))
            for patch in prompt_patches
        ]
        prompt_patches.insert(0, bos_patch)
        input_patches = torch.tensor(prompt_patches, device=device).reshape(1, -1)

        tunebody_flag = False

        with torch.inference_mode():
            while True:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    predicted_patch = model.generate(
                        input_patches.unsqueeze(0),
                        tags=valid_tags,
                        top_k=TOP_K,
                        top_p=TOP_P,
                        temperature=TEMPERATURE
                    )

                # Handle first tunebody token
                if not tunebody_flag:
                    decoded = patchilizer.decode([predicted_patch])
                    # 检测乐谱区开始：声部标记或 ABC 头
                    if any(m in decoded for m in ['[V:', '[r:', 'M:', 'L:', 'K:']):
                        tunebody_flag = True
                        logger.info(f"[TUNEBODY] Detected: {decoded[:50]}")
                        # 不再强制注入 [r:0/，让模型自然生成

                # Check for end token
                if (len(predicted_patch) >= 2 and 
                    predicted_patch[0] == patchilizer.bos_token_id and 
                    predicted_patch[1] == patchilizer.eos_token_id):
                    break

                # Append generated characters
                next_patch = patchilizer.decode([predicted_patch])
                for char in next_patch:
                    byte_list.append(char)
                    if tunebody_flag:
                        context_tunebody_byte_list.append(char)
                    else:
                        metadata_byte_list.append(char)
                    print(char, end='', flush=True)

                # Pad patch to fixed length
                padded_patch = []
                eos_found = False
                for j, token in enumerate(predicted_patch):
                    if eos_found:
                        padded_patch.append(patchilizer.special_token_id)
                    else:
                        padded_patch.append(token)
                        if token == patchilizer.eos_token_id:
                            eos_found = True
                while len(padded_patch) < PATCH_SIZE:
                    padded_patch.append(patchilizer.special_token_id)

                # Update input
                predicted_tensor = torch.tensor([padded_patch], device=device)
                input_patches = torch.cat([input_patches, predicted_tensor], dim=1)

                # Safety checks
                if len(byte_list) > 102400 or (time.time() - start_time) > 600:
                    failure_flag = True
                    break

                # Streaming context window management
                if input_patches.shape[1] >= PATCH_LENGTH * PATCH_SIZE:
                    print('\nStream generating...', flush=True)
                    metadata = ''.join(metadata_byte_list)
                    context_tunebody = ''.join(context_tunebody_byte_list)
                    if '\n' not in context_tunebody:
                        break

                    lines = context_tunebody.strip().split('\n')
                    if not context_tunebody.endswith('\n'):
                        lines = [l + '\n' for l in lines[:-1]] + [lines[-1]]
                    else:
                        lines = [l + '\n' for l in lines]

                    cut_index = len(lines) // 2
                    abc_code_slice = metadata + ''.join(lines[-cut_index:])
                    input_patches = patchilizer.encode_generate(abc_code_slice)
                    input_patches = torch.tensor([item for sublist in input_patches for item in sublist], device=device).reshape(1, -1)
                    context_tunebody_byte_list = list(''.join(lines[-cut_index:]))

            if not failure_flag:
                abc_text = ''.join(byte_list)
                abc_lines = [line + '\n' for line in abc_text.split('\n') if line.strip()]
                try:
                    unreduced_abc_lines = rest_unreduce(abc_lines)
                except Exception as e:
                    logger.error(f"Rest unreduce failed: {e}")
                    failure_flag = True
                else:
                    # Remove non-%% metadata lines
                    filtered_lines = [
                        line for line in unreduced_abc_lines 
                        if not (line.startswith('%') and not line.startswith('%%'))
                    ]
                    final_abc = 'X:1\n' + ''.join(filtered_lines)
                    return final_abc

        if failure_flag:
            logger.warning("Generation failed. Retrying...")
            time.sleep(1)


if __name__ == '__main__':
    # Example usage
    result = inference_from_tags(['classical', 'solo', 'violin', 'romantic'])
    print("\nGenerated ABC:\n", result)
