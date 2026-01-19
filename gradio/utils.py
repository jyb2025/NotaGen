import torch
import random
import bisect
import json
import re
import numpy as np
from config import *
from transformers import GPT2Model, GPT2LMHeadModel, PreTrainedModel
from samplings import top_p_sampling, top_k_sampling, temperature_sampling


# === 构建完整的标签词汇表（NotaGen-X 使用）===
def _build_tag_vocab():
    tags = []
    # 1. 主要流派（22种）
    tags.extend([
        'classical', 'jazz', 'rock', 'pop', 'folk', 'reggae', 'rap', 'country',
        'blues', 'electronic', 'hiphop', 'metal', 'edm', 'r&b', 'world', 'christian',
        'children', 'disco', 'soul', 'experimental', 'latin', 'newage',
    ])
    # 2. 技术特征（36种）
    tags.extend(['very_simple', 'simple', 'medium', 'complex', 'very_complex'])
    tags.extend(['very_slow', 'slow', 'medium', 'fast', 'very_fast'])
    tags.extend(['very_soft', 'soft', 'medium', 'loud', 'very_loud'])
    tags.extend(['legato', 'staccato', 'mixed'])
    tags.extend(['simple', 'syncopated', 'complex', 'irregular'])
    tags.extend(['diatonic', 'chromatic', 'modal', 'atonal', 'jazz_harmony'])
    tags.extend(['monophonic', 'homophonic', 'polyphonic', 'heterophonic'])
    tags.extend(['binary', 'ternary', 'rondo', 'theme_variations', 'through_composed'])
    # 3. 乐器相关（32种）
    tags.extend(['solo', 'duet', 'trio', 'quartet', 'small_ensemble', 'large_ensemble', 'orchestra'])
    tags.extend(['strings', 'woodwinds', 'brass', 'percussion', 'keyboard', 'voice', 'electronic'])
    tags.extend([
        'piano', 'guitar', 'violin', 'cello', 'flute', 'clarinet', 'viola', 'ukulele',
        'trumpet', 'saxophone', 'drums', 'bass', 'organ', 'harp', 'dizi',
        'accordion', 'mandolin', 'banjo', 'harmonica', 'oboe'
    ])
    # 4. 情绪情感（23种）
    tags.extend(['happy', 'sad', 'angry', 'peaceful', 'energetic', 'melancholic', 'romantic', 'dramatic', 'gentle'])
    tags.extend(['calm', 'moderate', 'intense', 'passionate', 'tense'])
    tags.extend(['playful', 'solemn', 'mysterious', 'heroic', 'nostalgic', 'dreamy', 'aggressive', 'graceful', 'horrifying'])
    # 5. 文化地域（21种）
    tags.extend(['europe', 'north_america', 'south_america', 'asia', 'africa', 'middle_east', 'oceania'])
    tags.extend(['medieval', 'renaissance', 'baroque', 'classical', 'romantic', '20th_century', 'contemporary'])
    tags.extend(['celtic', 'flamenco', 'tango', 'samba', 'bluegrass', 'klezmer', 'gamelan'])
    # 6. 功能用途（14种）
    tags.extend(['etude', 'scale_exercise'])
    tags.extend(['recital', 'competition', 'audition', 'worship', 'ceremonial', 'dance_accompaniment'])
    tags.extend(['background', 'focus', 'relaxation', 'meditation', 'workout', 'party'])
    return {tag: idx for idx, tag in enumerate(tags)}


class Patchilizer:
    def __init__(self, stream=PATCH_STREAM):
        self.stream = stream
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.special_token_id = 0

    def split_bars(self, body_lines):
        """
        Split a body of music into individual bars.
        """
        new_bars = []
        try:
            for line in body_lines:
                line_bars = re.split(self.regexPattern, line)
                line_bars = list(filter(None, line_bars))
                new_line_bars = []

                if len(line_bars) == 1:
                    new_line_bars = line_bars
                else:
                    if line_bars[0] in self.delimiters:
                        new_line_bars = [line_bars[i] + line_bars[i + 1] for i in range(0, len(line_bars), 2)]
                    else:
                        new_line_bars = [line_bars[0]] + [line_bars[i] + line_bars[i + 1] for i in range(1, len(line_bars), 2)]
                    if 'V' not in new_line_bars[-1]:
                        new_line_bars[-2] += new_line_bars[-1]
                        new_line_bars = new_line_bars[:-1]
                new_bars += new_line_bars
        except:
            pass

        return new_bars

    def split_patches(self, abc_text, patch_size=PATCH_SIZE, generate_last=False):
        # === 关键修复：强制只保留 ASCII 字符 (0-127) ===
        abc_text = ''.join(c for c in abc_text if 0 <= ord(c) < 128)
        # ==============================================
        if not generate_last and len(abc_text) % patch_size != 0:
            abc_text += chr(self.eos_token_id)
        patches = [abc_text[i : i + patch_size] for i in range(0, len(abc_text), patch_size)]
        return patches

    def patch2chars(self, patch):
        """
        Convert a patch into a bar.
        """
        bytes = ''
        for idx in patch:
            if idx == self.eos_token_id:
                break
            if idx < self.eos_token_id:
                pass
            bytes += chr(idx)
        return bytes

    def patchilize_metadata(self, metadata_lines):
        metadata_patches = []
        for line in metadata_lines:
            # === 关键修复：强制只保留 ASCII 字符 ===
            line = ''.join(c for c in line if 0 <= ord(c) < 128)
            # ======================================
            metadata_patches += self.split_patches(line)
        return metadata_patches

    def patchilize_tunebody(self, tunebody_lines, encode_mode='train'):
        tunebody_patches = []
        bars = self.split_bars(tunebody_lines)
        if encode_mode == 'train':
            for bar in bars:
                tunebody_patches += self.split_patches(bar)
        elif encode_mode == 'generate':
            for bar in bars[:-1]:
                tunebody_patches += self.split_patches(bar)
            tunebody_patches += self.split_patches(bars[-1], generate_last=True)
        return tunebody_patches

    def encode_train(self, abc_text, patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE, add_special_patches=True, cut=True):
        # === 过滤非 ASCII（双重保险）===
        abc_text = ''.join(c for c in abc_text if 0 <= ord(c) < 128)
        # ============================

        lines = abc_text.split('\n')
        lines = list(filter(None, lines))
        lines = [line + '\n' for line in lines]

        # 提取标签（NotaGen-X 格式）
        tags = []
        tunebody_index = 0
        for i, line in enumerate(lines):
            if line.startswith('%'):
                if line.strip() == '%end':
                    tunebody_index = i + 1
                    break
                tag = line[1:].strip()
                if tag and tag in _build_tag_vocab():
                    tags.append(tag)
            else:
                tunebody_index = i
                break

        if tunebody_index == 0:
            for i, line in enumerate(lines):
                if line.startswith('V:'):
                    tunebody_index = i
                    break

        metadata_lines = lines[:tunebody_index]
        tunebody_lines = lines[tunebody_index:]

        if self.stream:
            tunebody_lines = ['[r:' + str(line_index) + '/' + str(len(tunebody_lines) - line_index - 1) + ']' + line for line_index, line in
                                enumerate(tunebody_lines)]

        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='train')

        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)
            eos_patch = chr(self.bos_token_id) + chr(self.eos_token_id) * (patch_size - 1)
            metadata_patches = [bos_patch] + metadata_patches
            tunebody_patches = tunebody_patches + [eos_patch]

        if self.stream:
            if len(metadata_patches) + len(tunebody_patches) > patch_length:
                available_cut_indexes = [0] + [index + 1 for index, patch in enumerate(tunebody_patches) if '\n' in patch]
                line_index_for_cut_index = list(range(len(available_cut_indexes)))
                end_index = len(metadata_patches) + len(tunebody_patches) - patch_length
                biggest_index = bisect.bisect_left(available_cut_indexes, end_index)
                available_cut_indexes = available_cut_indexes[:biggest_index + 1]

                if len(available_cut_indexes) == 1:
                    choices = ['head']
                elif len(available_cut_indexes) == 2:
                    choices = ['head', 'tail']
                else:
                    choices = ['head', 'tail', 'middle']
                choice = random.choice(choices)
                if choice == 'head':
                    patches = metadata_patches + tunebody_patches[0:]
                else:
                    if choice == 'tail':
                        cut_index = len(available_cut_indexes) - 1
                    else:
                        cut_index = random.choice(range(1, len(available_cut_indexes) - 1))

                    line_index = line_index_for_cut_index[cut_index]
                    stream_tunebody_lines = tunebody_lines[line_index:]
                    stream_tunebody_patches = self.patchilize_tunebody(stream_tunebody_lines, encode_mode='train')
                    if add_special_patches:
                        stream_tunebody_patches = stream_tunebody_patches + [eos_patch]
                    patches = metadata_patches + stream_tunebody_patches
            else:
                patches = metadata_patches + tunebody_patches
        else:
            patches = metadata_patches + tunebody_patches

        if cut:
            patches = patches[:patch_length]

        # encode to ids
        id_patches = []
        for patch in patches:
            id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
            id_patches.append(id_patch)

        return id_patches, tags

    def encode_generate(self, abc_code, patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE, add_special_patches=True):
        # === 过滤非 ASCII（双重保险）===
        abc_code = ''.join(c for c in abc_code if 0 <= ord(c) < 128)
        # ============================

        lines = abc_code.split('\n')
        lines = list(filter(None, lines))

        tunebody_index = None
        for i, line in enumerate(lines):
            if line.startswith('V:') or line.startswith('[r:'):
                tunebody_index = i
                break

        if tunebody_index is None:
            tunebody_index = 0

        metadata_lines = lines[:tunebody_index]
        tunebody_lines = lines[tunebody_index:]

        metadata_lines = [line + '\n' for line in metadata_lines]
        if self.stream:
            if not abc_code.endswith('\n'):
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines) - 1)] + [tunebody_lines[-1]]
            else:
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines))]
        else:
            tunebody_lines = [line + '\n' for line in tunebody_lines]

        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='generate')

        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)
            metadata_patches = [bos_patch] + metadata_patches

        patches = metadata_patches + tunebody_patches
        patches = patches[:patch_length]

        # encode to ids
        id_patches = []
        for patch in patches:
            if len(patch) < PATCH_SIZE and patch[-1] != chr(self.eos_token_id):
                id_patch = [ord(c) for c in patch]
            else:
                id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
            id_patches.append(id_patch)

        return id_patches

    def decode(self, patches):
        return ''.join(self.patch2chars(patch) for patch in patches)


class PatchLevelDecoder(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.patch_embedding = torch.nn.Linear(PATCH_SIZE * 128, config.n_embd)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.base = GPT2Model(config)

    def forward(self, patches: torch.Tensor, masks=None) -> torch.Tensor:
        patches = torch.nn.functional.one_hot(patches, num_classes=128).to(self.dtype)
        patches = patches.reshape(len(patches), -1, PATCH_SIZE * 128)
        patches = self.patch_embedding(patches.to(self.device))

        if masks is None:
            return self.base(inputs_embeds=patches)
        else:
            return self.base(inputs_embeds=patches, attention_mask=masks)


class CharLevelDecoder(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.special_token_id = 0
        self.bos_token_id = 1
        self.base = GPT2LMHeadModel(config)

    def forward(self, encoded_patches: torch.Tensor, target_patches: torch.Tensor):
        target_patches = torch.cat((torch.ones_like(target_patches[:,0:1])*self.bos_token_id, target_patches), dim=1)
        target_masks = target_patches == self.special_token_id
        labels = target_patches.clone().masked_fill_(target_masks, -100)
        target_masks = torch.ones_like(labels)
        target_masks = target_masks.masked_fill_(labels == -100, 0)

        if PATCH_SAMPLING_BATCH_SIZE != 0 and PATCH_SAMPLING_BATCH_SIZE < target_patches.shape[0]:
            indices = list(range(len(target_patches)))
            random.shuffle(indices)
            selected_indices = sorted(indices[:PATCH_SAMPLING_BATCH_SIZE])
            target_patches = target_patches[selected_indices,:]
            target_masks = target_masks[selected_indices,:]
            encoded_patches = encoded_patches[selected_indices,:]

        inputs_embeds = torch.nn.functional.embedding(target_patches, self.base.transformer.wte.weight)
        inputs_embeds = torch.cat((encoded_patches.unsqueeze(1), inputs_embeds[:,1:,:]), dim=1)

        output = self.base(inputs_embeds=inputs_embeds, attention_mask=target_masks, labels=labels)
        return output

    def generate(self, encoded_patch: torch.Tensor, tokens: torch.Tensor):
        encoded_patch = encoded_patch.reshape(1, 1, -1)
        tokens = tokens.reshape(1, -1)
        tokens = torch.nn.functional.embedding(tokens, self.base.transformer.wte.weight)
        tokens = torch.cat((encoded_patch, tokens[:,1:,:]), dim=1)
        outputs = self.base(inputs_embeds=tokens)
        probs = torch.nn.functional.softmax(outputs.logits.squeeze(0)[-1], dim=-1)
        return probs


def safe_normalize_probs(probs):
    epsilon = 1e-12
    probs = np.array(probs, dtype=np.float64)
    probs = np.where(np.isnan(probs) | (probs < 0), 0, probs)
    probs = probs + epsilon
    s = probs.sum()
    if s > 0:
        probs = probs / s
    else:
        probs = np.zeros_like(probs)
        probs[0] = 1.0
    return probs


class NotaGenLMHeadModel(PreTrainedModel):
    """
    NotaGen-X 支持标签条件生成
    """
    def __init__(self, encoder_config, decoder_config):
        super().__init__(encoder_config)
        self.special_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.patch_level_decoder = PatchLevelDecoder(encoder_config)
        self.char_level_decoder = CharLevelDecoder(decoder_config)
        
        # 标签嵌入层
        self.tag_to_id = _build_tag_vocab()
        self.tag_embedding = torch.nn.Embedding(len(self.tag_to_id), HIDDEN_SIZE)
        torch.nn.init.normal_(self.tag_embedding.weight, std=0.02)

    def embed_tags(self, tags_list):
        """将标签列表转换为嵌入向量"""
        device = next(self.parameters()).device
        batch_embeds = []
        for tags in tags_list:
            valid_tag_ids = [self.tag_to_id[tag] for tag in tags if tag in self.tag_to_id]
            if valid_tag_ids:
                tag_tensor = torch.tensor(valid_tag_ids, device=device)
                embed = self.tag_embedding(tag_tensor).mean(dim=0)
            else:
                embed = torch.zeros(HIDDEN_SIZE, device=device)
            batch_embeds.append(embed)
        return torch.stack(batch_embeds)

    def forward(self, patches: torch.Tensor, masks: torch.Tensor, tags=None):
        patches = patches.reshape(len(patches), -1, PATCH_SIZE)
        encoded_patches = self.patch_level_decoder(patches, masks)["last_hidden_state"]
        
        if tags is not None:
            tag_embeds = self.embed_tags(tags)
            encoded_patches[:, 0] = encoded_patches[:, 0] + tag_embeds
        
        left_shift_masks = masks * (masks.flip(1).cumsum(1).flip(1) > 1)
        masks[:, 0] = 0
        
        encoded_patches = encoded_patches[left_shift_masks == 1]
        patches = patches[masks == 1]        

        return self.char_level_decoder(encoded_patches, patches)
        
    def generate(self, patches: torch.Tensor, tags=None, top_k=0, top_p=1, temperature=1.0):
        if patches.shape[-1] % PATCH_SIZE != 0:
            tokens = patches[:,:,-(patches.shape[-1]%PATCH_SIZE):].squeeze(0, 1)
            tokens = torch.cat((torch.tensor([self.bos_token_id], device=self.device), tokens), dim=-1)
            patches = patches[:,:,:-(patches.shape[-1]%PATCH_SIZE)]
        else:
            tokens = torch.tensor([self.bos_token_id], device=self.device)

        patches = patches.reshape(len(patches), -1, PATCH_SIZE)
        encoded_patches = self.patch_level_decoder(patches)["last_hidden_state"]
        
        if tags is not None:
            tag_embed = self.embed_tags([tags])
            encoded_patches[:, 0] = encoded_patches[:, 0] + tag_embed.squeeze(0)

        generated_patch = []            
        while True:
            prob = self.char_level_decoder.generate(encoded_patches[0][-1], tokens).cpu().detach().numpy()
            prob = safe_normalize_probs(prob)
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True)
            prob = safe_normalize_probs(prob)
            prob = top_p_sampling(prob, top_p=top_p, return_probs=True)
            prob = safe_normalize_probs(prob)
            token = temperature_sampling(prob, temperature=temperature)
            generated_patch.append(token)

            if len(tokens) >= PATCH_SIZE:
                break
            else:
                tokens = torch.cat((tokens, torch.tensor([token], device=self.device)), dim=0)
        
        return generated_patch
