import os
import gc
import time
import math
import json
import torch
import random
import numpy as np
from abctoolkit.transpose import Key2index, Key2Mode
from utils import *
from config import *
from tqdm import tqdm
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, get_scheduler, get_constant_schedule_with_warmup
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

Index2Key = {index: key for key, index in Key2index.items() if index not in [1, 11]}
Mode2Key = {mode: key for key, mode_list in Key2Mode.items() for mode in mode_list}

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

# 设置随机种子
def set_seed(seed: int = 42):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED if hasattr(globals(), 'SEED') else 42)

batch_size = BATCH_SIZE

# 配置类
@dataclass
class TrainingConfig:
    """训练配置"""
    num_epochs: int = NUM_EPOCHS
    batch_size: int = BATCH_SIZE
    accumulation_steps: int = ACCUMULATION_STEPS
    learning_rate: float = LEARNING_RATE
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    save_every: int = 1000  # 每多少步保存一次检查点
    log_every: int = 100    # 每多少步记录一次日志

config = TrainingConfig()

patchilizer = Patchilizer()

# 模型配置
patch_config = GPT2Config(
    num_hidden_layers=PATCH_NUM_LAYERS,
    max_length=PATCH_LENGTH,
    max_position_embeddings=PATCH_LENGTH,
    n_embd=HIDDEN_SIZE,
    num_attention_heads=HIDDEN_SIZE // 64,
    vocab_size=1
)
char_config = GPT2Config(
    num_hidden_layers=CHAR_NUM_LAYERS,
    max_length=PATCH_SIZE + 1,
    max_position_embeddings=PATCH_SIZE + 1,
    hidden_size=HIDDEN_SIZE,
    num_attention_heads=HIDDEN_SIZE // 64,
    vocab_size=128
)

# 创建模型（现在支持标签）
model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=char_config)
model = model.to(device)

# 打印参数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"可训练参数数量: {total_params:,}")

# 优化器和梯度缩放器
scaler = torch.amp.GradScaler('cuda')
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, List[str]]]) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
    """批量处理函数 - 支持标签"""
    input_patches, input_masks, tags = zip(*batch)
    
    # 使用堆叠而不是填充，因为所有序列长度相同
    input_patches = torch.stack(input_patches, dim=0)
    input_masks = torch.stack(input_masks, dim=0)
    
    return input_patches.to(device), input_masks.to(device), list(tags)

def create_minibatches(input_patches: torch.Tensor, 
                      input_masks: torch.Tensor,
                      tags: List[List[str]],
                      minibatch_size: int,
                      is_training: bool = True) -> List[Tuple[torch.Tensor, torch.Tensor, List[List[str]]]]:
    """将批次拆分为小批次 - 支持标签"""
    num_samples = len(input_patches)
    
    if not is_training or minibatch_size >= num_samples:
        return [(input_patches, input_masks, tags)]
    
    # 训练时随机打乱
    if is_training:
        indices = torch.randperm(num_samples)
        input_patches = input_patches[indices]
        input_masks = input_masks[indices]
        tags = [tags[i] for i in indices]
    
    minibatches = []
    for start_idx in range(0, num_samples, minibatch_size):
        end_idx = min(start_idx + minibatch_size, num_samples)
        minibatch = (
            input_patches[start_idx:end_idx],
            input_masks[start_idx:end_idx],
            tags[start_idx:end_idx]
        )
        minibatches.append(minibatch)
    
    return minibatches

class NotaGenDataset(Dataset):
    """NotaGen数据集类 - 支持标签提取"""
    
    def __init__(self, filenames: List[Dict], cache_size: int = 1000):
        self.filenames = filenames
        self.cache = {}
        self.cache_size = cache_size
        self.transposition_probs = [1/16, 2/16, 3/16, 4/16, 3/16, 2/16, 1/16]
        self.transposition_cum_probs = np.cumsum(self.transposition_probs)
        
    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        # 检查缓存
        if idx in self.cache:
            return self.cache[idx]
        
        try:
            file_info = self.filenames[idx]
            filepath = file_info['path']
            ori_key = Mode2Key[file_info['key']]
            
            # 选择转调的调性
            ori_key_index = Key2index[ori_key]
            available_index = [(ori_key_index + offset) % 12 for offset in range(-3, 4)]
            
            # 使用累积概率选择目标调性
            rand_val = random.random()
            selected_idx = np.searchsorted(self.transposition_cum_probs, rand_val)
            des_key_index = available_index[selected_idx]
            
            # 处理特殊调性名称
            des_key = self._get_key_name(des_key_index)
            
            # 构建目标文件路径
            folder = os.path.dirname(filepath)
            name = os.path.basename(filepath)
            des_filepath = os.path.join(folder, des_key, f"{name}_{des_key}.abc")
            
            # 读取并编码文件（现在会返回标签）
            with open(des_filepath, 'r', encoding='utf-8') as f:
                abc_text = f.read()
            
            file_bytes, tags = patchilizer.encode_train(abc_text)  # ← 关键：获取标签
            file_masks = [1] * len(file_bytes)
            
            file_bytes = torch.tensor(file_bytes, dtype=torch.long)
            file_masks = torch.tensor(file_masks, dtype=torch.long)
            
            # 缓存结果
            if len(self.cache) < self.cache_size:
                self.cache[idx] = (file_bytes, file_masks, tags)
            
            return file_bytes, file_masks, tags
            
        except Exception as e:
            logger.error(f"处理文件 {filepath} 时出错: {e}")
            # 返回空数据，让collate_fn处理
            empty_patch = torch.zeros((PATCH_LENGTH, PATCH_SIZE), dtype=torch.long)
            empty_mask = torch.zeros(PATCH_LENGTH, dtype=torch.long)
            return empty_patch, empty_mask, []
    
    def _get_key_name(self, key_index: int) -> str:
        """获取调性名称"""
        if key_index == 1:
            return 'Db' if random.random() < 0.8 else 'C#'
        elif key_index == 11:
            return 'B' if random.random() < 0.8 else 'Cb'
        elif key_index == 6:
            return 'F#' if random.random() < 0.5 else 'Gb'
        else:
            return Index2Key.get(key_index, 'C')

def train_epoch(epoch: int, 
                dataloader: DataLoader, 
                model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                scaler: GradScaler,
                scheduler,
                global_step: int) -> Tuple[float, int]:
    """训练一个epoch - 支持标签"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", 
                       disable=not logger.isEnabledFor(logging.INFO))
    
    for batch_idx, batch in enumerate(progress_bar):
        input_patches, input_masks, tags = batch
        
        # 梯度累积
        accumulated_loss = 0
        optimizer.zero_grad()
        
        minibatches = create_minibatches(
            input_patches, input_masks, tags,
            config.batch_size // config.accumulation_steps,
            is_training=True
        )
        
        for minibatch in minibatches:
            mb_patches, mb_masks, mb_tags = minibatch
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                loss = model(mb_patches, mb_masks, tags=mb_tags).loss / config.accumulation_steps  # ← 传入标签
            
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()
        
        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # 更新参数
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # 统计
        batch_size_actual = len(input_patches)
        total_loss += accumulated_loss * batch_size_actual
        total_samples += batch_size_actual
        
        # 更新进度条
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
        
        global_step += 1
        
        # 定期日志
        if global_step % config.log_every == 0:
            logger.info(f"Step {global_step}: loss={avg_loss:.4f}, "
                       f"lr={scheduler.get_last_lr()[0]:.2e}")
        
        # 定期保存检查点（并清理旧的）
        if global_step % config.save_every == 0:
            checkpoint_path = f"checkpoint_step_{global_step}.pt"
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, 
                           epoch, global_step, avg_loss)
            cleanup_old_checkpoints(global_step, keep_last=2)
    
    avg_epoch_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_epoch_loss, global_step

def evaluate_epoch(dataloader: DataLoader, model: torch.nn.Module) -> float:
    """验证一个epoch - 支持标签"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="[Eval]", 
                       disable=not logger.isEnabledFor(logging.INFO))
    
    with torch.no_grad():
        for batch in progress_bar:
            input_patches, input_masks, tags = batch
            
            # 验证时不需要梯度累积
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                loss = model(input_patches, input_masks, tags=tags).loss  # ← 传入标签
            
            batch_size_actual = len(input_patches)
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual
            
            # 更新进度条
            avg_loss = total_loss / total_samples if total_samples > 0 else 0
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_loss

def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   scheduler, epoch: int, step: int, loss: float):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config.__dict__ if hasattr(config, '__dict__') else {}
    }
    torch.save(checkpoint, path)
    logger.info(f"检查点已保存: {path}")

def cleanup_old_checkpoints(current_step: int, keep_last: int = 2):
    """清理旧的检查点文件，只保留最近 keep_last 个"""
    checkpoint_files = []
    for file in os.listdir('.'):
        if file.startswith('checkpoint_step_') and file.endswith('.pt'):
            try:
                step = int(file.replace('checkpoint_step_', '').replace('.pt', ''))
                checkpoint_files.append((step, file))
            except ValueError:
                continue
    
    checkpoint_files.sort(key=lambda x: x[0])
    files_to_keep = {filename for step, filename in checkpoint_files[-keep_last:]}
    
    deleted_count = 0
    total_size = 0
    for step, filename in checkpoint_files:
        if filename not in files_to_keep and step != current_step:
            try:
                file_size = os.path.getsize(filename)
                os.remove(filename)
                deleted_count += 1
                total_size += file_size
                logger.info(f"Deleted old checkpoint: {filename} ({file_size / (1024**3):.2f} GB)")
            except Exception as e:
                logger.warning(f"Failed to delete {filename}: {e}")
    
    if deleted_count > 0:
        logger.info(f"Cleanup: Removed {deleted_count} old checkpoints, freed {total_size / (1024**3):.2f} GB")

def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   scheduler, device: torch.device):
    """加载检查点"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"检查点不存在: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"加载检查点: epoch={checkpoint['epoch']}, step={checkpoint['step']}, "
               f"loss={checkpoint['loss']:.4f}")
    
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']

def split_dataset(files: List[Dict], eval_ratio: float = 0.1, 
                 min_eval_samples: int = 100) -> Tuple[List[Dict], List[Dict]]:
    """分割数据集，确保验证集有足够样本"""
    if len(files) < min_eval_samples * 2:
        # 样本太少，使用留一法
        eval_size = max(1, int(len(files) * eval_ratio))
    else:
        eval_size = max(min_eval_samples, int(len(files) * eval_ratio))
    
    random.shuffle(files)
    train_files = files[eval_size:]
    eval_files = files[:eval_size]
    
    return train_files, eval_files

# 主训练循环
if __name__ == "__main__":
    logger.info("=== NotaGen 训练开始（支持标签条件生成）===")
    
    start_time = time.time()
    
    # 加载数据
    try:
        with open(DATA_TRAIN_INDEX_PATH, "r", encoding="utf-8") as f:
            logger.info("加载训练数据...")
            all_files = [json.loads(line) for line in f if line.strip()]
        
        # 分割训练/验证集
        with open(DATA_EVAL_INDEX_PATH, "r", encoding="utf-8") as f:
            eval_lines = f.readlines()
            if len(eval_lines) == 0:
                logger.info("自动分割训练/验证集...")
                train_files, eval_files = split_dataset(all_files)
            else:
                train_files = all_files
                eval_files = [json.loads(line) for line in eval_lines if line.strip()]
        
        # 确保批次大小对齐
        train_batches = len(train_files) // config.batch_size
        eval_batches = len(eval_files) // config.batch_size
        
        train_files = train_files[:train_batches * config.batch_size]
        eval_files = eval_files[:eval_batches * config.batch_size]
        
        logger.info(f"训练样本: {len(train_files)} ({train_batches}批次)")
        logger.info(f"验证样本: {len(eval_files)} ({eval_batches}批次)")
        
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        raise
    
    # 创建数据集和数据加载器
    train_dataset = NotaGenDataset(train_files)
    eval_dataset = NotaGenDataset(eval_files)

    # === 修改：Windows 单进程模式 ===
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_batch,
        shuffle=True,
        num_workers=0,      # ← 关键：禁用多进程
        pin_memory=False    # ← 关键：禁用 pin_memory
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_batch,
        shuffle=False,
        num_workers=0,      # ← 关键：禁用多进程
        pin_memory=False    # ← 关键：禁用 pin_memory
    ) 
    
    # 学习率调度器
    total_steps = len(train_dataloader) * config.num_epochs
    scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps
    )
    
    # 加载预训练权重或检查点
    start_epoch = 0
    global_step = 0
    best_eval_loss = float('inf')
    
    if LOAD_FROM_CHECKPOINT and os.path.exists(WEIGHTS_PATH):
        logger.info(f"加载检查点: {WEIGHTS_PATH}")
        start_epoch, global_step, best_eval_loss = load_checkpoint(
            WEIGHTS_PATH, model, optimizer, scheduler, device
        )
        start_epoch += 1  # 从下一个epoch开始

    elif os.path.exists(PRETRAINED_PATH):
        logger.info(f"加载预训练权重: {PRETRAINED_PATH}")
        checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu', weights_only=False)
    
        # === 关键修改：兼容新增的 tag_embedding 层 ===
        pretrained_dict = checkpoint['model']
        model_dict = model.state_dict()
    
        # 过滤：只保留模型中存在的键，并且形状匹配
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }
    
        # 加载权重（strict=False 允许缺失键）
        model.load_state_dict(pretrained_dict, strict=False)
        logger.info("预训练权重加载完成（标签嵌入层将随机初始化）")
    
    logger.info(f"训练配置: {config}")
    logger.info("-" * 50)
    
    # 训练循环
    for epoch in range(start_epoch, config.num_epochs + 1):
        logger.info(f"\n{'='*20} Epoch {epoch}/{config.num_epochs} {'='*20}")
        epoch_start_time = time.time()
        
        # 训练
        train_loss, global_step = train_epoch(
            epoch, train_dataloader, model, optimizer, 
            scaler, scheduler, global_step
        )
        
        # 验证
        eval_loss = evaluate_epoch(eval_dataloader, model)
        
        epoch_time = time.time() - epoch_start_time
        
        # 记录日志
        log_msg = (f"Epoch {epoch}: "
                  f"train_loss={train_loss:.6f}, "
                  f"eval_loss={eval_loss:.6f}, "
                  f"time={epoch_time:.1f}s")
        logger.info(log_msg)
        
        with open(LOGS_PATH, 'a', encoding='utf-8') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {log_msg}\n")
        
        # 保存最佳模型（只保留1个）
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            # 直接覆盖保存，不保留历史最佳模型
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_sched': scheduler.state_dict(),
                'epoch': epoch,
                'best_epoch': epoch,
                'min_eval_loss': best_eval_loss
            }
            torch.save(checkpoint, WEIGHTS_PATH)
            logger.info(f"Best model updated (loss={eval_loss:.6f})")
        
        # 定期清理内存
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
    
    # 训练完成
    total_time = time.time() - start_time
    logger.info(f"\n{'='*50}")
    logger.info(f"训练完成! 总时间: {total_time:.1f}s")
    logger.info(f"最佳验证损失: {best_eval_loss:.6f}")
    logger.info(f"最终模型: {WEIGHTS_PATH}")
    logger.info(f"训练日志: {LOGS_PATH}")
