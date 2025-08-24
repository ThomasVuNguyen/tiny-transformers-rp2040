"""
GPU-Optimized Transformer Training Script for RP2040 Production Models
Based on comprehensive architectural studies from log.md

🚀 PRODUCTION MODELS (Based on log.md findings):
- rp2040-optimized (15-20K): vocab=512, dim=8, layers=3, heads=8, hidden=256 (32x FFN)
- rp2040-speed (8-12K):     vocab=256, dim=6, layers=2, heads=4, hidden=192 (32x FFN)  
- rp2040-quality (25-35K):  vocab=1024, dim=12, layers=4, heads=12, hidden=384 (32x FFN)

📚 Dataset Integration:
- Uses TinyStories dataset with 10% increments (10%, 20%, 30%, ..., 100%)
- GPU-accelerated training with PyTorch
- Automatic mixed precision for memory efficiency
- Gradient accumulation for larger effective batch sizes

🎯 Key Features:
- GPU memory optimization
- Automatic learning rate scheduling
- Early stopping with validation
- Model checkpointing
- Training metrics logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import numpy as np
import os
import json
import time
from tqdm import tqdm
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # CUDA optimizations for maximum GPU utilization
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    torch.backends.cudnn.deterministic = False  # Disable for performance
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32 for Ampere+
    torch.backends.cudnn.allow_tf32 = True  # Enable TensorFloat-32 for cuDNN
    
    # Set memory fraction to use most of GPU memory
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
    
    # Enable memory pool for faster allocations
    torch.cuda.empty_cache()
    logger.info("CUDA optimizations enabled for maximum GPU utilization")

# Production model configurations from log.md
PRODUCTION_CONFIGS = {
    'rp2040-optimized': {
        'vocab_size': 512,          # Large enough for real text, minimal speed impact
        'dim': 8,                   # Sweet spot: good capacity, acceptable speed
        'hidden_dim': 256,          # 32x FFN ratio (moderate speed impact)
        'n_layers': 3,              # 3 layers: good depth, manageable speed loss
        'n_heads': 8,               # 8 heads: good attention, moderate speed impact
        'max_seq_len': 64,          # Longer sequences for serious text
        'description': 'OPTIMIZED: 15-20K parameters - Production-ready RP2040 model'
    },
    'rp2040-speed': {
        'vocab_size': 256,          # Smaller vocab for speed
        'dim': 6,                   # Narrower for speed
        'hidden_dim': 192,          # 32x FFN ratio
        'n_layers': 2,              # 2 layers for speed
        'n_heads': 4,               # 4 heads for speed
        'max_seq_len': 48,          # Shorter sequences
        'description': 'SPEED: 8-12K parameters - Fast RP2040 model'
    },
    'rp2040-quality': {
        'vocab_size': 1024,         # Large vocab for quality
        'dim': 12,                  # Wider for quality
        'hidden_dim': 384,          # 32x FFN ratio
        'n_layers': 4,              # 4 layers for quality
        'n_heads': 12,              # 12 heads for quality
        'max_seq_len': 96,          # Longer sequences
        'description': 'QUALITY: 25-35K parameters - High-quality RP2040 model'
    }
}

class TinyStoriesDataset(Dataset):
    """Dataset for TinyStories with configurable size"""
    
    def __init__(self, dataset_path: str, percent: int, max_seq_len: int, tokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.data = []
        
        # Load dataset
        logger.info(f"Loading {percent}% of TinyStories dataset...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into sentences
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        # Calculate how many sentences to use
        total_sentences = len(sentences)
        sentences_to_use = int(total_sentences * percent / 100)
        sentences = sentences[:sentences_to_use]
        
        logger.info(f"Using {sentences_to_use} sentences ({percent}% of {total_sentences})")
        
        # Create training examples with proper sequences
        for sentence in sentences:
            if len(sentence) < 5:  # Skip very short sentences
                continue
                
            tokens = self.tokenizer.encode(sentence)
            if len(tokens) > 1:
                # Create sequences where each position predicts the next token
                for i in range(1, len(tokens)):
                    # Get sequence up to current position
                    sequence = tokens[:i+1]  # Include the target token
                    
                    if len(sequence) <= max_seq_len:
                        self.data.append(sequence)
                    else:
                        # If sequence is too long, take the last max_seq_len tokens
                        self.data.append(sequence[-max_seq_len:])
        
        logger.info(f"Created {len(self.data)} training examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        
        # Create input and target sequences
        input_ids = sequence[:-1]  # All tokens except the last
        target_ids = sequence[1:]   # All tokens except the first
        
        # Pad input to max_seq_len - 1 (since we need one less for targets)
        if len(input_ids) < self.max_seq_len - 1:
            input_ids = input_ids + [0] * (self.max_seq_len - 1 - len(input_ids))
        else:
            input_ids = input_ids[:self.max_seq_len - 1]
        
        # Pad targets to max_seq_len - 1
        if len(target_ids) < self.max_seq_len - 1:
            target_ids = target_ids + [0] * (self.max_seq_len - 1 - len(target_ids))
        else:
            target_ids = target_ids[:self.max_seq_len - 1]
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

class SimpleTokenizer:
    """Simple tokenizer for demonstration - replace with your actual tokenizer"""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.char_to_id = {'<pad>': 0, '<s>': 1, ' ': 2, '<unk>': 3}
        self.id_to_char = {0: '<pad>', 1: '<s>', 2: ' ', 3: '<unk>'}
        
        # Add letters
        for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
            self.char_to_id[char] = i + 4
            self.id_to_char[i + 4] = char
        
        # Add common words
        common_words = ['the', 'and', 'a', 'to', 'of', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'use', 'an', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part']
        
        for word in common_words[:min(len(common_words), vocab_size - len(self.char_to_id))]:
            if word not in self.char_to_id:
                self.char_to_id[word] = len(self.char_to_id)
                self.id_to_char[len(self.char_to_id) - 1] = word
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        tokens = []
        words = text.lower().split()
        
        for word in words:
            if word in self.char_to_id:
                tokens.append(self.char_to_id[word])
            else:
                # Fallback to character-level
                for char in word:
                    if char in self.char_to_id:
                        tokens.append(self.char_to_id[char])
                    else:
                        tokens.append(self.char_to_id['<unk>'])
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text"""
        return ' '.join([self.id_to_char.get(token, '<unk>') for token in tokens])

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network"""
    
    def __init__(self, dim: int, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.attention_norm(x + self.dropout(attn_out))
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ffn_out))
        
        return x

class ProductionTransformer(nn.Module):
    """Production-ready transformer model based on log.md recommendations"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.vocab_size = config['vocab_size']
        self.dim = config['dim']
        self.hidden_dim = config['hidden_dim']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.max_seq_len = config['max_seq_len']
        
        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.dim)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(self.dim, self.hidden_dim, self.n_heads)
            for _ in range(self.n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.dim, self.vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Model parameters: {total_params:,} ({total_params/1000:.1f}K)")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        x = token_embeds + position_embeds
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits

class GPUTrainer:
    """GPU-optimized trainer for production models"""
    
    def __init__(self, model_name: str, config: Dict, dataset_percent: int):
        self.model_name = model_name
        self.config = config
        self.dataset_percent = dataset_percent
        
        # Initialize model
        self.model = ProductionTransformer(config).to(device)
        
        # Initialize tokenizer
        self.tokenizer = SimpleTokenizer(config['vocab_size'])
        
        # Training parameters
        self.batch_size = self._get_optimal_batch_size()
        self.learning_rate = 1e-4
        self.weight_decay = 1e-2
        
        # Gradient accumulation for larger effective batch size
        self.gradient_accumulation_steps = 4  # Default value, can be overridden by command line
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # Mixed precision training
        self.scaler = GradScaler('cuda')
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"Initialized trainer for {model_name} with {dataset_percent}% dataset")
        logger.info(f"Batch size: {self.batch_size}, Learning rate: {self.learning_rate}")
    
    def _get_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on model size and GPU memory"""
        if not torch.cuda.is_available():
            return 32
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Optimized for H100 and high-end GPUs
        if self.config['dim'] <= 8:
            if gpu_memory >= 80:  # H100 class
                return 2048  # Much larger batch size for H100
            elif gpu_memory >= 40:  # A100 class
                return 1024
            elif gpu_memory >= 20:  # RTX 4090 class
                return 512
            elif gpu_memory >= 8:
                return 256
            else:
                return 128
        elif self.config['dim'] <= 12:
            if gpu_memory >= 80:  # H100 class
                return 1024
            elif gpu_memory >= 40:  # A100 class
                return 512
            elif gpu_memory >= 20:  # RTX 4090 class
                return 256
            elif gpu_memory >= 8:
                return 128
            else:
                return 64
        else:
            if gpu_memory >= 80:  # H100 class
                return 512
            elif gpu_memory >= 40:  # A100 class
                return 256
            elif gpu_memory >= 20:  # RTX 4090 class
                return 128
            elif gpu_memory >= 8:
                return 64
            else:
                return 32
    
    def _adjust_batch_size_for_memory(self) -> int:
        """Dynamically adjust batch size based on actual GPU memory usage"""
        if not torch.cuda.is_available():
            return self.batch_size
        
        # Get current GPU memory usage
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")
        
        # If we're using less than 70% of GPU memory, increase batch size
        if allocated < total * 0.7:
            new_batch_size = min(self.batch_size * 2, 4096)  # Cap at 4096
            if new_batch_size != self.batch_size:
                logger.info(f"Increasing batch size from {self.batch_size} to {new_batch_size} for better GPU utilization")
                self.batch_size = new_batch_size
        
        return self.batch_size
    
    def prepare_data(self, dataset_path: str) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data"""
        # Create dataset
        dataset = TinyStoriesDataset(
            dataset_path=dataset_path,
            percent=self.dataset_percent,
            max_seq_len=self.config['max_seq_len'],
            tokenizer=self.tokenizer
        )
        
        # Split into train/val (90/10)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,  # Increased from 4 to 8
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=4,  # Prefetch 4 batches per worker
            drop_last=True  # Drop incomplete batches for consistent training
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,  # Increased from 4 to 8
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=4,  # Prefetch 4 batches per worker
            drop_last=True  # Drop incomplete batches for consistent validation
        )
        
        logger.info(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} validation samples")
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        # Adjust batch size for optimal GPU utilization
        if epoch == 0:  # Only adjust on first epoch
            self._adjust_batch_size_for_memory()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} (Train)")
        
        for batch_idx, (input_ids, targets) in enumerate(progress_bar):
            input_ids = input_ids.to(device, non_blocking=True)  # Non-blocking transfer
            targets = targets.to(device, non_blocking=True)      # Non-blocking transfer
            
            # Forward pass with mixed precision
            with autocast('cuda'):
                logits = self.model(input_ids)
                loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation: only step optimizer every N steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # Update progress bar with GPU utilization info
            if batch_idx % 100 == 0:  # Update GPU info every 100 batches
                gpu_util = torch.cuda.utilization(0)
                gpu_memory = torch.cuda.memory_allocated(0) / 1e9
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                    'batch_size': f'{self.batch_size * self.gradient_accumulation_steps}',
                    'GPU_util': f'{gpu_util}%',
                    'GPU_mem': f'{gpu_memory:.1f}GB'
                })
            else:
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                    'batch_size': f'{self.batch_size * self.gradient_accumulation_steps}'
                })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for input_ids, targets in tqdm(val_loader, desc="Validation"):
                input_ids = input_ids.to(device, non_blocking=True)  # Non-blocking transfer
                targets = targets.to(device, non_blocking=True)      # Non-blocking transfer
                
                with autocast('cuda'):
                    logits = self.model(input_ids)
                    loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100) -> Dict:
        """Main training loop"""
        logger.info(f"Starting training for {epochs} epochs...")
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Log progress
            logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(f"best_{self.model_name}_{self.dataset_percent}p.pt")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_{self.model_name}_{self.dataset_percent}p_epoch{epoch+1}.pt")
        
        # Save final model
        self.save_checkpoint(f"final_{self.model_name}_{self.dataset_percent}p.pt")
        
        # Save training history
        self.save_training_history()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'dataset_percent': self.dataset_percent,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_training_history(self):
        """Save training history to JSON"""
        history = {
            'model_name': self.model_name,
            'dataset_percent': self.dataset_percent,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        history_dir = Path("training_history")
        history_dir.mkdir(exist_ok=True)
        
        history_path = history_dir / f"{self.model_name}_{self.dataset_percent}p_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training history saved: {history_path}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='GPU Training for RP2040 Production Models')
    parser.add_argument('--model', type=str, choices=list(PRODUCTION_CONFIGS.keys()), 
                       default='rp2040-optimized', help='Model to train')
    parser.add_argument('--dataset-percent', type=int, choices=range(10, 101, 10), 
                       default=10, help='Dataset percentage to use (10, 20, 30, ..., 100)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--dataset-path', type=str, default='dataset/TinyStories-train.txt',
                       help='Path to TinyStories dataset')
    parser.add_argument('--batch-size', type=int, help='Override automatic batch size')
    parser.add_argument('--aggressive-opt', action='store_true', 
                       help='Enable aggressive GPU optimization (larger batches, more workers)')
    parser.add_argument('--grad-accum', type=int, default=4, 
                       help='Gradient accumulation steps (default: 4)')
    
    args = parser.parse_args()
    
    # Apply aggressive optimization if requested
    if args.aggressive_opt:
        # Set even more aggressive CUDA settings
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.set_per_process_memory_fraction(0.98)  # Use 98% of GPU memory
            logger.info("Aggressive GPU optimization enabled")
    
    logger.info("=== GPU Training for RP2040 Production Models ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset_percent}%")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Aggressive optimization: {args.aggressive_opt}")
    logger.info(f"Gradient accumulation: {args.grad_accum} steps")
    
    # Get model configuration
    config = PRODUCTION_CONFIGS[args.model]
    logger.info(f"Configuration: {config}")
    
    # Initialize trainer
    trainer = GPUTrainer(args.model, config, args.dataset_percent)
    
    # Override batch size if specified
    if args.batch_size:
        trainer.batch_size = args.batch_size
        logger.info(f"Using custom batch size: {args.batch_size}")
    
    # Override gradient accumulation if specified
    if args.grad_accum != 4:
        trainer.gradient_accumulation_steps = args.grad_accum
        logger.info(f"Using custom gradient accumulation: {args.grad_accum} steps")
    
    # Prepare data
    if not os.path.exists(args.dataset_path):
        logger.error(f"Dataset not found at {args.dataset_path}")
        return
    
    train_loader, val_loader = trainer.prepare_data(args.dataset_path)
    
    # Train model
    start_time = time.time()
    history = trainer.train(train_loader, val_loader, args.epochs)
    training_time = time.time() - start_time
    
    # Final summary
    logger.info("=== Training Complete ===")
    logger.info(f"Total training time: {training_time/3600:.2f} hours")
    logger.info(f"Best validation loss: {history['best_val_loss']:.4f}")
    logger.info(f"Final train loss: {history['train_losses'][-1]:.4f}")
    logger.info(f"Final validation loss: {history['val_losses'][-1]:.4f}")

if __name__ == "__main__":
    main()
