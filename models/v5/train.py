"""
Scalable Transformer Training Script
Test different model sizes to find RP2040 limits

Model Size Presets (actual parameter counts):
- story-1k (1.3K):    vocab=64,  dim=8,   layers=1, heads=2  
- chat-8k (8.2K):     vocab=128, dim=16,  layers=2, heads=4
- chat-10k (10.4K):   vocab=144, dim=18,  layers=2, heads=3
- chat-18k (18.4K):   vocab=192, dim=24,  layers=2, heads=4
- assistant-45k (45K): vocab=256, dim=32,  layers=3, heads=8
- expert-229k (229K):  vocab=512, dim=64,  layers=4, heads=16
- expert-1310k (1.3M): vocab=1024,dim=128, layers=6, heads=32

Names now reflect actual parameter counts, not approximations
"""

import numpy as np
import struct
import random
import time
import os
from collections import Counter
import json

# Model size presets - names reflect actual parameter counts
MODEL_CONFIGS = {
    # === 1K Parameter Architectural Variants (Speed Comparison Study) ===
    'story-1k-wide': {
        'vocab_size': 96,
        'dim': 6,
        'hidden_dim': 24,  # 4x dim
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 32,
        'description': '1.0K parameters - Wide vocab, narrow model'
    },
    'story-1k-balanced': {
        'vocab_size': 64,
        'dim': 8,
        'hidden_dim': 32,  # 4x dim
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 32,
        'description': '1.3K parameters - Balanced architecture'
    },
    'story-1k-deep': {
        'vocab_size': 32,
        'dim': 6,
        'hidden_dim': 24,  # 4x dim
        'n_layers': 2,
        'n_heads': 2,
        'max_seq_len': 32,
        'description': '1.1K parameters - Deep, narrow model'
    },
    'story-1k-narrow': {
        'vocab_size': 32,
        'dim': 8,
        'hidden_dim': 32,  # 4x dim
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 32,
        'description': '1.0K parameters - Narrow vocab, wider model'
    },
    'story-1k-fat': {
        'vocab_size': 32,
        'dim': 10,
        'hidden_dim': 40,  # 4x dim
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 32,
        'description': '1.5K parameters - High dimensional'
    },
    'story-1k-mini': {
        'vocab_size': 48,
        'dim': 6,
        'hidden_dim': 24,  # 4x dim
        'n_layers': 2,
        'n_heads': 2,
        'max_seq_len': 32,
        'description': '1.2K parameters - Multi-layer mini'
    },
    'story-1k-micro': {
        'vocab_size': 80,
        'dim': 6,
        'hidden_dim': 24,  # 4x dim
        'n_layers': 2,
        'n_heads': 2,
        'max_seq_len': 32,
        'description': '1.3K parameters - Micro transformer'
    },
    'story-1k-ultra': {
        'vocab_size': 48,
        'dim': 8,
        'hidden_dim': 32,  # 4x dim
        'n_layers': 1,
        'n_heads': 4,
        'max_seq_len': 32,
        'description': '1.2K parameters - Ultra-wide attention'
    },
    
    # === Extended 1K Variants - Vocabulary Size Study ===
    'story-1k-tiny-vocab': {
        'vocab_size': 16,
        'dim': 8,
        'hidden_dim': 32,  # 4x dim
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 32,
        'description': '1.2K parameters - Tiny vocab, normal model'
    },
    'story-1k-huge-vocab': {
        'vocab_size': 160,
        'dim': 4,
        'hidden_dim': 16,  # 4x dim
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 32,
        'description': '1.0K parameters - Huge vocab, tiny model'
    },
    
    # === Extended 1K Variants - Dimension Study ===
    'story-1k-super-narrow': {
        'vocab_size': 128,
        'dim': 4,
        'hidden_dim': 16,  # 4x dim
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 32,
        'description': '1.0K parameters - Super narrow dimensions'
    },
    'story-1k-wide-dim': {
        'vocab_size': 32,
        'dim': 12,
        'hidden_dim': 48,  # 4x dim
        'n_layers': 1,
        'n_heads': 3,
        'max_seq_len': 32,
        'description': '1.5K parameters - Wide dimensions'
    },
    'story-1k-extreme-dim': {
        'vocab_size': 24,
        'dim': 12,
        'hidden_dim': 48,  # 4x dim
        'n_layers': 1,
        'n_heads': 4,
        'max_seq_len': 32,
        'description': '1.4K parameters - Extreme dimensions'
    },
    
    # === Extended 1K Variants - Multi-Head Study ===
    'story-1k-single-head': {
        'vocab_size': 64,
        'dim': 8,
        'hidden_dim': 32,  # 4x dim
        'n_layers': 1,
        'n_heads': 1,
        'max_seq_len': 32,
        'description': '1.3K parameters - Single attention head'
    },
    'story-1k-mega-heads': {
        'vocab_size': 32,
        'dim': 8,
        'hidden_dim': 32,  # 4x dim
        'n_layers': 1,
        'n_heads': 8,
        'max_seq_len': 32,
        'description': '1.0K parameters - Many attention heads'
    },
    
    # === Extended 1K Variants - Layer Depth Study ===
    'story-1k-triple-layer': {
        'vocab_size': 24,
        'dim': 4,
        'hidden_dim': 16,  # 4x dim
        'n_layers': 3,
        'n_heads': 2,
        'max_seq_len': 32,
        'description': '1.0K parameters - Triple layer depth'
    },
    'story-1k-quad-layer': {
        'vocab_size': 16,
        'dim': 4,
        'hidden_dim': 16,  # 4x dim
        'n_layers': 4,
        'n_heads': 2,
        'max_seq_len': 32,
        'description': '1.1K parameters - Quad layer depth'
    },
    
    # === Extended 1K Variants - Hidden Dimension Ratios ===
    'story-1k-thin-hidden': {
        'vocab_size': 80,
        'dim': 10,
        'hidden_dim': 20,  # 2x dim (thin)
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 32,
        'description': '1.4K parameters - Thin hidden layer'
    },
    'story-1k-fat-hidden': {
        'vocab_size': 32,
        'dim': 6,
        'hidden_dim': 48,  # 8x dim (fat)
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 32,
        'description': '1.5K parameters - Fat hidden layer'
    },
    'story-1k-mega-hidden': {
        'vocab_size': 24,
        'dim': 4,
        'hidden_dim': 64,  # 16x dim (mega)
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 32,
        'description': '1.4K parameters - Mega hidden layer'
    },
    
    # === Extended 1K Variants - Extreme Architectures ===
    'story-1k-vocab-monster': {
        'vocab_size': 320,
        'dim': 2,
        'hidden_dim': 8,  # 4x dim
        'n_layers': 1,
        'n_heads': 1,
        'max_seq_len': 32,
        'description': '1.0K parameters - Vocabulary monster'
    },
    'story-1k-dimension-beast': {
        'vocab_size': 8,
        'dim': 16,
        'hidden_dim': 64,  # 4x dim
        'n_layers': 1,
        'n_heads': 4,
        'max_seq_len': 32,
        'description': '1.2K parameters - Dimension beast'
    },
    'story-1k-layer-tower': {
        'vocab_size': 12,
        'dim': 3,
        'hidden_dim': 12,  # 4x dim
        'n_layers': 6,
        'n_heads': 1,
        'max_seq_len': 32,
        'description': '1.1K parameters - Layer tower'
    },
    'story-1k-head-hydra': {
        'vocab_size': 24,
        'dim': 6,
        'hidden_dim': 24,  # 4x dim
        'n_layers': 1,
        'n_heads': 6,
        'max_seq_len': 32,
        'description': '1.2K parameters - Multi-head hydra'
    },
    
    # === Extended 1K Variants - Balanced Extremes ===
    'story-1k-all-tiny': {
        'vocab_size': 32,
        'dim': 4,
        'hidden_dim': 16,  # 4x dim
        'n_layers': 2,
        'n_heads': 2,
        'max_seq_len': 32,
        'description': '1.0K parameters - Everything tiny'
    },
    'story-1k-mixed-extreme': {
        'vocab_size': 128,
        'dim': 6,
        'hidden_dim': 12,  # 2x dim
        'n_layers': 2,
        'n_heads': 3,
        'max_seq_len': 32,
        'description': '1.4K parameters - Mixed extremes'
    },
    'story-1k-fibonacci': {
        'vocab_size': 55,  # Fibonacci number
        'dim': 8,
        'hidden_dim': 21,  # Fibonacci ratio
        'n_layers': 1,
        'n_heads': 3,
        'max_seq_len': 32,
        'description': '1.2K parameters - Fibonacci ratios'
    },
    'story-1k-prime-numbers': {
        'vocab_size': 37,  # Prime
        'dim': 7,          # Prime
        'hidden_dim': 29,  # Prime
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 32,
        'description': '1.1K parameters - Prime number architecture'
    },
    
    # === 3K Parameter Architectural Variants ===
    'story-3k-vocab-heavy': {
        'vocab_size': 256,
        'dim': 6,
        'hidden_dim': 24,  # 4x dim
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 36,
        'description': '3.0K parameters - Vocabulary heavy'
    },
    'story-3k-balanced': {
        'vocab_size': 96,
        'dim': 12,
        'hidden_dim': 48,  # 4x dim
        'n_layers': 1,
        'n_heads': 3,
        'max_seq_len': 36,
        'description': '2.9K parameters - Balanced 3K design'
    },
    'story-3k-dimension-focus': {
        'vocab_size': 48,
        'dim': 16,
        'hidden_dim': 64,  # 4x dim
        'n_layers': 1,
        'n_heads': 4,
        'max_seq_len': 36,
        'description': '3.1K parameters - Dimension focused'
    },
    'story-3k-deep': {
        'vocab_size': 64,
        'dim': 8,
        'hidden_dim': 32,  # 4x dim
        'n_layers': 2,
        'n_heads': 2,
        'max_seq_len': 36,
        'description': '2.8K parameters - Deep 3K model'
    },
    'story-3k-ultra-deep': {
        'vocab_size': 32,
        'dim': 6,
        'hidden_dim': 24,  # 4x dim
        'n_layers': 3,
        'n_heads': 2,
        'max_seq_len': 36,
        'description': '2.9K parameters - Ultra deep'
    },
    'story-3k-wide-attention': {
        'vocab_size': 64,
        'dim': 12,
        'hidden_dim': 48,  # 4x dim
        'n_layers': 1,
        'n_heads': 6,
        'max_seq_len': 36,
        'description': '3.1K parameters - Wide attention'
    },
    'story-3k-thin-hidden': {
        'vocab_size': 128,
        'dim': 12,
        'hidden_dim': 24,  # 2x dim
        'n_layers': 1,
        'n_heads': 3,
        'max_seq_len': 36,
        'description': '3.0K parameters - Thin hidden layer'
    },
    
    # === 5K Parameter Architectural Variants ===
    'story-5k-vocab-monster': {
        'vocab_size': 400,
        'dim': 8,
        'hidden_dim': 32,  # 4x dim
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 40,
        'description': '5.0K parameters - Vocabulary monster'
    },
    'story-5k-balanced': {
        'vocab_size': 128,
        'dim': 16,
        'hidden_dim': 64,  # 4x dim
        'n_layers': 1,
        'n_heads': 4,
        'max_seq_len': 40,
        'description': '5.1K parameters - Balanced 5K design'
    },
    'story-5k-dimension-beast': {
        'vocab_size': 64,
        'dim': 20,
        'hidden_dim': 80,  # 4x dim
        'n_layers': 1,
        'n_heads': 4,
        'max_seq_len': 40,
        'description': '5.2K parameters - Dimension beast'
    },
    'story-5k-deep': {
        'vocab_size': 96,
        'dim': 12,
        'hidden_dim': 48,  # 4x dim
        'n_layers': 2,
        'n_heads': 3,
        'max_seq_len': 40,
        'description': '4.9K parameters - Deep 5K model'
    },
    'story-5k-ultra-deep': {
        'vocab_size': 48,
        'dim': 8,
        'hidden_dim': 32,  # 4x dim
        'n_layers': 3,
        'n_heads': 2,
        'max_seq_len': 40,
        'description': '4.8K parameters - Ultra deep'
    },
    'story-5k-layer-tower': {
        'vocab_size': 32,
        'dim': 6,
        'hidden_dim': 24,  # 4x dim
        'n_layers': 4,
        'n_heads': 2,
        'max_seq_len': 40,
        'description': '4.9K parameters - Layer tower'
    },
    'story-5k-attention-hydra': {
        'vocab_size': 80,
        'dim': 16,
        'hidden_dim': 64,  # 4x dim
        'n_layers': 1,
        'n_heads': 8,
        'max_seq_len': 40,
        'description': '5.2K parameters - Attention hydra'
    },
    
    # === 7K Parameter Architectural Variants ===
    'story-7k-vocab-heavy': {
        'vocab_size': 512,
        'dim': 10,
        'hidden_dim': 40,  # 4x dim
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 44,
        'description': '7.0K parameters - Vocabulary heavy'
    },
    'story-7k-balanced': {
        'vocab_size': 160,
        'dim': 18,
        'hidden_dim': 72,  # 4x dim
        'n_layers': 1,
        'n_heads': 3,
        'max_seq_len': 44,
        'description': '7.2K parameters - Balanced 7K design'
    },
    'story-7k-dimension-focus': {
        'vocab_size': 96,
        'dim': 24,
        'hidden_dim': 96,  # 4x dim
        'n_layers': 1,
        'n_heads': 6,
        'max_seq_len': 44,
        'description': '7.0K parameters - Dimension focused'
    },
    'story-7k-deep': {
        'vocab_size': 128,
        'dim': 14,
        'hidden_dim': 56,  # 4x dim
        'n_layers': 2,
        'n_heads': 2,
        'max_seq_len': 44,
        'description': '7.0K parameters - Deep 7K model'
    },
    'story-7k-ultra-deep': {
        'vocab_size': 64,
        'dim': 10,
        'hidden_dim': 40,  # 4x dim
        'n_layers': 3,
        'n_heads': 2,
        'max_seq_len': 44,
        'description': '6.9K parameters - Ultra deep'
    },
    'story-7k-mega-deep': {
        'vocab_size': 48,
        'dim': 8,
        'hidden_dim': 32,  # 4x dim
        'n_layers': 4,
        'n_heads': 2,
        'max_seq_len': 44,
        'description': '6.9K parameters - Mega deep'
    },
    'story-7k-extreme-deep': {
        'vocab_size': 32,
        'dim': 6,
        'hidden_dim': 24,  # 4x dim
        'n_layers': 5,
        'n_heads': 2,
        'max_seq_len': 44,
        'description': '6.9K parameters - Extreme deep'
    },

    # === Original Size Categories ===
    'story-4k': {
        'vocab_size': 112,
        'dim': 14,
        'hidden_dim': 56,  # 4x dim
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 40,
        'description': '3.9K parameters - Story generation'
    },
    # === 8K Parameter Architectural Variants ===
    'chat-8k-vocab-monster': {
        'vocab_size': 640,
        'dim': 10,
        'hidden_dim': 40,  # 4x dim
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 48,
        'description': '8.0K parameters - Vocabulary monster'
    },
    'chat-8k-balanced': {
        'vocab_size': 128,
        'dim': 16,
        'hidden_dim': 64,
        'n_layers': 2,
        'n_heads': 4,
        'max_seq_len': 48,
        'description': '8.2K parameters - Balanced chat design'
    },
    'chat-8k-dimension-beast': {
        'vocab_size': 96,
        'dim': 24,
        'hidden_dim': 96,  # 4x dim
        'n_layers': 1,
        'n_heads': 6,
        'max_seq_len': 48,
        'description': '8.1K parameters - Dimension beast'
    },
    'chat-8k-deep': {
        'vocab_size': 128,
        'dim': 12,
        'hidden_dim': 48,  # 4x dim
        'n_layers': 3,
        'n_heads': 3,
        'max_seq_len': 48,
        'description': '7.9K parameters - Deep chat model'
    },
    'chat-8k-ultra-deep': {
        'vocab_size': 64,
        'dim': 10,
        'hidden_dim': 40,  # 4x dim
        'n_layers': 4,
        'n_heads': 2,
        'max_seq_len': 48,
        'description': '8.0K parameters - Ultra deep'
    },
    'chat-8k-layer-tower': {
        'vocab_size': 48,
        'dim': 8,
        'hidden_dim': 32,  # 4x dim
        'n_layers': 5,
        'n_heads': 2,
        'max_seq_len': 48,
        'description': '7.9K parameters - Layer tower'
    },
    'chat-8k-attention-hydra': {
        'vocab_size': 96,
        'dim': 18,
        'hidden_dim': 72,  # 4x dim
        'n_layers': 1,
        'n_heads': 9,
        'max_seq_len': 48,
        'description': '8.1K parameters - Attention hydra'
    },
    'chat-8k-extreme-wide': {
        'vocab_size': 64,
        'dim': 28,
        'hidden_dim': 112,  # 4x dim
        'n_layers': 1,
        'n_heads': 4,
        'max_seq_len': 48,
        'description': '8.0K parameters - Extreme wide'
    },
    # === 10K Parameter Architectural Variants ===
    'chat-10k-vocab-monster': {
        'vocab_size': 800,
        'dim': 10,
        'hidden_dim': 40,  # 4x dim
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 52,
        'description': '10.0K parameters - Vocabulary monster'
    },
    'chat-10k-balanced': {
        'vocab_size': 144,
        'dim': 18,
        'hidden_dim': 72,  # 4x dim
        'n_layers': 2,
        'n_heads': 3,
        'max_seq_len': 52,
        'description': '10.4K parameters - Balanced chat design'
    },
    'chat-10k-dimension-beast': {
        'vocab_size': 96,
        'dim': 28,
        'hidden_dim': 112,  # 4x dim
        'n_layers': 1,
        'n_heads': 4,
        'max_seq_len': 52,
        'description': '10.1K parameters - Dimension beast'
    },
    'chat-10k-deep': {
        'vocab_size': 128,
        'dim': 16,
        'hidden_dim': 64,  # 4x dim
        'n_layers': 3,
        'n_heads': 4,
        'max_seq_len': 52,
        'description': '10.2K parameters - Deep chat model'
    },
    'chat-10k-ultra-deep': {
        'vocab_size': 80,
        'dim': 12,
        'hidden_dim': 48,  # 4x dim
        'n_layers': 4,
        'n_heads': 3,
        'max_seq_len': 52,
        'description': '10.1K parameters - Ultra deep'
    },
    'chat-10k-layer-tower': {
        'vocab_size': 64,
        'dim': 10,
        'hidden_dim': 40,  # 4x dim
        'n_layers': 5,
        'n_heads': 2,
        'max_seq_len': 52,
        'description': '10.0K parameters - Layer tower'
    },
    'chat-10k-mega-deep': {
        'vocab_size': 48,
        'dim': 8,
        'hidden_dim': 32,  # 4x dim
        'n_layers': 6,
        'n_heads': 2,
        'max_seq_len': 52,
        'description': '10.0K parameters - Mega deep'
    },
    'chat-10k-attention-hydra': {
        'vocab_size': 96,
        'dim': 20,
        'hidden_dim': 80,  # 4x dim
        'n_layers': 1,
        'n_heads': 10,
        'max_seq_len': 52,
        'description': '10.2K parameters - Attention hydra'
    },
    'chat-10k-extreme-wide': {
        'vocab_size': 64,
        'dim': 32,
        'hidden_dim': 128,  # 4x dim
        'n_layers': 1,
        'n_heads': 8,
        'max_seq_len': 52,
        'description': '10.2K parameters - Extreme wide'
    },
    'chat-10k-mixed-extreme': {
        'vocab_size': 200,
        'dim': 16,
        'hidden_dim': 32,  # 2x dim
        'n_layers': 2,
        'n_heads': 8,
        'max_seq_len': 52,
        'description': '10.1K parameters - Mixed extremes'
    },
    'chat-13k': {
        'vocab_size': 160,
        'dim': 20,
        'hidden_dim': 80,  # 4x dim
        'n_layers': 2,
        'n_heads': 4,
        'max_seq_len': 56,
        'description': '12.8K parameters - Chat responses'
    },
    'chat-18k': {
        'vocab_size': 192,
        'dim': 24,
        'hidden_dim': 96,  # 4x dim
        'n_layers': 2,
        'n_heads': 4,
        'max_seq_len': 60,
        'description': '18.4K parameters - Chat responses'
    },
    'assistant-45k': {
        'vocab_size': 256,
        'dim': 32,
        'hidden_dim': 128,
        'n_layers': 3,
        'n_heads': 8,
        'max_seq_len': 64,
        'description': '45.1K parameters - Assistant tasks'
    },
    'assistant-57k': {
        'vocab_size': 288,
        'dim': 36,
        'hidden_dim': 144,  # 4x dim
        'n_layers': 3,
        'n_heads': 6,
        'max_seq_len': 68,
        'description': '57.0K parameters - Assistant tasks'
    },
    'assistant-70k': {
        'vocab_size': 320,
        'dim': 40,
        'hidden_dim': 160,  # 4x dim
        'n_layers': 3,
        'n_heads': 8,
        'max_seq_len': 72,
        'description': '70.4K parameters - Assistant tasks'
    },
    'assistant-101k': {
        'vocab_size': 384,
        'dim': 48,
        'hidden_dim': 192,  # 4x dim
        'n_layers': 3,
        'n_heads': 8,
        'max_seq_len': 80,
        'description': '101.4K parameters - Assistant tasks'
    },
    'expert-229k': {
        'vocab_size': 512,
        'dim': 64,
        'hidden_dim': 256,
        'n_layers': 4,
        'n_heads': 16,
        'max_seq_len': 96,
        'description': '229.4K parameters - Expert tasks'
    },
    'expert-1310k': {
        'vocab_size': 1024,
        'dim': 128,
        'hidden_dim': 512,
        'n_layers': 6,
        'n_heads': 32,
        'max_seq_len': 128,
        'description': '1310.7K parameters - Expert tasks'
    }
}

class ScalableTokenizer:
    """Tokenizer that adapts to different vocab sizes"""
    
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Build vocabulary scaled to vocab_size"""
        # Special tokens (always first 4)
        self.vocab[0] = "<pad>"
        self.vocab[1] = "<s>"
        self.vocab[2] = "</s>"
        self.vocab[3] = "<unk>"
        self.vocab[4] = " "  # Space is crucial
        
        token_id = 5
        
        # Common letters by frequency
        common_chars = "etaoinshrdlcumwfgypbvkjxqz"
        for char in common_chars:
            if token_id < self.vocab_size:
                self.vocab[token_id] = char
                self.reverse_vocab[char] = token_id
                token_id += 1
        
        # Punctuation and symbols
        punct = ".,!?'-;:()[]{}\"@#$%&*+=/<>\n\t"
        for char in punct:
            if token_id < self.vocab_size:
                self.vocab[token_id] = char
                self.reverse_vocab[char] = token_id
                token_id += 1
        
        # Common bigrams (for larger vocabularies)
        if self.vocab_size > 64:
            bigrams = [" th", " he", " in", " er", " an", " re", " ed", " nd", " ha", " to", 
                      " ou", " it", " is", " en", " as", " at", " es", " on", " hi", " st",
                      "ing", "ion", "ter", "ent", "ity", "ous", "ate", "ive"]
            
            for bigram in bigrams:
                if token_id < self.vocab_size:
                    self.vocab[token_id] = bigram
                    self.reverse_vocab[bigram] = token_id
                    token_id += 1
        
        # Common trigrams (for even larger vocabularies)
        if self.vocab_size > 200:
            trigrams = [" the", " and", " for", " are", " but", " not", " you", " all", 
                       " can", " had", " her", " was", " one", " our", " out", " day",
                       "ing ", "ion ", "tion", "ness", "ment", "able", "ible"]
            
            for trigram in trigrams:
                if token_id < self.vocab_size:
                    self.vocab[token_id] = trigram
                    self.reverse_vocab[trigram] = token_id
                    token_id += 1
        
        # Common words (for largest vocabularies)
        if self.vocab_size > 500:
            words = ["the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
                    "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
                    "how", "man", "new", "now", "old", "see", "two", "way", "who", "boy",
                    "did", "its", "let", "put", "say", "she", "too", "use", "about", "after",
                    "again", "before", "came", "come", "could", "each", "first", "from",
                    "good", "great", "here", "just", "know", "last", "like", "long",
                    "look", "made", "make", "many", "much", "never", "only", "other",
                    "right", "said", "same", "should", "since", "still", "such", "take",
                    "than", "them", "time", "very", "want", "water", "well", "were",
                    "what", "when", "where", "which", "while", "with", "work", "would"]
            
            for word in words:
                if token_id < self.vocab_size:
                    # Add both with and without leading space
                    if token_id < self.vocab_size - 1:
                        self.vocab[token_id] = " " + word
                        self.reverse_vocab[" " + word] = token_id
                        token_id += 1
                    
                    if token_id < self.vocab_size:
                        self.vocab[token_id] = word
                        self.reverse_vocab[word] = token_id
                        token_id += 1
        
        # Fill remaining slots with numbers if needed
        for i in range(10):
            if token_id < self.vocab_size:
                self.vocab[token_id] = str(i)
                self.reverse_vocab[str(i)] = token_id
                token_id += 1
        
        print(f"Built vocabulary: {len(self.vocab)}/{self.vocab_size} tokens")
        print("Sample vocab:", {k: v for k, v in list(self.vocab.items())[:20]})
    
    def encode(self, text):
        """Encode with greedy longest-match tokenization"""
        if not text:
            return [1]
        
        tokens = [1]  # BOS
        text = text.lower()
        i = 0
        
        while i < len(text):
            found = False
            # Try longer matches first (up to 8 chars)
            for length in range(min(8, len(text) - i), 0, -1):
                substr = text[i:i+length]
                if substr in self.reverse_vocab:
                    tokens.append(self.reverse_vocab[substr])
                    i += length
                    found = True
                    break
            
            if not found:
                # Single char or unknown
                char = text[i]
                if char in self.reverse_vocab:
                    tokens.append(self.reverse_vocab[char])
                else:
                    tokens.append(3)  # <unk>
                i += 1
        
        return tokens
    
    def decode(self, tokens):
        """Decode tokens to text"""
        text = ""
        for token in tokens:
            if token == 1:  # BOS
                continue
            elif token == 2:  # EOS
                break
            elif 0 <= token < len(self.vocab) and token in self.vocab:
                text += self.vocab[token]
        return text
    
    def save_vocab(self, filename):
        """Save vocabulary for RP2040"""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'wb') as f:
            f.write(struct.pack('H', self.vocab_size))
            for i in range(self.vocab_size):
                if i in self.vocab:
                    token_str = self.vocab[i].encode('utf-8')
                    token_len = min(len(token_str), 255)  # Max 255 bytes
                    f.write(struct.pack('B', token_len))
                    f.write(token_str[:token_len])
                else:
                    f.write(struct.pack('B', 0))
        print(f"Vocabulary saved to {filename}")

class ScalableTransformer:
    """Transformer that scales with config"""
    
    def __init__(self, config):
        self.config = config
        self.vocab_size = config['vocab_size']
        self.dim = config['dim']
        self.hidden_dim = config['hidden_dim']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.max_seq_len = config['max_seq_len']
        self.head_dim = self.dim // self.n_heads
        
        self._init_weights()
        self._count_parameters()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        print(f"Initializing {self.config['description']} model...")
        
        # Scale initialization based on model size
        embed_scale = 0.1
        attn_scale = 0.02 / np.sqrt(self.n_layers)
        ffn_scale = 0.02 / np.sqrt(self.n_layers)
        
        # Token embeddings
        self.token_embedding = np.random.normal(0, embed_scale, 
                                              (self.vocab_size, self.dim)).astype(np.float32)
        
        # Layer weights
        self.layers = []
        for layer_id in range(self.n_layers):
            layer = {
                # Layer norm
                'ln1_weight': np.ones(self.dim, dtype=np.float32),
                'ln2_weight': np.ones(self.dim, dtype=np.float32),
                
                # Attention
                'wq': np.random.normal(0, attn_scale, (self.dim, self.dim)).astype(np.float32),
                'wk': np.random.normal(0, attn_scale, (self.dim, self.dim)).astype(np.float32),
                'wv': np.random.normal(0, attn_scale, (self.dim, self.dim)).astype(np.float32),
                'wo': np.random.normal(0, attn_scale, (self.dim, self.dim)).astype(np.float32),
                
                # Feed-forward
                'w1': np.random.normal(0, ffn_scale, (self.dim, self.hidden_dim)).astype(np.float32),
                'w2': np.random.normal(0, ffn_scale, (self.hidden_dim, self.dim)).astype(np.float32),
            }
            self.layers.append(layer)
        
        # Final layer norm
        self.final_ln_weight = np.ones(self.dim, dtype=np.float32)
    
    def _count_parameters(self):
        """Count parameters"""
        embedding_params = self.vocab_size * self.dim
        layer_params = self.n_layers * (
            4 * self.dim * self.dim +  # attention: q,k,v,o
            self.dim * self.hidden_dim * 2  # ffn: w1, w2
        )
        total = embedding_params + layer_params
        
        print(f"Parameters: {total:,} ({total/1000:.1f}K)")
        print(f"  Embeddings: {embedding_params:,}")
        print(f"  Layers: {layer_params:,}")
        
        # Estimate memory usage
        memory_mb = total * 4 / (1024 * 1024)  # 4 bytes per float32
        memory_kb = total * 4 / 1024
        print(f"Estimated memory: {memory_kb:.1f}KB ({memory_mb:.2f}MB)")
        
        return total
    
    def layer_norm(self, x, weight, eps=1e-6):
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * weight
    
    def attention(self, x, layer):
        """Multi-head self-attention"""
        seq_len, d_model = x.shape
        
        # Linear projections
        Q = x @ layer['wq']
        K = x @ layer['wk']
        V = x @ layer['wv']
        
        # Multi-head attention (simplified - no head splitting for efficiency)
        scores = Q @ K.T / np.sqrt(d_model)
        
        # Causal mask
        mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
        scores = scores + mask
        
        # Softmax
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / (np.sum(attn_weights, axis=-1, keepdims=True) + 1e-8)
        
        # Apply attention
        out = attn_weights @ V
        
        # Output projection
        return out @ layer['wo']
    
    def feed_forward(self, x, layer):
        """Feed-forward network with SwiGLU-like activation"""
        hidden = x @ layer['w1']
        # SiLU activation: x * sigmoid(x)
        sigmoid_hidden = 1 / (1 + np.exp(-np.clip(hidden, -10, 10)))
        activated = hidden * sigmoid_hidden
        return activated @ layer['w2']
    
    def forward(self, tokens):
        """Forward pass"""
        # Embeddings
        x = self.token_embedding[tokens]
        
        # Transformer layers
        for layer in self.layers:
            # Pre-attention norm
            normed = self.layer_norm(x, layer['ln1_weight'])
            
            # Self-attention with residual
            x = x + self.attention(normed, layer)
            
            # Pre-FFN norm  
            normed = self.layer_norm(x, layer['ln2_weight'])
            
            # FFN with residual
            x = x + self.feed_forward(normed, layer)
        
        # Final norm
        x = self.layer_norm(x, self.final_ln_weight)
        
        # Output projection (last position only)
        logits = x[-1] @ self.token_embedding.T
        
        return logits

class ScalableTrainer:
    """Trainer for different model sizes"""
    
    def __init__(self, model_size='tiny'):
        self.config = MODEL_CONFIGS[model_size]
        self.model_size = model_size
        
        print(f"=== Training {model_size.upper()} model ===")
        print(f"Config: {self.config}")
        
        self.tokenizer = ScalableTokenizer(self.config['vocab_size'])
        self.model = ScalableTransformer(self.config)
        self.training_data = []
    
    def prepare_data(self):
        """Prepare training data"""
        # Expanded training data for larger models
        sentences = [
            "hello world how are you today",
            "the cat sat on the mat in the sun",
            "once upon a time there was a little cat",
            "good morning sunshine how are you doing today",
            "the quick brown fox jumps over the lazy dog",
            "i love you very much my dear friend",
            "hello there friend how are you doing today",
            "the cat likes to play in the garden",
            "good night sleep well my dear friend",
            "how are you today my good friend",
            "the sun is shining bright in the sky",
            "cats and dogs are very good friends",
            "once there was a happy little cat in the garden",
            "hello good morning how are you today",
            "the little cat sat by the warm fire",
            "i want to go to the store today",
            "the weather is very nice today sunshine",
            "can you help me with my homework please",
            "the children are playing in the playground",
            "my mother is cooking dinner in the kitchen",
            "the dog is running fast in the park",
            "we are going to visit our friends tomorrow",
            "the book is on the table by the window",
            "she is reading a story to her children",
            "the birds are singing beautiful songs outside",
            "winter is coming and the leaves are falling",
            "the computer is working very well today",
            "programming is fun when you understand it",
            "artificial intelligence is changing the world rapidly",
            "machine learning helps us solve difficult problems"
        ]
        
        # Create training examples
        self.training_data = []
        for sentence in sentences:
            tokens = self.tokenizer.encode(sentence)
            if len(tokens) > 1:
                for i in range(1, len(tokens)):
                    context = tokens[:i]
                    target = tokens[i]
                    if len(context) <= self.config['max_seq_len']:
                        self.training_data.append((context, target))
        
        print(f"Created {len(self.training_data)} training examples")
    
    def train(self, epochs=50, learning_rate=0.001):
        """Train the model"""
        print(f"Training for {epochs} epochs with lr={learning_rate}")
        
        losses = []
        best_loss = float('inf')
        
        # Adjust learning rate based on model size
        if self.config['dim'] > 32:
            learning_rate *= 0.5  # Lower LR for bigger models
        
        for epoch in range(epochs):
            epoch_loss = 0
            random.shuffle(self.training_data)
            
            for i, (context, target) in enumerate(self.training_data):
                try:
                    # Forward pass
                    logits = self.model.forward(np.array(context))
                    
                    # Compute loss
                    exp_logits = np.exp(logits - np.max(logits))
                    probs = exp_logits / (np.sum(exp_logits) + 1e-8)
                    loss = -np.log(probs[target] + 1e-8)
                    epoch_loss += loss
                    
                    # Simple gradient update on embeddings
                    error = 1.0 - probs[target]
                    
                    # Update target token (increase)
                    if target < self.model.vocab_size:
                        self.model.token_embedding[target] += learning_rate * error * 0.1
                    
                    # Update other tokens (decrease slightly)
                    for j in range(min(100, self.model.vocab_size)):  # Limit updates for speed
                        if j != target and probs[j] > 0.01:  # Only update probable tokens
                            self.model.token_embedding[j] -= learning_rate * probs[j] * 0.01
                
                except Exception as e:
                    print(f"Training error at epoch {epoch}, sample {i}: {e}")
                    continue
            
            avg_loss = epoch_loss / len(self.training_data)
            losses.append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            if epoch % 10 == 0 or epoch < 5:
                print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f} (best: {best_loss:.4f})")
                self.test_generation()
        
        print(f"Training complete! Final loss: {losses[-1]:.4f}")
        return losses
    
    def test_generation(self):
        """Test generation during training"""
        test_prompts = ["hello", "the cat", "good morning"]
        
        for prompt in test_prompts:
            try:
                tokens = self.tokenizer.encode(prompt)
                result_tokens = tokens[:]
                
                for _ in range(6):
                    if len(result_tokens) >= self.config['max_seq_len']:
                        break
                    
                    logits = self.model.forward(np.array(result_tokens))
                    
                    # Sample from top-5
                    top_indices = np.argsort(logits)[-5:]
                    top_logits = logits[top_indices]
                    probs = np.exp(top_logits - np.max(top_logits))
                    probs = probs / np.sum(probs)
                    
                    next_token = np.random.choice(top_indices, p=probs)
                    result_tokens.append(next_token)
                    
                    if next_token == 2:  # EOS
                        break
                
                result_text = self.tokenizer.decode(result_tokens)
                print(f"  '{prompt}' -> '{result_text}'")
                
            except Exception as e:
                print(f"  '{prompt}' -> [generation error: {e}]")
    
    def save_model(self):
        """Save model with size-specific filename"""
        # Create models directory if it doesn't exist
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"Created {models_dir} directory")
        
        # Create model-specific folder
        model_folder = os.path.join(models_dir, self.model_size)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
            print(f"Created {model_folder} directory")
        
        # Get actual parameter count for filename
        param_count = self.model._count_parameters()
        
        model_filename = os.path.join(model_folder, f"model_{param_count}p.bin")
        vocab_filename = os.path.join(model_folder, f"vocab_{param_count}p.bin")
        
        print(f"Saving {self.model_size} model...")
        
        # Save model config first
        config_filename = os.path.join(model_folder, f"config_{param_count}p.json")
        with open(config_filename, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save model weights
        with open(model_filename, 'wb') as f:
            # Header with config
            header = struct.pack("8I", 
                               self.config['vocab_size'], 
                               self.config['dim'],
                               self.config['hidden_dim'], 
                               self.config['n_layers'],
                               self.config['n_heads'], 
                               self.config['max_seq_len'],
                               0, 0)  # Reserved
            f.write(header)
            
            # Token embeddings
            f.write(self.model.token_embedding.astype(np.float32).tobytes())
            
            # Layer weights
            for layer in self.model.layers:
                f.write(layer['ln1_weight'].astype(np.float32).tobytes())
                f.write(layer['ln2_weight'].astype(np.float32).tobytes())
                f.write(layer['wq'].astype(np.float32).tobytes())
                f.write(layer['wk'].astype(np.float32).tobytes())
                f.write(layer['wv'].astype(np.float32).tobytes())
                f.write(layer['wo'].astype(np.float32).tobytes())
                f.write(layer['w1'].astype(np.float32).tobytes())
                f.write(layer['w2'].astype(np.float32).tobytes())
            
            # Final layer norm
            f.write(self.model.final_ln_weight.astype(np.float32).tobytes())
        
        # Save vocabulary
        self.tokenizer.save_vocab(vocab_filename)
        
        print(f"Files created in {model_folder}/:")
        print(f"  model_{param_count}p.bin - model weights")
        print(f"  vocab_{param_count}p.bin - vocabulary") 
        print(f"  config_{param_count}p.json - configuration")

def estimate_model_size(vocab_size, dim, hidden_dim, n_layers, n_heads):
    """Estimate model parameters and memory usage"""
    embedding_params = vocab_size * dim
    layer_params = n_layers * (
        4 * dim * dim +  # attention: q,k,v,o
        dim * hidden_dim * 2  # ffn: w1, w2
    )
    total_params = embedding_params + layer_params
    
    memory_kb = total_params * 4 / 1024  # 4 bytes per float32
    memory_mb = memory_kb / 1024
    
    print(f"Model Size Estimation:")
    print(f"  Vocab: {vocab_size}, Dim: {dim}, Hidden: {hidden_dim}")
    print(f"  Layers: {n_layers}, Heads: {n_heads}")
    print(f"  Parameters: {total_params:,} ({total_params/1000:.1f}K)")
    print(f"  Memory: {memory_kb:.1f}KB ({memory_mb:.2f}MB)")
    
    return total_params, memory_kb

def quick_size_test():
    """Quick test of different sizes without training"""
    print("=== Quick Model Size Test ===")
    print("Testing different configurations...")
    
    # Test some intermediate sizes with accurate naming
    test_configs = [
        (128, 16, 64, 2, 4),      # 8.2K params (chat-8k)
        (144, 18, 72, 2, 3),      # 10.4K params (chat-10k)
        (160, 20, 80, 2, 4),      # 12.8K params (chat-13k)
        (192, 24, 96, 2, 4),      # 18.4K params (chat-18k)
        (224, 28, 112, 2, 4),     # ~25K params
        (256, 32, 128, 3, 8),     # 45.1K params (assistant-45k)
        (288, 36, 144, 3, 6),     # 57.0K params (assistant-57k)
        (320, 40, 160, 3, 8),     # 70.4K params (assistant-70k)
        (384, 48, 192, 3, 8),     # 101.4K params (assistant-101k)
        (448, 56, 224, 3, 8),     # ~150K params
    ]
    
    print("\nSize estimates:")
    for config in test_configs:
        estimate_model_size(*config)
        print()

def test_all_sizes():
    """Test training all model sizes"""
    results = {}
    
    for size_name in ['story-1k-wide', 'story-1k-balanced', 'story-1k-deep', 'story-1k-narrow', 'story-1k-fat', 'story-1k-mini', 'story-1k-micro', 'story-1k-ultra', 'story-3k', 'story-4k', 'chat-8k', 'chat-10k', 'chat-13k', 'chat-18k', 'assistant-45k', 'assistant-57k', 'assistant-70k', 'assistant-101k', 'expert-229k', 'expert-1310k']:
        print(f"\n{'='*50}")
        print(f"TESTING {size_name.upper()} MODEL")
        print(f"{'='*50}")
        
        try:
            start_time = time.time()
            
            # Create and train model
            trainer = ScalableTrainer(size_name)
            trainer.prepare_data()
            
            # Adjust epochs based on size
            epochs = max(20, 100 // (trainer.config['dim'] // 8))
            losses = trainer.train(epochs=epochs, learning_rate=0.005)
            
            training_time = time.time() - start_time
            
            # Save model
            trainer.save_model()
            
            # Record results
            param_count = trainer.model._count_parameters()
            results[size_name] = {
                'parameters': param_count,
                'training_time': training_time,
                'final_loss': losses[-1],
                'config': trainer.config
            }
            
            print(f"{size_name.upper()} COMPLETE: {param_count:,} params, {training_time:.1f}s training")
            
        except Exception as e:
            print(f"FAILED {size_name}: {e}")
            results[size_name] = {'error': str(e)}
    
    # Print summary
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    
    for size, result in results.items():
        if 'error' in result:
            print(f"{size:15s}: FAILED - {result['error']}")
        else:
            print(f"{size:15s}: {result['parameters']:7,} params, "
                  f"{result['training_time']:5.1f}s, "
                  f"loss={result['final_loss']:.3f}")

def find_rp2040_limit():
    """Test models sequentially to find RP2040 memory limit"""
    print("=== Finding RP2040 Memory Limit ===")
    print("Testing models in order until one fails...")
    
    # Test sizes in ascending order of parameters (start with fastest 1K variant)
    test_sizes = ['story-1k-balanced', 'story-3k', 'story-4k', 'chat-8k', 'chat-10k', 'chat-13k', 'chat-18k', 'assistant-45k', 'assistant-57k', 'assistant-70k', 'assistant-101k', 'expert-229k', 'expert-1310k']
    
    working_models = []
    failed_models = []
    
    for size_name in test_sizes:
        print(f"\n{'='*50}")
        print(f"TESTING {size_name.upper()} MODEL")
        print(f"{'='*50}")
        
        try:
            start_time = time.time()
            
            # Create and train model
            trainer = ScalableTrainer(size_name)
            trainer.prepare_data()
            
            # Quick training (fewer epochs for speed)
            epochs = max(10, 50 // (trainer.config['dim'] // 8))
            losses = trainer.train(epochs=epochs, learning_rate=0.01)
            
            training_time = time.time() - start_time
            
            # Save model
            trainer.save_model()
            
            # Record results
            param_count = trainer.model._count_parameters()
            working_models.append({
                'size': size_name,
                'parameters': param_count,
                'training_time': training_time,
                'final_loss': losses[-1],
                'config': trainer.config
            })
            
            print(f"✓ {size_name.upper()} SUCCESS: {param_count:,} params, {training_time:.1f}s training")
            
        except Exception as e:
            print(f"✗ {size_name.upper()} FAILED: {e}")
            failed_models.append({
                'size': size_name,
                'error': str(e)
            })
            
            # Ask if user wants to continue testing
            if input(f"\n{size_name} failed. Continue testing larger models? (y/n): ").lower() != 'y':
                break
    
    # Print summary
    print(f"\n{'='*50}")
    print("RP2040 LIMIT TESTING SUMMARY")
    print(f"{'='*50}")
    
    if working_models:
        print("WORKING MODELS:")
        for model in working_models:
            print(f"  ✓ {model['size']:12s}: {model['parameters']:7,} params")
        
        largest_working = max(working_models, key=lambda x: x['parameters'])
        print(f"\nLARGEST WORKING: {largest_working['size']} ({largest_working['parameters']:,} params)")
    
    if failed_models:
        print("\nFAILED MODELS:")
        for model in failed_models:
            print(f"  ✗ {model['size']:12s}: {model['error']}")
    
    if working_models and failed_models:
        print(f"\nESTIMATED RP2040 LIMIT: Between {working_models[-1]['parameters']:,} and {failed_models[0]['parameters']:,} parameters")

def custom_model():
    """Create and train a custom model size"""
    print("=== Custom Model Creation ===")
    
    try:
        print("Enter model parameters:")
        vocab_size = int(input("Vocab size (64-1024): "))
        dim = int(input("Model dimension (8-128): "))
        hidden_dim = int(input("Hidden dimension (32-512): "))
        n_layers = int(input("Number of layers (1-6): "))
        n_heads = int(input("Number of heads (2-32): "))
        max_seq_len = int(input("Max sequence length (32-128): "))
        
        # Validate inputs
        if not (64 <= vocab_size <= 1024 and 8 <= dim <= 128 and 
                32 <= hidden_dim <= 512 and 1 <= n_layers <= 6 and 
                2 <= n_heads <= 32 and 32 <= max_seq_len <= 128):
            print("Invalid parameters! Using safe defaults.")
            vocab_size, dim, hidden_dim, n_layers, n_heads, max_seq_len = 128, 16, 64, 2, 4, 48
        
        # Estimate size
        total_params, memory_kb = estimate_model_size(vocab_size, dim, hidden_dim, n_layers, n_heads)
        
        if memory_kb > 200:  # Warn if >200KB
            print(f"⚠️  Warning: This model will use {memory_kb:.1f}KB - may be too large for RP2040!")
            if input("Continue anyway? (y/n): ").lower() != 'y':
                return
        
        # Create custom config
        custom_config = {
            'vocab_size': vocab_size,
            'dim': dim,
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'max_seq_len': max_seq_len,
            'description': f'~{total_params//1000}K parameters (custom)'
        }
        
        # Create and train
        print(f"\nCreating custom model with {total_params:,} parameters...")
        trainer = ScalableTrainer('custom')
        trainer.config = custom_config
        trainer.model = ScalableTransformer(custom_config)
        
        trainer.prepare_data()
        epochs = max(20, 100 // max(1, dim // 8))
        trainer.train(epochs=epochs, learning_rate=0.005)
        
        # Create models directory if it doesn't exist
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"Created {models_dir} directory")
        
        # Create model-specific folder
        model_folder = os.path.join(models_dir, 'custom')
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
            print(f"Created {model_folder} directory")
        
        # Save with custom name
        model_filename = os.path.join(model_folder, f"model_{total_params}p.bin")
        vocab_filename = os.path.join(model_folder, f"vocab_{total_params}p.bin")
        config_filename = os.path.join(model_folder, f"config_{total_params}p.json")
        
        # Save config
        with open(config_filename, 'w') as f:
            json.dump(custom_config, f, indent=2)
        
        # Save model weights
        with open(model_filename, 'wb') as f:
            header = struct.pack("8I", 
                               custom_config['vocab_size'], 
                               custom_config['dim'],
                               custom_config['hidden_dim'], 
                               custom_config['n_layers'],
                               custom_config['n_heads'], 
                               custom_config['max_seq_len'],
                               0, 0)
            f.write(header)
            
            f.write(trainer.model.token_embedding.astype(np.float32).tobytes())
            
            for layer in trainer.model.layers:
                f.write(layer['ln1_weight'].astype(np.float32).tobytes())
                f.write(layer['ln2_weight'].astype(np.float32).tobytes())
                f.write(layer['wq'].astype(np.float32).tobytes())
                f.write(layer['wk'].astype(np.float32).tobytes())
                f.write(layer['wv'].astype(np.float32).tobytes())
                f.write(layer['wo'].astype(np.float32).tobytes())
                f.write(layer['w1'].astype(np.float32).tobytes())
                f.write(layer['w2'].astype(np.float32).tobytes())
            
            f.write(trainer.model.final_ln_weight.astype(np.float32).tobytes())
        
        # Save vocabulary
        trainer.tokenizer.save_vocab(vocab_filename)
        
        print(f"\nCustom model saved:")
        print(f"  {os.path.basename(model_filename)}")
        print(f"  {os.path.basename(vocab_filename)}")
        print(f"  {os.path.basename(config_filename)}")
        
    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"Error creating custom model: {e}")

def list_models():
    """List existing models in the models folder"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models folder found. Train a model first!")
        return
    
    print(f"=== Models in {models_dir}/ ===")
    
    # Find all model folders
    model_folders = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains model files
            model_files = [f for f in os.listdir(item_path) if f.startswith("model_") and f.endswith(".bin")]
            if model_files:
                model_folders.append(item)
    
    if not model_folders:
        print("No trained models found.")
        return
    
    print(f"Found {len(model_folders)} model folders:")
    
    # Process each model folder
    for model_name in sorted(model_folders):
        model_folder = os.path.join(models_dir, model_name)
        print(f"\n  {model_name}:")
        
        # Find all files in this model's folder
        files = os.listdir(model_folder)
        model_files = [f for f in files if f.startswith("model_") and f.endswith(".bin")]
        vocab_files = [f for f in files if f.startswith("vocab_") and f.endswith(".bin")]
        config_files = [f for f in files if f.startswith("config_") and f.endswith(".json")]
        
        # Extract parameter count from model filename
        if model_files:
            model_file = model_files[0]  # Should be only one
            param_str = model_file[6:-4]  # Remove "model_" prefix and ".bin" suffix
            if param_str.endswith('p'):
                try:
                    param_count = int(param_str[:-1])
                    print(f"    Parameters: {param_count:,}")
                    
                    # Show file sizes
                    for file_type, file_list in [("Model", model_files), ("Vocab", vocab_files), ("Config", config_files)]:
                        for filename in file_list:
                            filepath = os.path.join(model_folder, filename)
                            if os.path.exists(filepath):
                                size_kb = os.path.getsize(filepath) / 1024
                                print(f"    {file_type}: {filename} ({size_kb:.1f}KB)")
                    
                    # Calculate total size for this model
                    total_size = 0
                    for filename in model_files + vocab_files + config_files:
                        filepath = os.path.join(model_folder, filename)
                        if os.path.exists(filepath):
                            total_size += os.path.getsize(filepath)
                    
                    total_kb = total_size / 1024
                    if total_kb > 1024:
                        size_str = f"{total_kb/1024:.2f}MB"
                    else:
                        size_str = f"{total_kb:.1f}KB"
                    print(f"    Total size: {size_str}")
                    
                except ValueError:
                    print(f"    [Could not parse parameter count from {model_file}]")
        else:
            print(f"    [No model files found]")

def cleanup_models():
    """Clean up models folder - remove all model files"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models folder found.")
        return
    
    print(f"=== Cleaning up {models_dir}/ ===")
    
    # Find all model folders
    model_folders = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains model files
            model_files = [f for f in os.listdir(item_path) if f.endswith(('.bin', '.json'))]
            if model_files:
                model_folders.append(item)
    
    if not model_folders:
        print("No model folders found.")
        return
    
    print(f"Found {len(model_folders)} model folders:")
    
    # Display grouped files by model folder
    total_files = 0
    for model_name in sorted(model_folders):
        model_folder = os.path.join(models_dir, model_name)
        files = os.listdir(model_folder)
        model_files = [f for f in files if f.endswith(('.bin', '.json'))]
        
        if model_files:
            print(f"  {model_name}:")
            total_size = 0
            
            for filename in sorted(model_files):
                filepath = os.path.join(model_folder, filename)
                size_kb = os.path.getsize(filepath) / 1024
                total_size += size_kb
                print(f"    {filename} ({size_kb:.1f}KB)")
            
            total_mb = total_size / 1024
            if total_mb > 1:
                print(f"    Total: {total_mb:.2f}MB")
            else:
                print(f"    Total: {total_size:.1f}KB")
            print()
            
            total_files += len(model_files)
    
    confirm = input(f"\nRemove all {total_files} files from {len(model_folders)} model folders? (y/N): ").strip().lower()
    if confirm == 'y':
        removed_count = 0
        removed_folders = 0
        
        for model_name in model_folders:
            model_folder = os.path.join(models_dir, model_name)
            files = os.listdir(model_folder)
            
            # Remove all files in the model folder
            for filename in files:
                if filename.endswith(('.bin', '.json')):
                    filepath = os.path.join(model_folder, filename)
                    try:
                        os.remove(filepath)
                        removed_count += 1
                    except Exception as e:
                        print(f"  Error removing {filename}: {e}")
            
            # Try to remove the empty folder
            try:
                os.rmdir(model_folder)
                removed_folders += 1
                print(f"  Removed folder: {model_name}/")
            except:
                print(f"  Kept folder: {model_name}/ (not empty)")
        
        print(f"\nRemoved {removed_count} files and {removed_folders} folders.")
        
        # Remove empty models directory
        try:
            os.rmdir(models_dir)
            print(f"Removed empty {models_dir}/ directory")
        except:
            print(f"Kept {models_dir}/ directory (not empty)")
    else:
        print("Cleanup cancelled.")

def show_disk_usage():
    """Show disk usage of models folder"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models folder found.")
        return
    
    print(f"=== Disk Usage for {models_dir}/ ===")
    
    total_size = 0
    file_count = 0
    
    # Get all files and their sizes
    files_info = []
    model_groups = {}
    
    # Process each model folder
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains model files
            model_files = [f for f in os.listdir(item_path) if f.startswith("model_") and f.endswith(".bin")]
            if model_files:
                # This is a model folder
                model_name = item
                folder_size = 0
                folder_files = 0
                
                # Process all files in this model folder
                for filename in os.listdir(item_path):
                    filepath = os.path.join(item_path, filename)
                    if os.path.isfile(filepath):
                        size = os.path.getsize(filepath)
                        folder_size += size
                        folder_files += 1
                        total_size += size
                        file_count += 1
                        files_info.append((f"{model_name}/{filename}", size))
                
                # Extract parameter count from model filename for grouping
                if model_files:
                    model_file = model_files[0]
                    param_str = model_file[6:-4]  # Remove "model_" prefix and ".bin" suffix
                    if param_str.endswith('p'):
                        try:
                            param_count = int(param_str[:-1])
                            if model_name not in model_groups:
                                model_groups[model_name] = {'param_count': param_count, 'total_size': 0, 'file_count': 0}
                            model_groups[model_name]['total_size'] += folder_size
                            model_groups[model_name]['file_count'] += folder_files
                        except ValueError:
                            pass
    
    if not files_info:
        print("No files found.")
        return
    
    # Sort by size (largest first)
    files_info.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Total files: {file_count}")
    print(f"Total size: {total_size/1024:.1f}KB ({total_size/(1024*1024):.2f}MB)")
    print()
    
    # Show summary by model type
    if model_groups:
        print("Models by parameter count:")
        for model_name in sorted(model_groups.keys(), key=lambda x: model_groups[x]['param_count']):
            info = model_groups[model_name]
            size_kb = info['total_size'] / 1024
            if size_kb > 1024:
                size_str = f"{size_kb/1024:.2f}MB"
            else:
                size_str = f"{size_kb:.1f}KB"
            print(f"  {model_name:20s} ({info['param_count']:6,} params): {size_str:>8s}")
        print()
    
    print("All files by size:")
    for filepath, size in files_info:
        size_kb = size / 1024
        if size_kb > 1024:
            size_str = f"{size_kb/1024:.2f}MB"
        else:
            size_str = f"{size_kb:.1f}KB"
        print(f"  {filepath:35s}: {size_str:>8s}")
    
    print()
    print(f"Average file size: {total_size/file_count/1024:.1f}KB")

def find_models_by_params(min_params=None, max_params=None):
    """Find models within a parameter count range"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models folder found.")
        return
    
    print(f"=== Finding Models by Parameter Count ===")
    
    if min_params is None and max_params is None:
        min_params = int(input("Minimum parameters (or press Enter for no limit): ") or "0")
        max_params = int(input("Maximum parameters (or press Enter for no limit): ") or "999999")
    
    print(f"Searching for models with {min_params:,} to {max_params:,} parameters...")
    print()
    
    # Find all model files
    model_files = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains model files
            folder_model_files = [f for f in os.listdir(item_path) if f.startswith("model_") and f.endswith(".bin")]
            for model_file in folder_model_files:
                model_files.append((item, model_file))  # (folder_name, filename)
    
    if not model_files:
        print("No model files found.")
        return
    
    # Extract model info and filter by parameter count
    matching_models = []
    for folder_name, filename in model_files:
        # Extract parameter count from model filename (e.g., "model_1024p.bin" -> 1024)
        param_str = filename[6:-4]  # Remove "model_" prefix and ".bin" suffix
        
        if param_str.endswith('p'):
            try:
                param_count = int(param_str[:-1])  # Remove 'p' and convert to int
                if min_params <= param_count <= max_params:
                    # Check if all related files exist
                    model_file = f"model_{param_count}p.bin"
                    vocab_file = f"vocab_{param_count}p.bin"
                    config_file = f"config_{param_count}p.json"
                    
                    model_path = os.path.join(models_dir, folder_name, model_file)
                    vocab_path = os.path.join(models_dir, folder_name, vocab_file)
                    config_path = os.path.join(models_dir, folder_name, config_file)
                    
                    if all(os.path.exists(p) for p in [model_path, vocab_path, config_path]):
                        model_size = os.path.getsize(model_path)
                        vocab_size = os.path.getsize(vocab_path)
                        total_size = model_size + vocab_size
                        
                        matching_models.append({
                            'type': folder_name,
                            'params': param_count,
                            'model_size': model_size,
                            'vocab_size': vocab_size,
                            'total_size': total_size,
                            'files': [f"{folder_name}/{model_file}", f"{folder_name}/{vocab_file}", f"{folder_name}/{config_file}"]
                        })
            except ValueError:
                continue
    
    if not matching_models:
        print(f"No models found with {min_params:,} to {max_params:,} parameters.")
        return
    
    # Sort by parameter count
    matching_models.sort(key=lambda x: x['params'])
    
    print(f"Found {len(matching_models)} matching models:")
    print()
    
    for model in matching_models:
        total_kb = model['total_size'] / 1024
        if total_kb > 1024:
            size_str = f"{total_kb/1024:.2f}MB"
        else:
            size_str = f"{total_kb:.1f}KB"
        
        print(f"  {model['type']:20s} ({model['params']:6,} params): {size_str:>8s}")
        print(f"    Files: {', '.join(model['files'])}")
        print()
    
    # Summary
    total_params = sum(m['params'] for m in matching_models)
    total_size = sum(m['total_size'] for m in matching_models)
    total_kb = total_size / 1024
    
    print(f"Summary:")
    print(f"  Total models: {len(matching_models)}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Total disk usage: {total_kb:.1f}KB ({total_kb/1024:.2f}MB)")
    
    # RP2040 recommendations
    print(f"\nRP2040 Recommendations:")
    working_models = [m for m in matching_models if m['params'] <= 15000]  # Conservative estimate
    if working_models:
        largest_working = max(working_models, key=lambda x: x['params'])
        print(f"  ✓ Models likely to work: {len(working_models)} (up to {largest_working['params']:,} params)")
    else:
        print(f"  ⚠️  No models likely to work on RP2040")
    
    risky_models = [m for m in matching_models if 15000 < m['params'] <= 30000]
    if risky_models:
        print(f"  ⚠️  Models to test carefully: {len(risky_models)} ({min(m['params'] for m in risky_models):,} - {max(m['params'] for m in risky_models):,} params)")
    
    large_models = [m for m in matching_models if m['params'] > 30000]
    if large_models:
        print(f"  ✗ Models unlikely to work: {len(large_models)} ({min(m['params'] for m in large_models):,} - {max(m['params'] for m in large_models):,} params)")

def test_1k_variants():
    """Test all 1K parameter variants to compare architectural impact on speed"""
    print("=== Testing 1K Parameter Architectural Variants ===")
    print("Comparing different architectures with similar parameter counts")
    
    # Get all 1K variants
    variants = [name for name in MODEL_CONFIGS.keys() if name.startswith('story-1k')]
    
    print(f"\nFound {len(variants)} variants to test:")
    for variant in variants:
        config = MODEL_CONFIGS[variant]
        params = estimate_model_size(config['vocab_size'], config['dim'], 
                                   config['hidden_dim'], config['n_layers'], 
                                   config['n_heads'])[0]
        print(f"  {variant:20s}: {params:4d} params - {config['description']}")
    
    results = {}
    
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"TESTING {variant.upper()}")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            
            # Create and train model
            trainer = ScalableTrainer(variant)
            trainer.prepare_data()
            
            # Quick training for comparison
            epochs = 30  # Consistent training for all variants
            losses = trainer.train(epochs=epochs, learning_rate=0.01)
            
            training_time = time.time() - start_time
            
            # Save model
            trainer.save_model()
            
            # Record results
            param_count = trainer.model._count_parameters()
            config = trainer.config
            
            results[variant] = {
                'parameters': param_count,
                'training_time': training_time,
                'final_loss': losses[-1],
                'vocab_size': config['vocab_size'],
                'dim': config['dim'],
                'hidden_dim': config['hidden_dim'],
                'n_layers': config['n_layers'],
                'n_heads': config['n_heads'],
                'config': config
            }
            
            print(f"✓ {variant.upper()} SUCCESS: {param_count:,} params, {training_time:.1f}s training")
            
        except Exception as e:
            print(f"✗ {variant.upper()} FAILED: {e}")
            results[variant] = {'error': str(e)}
    
    # Print detailed comparison
    print(f"\n{'='*80}")
    print("1K PARAMETER ARCHITECTURAL COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Variant':20s} {'Params':6s} {'Vocab':5s} {'Dim':4s} {'Hid':4s} {'Lay':3s} {'Heads':5s} {'Loss':6s} {'Time':6s}")
    print("-" * 80)
    
    successful_variants = []
    for variant, result in results.items():
        if 'error' not in result:
            successful_variants.append((variant, result))
            print(f"{variant:20s} {result['parameters']:6,} {result['vocab_size']:5d} "
                  f"{result['dim']:4d} {result['hidden_dim']:4d} {result['n_layers']:3d} "
                  f"{result['n_heads']:5d} {result['final_loss']:6.3f} {result['training_time']:6.1f}s")
    
    # Analysis
    if successful_variants:
        print(f"\n📊 ARCHITECTURAL ANALYSIS:")
        
        # Sort by parameter count
        successful_variants.sort(key=lambda x: x[1]['parameters'])
        
        print(f"\nParameter count range:")
        min_params = successful_variants[0][1]['parameters']
        max_params = successful_variants[-1][1]['parameters']
        print(f"  Min: {min_params:,} params ({successful_variants[0][0]})")
        print(f"  Max: {max_params:,} params ({successful_variants[-1][0]})")
        
        # Sort by training time (speed proxy)
        by_speed = sorted(successful_variants, key=lambda x: x[1]['training_time'])
        print(f"\nTraining speed ranking (fastest to slowest):")
        for i, (variant, result) in enumerate(by_speed):
            print(f"  {i+1}. {variant:20s}: {result['training_time']:5.1f}s "
                  f"(v={result['vocab_size']}, d={result['dim']}, l={result['n_layers']})")
        
        # Sort by loss (quality proxy)
        by_loss = sorted(successful_variants, key=lambda x: x[1]['final_loss'])
        print(f"\nTraining quality ranking (best to worst loss):")
        for i, (variant, result) in enumerate(by_loss):
            print(f"  {i+1}. {variant:20s}: {result['final_loss']:.3f} loss "
                  f"(v={result['vocab_size']}, d={result['dim']}, l={result['n_layers']})")
        
        print(f"\n🎯 SPEED vs ARCHITECTURE INSIGHTS:")
        print("  - Wide vocab models: Larger embedding matrices")
        print("  - Deep models: More layer computations")
        print("  - High-dim models: Expensive attention/FFN")
        print("  - Multi-head models: More attention computations")
        
        return results
    else:
        print("❌ No variants completed successfully!")
        return {}

def test_3k_variants():
    """Test all 3K parameter variants"""
    print("=== Testing 3K Parameter Architectural Variants ===")
    variants = [name for name in MODEL_CONFIGS.keys() if name.startswith('story-3k')]
    return _test_parameter_range(variants, "3K")

def test_5k_variants():
    """Test all 5K parameter variants"""
    print("=== Testing 5K Parameter Architectural Variants ===")
    variants = [name for name in MODEL_CONFIGS.keys() if name.startswith('story-5k')]
    return _test_parameter_range(variants, "5K")

def test_7k_variants():
    """Test all 7K parameter variants"""
    print("=== Testing 7K Parameter Architectural Variants ===")
    variants = [name for name in MODEL_CONFIGS.keys() if name.startswith('story-7k')]
    return _test_parameter_range(variants, "7K")

def test_8k_variants():
    """Test all 8K parameter variants"""
    print("=== Testing 8K Parameter Architectural Variants ===")
    variants = [name for name in MODEL_CONFIGS.keys() if name.startswith('chat-8k')]
    return _test_parameter_range(variants, "8K")

def test_10k_variants():
    """Test all 10K parameter variants"""
    print("=== Testing 10K Parameter Architectural Variants ===")
    variants = [name for name in MODEL_CONFIGS.keys() if name.startswith('chat-10k')]
    return _test_parameter_range(variants, "10K")

def test_all_variants():
    """Test ALL architectural variants across all parameter ranges"""
    print("=== COMPREHENSIVE ARCHITECTURAL STUDY ===")
    print("Testing ALL variants from 1K to 10K parameters")
    
    all_variants = []
    for name in MODEL_CONFIGS.keys():
        if any(name.startswith(prefix) for prefix in ['story-1k', 'story-3k', 'story-5k', 'story-7k', 'chat-8k', 'chat-10k']):
            all_variants.append(name)
    
    print(f"Found {len(all_variants)} architectural variants to test!")
    return _test_parameter_range(all_variants, "ALL")

def _test_parameter_range(variants, range_name):
    """Helper function to test a range of parameter variants"""
    print(f"\nFound {len(variants)} variants to test:")
    for variant in variants:
        config = MODEL_CONFIGS[variant]
        params = estimate_model_size(config['vocab_size'], config['dim'], 
                                   config['hidden_dim'], config['n_layers'], 
                                   config['n_heads'])[0]
        print(f"  {variant:25s}: {params:5,} params - {config['description']}")
    
    results = {}
    
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"TESTING {variant.upper()}")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            
            # Create and train model
            trainer = ScalableTrainer(variant)
            trainer.prepare_data()
            
            # Quick training for comparison
            epochs = 30  # Consistent training for all variants
            losses = trainer.train(epochs=epochs, learning_rate=0.01)
            
            training_time = time.time() - start_time
            
            # Save model
            trainer.save_model()
            
            # Record results
            param_count = trainer.model._count_parameters()
            config = trainer.config
            
            results[variant] = {
                'parameters': param_count,
                'training_time': training_time,
                'final_loss': losses[-1],
                'vocab_size': config['vocab_size'],
                'dim': config['dim'],
                'hidden_dim': config['hidden_dim'],
                'n_layers': config['n_layers'],
                'n_heads': config['n_heads'],
                'config': config
            }
            
            print(f"✓ {variant.upper()} SUCCESS: {param_count:,} params, {training_time:.1f}s training")
            
        except Exception as e:
            print(f"✗ {variant.upper()} FAILED: {e}")
            results[variant] = {'error': str(e)}
    
    # Print detailed comparison
    print(f"\n{'='*80}")
    print(f"{range_name} PARAMETER ARCHITECTURAL COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Variant':25s} {'Params':6s} {'Vocab':5s} {'Dim':4s} {'Hid':4s} {'Lay':3s} {'Heads':5s} {'Loss':6s} {'Time':6s}")
    print("-" * 80)
    
    successful_variants = []
    for variant, result in results.items():
        if 'error' not in result:
            successful_variants.append((variant, result))
            print(f"{variant:25s} {result['parameters']:6,} {result['vocab_size']:5d} "
                  f"{result['dim']:4d} {result['hidden_dim']:4d} {result['n_layers']:3d} "
                  f"{result['n_heads']:5d} {result['final_loss']:6.3f} {result['training_time']:6.1f}s")
    
    # Analysis
    if successful_variants:
        print(f"\n📊 {range_name} ARCHITECTURAL ANALYSIS:")
        
        # Sort by parameter count
        successful_variants.sort(key=lambda x: x[1]['parameters'])
        
        print(f"\nParameter count range:")
        min_params = successful_variants[0][1]['parameters']
        max_params = successful_variants[-1][1]['parameters']
        print(f"  Min: {min_params:,} params ({successful_variants[0][0]})")
        print(f"  Max: {max_params:,} params ({successful_variants[-1][0]})")
        
        # Sort by training time (speed proxy)
        by_speed = sorted(successful_variants, key=lambda x: x[1]['training_time'])
        print(f"\nTraining speed ranking (fastest to slowest):")
        for i, (variant, result) in enumerate(by_speed[:10]):  # Top 10
            print(f"  {i+1:2d}. {variant:25s}: {result['training_time']:5.1f}s "
                  f"(v={result['vocab_size']:3d}, d={result['dim']:2d}, l={result['n_layers']})")
        
        # Sort by loss (quality proxy)
        by_loss = sorted(successful_variants, key=lambda x: x[1]['final_loss'])
        print(f"\nTraining quality ranking (best to worst loss):")
        for i, (variant, result) in enumerate(by_loss[:10]):  # Top 10
            print(f"  {i+1:2d}. {variant:25s}: {result['final_loss']:.3f} loss "
                  f"(v={result['vocab_size']:3d}, d={result['dim']:2d}, l={result['n_layers']})")
        
        print(f"\n🎯 {range_name} PARAMETER INSIGHTS:")
        print("  - Vocabulary size impact on embedding lookup speed")
        print("  - Dimension size impact on attention/FFN computation")
        print("  - Layer depth impact on sequential processing")
        print("  - Attention head impact on parallel computation")
        
        return results
    else:
        print(f"❌ No {range_name} variants completed successfully!")
        return {}

def find_rp2040_models():
    """Quick find models likely to work on RP2040"""
    print("=== Finding RP2040-Compatible Models ===")
    print("Searching for models with ≤15K parameters (likely to work)...")
    find_models_by_params(0, 15000)
    
    print(f"\n{'='*50}")
    print("Searching for models with 15K-30K parameters (test carefully)...")
    find_models_by_params(15000, 30000)

def main():
    """Main training function"""
    print("=== Scalable Transformer Training ===")
    print("Testing different model sizes for RP2040")
    print("All models will be saved in the 'models/' folder")
    
    # Ask which size to train
    print("\nAvailable sizes:")
    for name, config in MODEL_CONFIGS.items():
        print(f"  {name:15s}: {config['description']}")
    
    print("\nArchitectural Studies:")
    print("  1k_test  : Test all 1K architectural variants")
    print("  3k_test  : Test all 3K architectural variants")
    print("  5k_test  : Test all 5K architectural variants")
    print("  7k_test  : Test all 7K architectural variants")
    print("  8k_test  : Test all 8K architectural variants")
    print("  10k_test : Test all 10K architectural variants")
    print("  all_variants : Test ALL architectural variants (1K-10K)")
    
    print("\nOther options:")
    print("  all      : Train all predefined sizes")
    print("  limit    : Find RP2040 memory limit")
    print("  quick    : Quick size estimation")
    print("  custom   : Create custom model size")
    print("  test     : Access testing options")
    print("  list     : List existing models")
    print("  test_all : Test all sizes (alias for 'all')")
    print("  cleanup  : Clean up all model files")
    print("  disk     : Show disk usage of models folder")
    print("  find_params : Find models by parameter count range")
    print("  rp2040_quick : Quick find models likely to work on RP2040")
    
    # For automated testing, train all sizes
    choice = input("\nEnter size or option: ").strip().lower()
    
    if choice == 'all':
        test_all_sizes()
    elif choice == '1k_test':
        test_1k_variants()
    elif choice == '3k_test':
        test_3k_variants()
    elif choice == '5k_test':
        test_5k_variants()
    elif choice == '7k_test':
        test_7k_variants()
    elif choice == '8k_test':
        test_8k_variants()
    elif choice == '10k_test':
        test_10k_variants()
    elif choice == 'all_variants':
        test_all_variants()
    elif choice == 'limit':
        find_rp2040_limit()
    elif choice == 'quick':
        quick_size_test()
    elif choice == 'custom':
        custom_model()
    elif choice == 'test':
        print("Available test options:")
        print("  - 'limit': Find RP2040 memory limit")
        print("  - 'quick': Quick size estimation")
        print("  - 'all': Train all predefined sizes")
        print("  - Or enter a specific size name")
        test_choice = input("Enter test option: ").strip().lower()
        if test_choice == 'limit':
            find_rp2040_limit()
        elif test_choice == 'quick':
            quick_size_test()
        elif test_choice == 'all':
            test_all_sizes()
        elif test_choice in MODEL_CONFIGS:
            trainer = ScalableTrainer(test_choice)
            trainer.prepare_data()
            epochs = max(50, 200 // max(1, trainer.config['dim'] // 16))
            trainer.train(epochs=epochs, learning_rate=0.005)
            trainer.save_model()
            print(f"\n{test_choice.upper()} model training complete!")
        else:
            print("Invalid test option")
    elif choice == 'list':
        list_models()
    elif choice == 'test_all':
        test_all_sizes()
    elif choice == 'cleanup':
        cleanup_models()
    elif choice == 'disk':
        show_disk_usage()
    elif choice == 'find_params':
        find_models_by_params()
    elif choice == 'rp2040_quick':
        find_rp2040_models()
    elif choice in MODEL_CONFIGS:
        trainer = ScalableTrainer(choice)
        trainer.prepare_data()
        
        # Train with size-appropriate epochs
        epochs = max(50, 200 // max(1, trainer.config['dim'] // 16))
        trainer.train(epochs=epochs, learning_rate=0.005)
        
        trainer.save_model()
        print(f"\n{choice.upper()} model training complete!")
    else:
        print("Invalid choice. Training tiny model as default.")
        trainer = ScalableTrainer('tiny')
        trainer.prepare_data()
        trainer.train(epochs=100, learning_rate=0.01)
        trainer.save_model()

if __name__ == "__main__":
    main()