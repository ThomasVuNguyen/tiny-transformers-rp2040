#!/usr/bin/env python3
"""
Create a minimal test model for RP2040 to verify the binary format works correctly.
This creates a tiny model with known weights to test the conversion pipeline.
"""

import struct
import json
import numpy as np
import os
from pathlib import Path

def create_minimal_test_model():
    """Create a minimal test model that should work on RP2040"""
    
    # Minimal configuration that should work
    config = {
        'vocab_size': 64,      # Very small vocab
        'dim': 4,              # Very small dimension (divisible by n_heads)
        'hidden_dim': 16,      # Small hidden dim (4x multiplier)
        'n_layers': 1,         # Just one layer
        'n_heads': 2,          # 2 heads (4 is divisible by 2)
        'max_seq_len': 16,     # Short sequences
        'description': 'Minimal test model for RP2040'
    }
    
    print(f"🔧 Creating minimal test model with config: {config}")
    
    # Create output directory
    output_dir = Path("models/test-minimal")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create binary model file
    model_file = output_dir / f"model_{config['vocab_size']}p.bin"
    
    with open(model_file, 'wb') as f:
        print("📝 Writing header...")
        # Write 32-byte header
        header = struct.pack("8I",
            config['vocab_size'],    # 0: vocab_size
            config['dim'],           # 4: dim
            config['hidden_dim'],    # 8: hidden_dim
            config['n_layers'],      # 12: n_layers
            config['n_heads'],       # 16: n_heads
            config['max_seq_len'],   # 20: max_seq_len
            0,                       # 24: dropout (unused)
            0                        # 28: activation (unused)
        )
        f.write(header)
        
        # Write quantization metadata (4 bytes) - float32
        print("📝 Writing quantization info...")
        scale_factor = 1.0  # No quantization
        f.write(struct.pack("f", scale_factor))
        
        print("📝 Writing embeddings...")
        # Write token embeddings: vocab_size x dim
        # Each embedding is a row of `dim` float32 values
        for i in range(config['vocab_size']):
            # Create simple embeddings: just use token index as pattern
            embedding = [float(i % 10) / 10.0] * config['dim']
            for val in embedding:
                f.write(struct.pack('f', val))
        
        print("📝 Writing transformer layer...")
        # Write single transformer layer
        layer_idx = 0
        
        # Layer norms: 2 vectors of `dim` float32 values each
        print(f"  Writing layer norms...")
        ln1_weight = [1.0] * config['dim']  # Layer norm 1
        ln2_weight = [1.0] * config['dim']  # Layer norm 2
        
        for val in ln1_weight:
            f.write(struct.pack('f', val))
        for val in ln2_weight:
            f.write(struct.pack('f', val))
        
        # Attention weights: wq, wk, wv, wo
        # Each is dim x dim matrix, written row by row
        attention_weights = ['wq', 'wk', 'wv', 'wo']
        for weight_name in attention_weights:
            print(f"  Writing {weight_name}...")
            for row_idx in range(config['dim']):
                # Simple pattern: identity-like matrices with small values
                row = []
                for col_idx in range(config['dim']):
                    if row_idx == col_idx:
                        row.append(0.1)  # Diagonal
                    else:
                        row.append(0.01)  # Off-diagonal
                
                for val in row:
                    f.write(struct.pack('f', val))
        
        # Feed-forward weights
        print(f"  Writing w1 (dim={config['dim']} x hidden_dim={config['hidden_dim']})...")
        # w1: dim x hidden_dim matrix
        for row_idx in range(config['dim']):
            row = [0.1] * config['hidden_dim']  # Simple constant values
            for val in row:
                f.write(struct.pack('f', val))
        
        print(f"  Writing w2 (hidden_dim={config['hidden_dim']} x dim={config['dim']})...")
        # w2: hidden_dim x dim matrix  
        for row_idx in range(config['hidden_dim']):
            row = [0.1] * config['dim']  # Simple constant values
            for val in row:
                f.write(struct.pack('f', val))
        
        # Final layer norm
        print("📝 Writing final layer norm...")
        final_ln = [1.0] * config['dim']
        for val in final_ln:
            f.write(struct.pack('f', val))
    
    print(f"✅ Model file created: {model_file}")
    
    # Calculate and verify file size
    file_size = os.path.getsize(model_file)
    expected_size = (
        32 +  # Header
        4 +   # Scale factor
        config['vocab_size'] * config['dim'] * 4 +  # Embeddings
        config['n_layers'] * (
            config['dim'] * 2 * 4 +  # Layer norms (ln1, ln2)
            config['dim'] * config['dim'] * 4 * 4 +  # Attention weights (wq,wk,wv,wo)
            config['dim'] * config['hidden_dim'] * 4 +  # w1
            config['hidden_dim'] * config['dim'] * 4    # w2
        ) +
        config['dim'] * 4  # Final layer norm
    )
    
    print(f"📊 File size: {file_size:,} bytes")
    print(f"📊 Expected: {expected_size:,} bytes")
    if file_size == expected_size:
        print("✅ File size matches expected!")
    else:
        print(f"⚠️  Size mismatch: {file_size - expected_size:,} bytes difference")
    
    # Create vocabulary file
    vocab_file = output_dir / f"vocab_{config['vocab_size']}p.bin"
    print(f"📚 Creating vocabulary file: {vocab_file}")
    
    with open(vocab_file, 'wb') as f:
        # Write vocab size (2 bytes)
        f.write(struct.pack('H', config['vocab_size']))
        
        # Create simple vocabulary
        tokens = ["<pad>", "<s>", "</s>", "<unk>", " "] + \
                [chr(ord('a') + i) for i in range(26)] + \
                [str(i) for i in range(10)]
        
        # Pad to vocab_size
        while len(tokens) < config['vocab_size']:
            tokens.append(f"tok_{len(tokens)}")
        
        # Write each token
        for i in range(config['vocab_size']):
            token = tokens[i] if i < len(tokens) else f"tok_{i}"
            token_bytes = token.encode('utf-8')
            length = min(len(token_bytes), 255)
            f.write(struct.pack('B', length))
            f.write(token_bytes[:length])
    
    print(f"✅ Vocabulary file created: {vocab_file}")
    
    # Create config file
    config_file = output_dir / f"config_{config['vocab_size']}p.json"
    print(f"⚙️  Creating config file: {config_file}")
    
    config_data = {
        "model_name": "test-minimal",
        "vocab_size": config['vocab_size'],
        "dim": config['dim'],
        "hidden_dim": config['hidden_dim'],
        "n_layers": config['n_layers'],
        "n_heads": config['n_heads'],
        "max_seq_len": config['max_seq_len'],
        "quantization": "float32",
        "description": config['description'],
        "source": "Test model generator"
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"✅ Config file created: {config_file}")
    
    # Show final structure
    print(f"\n📁 Test model created successfully!")
    print(f"  {output_dir}/")
    print(f"  ├── model_{config['vocab_size']}p.bin")
    print(f"  ├── vocab_{config['vocab_size']}p.bin")
    print(f"  └── config_{config['vocab_size']}p.json")
    
    print(f"\n🎯 To test on RP2040:")
    print(f"1. Copy the 'test-minimal' folder to your RP2040's models/ directory")
    print(f"2. Run: python inference.py")
    print(f"3. Select the test-minimal model")
    
    return True

if __name__ == "__main__":
    print("🚀 Creating minimal test model for RP2040...")
    success = create_minimal_test_model()
    if success:
        print("\n🎉 Test model created successfully!")
        print("This model should load without errors on RP2040.")
    else:
        print("\n❌ Failed to create test model")
