#!/usr/bin/env python3
"""
Simple model converter: python model-convert.py model-name

Converts a trained PyTorch model from checkpoints/model-name.pt to 
models/model-name/ with all necessary files for RP2040 inference.

Usage:
    python model-convert.py rp2040-speed
    python model-convert.py my-custom-model
"""

import os
import sys
import torch
import struct
import json
import numpy as np
from pathlib import Path

def convert_model(model_name):
    """Convert a model from checkpoints to RP2040 format"""
    print(f"🚀 Converting {model_name} to RP2040 format...")
    
    # Input checkpoint path
    checkpoint_path = f"checkpoints/{model_name}.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        if os.path.exists("checkpoints"):
            for file in os.listdir("checkpoints"):
                if file.endswith(".pt"):
                    print(f"  - {file[:-3]}")
        return False
    
    # Create output directory
    output_dir = Path(f"models/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Load checkpoint
        print(f"📥 Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'config' not in checkpoint:
            print("❌ Checkpoint missing config, using defaults...")
            config = {
                'vocab_size': 256,
                'dim': 8,
                'hidden_dim': 192,
                'n_layers': 2,
                'n_heads': 4,
                'max_seq_len': 48,
                'description': f'{model_name} model'
            }
        else:
            config = checkpoint['config']
        
        print(f"✅ Model config: {config}")
        
        # Convert model to binary format
        model_output = output_dir / f"model_{config['vocab_size']}p.bin"
        print(f"🔧 Converting model to {model_output}...")
        
        with open(model_output, 'wb') as f:
            # Write header (32 bytes)
            header_data = struct.pack("8I",
                config['vocab_size'],
                config['dim'],
                config['hidden_dim'],
                config['n_layers'],
                config['n_heads'],
                config['max_seq_len'],
                0,  # dropout
                0   # activation
            )
            f.write(header_data)
            
            # Write quantization metadata (4 bytes) - float32
            scale_factor = 1.0
            f.write(struct.pack("f", scale_factor))
            
            # Write embeddings
            embedding_key = 'token_embedding.weight'
            if embedding_key in checkpoint['model_state_dict']:
                embeddings = checkpoint['model_state_dict'][embedding_key]
                print(f"  Writing embeddings: {embeddings.shape}")
                
                for i in range(embeddings.shape[0]):
                    row = embeddings[i].flatten()
                    row_bytes = row.numpy().tobytes()
                    f.write(row_bytes)
            
            # Write transformer layers
            n_layers = config['n_layers']
            dim = config['dim']
            hidden_dim = config['hidden_dim']
            
            for layer_idx in range(n_layers):
                print(f"  Writing layer {layer_idx + 1}/{n_layers}")
                
                # Layer norm weights
                ln1_key = f'layers.{layer_idx}.attention_norm.weight'
                ln2_key = f'layers.{layer_idx}.ffn_norm.weight'
                
                if ln1_key in checkpoint['model_state_dict']:
                    ln1_weight = checkpoint['model_state_dict'][ln1_key]
                    ln2_weight = checkpoint['model_state_dict'][ln2_key]
                    
                    f.write(ln1_weight.numpy().tobytes())
                    f.write(ln2_weight.numpy().tobytes())
                
                # Attention weights: wq, wk, wv, wo
                attention_keys = ['wq', 'wk', 'wv', 'wo']
                for key in attention_keys:
                    weight_key = f'layers.{layer_idx}.attention.{key}.weight'
                    if weight_key in checkpoint['model_state_dict']:
                        weight = checkpoint['model_state_dict'][weight_key]
                        
                        for i in range(weight.shape[0]):
                            row = weight[i].flatten()
                            row_bytes = row.numpy().tobytes()
                            f.write(row_bytes)
                
                # Feed-forward weights: w1, w2
                ffn_keys = ['w1', 'w2']
                for key in ffn_keys:
                    weight_key = f'layers.{layer_idx}.ffn.{key}.weight'
                    if weight_key in checkpoint['model_state_dict']:
                        weight = checkpoint['model_state_dict'][weight_key]
                        
                        for i in range(weight.shape[0]):
                            row = weight[i].flatten()
                            row_bytes = row.numpy().tobytes()
                            f.write(row_bytes)
            
            # Write final layer norm (zeros for now)
            final_ln = np.zeros(dim, dtype=np.float32)
            f.write(final_ln.tobytes())
        
        print(f"✅ Model converted: {model_output}")
        
        # Create vocabulary
        vocab_output = output_dir / f"vocab_{config['vocab_size']}p.bin"
        print(f"📚 Creating vocabulary...")
        
        with open(vocab_output, 'wb') as f:
            vocab_size = config['vocab_size']
            
            # Write vocab size as 2-byte unsigned short
            f.write(struct.pack('H', vocab_size))
            
            # Create basic vocabulary
            special_tokens = ["<pad>", "<s>", "</s>", "<unk>", " "]
            chars = "etaoinshrdlcumwfgypbvkjxqz.,!?'-"
            common_words = ['the', 'and', 'a', 'to', 'of', 'in', 'is', 'it', 'you', 'that']
            
            all_tokens = special_tokens + list(chars) + common_words
            
            for i in range(vocab_size):
                if i < len(all_tokens):
                    token = all_tokens[i]
                else:
                    token = f"token_{i}"
                
                token_bytes = token.encode('utf-8')
                length = min(len(token_bytes), 255)
                f.write(struct.pack('B', length))
                f.write(token_bytes[:length])
        
        print(f"✅ Vocabulary created: {vocab_output}")
        
        # Create config file
        config_output = output_dir / f"config_{config['vocab_size']}p.json"
        print(f"⚙️  Creating config file...")
        
        config_data = {
            "model_name": model_name,
            "vocab_size": config['vocab_size'],
            "dim": config['dim'],
            "hidden_dim": config['hidden_dim'],
            "n_layers": config['n_layers'],
            "n_heads": config['n_heads'],
            "max_seq_len": config['max_seq_len'],
            "quantization": "float32",
            "description": config.get('description', f'{model_name} model'),
            "source": "PyTorch training"
        }
        
        with open(config_output, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✅ Config created: {config_output}")
        
        # Show final structure
        print(f"\n📁 Conversion complete! Files created:")
        print(f"  {output_dir}/")
        print(f"  ├── model_{config['vocab_size']}p.bin")
        print(f"  ├── vocab_{config['vocab_size']}p.bin")
        print(f"  └── config_{config['vocab_size']}p.json")
        
        # Show deployment instructions
        print(f"\n🎯 To deploy on RP2040:")
        print(f"1. Copy the '{model_name}' folder to your RP2040's models/ directory")
        print(f"2. Run: python inference.py")
        print(f"3. The script will auto-detect your model!")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python model-convert.py <model-name>")
        print("Example: python model-convert.py rp2040-speed")
        print("\nThis will convert checkpoints/model-name.pt to models/model-name/")
        return 1
    
    model_name = sys.argv[1]
    
    # Check if checkpoints directory exists
    if not os.path.exists("checkpoints"):
        print("❌ checkpoints/ directory not found!")
        print("Please train a model first using train-gpu2.py")
        return 1
    
    # Convert the model
    success = convert_model(model_name)
    
    if success:
        print(f"\n🎉 {model_name} successfully converted to RP2040 format!")
        return 0
    else:
        print(f"\n❌ Failed to convert {model_name}")
        return 1

if __name__ == "__main__":
    exit(main())
