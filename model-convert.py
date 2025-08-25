#!/usr/bin/env python3
"""
Smart model converter: Automatically converts all trained PyTorch models to RP2040 format

Converts trained PyTorch models from checkpoints/ to models/ with all necessary files 
for RP2040 inference (binary model, vocabulary, and config).

Usage:
    python model-convert.py                    # Convert ALL models in checkpoints/
    python model-convert.py rp2040-speed      # Convert specific model only

Features:
    - Auto-scans checkpoints directory
    - Handles both naming formats (model_name.pt and best_model_name_dataset.pt)
    - Preserves full model names including dataset percentages
    - Batch conversion with progress tracking
    - Creates complete RP2040 deployment package
"""

import os
import sys
import torch
import struct
import json
import numpy as np
from pathlib import Path

def convert_model(model_name, checkpoint_filename=None):
    """Convert a model from checkpoints to RP2040 format"""
    print(f"🚀 Converting {model_name} to RP2040 format...")
    
    # Input checkpoint path - handle both naming formats
    if checkpoint_filename:
        # Use the specific filename found during scanning
        checkpoint_path = f"checkpoints/{checkpoint_filename}"
    else:
        # Try to find the checkpoint automatically
        checkpoint_path = f"checkpoints/{model_name}.pt"
        if not os.path.exists(checkpoint_path):
            # Try with "best_" prefix and dataset suffix
            checkpoint_path = f"checkpoints/best_{model_name}_*.pt"
            # Find matching files
            import glob
            matching_files = glob.glob(checkpoint_path)
            if matching_files:
                checkpoint_path = matching_files[0]  # Use first match
            else:
                # Try other common patterns
                patterns = [
                    f"checkpoints/best_{model_name}_*.pt",
                    f"checkpoints/{model_name}_*.pt",
                    f"checkpoints/{model_name}.pt"
                ]
                for pattern in patterns:
                    matching_files = glob.glob(pattern)
                    if matching_files:
                        checkpoint_path = matching_files[0]
                        break
                else:
                    checkpoint_path = f"checkpoints/{model_name}.pt"  # Final fallback
    
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
                
                # Write embeddings row by row (vocab_size x dim)
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
                
                # Layer norm weights (dim each)
                ln1_key = f'layers.{layer_idx}.attention_norm.weight'
                ln2_key = f'layers.{layer_idx}.ffn_norm.weight'
                
                if ln1_key in checkpoint['model_state_dict']:
                    ln1_weight = checkpoint['model_state_dict'][ln1_key]
                    ln2_weight = checkpoint['model_state_dict'][ln2_key]
                    
                    # Write as float32 (4 bytes per value)
                    f.write(ln1_weight.numpy().astype(np.float32).tobytes())
                    f.write(ln2_weight.numpy().astype(np.float32).tobytes())
                
                # Handle PyTorch MultiheadAttention format
                # in_proj_weight contains concatenated [wq, wk, wv] weights
                in_proj_key = f'layers.{layer_idx}.attention.in_proj_weight'
                out_proj_key = f'layers.{layer_idx}.attention.out_proj.weight'
                
                if in_proj_key in checkpoint['model_state_dict']:
                    in_proj = checkpoint['model_state_dict'][in_proj_key]
                    out_proj = checkpoint['model_state_dict'][out_proj_key]
                    
                    print(f"    Converting PyTorch attention weights:")
                    print(f"      in_proj: {in_proj.shape} -> split into wq, wk, wv")
                    print(f"      out_proj: {out_proj.shape} -> wo")
                    
                    # Split in_proj into wq, wk, wv (each is dim x dim)
                    dim = config['dim']
                    wq = in_proj[:dim, :]      # First dim rows
                    wk = in_proj[dim:2*dim, :] # Second dim rows  
                    wv = in_proj[2*dim:, :]    # Last dim rows
                    wo = out_proj              # Output projection
                    
                    # Write wq, wk, wv, wo in order
                    for weight_name, weight in [('wq', wq), ('wk', wk), ('wv', wv), ('wo', wo)]:
                        print(f"    Writing {weight_name}: {weight.shape}")
                        for i in range(weight.shape[0]):
                            row = weight[i].flatten()
                            row_bytes = row.numpy().astype(np.float32).tobytes()
                            f.write(row_bytes)
                
                # Handle PyTorch FFN format (Sequential layers)
                # ffn.0.weight is w1 (first linear layer): hidden_dim x dim  
                # ffn.3.weight is w2 (second linear layer): dim x hidden_dim
                w1_key = f'layers.{layer_idx}.ffn.0.weight'  # PyTorch format
                w2_key = f'layers.{layer_idx}.ffn.3.weight'  # PyTorch format
                
                if w1_key in checkpoint['model_state_dict']:
                    w1_pytorch = checkpoint['model_state_dict'][w1_key]  # [hidden_dim, dim]
                    w2_pytorch = checkpoint['model_state_dict'][w2_key]  # [dim, hidden_dim]
                    
                    print(f"    Converting PyTorch FFN weights:")
                    print(f"      ffn.0 (w1): {w1_pytorch.shape}")
                    print(f"      ffn.3 (w2): {w2_pytorch.shape}")
                    
                    # For inference.py, we need:
                    # w1: dim x hidden_dim (transpose PyTorch w1)
                    # w2: hidden_dim x dim (transpose PyTorch w2)
                    w1_transposed = w1_pytorch.T  # [dim, hidden_dim]
                    w2_transposed = w2_pytorch.T  # [hidden_dim, dim]
                    
                    print(f"    Writing w1: {w1_transposed.shape} (transposed)")
                    # Write w1 row by row (dim rows, each row has hidden_dim values)
                    for i in range(w1_transposed.shape[0]):
                        row = w1_transposed[i].flatten()
                        row_bytes = row.numpy().astype(np.float32).tobytes()
                        f.write(row_bytes)
                    
                    print(f"    Writing w2: {w2_transposed.shape} (transposed)")
                    # Write w2 row by row (hidden_dim rows, each row has dim values)
                    for i in range(w2_transposed.shape[0]):
                        row = w2_transposed[i].flatten()
                        row_bytes = row.numpy().astype(np.float32).tobytes()
                        f.write(row_bytes)
            
            # Write final layer norm (dim values)
            final_ln = np.zeros(dim, dtype=np.float32)
            f.write(final_ln.tobytes())
        
        print(f"✅ Model converted: {model_output}")
        
        # Verify file size
        file_size = os.path.getsize(model_output)
        expected_size = (32 + 4 +  # Header + scale factor
                        config['vocab_size'] * config['dim'] * 4 +  # Embeddings
                        config['n_layers'] * (config['dim'] * 2 * 4 +  # Layer norms
                                            config['dim'] * config['dim'] * 4 * 4 +  # Attention weights
                                            config['dim'] * config['hidden_dim'] * 4 +  # w1
                                            config['hidden_dim'] * config['dim'] * 4) +  # w2
                        config['dim'] * 4)  # Final layer norm
        
        print(f"📊 File size: {file_size:,} bytes")
        print(f"📊 Expected: {expected_size:,} bytes")
        print(f"📊 Difference: {file_size - expected_size:,} bytes")
        
        # Create vocabulary
        vocab_output = output_dir / f"vocab_{config['vocab_size']}p.bin"
        print(f"📚 Creating vocabulary...")
        
        with open(vocab_output, 'wb') as f:
            vocab_size = config['vocab_size']
            
            # Write vocab size as 2-byte unsigned short
            f.write(struct.pack('H', vocab_size))
            
            # Create proper English vocabulary based on TinyStories training
            def create_proper_vocabulary(vocab_size):
                """Create a proper English vocabulary instead of generic token_X"""
                # Special tokens (essential for model operation)
                special_tokens = ["<pad>", "<s>", "</s>", "<unk>", " "]
                
                # Most common English words from TinyStories dataset
                common_words = [
                    "the", "and", "a", "to", "of", "in", "is", "it", "you", "that",
                    "he", "was", "for", "on", "are", "as", "with", "his", "they", "at",
                    "be", "this", "have", "from", "or", "one", "had", "by", "word", "but",
                    "not", "what", "all", "were", "we", "when", "your", "can", "said",
                    "there", "use", "an", "each", "which", "she", "do", "how", "their",
                    "if", "up", "out", "many", "then", "them", "these", "so", "some",
                    "her", "would", "make", "like", "into", "him", "time", "two", "more",
                    "go", "no", "way", "could", "my", "than", "first", "been", "call",
                    "who", "its", "now", "find", "long", "down", "day", "did", "get",
                    "come", "made", "may", "part", "cat", "dog", "boy", "girl", "man",
                    "woman", "house", "car", "tree", "water", "food", "book", "school",
                    "friend", "family", "mother", "father", "baby", "child", "people",
                    "good", "bad", "big", "small", "new", "old", "young", "happy", "sad",
                    "good", "bad", "big", "small", "new", "old", "young", "happy", "sad",
                    "run", "walk", "eat", "sleep", "play", "work", "read", "write",
                    "see", "hear", "know", "think", "feel", "want", "need", "help",
                    "look", "watch", "listen", "talk", "speak", "tell", "ask", "answer",
                    "give", "take", "put", "bring", "send", "open", "close", "start",
                    "stop", "begin", "end", "come", "go", "leave", "arrive", "return"
                ]
                
                # Common letters and punctuation
                letters = list("abcdefghijklmnopqrstuvwxyz")
                punctuation = [".", ",", "!", "?", "'", "-", ":", ";", "(", ")", "\""]
                
                # Combine all tokens in priority order
                all_tokens = special_tokens + common_words + letters + punctuation
                
                # Ensure we don't exceed vocab_size
                if len(all_tokens) > vocab_size:
                    all_tokens = all_tokens[:vocab_size]
                elif len(all_tokens) < vocab_size:
                    # Add more common words to fill vocabulary
                    extra_words = [
                        "about", "after", "again", "against", "all", "also", "always",
                        "another", "any", "around", "away", "back", "because", "before",
                        "best", "better", "between", "both", "call", "came", "come",
                        "could", "day", "did", "different", "does", "done", "each",
                        "even", "every", "first", "found", "get", "give", "goes",
                        "going", "got", "great", "had", "has", "have", "help", "here",
                        "high", "home", "house", "into", "just", "keep", "kind", "know",
                        "last", "left", "life", "like", "little", "long", "look", "made",
                        "make", "many", "may", "mean", "men", "might", "more", "most",
                        "move", "much", "must", "name", "never", "new", "next", "night",
                        "now", "number", "off", "often", "once", "only", "other", "over",
                        "own", "part", "people", "place", "put", "right", "same", "say",
                        "see", "seem", "should", "show", "small", "some", "still",
                        "such", "take", "tell", "than", "that", "them", "then", "there",
                        "they", "thing", "think", "this", "those", "through", "time",
                        "today", "together", "too", "under", "until", "upon", "very",
                        "want", "way", "well", "went", "were", "what", "when", "where",
                        "which", "while", "who", "why", "will", "with", "without",
                        "work", "world", "would", "year", "years", "young"
                    ]
                    
                    # Add extra words until we reach vocab_size
                    for word in extra_words:
                        if len(all_tokens) >= vocab_size:
                            break
                        if word not in all_tokens:
                            all_tokens.append(word)
                
                return all_tokens
            
            # Generate proper vocabulary
            all_tokens = create_proper_vocabulary(vocab_size)
            print(f"  Created vocabulary with {len(all_tokens)} tokens")
            print(f"  Special tokens: {all_tokens[:5]}")
            print(f"  Sample words: {all_tokens[5:15]}")
            
            # Write tokens to file
            for i in range(vocab_size):
                if i < len(all_tokens):
                    token = all_tokens[i]
                else:
                    # Fallback: create meaningful token names for any remaining slots
                    token = f"word_{i}"
                
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

def scan_checkpoints():
    """Scan checkpoints directory for all available models"""
    if not os.path.exists("checkpoints"):
        print("❌ checkpoints/ directory not found!")
        print("Please train a model first using train-gpu2.py")
        return []
    
    checkpoint_files = []
    for file in os.listdir("checkpoints"):
        if file.endswith(".pt"):
            # Extract model name from filename
            # Handle both formats: model_name.pt and best_model_name_dataset.pt
            if file.startswith("best_"):
                # Format: best_rp2040-speed_40p.pt -> rp2040-speed_40p
                # Keep the full name including dataset percentage
                model_name = file[5:-3]  # Remove "best_" and ".pt", keep everything else
                checkpoint_files.append((model_name, file))
            else:
                # Format: model_name.pt -> model_name
                model_name = file[:-3]
                checkpoint_files.append((model_name, file))
    
    return checkpoint_files

def main():
    """Main function"""
    if len(sys.argv) == 2:
        # Single model conversion
        model_name = sys.argv[1]
        
        # Check if checkpoints directory exists
        if not os.path.exists("checkpoints"):
            print("❌ checkpoints/ directory not found!")
            print("Please train a model first using train-gpu2.py")
            return 1
        
        # Find the actual checkpoint file for this model
        import glob
        checkpoint_patterns = [
            f"checkpoints/best_{model_name}_*.pt",
            f"checkpoints/{model_name}_*.pt",
            f"checkpoints/{model_name}.pt",
            f"checkpoints/best_{model_name}.pt"
        ]
        
        checkpoint_filename = None
        for pattern in checkpoint_patterns:
            matching_files = glob.glob(pattern)
            if matching_files:
                checkpoint_filename = os.path.basename(matching_files[0])
                break
        
        # If still no match, try to find any checkpoint that contains the model name
        if not checkpoint_filename:
            for file in os.listdir("checkpoints"):
                if file.endswith(".pt") and model_name.replace("_", "") in file.replace("_", "").replace("-", ""):
                    checkpoint_filename = file
                    break
        
        if not checkpoint_filename:
            print(f"❌ No checkpoint found for model '{model_name}'")
            print("Available checkpoints:")
            for file in os.listdir("checkpoints"):
                if file.endswith(".pt"):
                    print(f"  - {file[:-3]}")
            return 1
        
        print(f"📁 Found checkpoint: {checkpoint_filename}")
        
        # Convert the specific model
        success = convert_model(model_name, checkpoint_filename=checkpoint_filename)
        
        if success:
            print(f"\n🎉 {model_name} successfully converted to RP2040 format!")
            return 0
        else:
            print(f"\n❌ Failed to convert {model_name}")
            return 1
    
    elif len(sys.argv) == 1:
        # Auto-scan and convert all models
        print("🔍 Scanning checkpoints directory for models to convert...")
        checkpoint_files = scan_checkpoints()
        
        if not checkpoint_files:
            print("❌ No checkpoint files found in checkpoints/ directory!")
            print("Please train a model first using train-gpu2.py")
            return 1
        
        print(f"📁 Found {len(checkpoint_files)} checkpoint(s):")
        for model_name, filename in checkpoint_files:
            print(f"  - {model_name} ({filename})")
        
        print(f"\n🚀 Starting batch conversion of all models...")
        
        successful_conversions = 0
        failed_conversions = 0
        
        for model_name, filename in checkpoint_files:
            print(f"\n{'='*60}")
            print(f"🔄 Converting {model_name} from {filename}...")
            print(f"{'='*60}")
            
            try:
                success = convert_model(model_name, checkpoint_filename=filename)
                if success:
                    successful_conversions += 1
                    print(f"✅ {model_name} converted successfully!")
                else:
                    failed_conversions += 1
                    print(f"❌ {model_name} conversion failed!")
            except Exception as e:
                failed_conversions += 1
                print(f"❌ {model_name} conversion failed with error: {e}")
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"🎯 BATCH CONVERSION COMPLETE!")
        print(f"{'='*60}")
        print(f"✅ Successful: {successful_conversions}")
        print(f"❌ Failed: {failed_conversions}")
        print(f"📊 Total: {len(checkpoint_files)}")
        
        if successful_conversions > 0:
            print(f"\n🎉 {successful_conversions} model(s) ready for RP2040 deployment!")
            print(f"📁 Check the models/ directory for converted files")
        
        return 0 if failed_conversions == 0 else 1
    
    else:
        print("Usage:")
        print("  python model-convert.py                    # Convert ALL models")
        print("  python model-convert.py <model-name>      # Convert specific model")
        print("\nExamples:")
        print("  python model-convert.py                    # Convert all checkpoints")
        print("  python model-convert.py rp2040-speed      # Convert specific model")
        return 1

if __name__ == "__main__":
    exit(main())
