#!/usr/bin/env python3
"""
Debug script to examine checkpoint structure and identify conversion issues
"""

import torch
import os

def debug_checkpoint(checkpoint_path):
    """Debug checkpoint structure"""
    print(f"🔍 Debugging checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Checkpoint loaded successfully")
        
        # Print top-level keys
        print(f"\n📋 Top-level keys:")
        for key in checkpoint.keys():
            print(f"  - {key}: {type(checkpoint[key])}")
        
        # Check config
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"\n⚙️  Model config:")
            for key, value in config.items():
                print(f"  {key}: {value}")
        else:
            print(f"\n⚠️  No config found in checkpoint")
        
        # Check model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"\n🏗️  Model state dict keys:")
            for key in sorted(state_dict.keys()):
                tensor = state_dict[key]
                print(f"  {key}: {tensor.shape} ({tensor.dtype})")
        else:
            print(f"\n⚠️  No model_state_dict found in checkpoint")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_checkpoint("checkpoints/best_rp2040-optimized_10p.pt")
