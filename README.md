# tiny-transformers-rp2040

Running very tiny transformers on RP2040 microcontroller with GPU training and simple conversion.

## 🚀 Quick Start

### 1. Train a Model
```bash
# Install dependencies
python install-gpu-deps.py

# Train a model
python train-gpu2.py --model rp2040-speed --dataset-percent 10 --epochs 100
```

### 2. Convert to RP2040 Format
```bash
# Simple one-command conversion
python model-convert.py rp2040-speed
```

### 3. Deploy on RP2040
```bash
# Copy models/rp2040-speed/ folder to your RP2040
# Run inference
python inference.py
```

## 📁 Project Structure

- **`train-gpu2.py`** - GPU training script for RP2040 models
- **`model-convert.py`** - Simple model conversion to RP2040 format
- **`inference.py`** - RP2040 inference engine
- **`models/`** - Converted models for RP2040 deployment
- **`checkpoints/`** - PyTorch training checkpoints

## 🎯 Key Features

- **GPU Training** - Train models on GPU with PyTorch
- **Simple Conversion** - One command converts to RP2040 format
- **Auto-Detection** - RP2040 automatically finds and loads models
- **Memory Optimized** - Designed for RP2040's 256KB RAM

## 📚 Documentation

- **`README-GPU-TRAINING.md`** - Complete GPU training guide
- **`requirements-gpu.txt`** - GPU training dependencies

## 🔄 Workflow

```
GPU Training → Simple Conversion → RP2040 Deployment
     ↓              ↓                    ↓
train-gpu2.py → model-convert.py → inference.py
```

Happy training and deploying! 🚀
