# GPU Training for RP2040 Production Models

This script provides GPU-accelerated training for the production-ready RP2040 transformer models based on comprehensive architectural studies from `log.md`.

## 🚀 Production Models

Based on the findings in `log.md`, we have three optimized architectures:

| Model | Parameters | Architecture | Use Case |
|-------|------------|--------------|----------|
| `rp2040-optimized` | 15-20K | vocab=512, dim=8, layers=3, heads=8, hidden=256 | **Production-ready** - Best balance of speed and quality |
| `rp2040-speed` | 8-12K | vocab=256, dim=8, layers=2, heads=4, hidden=192 | **Speed-focused** - Fastest inference on RP2040 |
| `rp2040-quality` | 25-35K | vocab=1024, dim=12, layers=4, heads=12, hidden=384 | **Quality-focused** - Highest quality text generation |

## 🎯 Key Features

- **GPU Acceleration**: Uses PyTorch with CUDA for 10-100x speedup
- **Dataset Increments**: Train with 10%, 20%, 30%, ..., 100% of TinyStories dataset
- **Mixed Precision**: Automatic mixed precision (AMP) for memory efficiency
- **Smart Batching**: Automatic batch size optimization based on GPU memory
- **Early Stopping**: Prevents overfitting with validation-based early stopping
- **Checkpointing**: Saves best models and regular checkpoints
- **Training History**: Logs all training metrics for analysis
- **Configuration Validation**: Automatically validates model configurations for PyTorch compatibility

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (4GB+ VRAM recommended)
- **RAM**: 8GB+ system RAM
- **Storage**: 2GB+ free space for dataset and models

### Software
```bash
# Install PyTorch with CUDA support
pip install -r requirements-gpu.txt

# Or install manually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy tqdm pynvml
```

## 🚀 Quick Installation

### Option 1: Automatic Installation (Recommended)
```bash
python install-gpu-deps.py
```

### Option 2: Manual Installation
```bash
pip install -r requirements-gpu.txt
```

### Option 3: Install Core Dependencies Only
```bash
pip install torch torchvision torchaudio numpy tqdm pynvml
```

**Note**: `pynvml` is required for GPU utilization monitoring. Without it, you'll see "N/A" for GPU utilization in training progress.

## 🚀 Quick Start

### 1. Basic Training (10% dataset)
```bash
python train-gpu.py --model rp2040-optimized --dataset-percent 10 --epochs 50
```

### 2. Speed Model with 30% dataset
```bash
python train-gpu.py --model rp2040-speed --dataset-percent 30 --epochs 100
```

### 3. Quality Model with 50% dataset
```bash
python train-gpu.py --model rp2040-quality --dataset-percent 50 --epochs 150
```

### 4. Full Dataset Training
```bash
python train-gpu.py --model rp2040-optimized --dataset-percent 100 --epochs 200
```

## 📊 Command Line Options

```bash
python train-gpu2.py [OPTIONS]

Options:
  --model MODEL           Model to train (rp2040-optimized, rp2040-speed, rp2040-quality)
  --dataset-percent INT   Dataset percentage (10, 20, 30, ..., 100)
  --epochs INT           Number of training epochs (default: 100)
  --dataset-path PATH    Path to TinyStories dataset (default: dataset/TinyStories-train.txt)
  --batch-size INT       Override automatic batch size
  --aggressive-opt       Enable aggressive GPU optimization (higher utilization)
  --memory-opt          Enable memory optimization (lower RAM usage)
  --grad-accum INT      Gradient accumulation steps (default: 4)
  -h, --help            Show help message
```

### 🎯 Practical Examples

#### **Basic Training Examples**
```bash
# Quick test with 10% dataset
python train-gpu2.py --model rp2040-speed --dataset-percent 10 --epochs 50

# Production training with 50% dataset
python train-gpu2.py --model rp2040-optimized --dataset-percent 50 --epochs 100

# High-quality model with full dataset
python train-gpu2.py --model rp2040-quality --dataset-percent 100 --epochs 200
```

#### **Performance Optimization Examples**
```bash
# Maximum GPU utilization (recommended for powerful GPUs)
python train-gpu2.py --model rp2040-speed --dataset-percent 30 --aggressive-opt

# Memory-efficient training (for limited RAM systems)
python train-gpu2.py --model rp2040-optimized --dataset-percent 25 --memory-opt

# Custom batch size for specific GPU memory
python train-gpu2.py --model rp2040-speed --dataset-percent 20 --batch-size 256

# Larger effective batch size with gradient accumulation
python train-gpu2.py --model rp2040-optimized --dataset-percent 40 --grad-accum 8
```

#### **Training Progression Examples**
```bash
# Step 1: Quick validation (5-10 minutes)
python train-gpu2.py --model rp2040-speed --dataset-percent 10 --epochs 20

# Step 2: Medium training (30-60 minutes)
python train-gpu2.py --model rp2040-speed --dataset-percent 25 --epochs 100

# Step 3: Production training (2-4 hours)
python train-gpu2.py --model rp2040-optimized --dataset-percent 50 --epochs 150
```

## ⏱️ Expected Training Times

Training times depend on your GPU and dataset size:

| Dataset % | GPU Memory | Expected Time (100 epochs) |
|-----------|------------|----------------------------|
| 10%       | 4GB+       | 30-60 minutes              |
| 25%       | 4GB+       | 1-2 hours                  |
| 50%       | 6GB+       | 2-4 hours                  |
| 75%       | 8GB+       | 4-8 hours                  |
| 100%      | 8GB+       | 6-12 hours                 |

## 📁 Output Files & Model Versions

### 🎯 Understanding Model Versions Created During Training

`train-gpu2.py` creates **3 different types of model files** during training:

#### **A. 🏆 Best Model (Most Important)**
- **File**: `best_{model_name}_{dataset_percent}p.pt`
- **Example**: `best_rp2040-speed_10p.pt`
- **When**: Saved every time validation loss improves
- **Purpose**: The best-performing model during training
- **Status**: **This is the one you want to use for conversion!**

#### **B. 🔄 Periodic Checkpoints**
- **File**: `checkpoint_{model_name}_{dataset_percent}p_epoch{N}.pt`
- **Example**: `checkpoint_rp2040-speed_10p_epoch10.pt`, `checkpoint_rp2040-speed_10p_epoch20.pt`
- **When**: Saved every 10 epochs
- **Purpose**: Recovery points in case training crashes
- **Status**: Only created if training runs for 10+ epochs

#### **C. 🏁 Final Model**
- **File**: `final_{model_name}_{dataset_percent}p.pt`
- **Example**: `final_rp2040-speed_10p.pt`
- **When**: Saved at the end of training (regardless of performance)
- **Purpose**: The final state when training stops
- **Status**: Only created if training completes without early stopping

### 📂 Directory Structure

The script creates several directories:

```
checkpoints/           # Model checkpoints (.pt files)
├── best_rp2040-speed_10p.pt           # 🏆 BEST MODEL (use this one!)
├── checkpoint_rp2040-speed_10p_epoch10.pt  # Periodic checkpoint
├── checkpoint_rp2040-speed_10p_epoch20.pt  # Periodic checkpoint
└── final_rp2040-speed_10p.pt          # Final model

training_history/      # Training metrics (JSON files)
└── rp2040-speed_10p_history.json     # Loss curves and training stats
```

### 🤔 Why You Might Only See 1 Model File

**Common scenario**: You only see `best_rp2040-speed_10p.pt` in your `checkpoints/` folder.

**This happens because:**
1. **Early Stopping**: Training stopped early due to validation loss not improving
2. **Short Training**: Training didn't reach 10 epochs (no periodic checkpoints)
3. **Best Model Only**: The `best_*` file is the most important one anyway!

### 📊 What's Inside Each Checkpoint File

Each `.pt` file contains:
```python
checkpoint = {
    'model_state_dict': ...,      # Neural network weights (the important part!)
    'optimizer_state_dict': ...,  # Adam optimizer state
    'scheduler_state_dict': ...,  # Learning rate scheduler state
    'config': ...,               # Model architecture configuration
    'dataset_percent': ...,       # What % of dataset was used
    'train_losses': ...,          # Training loss history
    'val_losses': ...             # Validation loss history
}
```

## 🚀 Complete Training Process Guide

### 🎯 Training Flow Overview

When you run `train-gpu2.py`, here's exactly what happens:

```bash
python train-gpu2.py --model rp2040-speed --dataset-percent 10 --epochs 100
```

**Step-by-Step Process:**

1. **🚀 Initialization**
   - Load model configuration (`rp2040-speed`)
   - Initialize GPU optimizations (CUDA, mixed precision)
   - Create model with specified architecture
   - Set up optimizer (AdamW) and learning rate scheduler

2. **📚 Data Preparation**
   - Load TinyStories dataset
   - Use specified percentage (10% = subset of data)
   - Split into 90% training, 10% validation
   - Create optimized DataLoaders with GPU acceleration

3. **🔄 Training Loop** (for each epoch):
   ```python
   for epoch in range(epochs):
       # Train one epoch
       train_loss = train_epoch()
       
       # Validate
       val_loss = validate()
       
       # Save best model if validation improved
       if val_loss < best_val_loss:
           save_checkpoint("best_rp2040-speed_10p.pt")  # 🏆
       
       # Early stopping check (patience = 10 epochs)
       if no_improvement_for_10_epochs:
           break  # Stop training early
       
       # Save periodic checkpoint every 10 epochs
       if epoch % 10 == 0:
           save_checkpoint("checkpoint_..._epoch10.pt")  # 🔄
   ```

4. **🏁 Training Completion**
   - Save final model (if training completed normally)
   - Save training history (loss curves, metrics)
   - Display final statistics

### 📊 Real-Time Monitoring

During training, you'll see progress like this:

```
Epoch 1 (Train): 100%|██████| 1250/1250 [02:15<00:00, 9.23it/s, loss=4.2156, avg_loss=4.3421, batch_size=512, GPU_util=87%, GPU_mem=3.2GB]
Validation: 100%|██████| 139/139 [00:08<00:00, 16.85it/s]
Epoch 1: Train Loss: 4.3421, Val Loss: 4.1234

Epoch 2 (Train): 100%|██████| 1250/1250 [02:12<00:00, 9.45it/s, loss=3.8765, avg_loss=3.9123, batch_size=512, GPU_util=89%, GPU_mem=3.2GB]
Validation: 100%|██████| 139/139 [00:08<00:00, 17.23it/s]
Epoch 2: Train Loss: 3.9123, Val Loss: 3.8456
```

**Key Metrics Shown:**
- **loss**: Current batch loss
- **avg_loss**: Average loss for the epoch
- **batch_size**: Effective batch size (including gradient accumulation)
- **GPU_util**: GPU utilization percentage
- **GPU_mem**: GPU memory usage

### 🎯 Early Stopping Behavior

**Why training might stop early:**

```python
# Early stopping logic
patience = 10  # Wait 10 epochs for improvement
patience_counter = 0

if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0  # Reset counter
    save_best_model()     # 🏆 Save new best model
else:
    patience_counter += 1  # Increment counter

if patience_counter >= 10:
    print("Early stopping - validation loss hasn't improved for 10 epochs")
    break  # Stop training
```

**This is GOOD behavior** - it prevents overfitting and saves the best model!

### 📈 Training Optimization Features

#### **🚀 GPU Memory Optimization**
- **Dynamic Batch Sizing**: Automatically adjusts batch size based on GPU memory
- **Mixed Precision**: Uses FP16 to reduce memory usage by ~50%
- **Gradient Accumulation**: Simulates larger batch sizes without memory increase
- **Memory Cleanup**: Clears GPU cache between epochs

#### **⚡ Performance Optimizations**
- **CUDA Optimizations**: Enables TensorFloat-32, cuDNN auto-tuner
- **Non-blocking Transfers**: Overlaps data transfer with computation
- **Persistent Workers**: Keeps data loading workers alive between epochs
- **Prefetch Factor**: Pre-loads batches for faster training

#### **🎛️ Training Stability**
- **Gradient Clipping**: Prevents gradient explosion (max_norm=1.0)
- **Learning Rate Scheduling**: Cosine annealing for smooth convergence
- **Validation Monitoring**: Tracks overfitting with separate validation set

## 🔧 Advanced Usage

### Custom Batch Size
```bash
python train-gpu.py --model rp2040-optimized --dataset-percent 50 --batch-size 64
```

### Different Dataset Path
```bash
python train-gpu.py --model rp2040-speed --dataset-percent 25 --dataset-path /path/to/your/dataset.txt
```

### Quick Testing (Small dataset)
```bash
python train-gpu.py --model rp2040-optimized --dataset-percent 10 --epochs 10
```

## 📈 Training Progression

### Recommended Training Sequence

1. **Start Small**: Train with 10% dataset to verify setup
   ```bash
   python train-gpu.py --model rp2040-optimized --dataset-percent 10 --epochs 50
   ```

2. **Scale Up**: Increase dataset size gradually
   ```bash
   python train-gpu.py --model rp2040-optimized --dataset-percent 25 --epochs 100
   python train-gpu.py --model rp2040-optimized --dataset-percent 50 --epochs 150
   ```

3. **Full Training**: Train with complete dataset
   ```bash
   python train-gpu.py --model rp2040-optimized --dataset-percent 100 --epochs 200
   ```

### Model Comparison

Train all three models with the same dataset percentage to compare:

```bash
# Train all models with 25% dataset
python train-gpu.py --model rp2040-speed --dataset-percent 25 --epochs 100
python train-gpu.py --model rp2040-optimized --dataset-percent 25 --epochs 100
python train-gpu.py --model rp2040-quality --dataset-percent 25 --epochs 100
```

## 🐛 Troubleshooting

### 🤔 Model File Questions

#### **"Why do I only see 1 model file?"**
- **Normal**: You only see `best_rp2040-speed_10p.pt`
- **Reason**: Early stopping prevented other files from being created
- **Solution**: This is the best model - use it for conversion!

#### **"Where are the checkpoint files?"**
- **Missing**: `checkpoint_*_epoch10.pt` files
- **Reason**: Training didn't reach 10 epochs (early stopping)
- **Solution**: Normal behavior - the best model is what matters

#### **"Training stopped early - is this bad?"**
- **Answer**: NO! Early stopping is GOOD
- **Reason**: Prevents overfitting, saves the best model
- **Result**: You get the optimal model for RP2040 deployment

### 🚀 Training Process Issues

#### **"Training is very slow"**
```bash
# Check GPU utilization
nvidia-smi

# Use aggressive optimization
python train-gpu2.py --model rp2040-speed --dataset-percent 10 --aggressive-opt

# Increase batch size if you have GPU memory
python train-gpu2.py --model rp2040-speed --dataset-percent 10 --batch-size 1024
```

#### **"GPU utilization shows N/A"**
- **Error**: `ModuleNotFoundError: pynvml does not seem to be installed`
- **Solution**: `pip install pynvml`
- **Alternative**: Training continues normally, just no GPU% shown

#### **"Training loss not decreasing"**
- **Check**: Validation loss in logs
- **Try**: Smaller learning rate or different model
- **Normal**: Loss might plateau - early stopping will handle it

### 💾 GPU Memory Issues

#### **Out of Memory Errors**
```bash
# Reduce batch size
python train-gpu2.py --model rp2040-speed --dataset-percent 10 --batch-size 32

# Use memory optimization
python train-gpu2.py --model rp2040-speed --dataset-percent 10 --memory-opt

# Use smaller model
python train-gpu2.py --model rp2040-speed --dataset-percent 10  # Instead of rp2040-quality

# Use smaller dataset
python train-gpu2.py --model rp2040-speed --dataset-percent 10  # Instead of 50%
```

#### **High RAM Usage**
```bash
# Enable memory optimization
python train-gpu2.py --model rp2040-speed --dataset-percent 10 --memory-opt

# Reduce dataset percentage
python train-gpu2.py --model rp2040-speed --dataset-percent 10  # Start small
```

### 🔧 CUDA & PyTorch Issues

#### **CUDA Errors**
- Ensure PyTorch is installed with CUDA support
- Check GPU driver compatibility
- Verify CUDA toolkit installation
- Try: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

#### **"No CUDA devices available"**
- Check: `nvidia-smi` works
- Verify: PyTorch CUDA installation
- Test: `python -c "import torch; print(torch.cuda.is_available())"`

### 📊 Dataset Issues

#### **"Dataset not found"**
- **Check**: `dataset/TinyStories-train.txt` exists
- **Download**: Get TinyStories dataset
- **Custom path**: Use `--dataset-path /path/to/your/dataset.txt`

#### **"Training data seems wrong"**
- **Check**: Dataset format (one sentence per line or paragraph)
- **Verify**: File encoding (should be UTF-8)
- **Test**: Start with `--dataset-percent 10` to verify

## 📊 Monitoring Training

### Real-time Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop
```

### Training Logs
The script provides detailed logging:
- Training loss per epoch
- Validation loss per epoch
- GPU memory usage
- Training time estimates

## 🔄 Model Conversion

After training, convert PyTorch models to RP2040 format using our simple conversion script:

### Simple Conversion
```bash
# Convert any trained model to RP2040 format
python model-convert.py rp2040-speed
python model-convert.py rp2040-optimized
python model-convert.py my-custom-model
```

### What It Does
1. **Takes** `checkpoints/model-name.pt`
2. **Creates** `models/model-name/` folder
3. **Generates** all necessary files for RP2040 inference

### File Structure for RP2040
```
models/
└── rp2040-speed/
    ├── model_256p.bin      # Converted model weights
    ├── vocab_256p.bin      # Vocabulary file
    └── config_256p.json    # Model configuration
```

### Complete Workflow
```bash
# 1. Train your model
python train-gpu2.py --model rp2040-speed --dataset-percent 10 --epochs 100

# 2. Convert to RP2040 format (one command!)
python model-convert.py rp2040-speed

# 3. Copy to RP2040 and test
# Copy models/rp2040-speed/ folder to your RP2040
# Run: python inference.py
```

## 📚 Technical Details

### Architecture Features
- **Multi-head Attention**: Configurable number of attention heads
- **Feed-forward Networks**: 32x hidden dimension ratios for computational power
- **Layer Normalization**: Stable training with proper normalization
- **Position Embeddings**: Learnable position encodings
- **Dropout**: Regularization to prevent overfitting

### Training Optimizations
- **Mixed Precision**: FP16 training for memory efficiency
- **Gradient Clipping**: Prevents gradient explosion
- **Learning Rate Scheduling**: Cosine annealing for optimal convergence
- **Early Stopping**: Prevents overfitting
- **Checkpointing**: Resume training from any point

### Data Processing
- **Efficient Tokenization**: Character and word-level tokenization
- **Smart Padding**: Dynamic sequence length handling
- **Data Augmentation**: Random sentence sampling
- **Validation Split**: 90/10 train/validation split

## 🎯 Next Steps

1. **Install Dependencies**: `pip install -r requirements-gpu.txt`
2. **Verify GPU**: Ensure CUDA is working
3. **Start Training**: Begin with small dataset (10%)
4. **Scale Up**: Gradually increase dataset size
5. **Compare Models**: Train all three architectures
6. **Convert Models**: Convert to RP2040 format for deployment

## 📞 Support

For issues or questions:
1. Check GPU compatibility and CUDA installation
2. Verify dataset path and format
3. Monitor GPU memory usage
4. Check training logs for error messages

Happy training! 🚀
