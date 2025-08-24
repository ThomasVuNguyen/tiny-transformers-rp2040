# GPU Training for RP2040 Production Models

This script provides GPU-accelerated training for the production-ready RP2040 transformer models based on comprehensive architectural studies from `log.md`.

## 🚀 Production Models

Based on the findings in `log.md`, we have three optimized architectures:

| Model | Parameters | Architecture | Use Case |
|-------|------------|--------------|----------|
| `rp2040-optimized` | 15-20K | vocab=512, dim=8, layers=3, heads=8, hidden=256 | **Production-ready** - Best balance of speed and quality |
| `rp2040-speed` | 8-12K | vocab=256, dim=6, layers=2, heads=4, hidden=192 | **Speed-focused** - Fastest inference on RP2040 |
| `rp2040-quality` | 25-35K | vocab=1024, dim=12, layers=4, heads=12, hidden=384 | **Quality-focused** - Highest quality text generation |

## 🎯 Key Features

- **GPU Acceleration**: Uses PyTorch with CUDA for 10-100x speedup
- **Dataset Increments**: Train with 10%, 20%, 30%, ..., 100% of TinyStories dataset
- **Mixed Precision**: Automatic mixed precision (AMP) for memory efficiency
- **Smart Batching**: Automatic batch size optimization based on GPU memory
- **Early Stopping**: Prevents overfitting with validation-based early stopping
- **Checkpointing**: Saves best models and regular checkpoints
- **Training History**: Logs all training metrics for analysis

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
pip install numpy tqdm
```

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
python train-gpu.py [OPTIONS]

Options:
  --model MODEL           Model to train (rp2040-optimized, rp2040-speed, rp2040-quality)
  --dataset-percent INT   Dataset percentage (10, 20, 30, ..., 100)
  --epochs INT           Number of training epochs (default: 100)
  --dataset-path PATH    Path to TinyStories dataset (default: dataset/TinyStories-train.txt)
  --batch-size INT       Override automatic batch size
  -h, --help            Show help message
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

## 📁 Output Files

The script creates several directories:

```
checkpoints/           # Model checkpoints
├── best_rp2040-optimized_25p.pt
├── checkpoint_rp2040-optimized_25p_epoch10.pt
└── final_rp2040-optimized_25p.pt

training_history/      # Training metrics
└── rp2040-optimized_25p_history.json
```

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

### GPU Memory Issues
- Reduce batch size: `--batch-size 16`
- Use smaller dataset: `--dataset-percent 10`
- Use smaller model: `--model rp2040-speed`

### CUDA Errors
- Ensure PyTorch is installed with CUDA support
- Check GPU driver compatibility
- Verify CUDA toolkit installation

### Slow Training
- Check GPU utilization with `nvidia-smi`
- Ensure mixed precision is working
- Verify data loading is not bottleneck

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

After training, convert PyTorch models to RP2040 format:

```python
# Example conversion script (to be implemented)
from train_gpu import ProductionTransformer
import torch

# Load trained model
checkpoint = torch.load('checkpoints/best_rp2040-optimized_50p.pt')
model = ProductionTransformer(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])

# Convert to RP2040 format
# ... conversion logic ...
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
