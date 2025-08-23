"""
Scalable RP2040 Inference Script for CircuitPython
Automatically detects and loads any model size

Supports model sizes from 1K to 300K+ parameters
Will test memory limits and inference speed

Usage:
1. Copy model files from Windows training to models/ folder:
   - models/[model-name]/model_[params]p.bin
   - models/[model-name]/vocab_[params]p.bin
   - models/[model-name]/config_[params]p.json
2. This script will auto-detect available models
3. You can test multiple models to find RP2040 limits

CircuitPython Notes:
- Uses os.listdir() for directory operations
- CircuitPython-compatible file operations
- Memory management optimized for RP2040
- No pre-configuration needed - just scan and use
"""

import gc
import struct
import math
import time
import microcontroller
import json
import os

# ============================================================================
# FAST MATH APPROXIMATIONS FOR RP2040 PERFORMANCE
# ============================================================================

def fast_exp(x):
    """Fast exponential approximation for RP2040"""
    # Clamp input to prevent overflow
    if x > 10.0:
        return 22026.465794806718  # e^10
    elif x < -10.0:
        return 0.000045399929762484854  # e^-10
    
    # Taylor series approximation (faster than math.exp on RP2040)
    # e^x ≈ 1 + x + x²/2 + x³/6 + x⁴/24
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    return 1.0 + x + x2 * 0.5 + x3 * 0.16666666666666666 + x4 * 0.041666666666666664

def fast_sigmoid(x):
    """Fast sigmoid approximation optimized for RP2040"""
    # Clamp to prevent overflow
    if x > 10.0:
        return 0.9999546021312976  # sigmoid(10)
    elif x < -10.0:
        return 0.000045397868702434395  # sigmoid(-10)
    
    # Use fast_exp for better performance
    exp_neg_x = fast_exp(-x)
    return 1.0 / (1.0 + exp_neg_x)

def fast_sqrt(x):
    """Fast square root using Newton's method (2 iterations)"""
    if x <= 0.0:
        return 0.0
    
    # Initial guess using bit manipulation approximation
    # For most values in neural networks, this is quite good
    if x >= 1.0:
        guess = x * 0.5
    else:
        guess = x * 2.0
    
    # Two Newton iterations: x_new = 0.5 * (x + n/x)
    guess = 0.5 * (guess + x / guess)
    guess = 0.5 * (guess + x / guess)
    
    return guess

# ============================================================================
# CIRCUITPYTHON CONFIGURATION - Edit these variables as needed
# ============================================================================

# Model folder structure
MODELS_DIR = "models"  # Main models directory

# File naming patterns (used for detection)
MODEL_FILE_PATTERN = "model_{params}p.bin"      # e.g., "model_1024p.bin"
VOCAB_FILE_PATTERN = "vocab_{params}p.bin"     # e.g., "vocab_1024p.bin"
CONFIG_FILE_PATTERN = "config_{params}p.json"  # e.g., "config_1024p.json"

# Legacy file patterns (for backward compatibility)
LEGACY_MODEL_PATTERN = "model_{name}.bin"      # e.g., "model_tiny.bin"
LEGACY_VOCAB_PATTERN = "vocab_{name}.bin"     # e.g., "vocab_tiny.bin"

# ============================================================================
# END CONFIGURATION
# ============================================================================

class MemoryManager:
    """Enhanced memory management with detailed reporting for CircuitPython"""
    
    def __init__(self):
        self.initial_free = gc.mem_free()
        self.peak_usage = 0
        print(f"=== RP2040 Memory Info ===")
        print(f"Initial free: {self.initial_free:,} bytes")
        print(f"Total RAM: ~256KB")
        
        # CircuitPython specific info
        try:
            import sys
            print(f"CircuitPython version: {sys.version}")
        except:
            pass
    
    def check(self, label="", critical_threshold=20000):
        """Check memory with warnings - optimized for CircuitPython"""
        free = gc.mem_free()
        used = self.initial_free - free
        self.peak_usage = max(self.peak_usage, used)
        
        status = "OK"
        if free < critical_threshold:
            status = "⚠️  LOW"
        elif free < critical_threshold // 2:
            status = "🚨 CRITICAL"
        
        print(f"{label:20s}: {free:6,} free, {used:6,} used [{status}]")
        
        if free < critical_threshold:
            print(f"  WARNING: Less than {critical_threshold:,} bytes free!")
            self.cleanup()
        
        return free
    
    def cleanup(self):
        """Aggressive cleanup for CircuitPython"""
        before = gc.mem_free()
        gc.collect()
        after = gc.mem_free()
        freed = after - before
        if freed > 1000:
            print(f"  GC freed: {freed:,} bytes")
        return after
    
    def report_peak(self):
        """Report peak memory usage"""
        print(f"Peak memory usage: {self.peak_usage:,} bytes")
        print(f"Memory efficiency: {self.peak_usage/self.initial_free*100:.1f}%")
        
        # CircuitPython recommendations
        if self.peak_usage > 200000:  # >200KB
            print("⚠️  High memory usage - consider smaller models")
        elif self.peak_usage > 150000:  # >150KB
            print("⚠️  Moderate memory usage - test carefully")
        else:
            print("✅ Memory usage looks good for RP2040")

mem = MemoryManager()

class ScalableTokenizer:
    """Tokenizer that works with any vocab size"""
    
    def __init__(self, vocab_file):
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0
        
        if not self._load_vocab(vocab_file):
            print(f"Failed to load {vocab_file}, creating fallback vocab")
            self._create_fallback_vocab()
    
    def _load_vocab(self, vocab_file):
        """Load vocabulary from binary file"""
        try:
            with open(vocab_file, 'rb') as f:
                # Read vocab size
                size_bytes = f.read(2)
                if len(size_bytes) != 2:
                    return False
                
                self.vocab_size = struct.unpack('H', size_bytes)[0]
                print(f"Loading vocabulary: {self.vocab_size} tokens")
                
                # Read tokens
                loaded = 0
                for i in range(self.vocab_size):
                    len_byte = f.read(1)
                    if len(len_byte) != 1:
                        break
                    
                    token_len = struct.unpack('B', len_byte)[0]
                    
                    if token_len > 0:
                        token_bytes = f.read(token_len)
                        if len(token_bytes) == token_len:
                            try:
                                token_str = token_bytes.decode('utf-8')
                                self.vocab[i] = token_str
                                self.reverse_vocab[token_str] = i
                                loaded += 1
                            except:
                                pass
                    else:
                        self.vocab[i] = f"<empty_{i}>"
                
                print(f"Loaded {loaded}/{self.vocab_size} tokens successfully")
                return True
                
        except Exception as e:
            print(f"Error loading vocab: {e}")
            return False
    
    def _create_fallback_vocab(self):
        """Create basic fallback vocabulary"""
        self.vocab_size = 64
        self.vocab[0] = "<pad>"
        self.vocab[1] = "<s>"
        self.vocab[2] = "</s>"
        self.vocab[3] = "<unk>"
        self.vocab[4] = " "
        
        chars = "etaoinshrdlcumwfgypbvkjxqz.,!?'-"
        for i, char in enumerate(chars):
            if i + 5 < self.vocab_size:
                self.vocab[i + 5] = char
                self.reverse_vocab[char] = i + 5
        
        print("Created fallback vocabulary")
    
    def encode(self, text):
        """Smart encoding with longest-match"""
        if not text:
            return [1]
        
        tokens = [1]  # BOS
        text = text.lower()
        i = 0
        
        while i < len(text):
            found = False
            # Try longer matches first
            for length in range(min(8, len(text) - i), 0, -1):
                substr = text[i:i+length]
                if substr in self.reverse_vocab:
                    tokens.append(self.reverse_vocab[substr])
                    i += length
                    found = True
                    break
            
            if not found:
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
            if token == 1:
                continue
            elif token == 2:
                break
            elif 0 <= token < len(self.vocab) and token in self.vocab:
                text += self.vocab[token]
        return text

class ScalableTransformer:
    """Transformer that adapts to any model size"""
    
    def __init__(self, model_file):
        # Model config (loaded from file)
        self.config = {}
        self.vocab_size = 0
        self.dim = 0
        self.hidden_dim = 0
        self.n_layers = 0
        self.n_heads = 0
        self.max_seq_len = 0
        
        # Model weights
        self.token_embedding = None
        self.layers = []
        self.final_ln_weight = None
        
        # KV cache for autoregressive generation (major performance optimization)
        self.kv_cache = None
        self.cache_seq_len = 0
        
        # Load model
        if not self._load_model(model_file):
            raise RuntimeError(f"Failed to load model from {model_file}")
    
    def _load_model(self, model_file):
        """Load model from binary file"""
        print(f"Loading model: {model_file}")
        
        # Aggressive memory cleanup before loading
        print("Preparing memory...")
        for _ in range(3):
            gc.collect()
        
        mem.check("Before model load")
        
        try:
            with open(model_file, 'rb') as f:
                # Read header
                header = f.read(32)
                if len(header) < 32:
                    print("Invalid header size")
                    return False
                
                # Parse config from header
                config_values = struct.unpack("8I", header)
                self.vocab_size = config_values[0]
                self.dim = config_values[1] 
                self.hidden_dim = config_values[2]
                self.n_layers = config_values[3]
                self.n_heads = config_values[4]
                self.max_seq_len = config_values[5]
                
                print(f"Model config:")
                print(f"  Vocab size: {self.vocab_size:,}")
                print(f"  Dimensions: {self.dim}")
                print(f"  Hidden dim: {self.hidden_dim}")
                print(f"  Layers: {self.n_layers}")
                print(f"  Heads: {self.n_heads}")
                print(f"  Max seq len: {self.max_seq_len}")
                
                # Calculate total parameters
                embed_params = self.vocab_size * self.dim
                layer_params = self.n_layers * (4 * self.dim * self.dim + self.dim * self.hidden_dim * 2)
                total_params = embed_params + layer_params
                
                print(f"  Parameters: {total_params:,} ({total_params/1000:.1f}K)")
                
                # Estimate memory needed
                memory_needed = total_params * 4  # 4 bytes per float32
                print(f"  Memory needed: {memory_needed:,} bytes ({memory_needed/1024:.1f}KB)")
                
                current_free = gc.mem_free()
                if memory_needed > current_free:
                    print(f"  ⚠️  WARNING: Need {memory_needed:,}B but only {current_free:,}B free!")
                    print("  Model may not fit in memory!")
                    
                    # Check if we have enough for chunked loading (need ~25% overhead)
                    overhead_needed = memory_needed * 1.25
                    if overhead_needed > current_free:
                        print(f"  ❌ Even with chunked loading, need ~{overhead_needed:,}B")
                        print(f"  Consider using a smaller model (< {current_free//5000}K parameters)")
                        return False
                    else:
                        print(f"  ✅ Will try chunked loading to reduce peak memory usage")
                
                # Additional memory fragmentation check
                if current_free < 50000:  # Less than 50KB free
                    print(f"  ⚠️  Low memory warning: Only {current_free:,}B free")
                    print("  Memory fragmentation may cause allocation failures")
                
                # Load token embeddings with chunked reading to avoid large allocations
                embed_size = self.vocab_size * self.dim
                print(f"Loading embeddings in chunks to avoid large allocation...")
                
                # Read embeddings row by row to avoid large memory allocation
                self.token_embedding = []
                bytes_per_row = self.dim * 4  # 4 bytes per float
                
                for i in range(self.vocab_size):
                    if i % 10 == 0:  # Progress indicator
                        gc.collect()  # Collect garbage every 10 rows
                    
                    row_bytes = f.read(bytes_per_row)
                    if len(row_bytes) != bytes_per_row:
                        print(f"Expected {bytes_per_row} bytes for row {i}, got {len(row_bytes)}")
                        return False
                    
                    # Unpack row and convert to list
                    row_floats = struct.unpack(f'{self.dim}f', row_bytes)
                    row = list(row_floats)  # Convert tuple to list
                    self.token_embedding.append(row)
                
                mem.cleanup()
                mem.check("After embeddings")
                
                # Load layers with memory optimization
                self.layers = []
                for layer_id in range(self.n_layers):
                    print(f"  Loading layer {layer_id + 1}/{self.n_layers}")
                    gc.collect()  # Clean up before each layer
                    
                    layer = {}
                    
                    # Layer norms (small, load normally)
                    ln1_bytes = f.read(self.dim * 4)
                    ln2_bytes = f.read(self.dim * 4)
                    layer['ln1_weight'] = list(struct.unpack(f'{self.dim}f', ln1_bytes))
                    layer['ln2_weight'] = list(struct.unpack(f'{self.dim}f', ln2_bytes))
                    
                    # Attention weights - load row by row to avoid large allocations
                    attn_size = self.dim * self.dim
                    for weight_name in ['wq', 'wk', 'wv', 'wo']:
                        print(f"    Loading {weight_name}...")
                        matrix = []
                        bytes_per_row = self.dim * 4
                        
                        for i in range(self.dim):
                            if i % 5 == 0:  # More frequent GC for attention matrices
                                gc.collect()
                            
                            row_bytes = f.read(bytes_per_row)
                            if len(row_bytes) != bytes_per_row:
                                print(f"Error reading {weight_name} row {i}")
                                return False
                            
                            row_floats = struct.unpack(f'{self.dim}f', row_bytes)
                            matrix.append(list(row_floats))
                        
                        layer[weight_name] = matrix
                    
                    # FFN weights - load row by row to avoid large allocations
                    print(f"    Loading w1...")
                    # W1: dim x hidden_dim
                    w1_matrix = []
                    w1_bytes_per_row = self.hidden_dim * 4
                    for i in range(self.dim):
                        if i % 3 == 0:
                            gc.collect()
                        
                        row_bytes = f.read(w1_bytes_per_row)
                        if len(row_bytes) != w1_bytes_per_row:
                            print(f"Error reading w1 row {i}")
                            return False
                        
                        row_floats = struct.unpack(f'{self.hidden_dim}f', row_bytes)
                        w1_matrix.append(list(row_floats))
                    layer['w1'] = w1_matrix
                    
                    print(f"    Loading w2...")
                    # W2: hidden_dim x dim  
                    w2_matrix = []
                    w2_bytes_per_row = self.dim * 4
                    for i in range(self.hidden_dim):
                        if i % 3 == 0:
                            gc.collect()
                        
                        row_bytes = f.read(w2_bytes_per_row)
                        if len(row_bytes) != w2_bytes_per_row:
                            print(f"Error reading w2 row {i}")
                            return False
                        
                        row_floats = struct.unpack(f'{self.dim}f', row_bytes)
                        w2_matrix.append(list(row_floats))
                    layer['w2'] = w2_matrix
                    
                    self.layers.append(layer)
                    mem.cleanup()
                
                # Final layer norm
                final_ln_bytes = f.read(self.dim * 4)
                self.final_ln_weight = list(struct.unpack(f'{self.dim}f', final_ln_bytes))
                
                print("✅ Model loaded successfully!")
                mem.check("After full model load")
                return True
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exception(type(e), e, e.__traceback__)
            return False
    
    def _matmul(self, a, b_matrix):
        """Optimized matrix multiplication with better memory access"""
        if len(b_matrix) == 0:
            return [0.0] * len(a)
        
        # Pre-allocate result for better performance
        cols = len(b_matrix[0])
        rows = len(a)
        result = [0.0] * cols
        
        # Optimized inner loop with better cache locality
        for j in range(cols):
            val = 0.0
            # Unroll loop for small dimensions (common case)
            if rows <= 32:  # Most of our models have dim <= 32
                for i in range(rows):
                    val += a[i] * b_matrix[i][j]
            else:
                # For larger dimensions, use chunked access
                for i in range(0, rows, 4):
                    end = min(i + 4, rows)
                    for k in range(i, end):
                        val += a[k] * b_matrix[k][j]
            result[j] = val
        return result
    
    def _layer_norm(self, x, weight, eps=1e-6):
        """Layer normalization with fast square root"""
        mean = sum(x) / len(x)
        variance = sum((val - mean) ** 2 for val in x) / len(x)
        std = fast_sqrt(variance + eps)  # Use fast sqrt
        inv_std = 1.0 / std if std > 0.0 else 0.0
        return [(val - mean) * inv_std * weight[i] for i, val in enumerate(x)]
    
    def _softmax(self, x):
        """Softmax with fast exponential approximation"""
        max_val = max(x)
        exp_vals = [fast_exp(val - max_val) for val in x]  # Use fast exp
        sum_exp = sum(exp_vals)
        if sum_exp == 0:
            return [1.0 / len(x)] * len(x)
        inv_sum = 1.0 / sum_exp  # Pre-compute inverse
        return [val * inv_sum for val in exp_vals]
    
    def _attention(self, x_seq, layer, layer_idx, use_cache=True):
        """Multi-head self-attention with KV caching optimization"""
        seq_len = len(x_seq)
        
        # For autoregressive generation, we can reuse cached K,V
        if use_cache and seq_len == 1 and self.cache_seq_len > 0:
            # Only compute for the new token
            x_new = x_seq[0]
            q_new = self._matmul(x_new, layer['wq'])
            k_new = self._matmul(x_new, layer['wk'])
            v_new = self._matmul(x_new, layer['wv'])
            
            # Get cached K,V
            cached_k, cached_v = self._get_cached_kv(layer_idx)
            
            # Update cache
            self._update_kv_cache(layer_idx, k_new, v_new)
            
            # Compute attention only for new position
            scores = []
            # Attend to all previous positions + current
            for j in range(len(cached_k)):
                score = sum(q_new[k] * cached_k[j][k] for k in range(self.dim))
                score = score / fast_sqrt(self.dim)  # Use fast sqrt
                scores.append(score)
            
            # Add current position
            score = sum(q_new[k] * k_new[k] for k in range(self.dim))
            score = score / fast_sqrt(self.dim)
            scores.append(score)
            
            # Softmax
            attn_weights = self._softmax(scores)
            
            # Weighted sum of values
            output = [0.0] * self.dim
            for j, weight in enumerate(attn_weights[:-1]):
                for k in range(self.dim):
                    output[k] += weight * cached_v[j][k]
            
            # Add contribution from new token
            weight = attn_weights[-1]
            for k in range(self.dim):
                output[k] += weight * v_new[k]
            
            # Output projection
            projected = self._matmul(output, layer['wo'])
            return [projected]
        
        else:
            # Full attention computation (first pass or no cache)
            # Q, K, V projections
            Q = [self._matmul(x, layer['wq']) for x in x_seq]
            K = [self._matmul(x, layer['wk']) for x in x_seq] 
            V = [self._matmul(x, layer['wv']) for x in x_seq]
            
            # Update cache if using cache
            if use_cache:
                for i in range(seq_len):
                    self._update_kv_cache(layer_idx, K[i], V[i])
            
            output_seq = []
            sqrt_dim = fast_sqrt(self.dim)  # Pre-compute
            
            for i in range(seq_len):
                # Attention scores for position i
                scores = []
                for j in range(i + 1):  # Causal mask
                    score = sum(Q[i][k] * K[j][k] for k in range(self.dim))
                    score = score / sqrt_dim
                    scores.append(score)
                
                # Pad future positions with large negative values
                for j in range(i + 1, seq_len):
                    scores.append(-1e9)
                
                # Softmax
                attn_weights = self._softmax(scores)
                
                # Weighted sum of values
                output = [0.0] * self.dim
                for j in range(i + 1):
                    weight = attn_weights[j]
                    for k in range(self.dim):
                        output[k] += weight * V[j][k]
                
                output_seq.append(output)
            
            # Output projection
            projected_seq = []
            for output in output_seq:
                projected = self._matmul(output, layer['wo'])
                projected_seq.append(projected)
            
            return projected_seq
    
    def _feed_forward(self, x, layer):
        """Feed-forward with fast SiLU activation"""
        # First projection
        hidden = self._matmul(x, layer['w1'])
        
        # Fast SiLU activation: x * sigmoid(x)
        for i in range(len(hidden)):
            sigmoid_val = fast_sigmoid(hidden[i])  # Use fast sigmoid
            hidden[i] = hidden[i] * sigmoid_val
        
        # Second projection
        output = self._matmul(hidden, layer['w2'])
        
        return output
    
    def _init_kv_cache(self, seq_len):
        """Initialize KV cache for the sequence"""
        if self.kv_cache is None or len(self.kv_cache) != self.n_layers:
            self.kv_cache = []
            for _ in range(self.n_layers):
                # Each layer has K and V caches
                layer_cache = {
                    'k': [],  # List of key vectors for each position
                    'v': []   # List of value vectors for each position
                }
                self.kv_cache.append(layer_cache)
        
        self.cache_seq_len = 0  # Reset cache length
    
    def _update_kv_cache(self, layer_idx, k_new, v_new):
        """Update KV cache with new key/value vectors"""
        if self.kv_cache and layer_idx < len(self.kv_cache):
            self.kv_cache[layer_idx]['k'].append(k_new)
            self.kv_cache[layer_idx]['v'].append(v_new)
    
    def _get_cached_kv(self, layer_idx):
        """Get cached K,V for a layer"""
        if self.kv_cache and layer_idx < len(self.kv_cache):
            return self.kv_cache[layer_idx]['k'], self.kv_cache[layer_idx]['v']
        return [], []
    
    def clear_cache(self):
        """Clear KV cache (call when starting new sequence)"""
        self.kv_cache = None
        self.cache_seq_len = 0
    
    def forward(self, tokens, use_cache=True):
        """Optimized forward pass with KV caching"""
        seq_len = len(tokens)
        if seq_len == 0:
            return [0.0] * self.vocab_size
        
        # Initialize cache for first call
        if use_cache and (self.kv_cache is None or seq_len > 1):
            self._init_kv_cache(seq_len)
            self.cache_seq_len = 0
        
        # Token embeddings
        x_seq = []
        for token in tokens:
            if 0 <= token < self.vocab_size:
                embedding = self.token_embedding[token][:]
            else:
                embedding = self.token_embedding[3][:]  # <unk>
            x_seq.append(embedding)
        
        # Transformer layers
        for layer_id, layer in enumerate(self.layers):
            # Pre-attention norm
            normed_seq = []
            for x in x_seq:
                normed = self._layer_norm(x, layer['ln1_weight'])
                normed_seq.append(normed)
            
            # Self-attention with caching
            attn_output = self._attention(normed_seq, layer, layer_id, use_cache)
            
            # Residual connection
            attn_seq_len = len(attn_output)
            for i in range(attn_seq_len):
                for j in range(self.dim):
                    x_seq[i][j] += attn_output[i][j]
            
            # Pre-FFN norm
            normed_seq = []
            for x in x_seq:
                normed = self._layer_norm(x, layer['ln2_weight'])
                normed_seq.append(normed)
            
            # Feed-forward
            ffn_output = []
            for normed in normed_seq:
                ff_out = self._feed_forward(normed, layer)
                ffn_output.append(ff_out)
            
            # Residual connection
            for i in range(len(ffn_output)):
                for j in range(self.dim):
                    x_seq[i][j] += ffn_output[i][j]
        
        # Update cache sequence length
        if use_cache:
            self.cache_seq_len += seq_len
        
        # Final layer norm
        last_hidden = self._layer_norm(x_seq[-1], self.final_ln_weight)
        
        # Optimized output projection (pre-allocate)
        logits = [0.0] * self.vocab_size
        for i in range(self.vocab_size):
            logit = 0.0
            for j in range(self.dim):
                logit += last_hidden[j] * self.token_embedding[i][j]
            logits[i] = logit
        
        return logits
    
    def sample(self, logits, temperature=1.0):
        """Sample next token"""
        if temperature <= 0.0:
            return logits.index(max(logits))
        
        scaled_logits = [l / temperature for l in logits]
        probs = self._softmax(scaled_logits)
        
        # Time-based random sampling
        rand_val = (time.monotonic_ns() % 10000) / 10000.0
        
        cumsum = 0.0
        for i, prob in enumerate(probs):
            cumsum += prob
            if rand_val < cumsum:
                return i
        
        return len(probs) - 1

class ModelDetector:
    """Detect and test available models"""
    
    def __init__(self):
        self.available_models = []
        self.detect_models()
    
    def detect_models(self):
        """Find all available model files"""
        print("=== Detecting Available Models ===")
        
        # First try new folder structure
        self._detect_folder_models()
        
        # Then try legacy file structure
        self._detect_legacy_models()
        
        if not self.available_models:
            print("❌ No model files found!")
            print(f"Expected either:")
            print(f"  New structure: {MODELS_DIR}/[model-name]/model_[params]p.bin")
            print(f"  Legacy structure: model_[name].bin and vocab_[name].bin")
        else:
            print(f"✅ Found {len(self.available_models)} models")
    
    def _detect_folder_models(self):
        """Automatically detect models in the models folder"""
        # Check if models directory exists by trying to list it
        try:
            items = os.listdir(MODELS_DIR)
        except OSError:
            print(f"Models directory '{MODELS_DIR}' not found")
            return
        
        print(f"Scanning {MODELS_DIR}/ directory for model folders...")
        
        try:
            # Get all items in models directory
            for item in items:
                # In CircuitPython, we need to check if it's a directory differently
                # Try to list the contents to see if it's a directory
                try:
                    folder_path = f"{MODELS_DIR}/{item}"
                    files = os.listdir(folder_path)
                    
                    # If we can list files, it's a directory
                    print(f"  Checking folder: {item}")
                    
                    # Look for model files in this folder
                    model_files = [f for f in files if f.startswith("model_") and f.endswith(".bin")]
                    
                    if model_files:
                        # Extract parameter count from filename
                        model_file = model_files[0]  # Should be only one
                        param_str = model_file[6:-4]  # Remove "model_" prefix and ".bin" suffix
                        
                        if param_str.endswith('p'):
                            try:
                                param_count = int(param_str[:-1])
                                
                                # Check if all required files exist by trying to open them
                                vocab_file = f"vocab_{param_count}p.bin"
                                config_file = f"config_{param_count}p.json"
                                
                                vocab_exists = False
                                config_exists = False
                                
                                # Check if vocab file exists
                                try:
                                    with open(f"{folder_path}/{vocab_file}", 'rb') as f:
                                        f.read(2)  # Try to read vocab size
                                    vocab_exists = True
                                except:
                                    pass
                                
                                # Check if config file exists
                                try:
                                    with open(f"{folder_path}/{config_file}", 'r') as f:
                                        f.read(1)  # Try to read first character
                                    config_exists = True
                                except:
                                    pass
                                
                                if vocab_exists and config_exists:
                                    # Generate description based on folder name and params
                                    description = self._generate_description(item, param_count)
                                    
                                    self.available_models.append({
                                        'name': item,
                                        'folder': item,
                                        'model_file': f"{folder_path}/{model_file}",
                                        'vocab_file': f"{folder_path}/{vocab_file}",
                                        'config_file': f"{folder_path}/{config_file}",
                                        'params': param_count,
                                        'description': description,
                                        'type': 'folder'
                                    })
                                    
                                    print(f"    ✅ Found model: {param_count:,} params - {description}")
                                else:
                                    missing = []
                                    if not vocab_exists:
                                        missing.append(vocab_file)
                                    if not config_exists:
                                        missing.append(config_file)
                                    print(f"    ⚠️  Missing files: {', '.join(missing)}")
                                    
                            except ValueError:
                                print(f"    ⚠️  Could not parse params from {model_file}")
                                continue
                        else:
                            print(f"    ⚠️  Model file doesn't match pattern: {model_file}")
                    else:
                        print(f"    ⚠️  No model files found")
                
                except OSError:
                    # This item is not a directory, skip it
                    print(f"  Skipping non-directory: {item}")
                except Exception as e:
                    print(f"    ⚠️  Error checking {item}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error scanning {MODELS_DIR}: {e}")
    
    def _generate_description(self, folder_name, param_count):
        """Generate a human-readable description for a model"""
        # Try to extract meaningful info from folder name
        if '-' in folder_name:
            parts = folder_name.split('-')
            if len(parts) >= 2:
                try:
                    size_part = parts[1]
                    if size_part.endswith('k') or size_part.endswith('K'):
                        size_num = int(size_part[:-1])
                        if size_num <= 5:
                            return f"Small {size_num}K parameter model"
                        elif size_num <= 20:
                            return f"Medium {size_num}K parameter model"
                        else:
                            return f"Large {size_num}K parameter model"
                except:
                    pass
        
        # Fallback description
        if param_count < 5000:
            return f"Small model ({param_count:,} params)"
        elif param_count < 20000:
            return f"Medium model ({param_count:,} params)"
        else:
            return f"Large model ({param_count:,} params)"
    
    def _detect_legacy_models(self):
        """Detect models using legacy file structure"""
        print("Checking for legacy model files...")
        
        # Common legacy model names to check
        legacy_names = ['tiny', 'small', 'medium', 'large', 'xlarge', 'custom']
        
        for model_name in legacy_names:
            model_file = LEGACY_MODEL_PATTERN.format(name=model_name)
            vocab_file = LEGACY_VOCAB_PATTERN.format(name=model_name)
            
            try:
                # Check if both files exist by trying to open them
                model_exists = False
                vocab_exists = False
                
                # Try to open model file
                try:
                    with open(model_file, 'rb') as f:
                        header = f.read(32)
                        if len(header) >= 32:
                            model_exists = True
                except:
                    pass
                
                # Try to open vocab file
                try:
                    with open(vocab_file, 'rb') as f:
                        f.read(2)  # Try to read vocab size
                    vocab_exists = True
                except:
                    pass
                
                if model_exists and vocab_exists:
                    # Extract parameter count from header
                    try:
                        with open(model_file, 'rb') as f:
                            header = f.read(32)
                            if len(header) >= 32:
                                # Extract parameter count from header
                                config_values = struct.unpack("8I", header)
                                vocab_size = config_values[0]
                                dim = config_values[1]
                                hidden_dim = config_values[2]
                                n_layers = config_values[3]
                                
                                # Calculate approximate parameter count
                                embed_params = vocab_size * dim
                                layer_params = n_layers * (4 * dim * dim + dim * hidden_dim * 2)
                                total_params = embed_params + layer_params
                                
                                self.available_models.append({
                                    'name': model_name,
                                    'folder': None,
                                    'model_file': model_file,
                                    'vocab_file': vocab_file,
                                    'config_file': None,
                                    'params': total_params,
                                    'description': f"Legacy {model_name} model (~{total_params//1000}K params)",
                                    'type': 'legacy'
                                })
                                
                                print(f"✅ Found legacy {model_name} (~{total_params//1000}K params)")
                    except Exception as e:
                        print(f"⚠️  Error reading {model_file}: {e}")
                        continue
                
            except Exception as e:
                continue
        
        if not any(m['type'] == 'legacy' for m in self.available_models):
            print("No legacy models found")
    
    def test_model_loading(self, model_info):
        """Test if a model can be loaded"""
        print(f"\n=== Testing {model_info['name'].upper()} Model ===")
        print(f"Description: {model_info['description']}")
        print(f"Parameters: {model_info['params']:,}")
        print(f"Type: {model_info['type']}")
        
        try:
            # Check memory before loading
            initial_free = mem.check("Before loading")
            
            # Load tokenizer
            tokenizer = ScalableTokenizer(model_info['vocab_file'])
            
            # Load model
            model = ScalableTransformer(model_info['model_file'])
            
            # Check memory after loading
            final_free = mem.check("After loading")
            memory_used = initial_free - final_free
            
            print(f"Memory used by model: {memory_used:,} bytes")
            
            # Quick inference test
            test_tokens = tokenizer.encode("hello")
            print(f"Test tokens: {test_tokens}")
            
            start_time = time.monotonic()
            logits = model.forward(test_tokens)
            inference_time = time.monotonic() - start_time
            
            print(f"Forward pass time: {inference_time*1000:.1f}ms")
            print(f"Estimated tokens/sec: {1.0/inference_time:.1f}")
            
            # Test sampling
            next_token = model.sample(logits, temperature=0.8)
            decoded = tokenizer.decode([next_token])
            print(f"Sample output: token {next_token} -> '{decoded}'")
            
            return {
                'success': True,
                'memory_used': memory_used,
                'inference_time': inference_time,
                'model': model,
                'tokenizer': tokenizer
            }
            
        except Exception as e:
            print(f"❌ Failed to load {model_info['name']}: {e}")
            return {'success': False, 'error': str(e)}

    def test_single_model(self, model_name):
        """Test a single specific model - useful for CircuitPython development"""
        print(f"\n=== Testing Single Model: {model_name} ===")
        
        # Find the model
        model_info = None
        for info in self.available_models:
            if info['name'] == model_name:
                model_info = info
                break
        
        if not model_info:
            print(f"❌ Model '{model_name}' not found!")
            print(f"Available models: {[m['name'] for m in self.available_models]}")
            return False
        
        print(f"Found model: {model_info['description']}")
        print(f"Parameters: {model_info['params']:,}")
        print(f"Type: {model_info['type']}")
        
        # Test loading
        result = self.test_model_loading(model_info)
        
        if result['success']:
            print(f"\n✅ {model_name} model test successful!")
            print(f"Memory used: {result['memory_used']:,} bytes")
            print(f"Inference speed: {1.0/result['inference_time']:.1f} tokens/sec")
            
            # Quick generation test
            try:
                print(f"\n--- Quick Generation Test ---")
                test_prompt = "hello"
                generated = self._quick_generate(result['model'], result['tokenizer'], test_prompt, max_tokens=5)
                print(f"Prompt: '{test_prompt}'")
                print(f"Generated: '{generated}'")
            except Exception as e:
                print(f"Generation test failed: {e}")
            
            # Clean up
            del result['model']
            del result['tokenizer']
            mem.cleanup()
            
            return True
        else:
            print(f"❌ {model_name} model test failed: {result.get('error', 'Unknown error')}")
            return False
    
    def _quick_generate(self, model, tokenizer, prompt, max_tokens=5):
        """Quick text generation for testing"""
        tokens = tokenizer.encode(prompt)
        generated_tokens = []
        
        for step in range(max_tokens):
            current_tokens = tokens + generated_tokens
            
            # Truncate if too long
            if len(current_tokens) > model.max_seq_len - 1:
                current_tokens = current_tokens[-(model.max_seq_len - 1):]
            
            # Forward pass
            logits = model.forward(current_tokens)
            
            # Sample
            next_token = model.sample(logits, temperature=0.8)
            
            if next_token == 2:  # EOS
                break
            
            generated_tokens.append(next_token)
        
        # Decode result
        full_tokens = tokens + generated_tokens
        return tokenizer.decode(full_tokens)

class ScalableInference:
    """Main inference engine that handles multiple model sizes"""
    
    def __init__(self):
        print("=== Scalable RP2040 Transformer Inference ===")
        print(f"RP2040: {microcontroller.cpu.frequency//1000000}MHz, {gc.mem_free():,} bytes free")
        
        self.detector = ModelDetector()
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        
        if not self.detector.available_models:
            raise RuntimeError("No models found!")
    
    def benchmark_all_models(self):
        """Benchmark all available models"""
        print(f"\n=== BENCHMARKING ALL MODELS ===")
        results = {}
        
        for model_info in self.detector.available_models:
            result = self.detector.test_model_loading(model_info)
            results[model_info['name']] = result
            
            # Clean up for next model
            if result['success']:
                del result['model']
                del result['tokenizer']
            
            mem.cleanup()
            print(f"Memory after cleanup: {gc.mem_free():,} bytes")
            time.sleep(1)  # Brief pause
        
        # Print summary
        print(f"\n=== BENCHMARK RESULTS ===")
        print(f"{'Model':15s} {'Type':8s} {'Params':8s} {'Status':8s} {'Memory':8s} {'Speed':12s}")
        print("-" * 70)
        
        for name, result in results.items():
            if result['success']:
                memory_kb = result['memory_used'] // 1024
                speed = 1.0 / result['inference_time']
                
                # Find model info for additional details
                model_info = None
                for info in self.detector.available_models:
                    if info['name'] == name:
                        model_info = info
                        break
                
                if model_info:
                    param_str = f"{model_info['params']//1000}K" if model_info['params'] >= 1000 else str(model_info['params'])
                    model_type = model_info['type'][:7]  # Truncate to fit column
                    print(f"{name:15s} {model_type:8s} {param_str:8s} ✅      {memory_kb:5d}KB {speed:8.1f} tok/s")
                else:
                    print(f"{name:15s} {'unknown':8s} {'unknown':8s} ✅      {memory_kb:5d}KB {speed:8.1f} tok/s")
            else:
                print(f"{name:15s} {'unknown':8s} {'unknown':8s} ❌      Failed    -")
        
        return results
    
    def load_model(self, model_name):
        """Load a specific model"""
        model_info = None
        for info in self.detector.available_models:
            if info['name'] == model_name:
                model_info = info
                break
        
        if not model_info:
            print(f"Model '{model_name}' not found!")
            return False
        
        print(f"\n=== Loading {model_name.upper()} Model ===")
        
        # Clean up previous model
        if self.current_model:
            del self.current_model
            del self.current_tokenizer
            mem.cleanup()
        
        try:
            # Load new model
            self.current_tokenizer = ScalableTokenizer(model_info['vocab_file'])
            self.current_model = ScalableTransformer(model_info['model_file'])
            self.current_model_name = model_name
            
            print(f"✅ {model_name.upper()} model loaded successfully!")
            mem.check(f"{model_name} loaded")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            # Ensure current_model state is clean after failure
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            return False
    
    def generate(self, prompt="hello", max_tokens=10, temperature=0.8):
        """Generate text with optimized caching"""
        if not self.current_model:
            print("No model loaded!")
            return ""
        
        print(f"\n=== Generating with {self.current_model_name.upper()} (Optimized) ===")
        print(f"Prompt: '{prompt}'")
        
        # Clear cache for new sequence
        self.current_model.clear_cache()
        
        # Encode prompt
        tokens = self.current_tokenizer.encode(prompt)
        print(f"Input tokens ({len(tokens)}): {tokens}")
        
        start_time = time.monotonic()
        generated_tokens = []
        
        # Process initial prompt (prefill phase)
        if len(tokens) > self.current_model.max_seq_len - max_tokens:
            tokens = tokens[-(self.current_model.max_seq_len - max_tokens):]
        
        # First forward pass with full prompt
        logits = self.current_model.forward(tokens, use_cache=True)
        
        for step in range(max_tokens):
            # Sample next token
            next_token = self.current_model.sample(logits, temperature)
            
            if next_token == 2:  # EOS
                break
            
            generated_tokens.append(next_token)
            
            # Forward pass for single token (decode phase) - this uses KV cache!
            logits = self.current_model.forward([next_token], use_cache=True)
            
            # Show progress
            if step > 0 and step % 3 == 0:
                partial = self.current_tokenizer.decode(tokens + generated_tokens)
                print(f"  Step {step}: {partial}")
        
        # Final result
        full_tokens = tokens + generated_tokens
        result = self.current_tokenizer.decode(full_tokens)
        
        elapsed = time.monotonic() - start_time
        speed = len(generated_tokens) / max(elapsed, 0.001)
        
        print(f"Result: '{result}'")
        print(f"Generated {len(generated_tokens)} tokens in {elapsed:.2f}s")
        print(f"Speed: {speed:.1f} tokens/sec")
        
        return result
    
    def benchmark_performance(self, prompt="hello world", iterations=3):
        """Benchmark performance improvements"""
        if not self.current_model:
            print("No model loaded!")
            return
        
        print(f"\n=== Performance Benchmark: {self.current_model_name.upper()} ===")
        print(f"Prompt: '{prompt}' | Iterations: {iterations}")
        
        tokens = self.current_tokenizer.encode(prompt)
        print(f"Input tokens: {len(tokens)}")
        
        total_time = 0.0
        total_tokens = 0
        
        for i in range(iterations):
            print(f"\nIteration {i+1}/{iterations}:")
            
            # Clear cache for fair comparison
            self.current_model.clear_cache()
            
            start_time = time.monotonic()
            
            # Prefill phase
            logits = self.current_model.forward(tokens, use_cache=True)
            prefill_time = time.monotonic() - start_time
            
            # Decode phase (generate 5 tokens)
            decode_times = []
            for step in range(5):
                step_start = time.monotonic()
                next_token = self.current_model.sample(logits, temperature=0.8)
                if next_token == 2:  # EOS
                    break
                logits = self.current_model.forward([next_token], use_cache=True)
                decode_times.append(time.monotonic() - step_start)
                total_tokens += 1
            
            iteration_time = time.monotonic() - start_time
            total_time += iteration_time
            
            avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0
            
            print(f"  Prefill: {prefill_time*1000:.1f}ms")
            print(f"  Decode avg: {avg_decode*1000:.1f}ms/token")
            print(f"  Total: {iteration_time*1000:.1f}ms")
            print(f"  Speed: {len(decode_times)/iteration_time:.1f} tok/s")
        
        avg_time = total_time / iterations
        avg_speed = total_tokens / total_time
        
        print(f"\n📊 Average Results:")
        print(f"  Time per iteration: {avg_time*1000:.1f}ms")
        print(f"  Average speed: {avg_speed:.1f} tokens/sec")
        print(f"  Memory usage: {(139296 - gc.mem_free())/1024:.1f}KB")
        
        return avg_speed
    
    def interactive_demo(self):
        """Interactive demo with all models"""
        print(f"\n=== INTERACTIVE DEMO ===")
        
        test_prompts = [
            "hello",
            "the cat", 
            "good morning",
            "once upon a time",
            "artificial intelligence"
        ]
        
        # Test each available model
        for model_info in self.detector.available_models:
            if self.load_model(model_info['name']):
                print(f"\n--- Testing {model_info['name'].upper()} model ---")
                
                for prompt in test_prompts[:3]:  # Test fewer prompts for larger models
                    try:
                        result = self.generate(prompt, max_tokens=6, temperature=0.9)
                        time.sleep(0.5)  # Brief pause
                    except Exception as e:
                        print(f"Error generating '{prompt}': {e}")
                        break
                
                mem.cleanup()
                print(f"Memory after {model_info['name']}: {gc.mem_free():,} bytes")
                
                # Pause between models
                time.sleep(1)
            else:
                print(f"\n--- Skipping {model_info['name'].upper()} model (failed to load) ---")
    
    def list_available_models(self):
        """List all available models with details"""
        print(f"\n=== Available Models ===")
        
        if not self.detector.available_models:
            print("No models found!")
            return
        
        # Group by type
        folder_models = [m for m in self.detector.available_models if m['type'] == 'folder']
        legacy_models = [m for m in self.detector.available_models if m['type'] == 'legacy']
        
        if folder_models:
            print(f"\n📁 Folder-based models ({len(folder_models)}):")
            print(f"{'Name':15s} {'Folder':20s} {'Params':8s} {'Description':40s}")
            print("-" * 85)
            for model in sorted(folder_models, key=lambda x: x['params']):
                param_str = f"{model['params']//1000}K" if model['params'] >= 1000 else str(model['params'])
                desc = model['description'][:37] + "..." if len(model['description']) > 40 else model['description']
                print(f"{model['name']:15s} {model['folder']:20s} {param_str:8s} {desc:40s}")
        
        if legacy_models:
            print(f"\n📄 Legacy models ({len(legacy_models)}):")
            print(f"{'Name':15s} {'Params':8s} {'Description':40s}")
            print("-" * 65)
            for model in sorted(legacy_models, key=lambda x: x['params']):
                param_str = f"{model['params']//1000}K" if model['params'] >= 1000 else str(model['params'])
                desc = model['description'][:37] + "..." if len(model['description']) > 40 else model['description']
                print(f"{model['name']:15s} {param_str:8s} {desc:40s}")
        
        # Show total counts
        total_params = sum(m['params'] for m in self.detector.available_models)
        print(f"\n📊 Summary:")
        print(f"  Total models: {len(self.detector.available_models)}")
        print(f"  Total parameters: {total_params:,} ({total_params/1000:.1f}K)")
        
        # RP2040 recommendations
        working_models = [m for m in self.detector.available_models if m['params'] <= 15000]
        risky_models = [m for m in self.detector.available_models if 15000 < m['params'] <= 30000]
        large_models = [m for m in self.detector.available_models if m['params'] > 30000]
        
        print(f"\n🎯 RP2040 Recommendations:")
        if working_models:
            largest_working = max(working_models, key=lambda x: x['params'])
            print(f"  ✅ Likely to work: {len(working_models)} models (up to {largest_working['params']:,} params)")
        
        if risky_models:
            print(f"  ⚠️  Test carefully: {len(risky_models)} models ({min(m['params'] for m in risky_models):,} - {max(m['params'] for m in risky_models):,} params)")
        
        if large_models:
            print(f"  ❌ Unlikely to work: {len(large_models)} models ({min(m['params'] for m in large_models):,} - {max(m['params'] for m in large_models):,} params)")
    
    def show_configuration(self):
        """Show current configuration"""
        print(f"\n=== Configuration ===")
        print(f"Models directory: {MODELS_DIR}")
        print(f"Detection mode: Automatic folder scanning")
        print(f"File patterns:")
        print(f"  Model: {MODEL_FILE_PATTERN}")
        print(f"  Vocab: {VOCAB_FILE_PATTERN}")
        print(f"  Config: {CONFIG_FILE_PATTERN}")
        print(f"  Legacy model: {LEGACY_MODEL_PATTERN}")
        print(f"  Legacy vocab: {LEGACY_VOCAB_PATTERN}")
        print(f"\nCircuitPython features:")
        print(f"  - Automatic model detection")
        print(f"  - No pre-configuration needed")
        print(f"  - Just add model folders to {MODELS_DIR}/")
        print(f"  - Supports both folder and legacy structures")
        print(f"  - CircuitPython 3.4.0+ compatible")

def main():
    """Main function with CircuitPython-friendly interface"""
    try:
        print("=== CircuitPython RP2040 Transformer Inference ===")
        print(f"RP2040: {microcontroller.cpu.frequency//1000000}MHz")
        
        # Test CircuitPython compatibility
        print("\n=== Testing CircuitPython Compatibility ===")
        try:
            # Test basic file operations
            test_dir = "test_scan"
            try:
                os.listdir(test_dir)
                print("✅ Directory listing works")
            except OSError:
                print("✅ Directory listing error handling works")
            
            # Test file operations
            try:
                with open("test_file.tmp", "w") as f:
                    f.write("test")
                with open("test_file.tmp", "r") as f:
                    content = f.read()
                if content == "test":
                    print("✅ File read/write works")
                # Clean up
                try:
                    os.remove("test_file.tmp")
                except:
                    pass
            except Exception as e:
                print(f"⚠️  File operations test: {e}")
            
            print("CircuitPython compatibility test complete")
            
        except Exception as e:
            print(f"⚠️  Compatibility test failed: {e}")
        
        # Create inference engine
        inference = ScalableInference()
        
        # Show configuration
        inference.show_configuration()
        
        # List available models
        inference.list_available_models()
        
        if not inference.detector.available_models:
            print("❌ No models found! Please add model folders to the models/ directory.")
            return
        
        # Simple CircuitPython interface
        print(f"\n=== CircuitPython Interface ===")
        print("Options:")
        print("  1. Test all models (benchmark)")
        print("  2. Test single model")
        print("  3. Interactive demo")
        print("  4. Exit")
        
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                # Benchmark all models
                results = inference.benchmark_all_models()
                
                # Find largest working model
                working_models = [name for name, result in results.items() if result['success']]
                
                if working_models:
                    print(f"\n✅ Working models: {', '.join(working_models)}")
                    
                    # Demo with largest working model
                    largest_model = working_models[-1]
                    print(f"\n=== DEMO WITH {largest_model.upper()} MODEL ===")
                    
                    if inference.load_model(largest_model):
                        inference.interactive_demo()
                else:
                    print("❌ No models could be loaded!")
            
            elif choice == "2":
                # Test single model
                print(f"\nAvailable models: {[m['name'] for m in inference.detector.available_models]}")
                model_name = input("Enter model name: ").strip()
                
                if model_name:
                    inference.detector.test_single_model(model_name)
                else:
                    print("No model name provided")
            
            elif choice == "3":
                # Interactive demo
                print(f"\nAvailable models: {[m['name'] for m in inference.detector.available_models]}")
                model_name = input("Enter model name for demo: ").strip()
                
                if model_name and inference.load_model(model_name):
                    inference.interactive_demo()
                else:
                    print("Invalid model name or failed to load")
            
            elif choice == "4":
                print("Exiting...")
                return
            
            else:
                print("Invalid choice, running full benchmark...")
                # Fall back to full benchmark
                results = inference.benchmark_all_models()
        
        except EOFError:
            # CircuitPython REPL - run full benchmark automatically
            print("\nRunning full benchmark (CircuitPython REPL mode)...")
            results = inference.benchmark_all_models()
            
            # Find largest working model
            working_models = [name for name, result in results.items() if result['success']]
            
            if working_models:
                print(f"\n✅ Working models: {', '.join(working_models)}")
                
                # Demo with largest working model
                largest_model = working_models[-1]
                print(f"\n=== DEMO WITH {largest_model.upper()} MODEL ===")
                
                if inference.load_model(largest_model):
                    inference.interactive_demo()
            else:
                print("❌ No models could be loaded!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exception(type(e), e, e.__traceback__)
    
    finally:
        mem.report_peak()
        mem.check("Final memory")
        print("\n=== CircuitPython Inference Complete ===")

if __name__ == "__main__":
    main()
