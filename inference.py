"""
Scalable RP2040 Inference Script
Automatically detects and loads any model size

Supports model sizes from 1K to 300K+ parameters
Will test memory limits and inference speed

Usage:
1. Copy model files from Windows training:
   - model_[size].bin (e.g., model_tiny.bin, model_small.bin)
   - vocab_[size].bin  
   - config_[size].json (optional)
2. This script will auto-detect available models
3. You can test multiple models to find RP2040 limits
"""

import gc
import struct
import math
import time
import microcontroller
import json

class MemoryManager:
    """Enhanced memory management with detailed reporting"""
    
    def __init__(self):
        self.initial_free = gc.mem_free()
        self.peak_usage = 0
        print(f"=== RP2040 Memory Info ===")
        print(f"Initial free: {self.initial_free:,} bytes")
        print(f"Total RAM: ~256KB")
    
    def check(self, label="", critical_threshold=20000):
        """Check memory with warnings"""
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
        """Aggressive cleanup"""
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
        
        # Load model
        if not self._load_model(model_file):
            raise RuntimeError(f"Failed to load model from {model_file}")
    
    def _load_model(self, model_file):
        """Load model from binary file"""
        print(f"Loading model: {model_file}")
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
                
                # Load token embeddings
                embed_size = self.vocab_size * self.dim
                embed_bytes = f.read(embed_size * 4)
                
                if len(embed_bytes) != embed_size * 4:
                    print(f"Expected {embed_size * 4} embedding bytes, got {len(embed_bytes)}")
                    return False
                
                # Convert to nested lists
                embed_floats = struct.unpack(f'{embed_size}f', embed_bytes)
                self.token_embedding = []
                for i in range(self.vocab_size):
                    row = []
                    for j in range(self.dim):
                        row.append(embed_floats[i * self.dim + j])
                    self.token_embedding.append(row)
                
                mem.cleanup()
                mem.check("After embeddings")
                
                # Load layers
                self.layers = []
                for layer_id in range(self.n_layers):
                    print(f"  Loading layer {layer_id + 1}/{self.n_layers}")
                    
                    layer = {}
                    
                    # Layer norms
                    ln1_bytes = f.read(self.dim * 4)
                    ln2_bytes = f.read(self.dim * 4)
                    layer['ln1_weight'] = list(struct.unpack(f'{self.dim}f', ln1_bytes))
                    layer['ln2_weight'] = list(struct.unpack(f'{self.dim}f', ln2_bytes))
                    
                    # Attention weights
                    attn_size = self.dim * self.dim
                    for weight_name in ['wq', 'wk', 'wv', 'wo']:
                        weight_bytes = f.read(attn_size * 4)
                        weight_floats = struct.unpack(f'{attn_size}f', weight_bytes)
                        # Convert to 2D matrix
                        matrix = []
                        for i in range(self.dim):
                            row = []
                            for j in range(self.dim):
                                row.append(weight_floats[i * self.dim + j])
                            matrix.append(row)
                        layer[weight_name] = matrix
                    
                    # FFN weights
                    # W1: dim x hidden_dim
                    w1_size = self.dim * self.hidden_dim
                    w1_bytes = f.read(w1_size * 4)
                    w1_floats = struct.unpack(f'{w1_size}f', w1_bytes)
                    w1_matrix = []
                    for i in range(self.dim):
                        row = []
                        for j in range(self.hidden_dim):
                            row.append(w1_floats[i * self.hidden_dim + j])
                        w1_matrix.append(row)
                    layer['w1'] = w1_matrix
                    
                    # W2: hidden_dim x dim  
                    w2_size = self.hidden_dim * self.dim
                    w2_bytes = f.read(w2_size * 4)
                    w2_floats = struct.unpack(f'{w2_size}f', w2_bytes)
                    w2_matrix = []
                    for i in range(self.hidden_dim):
                        row = []
                        for j in range(self.dim):
                            row.append(w2_floats[i * self.dim + j])
                        w2_matrix.append(row)
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
        """Optimized matrix multiplication"""
        if len(b_matrix) == 0:
            return [0.0] * len(a)
        
        result = []
        for j in range(len(b_matrix[0])):
            val = 0.0
            for i in range(len(a)):
                val += a[i] * b_matrix[i][j]
            result.append(val)
        return result
    
    def _layer_norm(self, x, weight, eps=1e-6):
        """Layer normalization"""
        mean = sum(x) / len(x)
        variance = sum((val - mean) ** 2 for val in x) / len(x)
        std = math.sqrt(variance + eps)
        return [(val - mean) / std * weight[i] for i, val in enumerate(x)]
    
    def _softmax(self, x):
        """Softmax with numerical stability"""
        max_val = max(x)
        exp_vals = [math.exp(val - max_val) for val in x]
        sum_exp = sum(exp_vals)
        if sum_exp == 0:
            return [1.0 / len(x)] * len(x)
        return [val / sum_exp for val in exp_vals]
    
    def _attention(self, x_seq, layer):
        """Multi-head self-attention (simplified)"""
        seq_len = len(x_seq)
        
        # Q, K, V projections
        Q = [self._matmul(x, layer['wq']) for x in x_seq]
        K = [self._matmul(x, layer['wk']) for x in x_seq] 
        V = [self._matmul(x, layer['wv']) for x in x_seq]
        
        output_seq = []
        
        for i in range(seq_len):
            # Attention scores for position i
            scores = []
            for j in range(i + 1):  # Causal mask
                score = sum(Q[i][k] * K[j][k] for k in range(self.dim))
                score = score / math.sqrt(self.dim)
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
        """Feed-forward with SiLU activation"""
        # First projection
        hidden = self._matmul(x, layer['w1'])
        
        # SiLU activation: x * sigmoid(x)
        for i in range(len(hidden)):
            sigmoid_val = 1.0 / (1.0 + math.exp(-max(-10, min(10, hidden[i]))))
            hidden[i] = hidden[i] * sigmoid_val
        
        # Second projection
        output = self._matmul(hidden, layer['w2'])
        
        return output
    
    def forward(self, tokens):
        """Forward pass through the model"""
        seq_len = len(tokens)
        if seq_len == 0:
            return [0.0] * self.vocab_size
        
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
            
            # Self-attention
            attn_output = self._attention(normed_seq, layer)
            
            # Residual connection
            for i in range(seq_len):
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
            for i in range(seq_len):
                for j in range(self.dim):
                    x_seq[i][j] += ffn_output[i][j]
        
        # Final layer norm
        last_hidden = self._layer_norm(x_seq[-1], self.final_ln_weight)
        
        # Output projection
        logits = []
        for i in range(self.vocab_size):
            logit = sum(last_hidden[j] * self.token_embedding[i][j] for j in range(self.dim))
            logits.append(logit)
        
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
        model_files = []
        
        # Try different naming patterns
        size_names = ['tiny', 'small', 'medium', 'large', 'xlarge']
        
        for size in size_names:
            model_file = f"model_{size}.bin"
            vocab_file = f"vocab_{size}.bin"
            
            try:
                # Check if both files exist
                with open(model_file, 'rb') as f:
                    f.read(4)  # Try to read header
                with open(vocab_file, 'rb') as f:
                    f.read(2)  # Try to read vocab size
                
                self.available_models.append({
                    'name': size,
                    'model_file': model_file,
                    'vocab_file': vocab_file
                })
                print(f"✅ Found {size} model: {model_file}")
                
            except:
                continue
        
        if not self.available_models:
            print("❌ No model files found!")
            print("Expected files: model_[size].bin and vocab_[size].bin")
    
    def test_model_loading(self, model_info):
        """Test if a model can be loaded"""
        print(f"\n=== Testing {model_info['name'].upper()} Model ===")
        
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
        print(f"{'Model':8s} {'Status':8s} {'Memory':8s} {'Speed':12s}")
        print("-" * 45)
        
        for name, result in results.items():
            if result['success']:
                memory_kb = result['memory_used'] // 1024
                speed = 1.0 / result['inference_time']
                print(f"{name:8s} ✅      {memory_kb:5d}KB {speed:8.1f} tok/s")
            else:
                print(f"{name:8s} ❌      Failed    -")
        
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
            return False
    
    def generate(self, prompt="hello", max_tokens=10, temperature=0.8):
        """Generate text with current model"""
        if not self.current_model:
            print("No model loaded!")
            return ""
        
        print(f"\n=== Generating with {self.current_model_name.upper()} ===")
        print(f"Prompt: '{prompt}'")
        
        # Encode prompt
        tokens = self.current_tokenizer.encode(prompt)
        print(f"Input tokens ({len(tokens)}): {tokens}")
        
        start_time = time.monotonic()
        generated_tokens = []
        
        for step in range(max_tokens):
            # Current sequence
            current_tokens = tokens + generated_tokens
            
            # Truncate if too long
            if len(current_tokens) > self.current_model.max_seq_len - 1:
                current_tokens = current_tokens[-(self.current_model.max_seq_len - 1):]
            
            # Forward pass
            logits = self.current_model.forward(current_tokens)
            
            # Sample
            next_token = self.current_model.sample(logits, temperature)
            
            if next_token == 2:  # EOS
                break
            
            generated_tokens.append(next_token)
            
            # Show progress
            if step > 0 and step % 3 == 0:
                partial = self.current_tokenizer.decode(current_tokens + [next_token])
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

def main():
    """Main function"""
    try:
        # Create inference engine
        inference = ScalableInference()
        
        # Benchmark all models
        results = inference.benchmark_all_models()
        
        # Find largest working model
        working_models = [name for name, result in results.items() if result['success']]
        
        if working_models:
            print(f"\n✅ Working models: {', '.join(working_models)}")
            
            # Demo with largest working model
            largest_model = working_models[-1]  # Assume ordered by size
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

if __name__ == "__main__":
    main()
