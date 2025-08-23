"""
RP2040 CircuitPython Inference Script
Loads pre-trained tiny transformer and runs inference only

Files needed on RP2040:
- tiny_model_trained.bin (from Windows training)
- vocab.bin (from Windows training)
- this script (main.py or code.py)

Memory usage: ~50-80KB
Speed: ~5-10 tokens/second
"""

import gc
import struct
import math
import time
import microcontroller

# Model configuration - must match training script exactly
VOCAB_SIZE = 64
DIM = 8  
HIDDEN_DIM = 16
N_LAYERS = 1
N_HEADS = 2
MAX_SEQ_LEN = 32
MAX_GEN_TOKENS = 15

class MemoryManager:
    """Memory management for RP2040"""
    def __init__(self):
        self.initial_free = gc.mem_free()
        print(f"Initial memory: {self.initial_free} bytes")
    
    def check(self, label=""):
        free = gc.mem_free()
        used = self.initial_free - free
        print(f"{label}: {free} free, {used} used")
        return free
    
    def cleanup(self):
        before = gc.mem_free()
        gc.collect()
        after = gc.mem_free()
        if after > before:
            print(f"GC: freed {after - before} bytes")

mem = MemoryManager()

class TinyTokenizer:
    """Tokenizer for RP2040 - loads from vocab.bin"""
    
    def __init__(self, vocab_file="vocab.bin"):
        self.vocab_size = VOCAB_SIZE
        self.vocab = {}
        self.reverse_vocab = {}
        
        if not self._load_vocab(vocab_file):
            print("Failed to load vocab, using fallback")
            self._create_fallback_vocab()
    
    def _load_vocab(self, vocab_file):
        """Load vocabulary from binary file"""
        try:
            with open(vocab_file, 'rb') as f:
                # Read vocab size
                size_bytes = f.read(2)
                if len(size_bytes) != 2:
                    return False
                    
                vocab_size = struct.unpack('H', size_bytes)[0]
                print(f"Loading vocab size: {vocab_size}")
                
                # Read each token
                for i in range(min(vocab_size, self.vocab_size)):
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
                            except:
                                pass
                
                print(f"Loaded {len(self.vocab)} tokens")
                return True
                
        except Exception as e:
            print(f"Error loading vocab: {e}")
            return False
    
    def _create_fallback_vocab(self):
        """Create fallback vocabulary if file load fails"""
        # Basic vocabulary matching training script
        self.vocab[0] = "<pad>"
        self.vocab[1] = "<s>"
        self.vocab[2] = "</s>" 
        self.vocab[3] = "<unk>"
        self.vocab[4] = " "
        
        # Common letters
        chars = "etaoinshrdlcumwfgypbvkjxqz.,!?'-"
        for i, char in enumerate(chars):
            if i + 5 < self.vocab_size:
                self.vocab[i + 5] = char
                self.reverse_vocab[char] = i + 5
        
        print("Created fallback vocabulary")
    
    def encode(self, text):
        """Encode text to token IDs"""
        if not text:
            return [1]
        
        tokens = [1]  # Start token
        text = text.lower()
        i = 0
        
        while i < len(text) and len(tokens) < MAX_SEQ_LEN:
            found = False
            # Try longer matches first (greedy)
            for length in range(min(4, len(text) - i), 0, -1):
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
            if token == 1:  # Start token
                continue
            elif token == 2:  # End token
                break
            elif token in self.vocab:
                text += self.vocab[token]
        return text

class TinyTransformerInference:
    """Inference-only tiny transformer for RP2040"""
    
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.dim = DIM
        self.hidden_dim = HIDDEN_DIM
        self.n_layers = N_LAYERS
        self.n_heads = N_HEADS
        
        # Pre-allocate arrays to avoid fragmentation
        self.x = [0.0] * (MAX_SEQ_LEN * DIM)  # Working space
        self.logits = [0.0] * VOCAB_SIZE
        
        # Model weights (will be loaded from file)
        self.token_embedding = None
        self.ln1_weight = None
        self.ln2_weight = None
        self.wq = None
        self.wk = None
        self.wv = None
        self.wo = None
        self.w1 = None
        self.w2 = None
    
    def load_weights(self, model_file="tiny_model_trained.bin"):
        """Load pre-trained weights from file"""
        print(f"Loading model from {model_file}")
        mem.check("Before model load")
        
        try:
            with open(model_file, 'rb') as f:
                # Read header
                header = f.read(32)
                if len(header) < 32:
                    raise ValueError("Invalid model file")
                
                vocab_size, dim, hidden_dim, n_layers, n_heads = struct.unpack("5I", header[:20])
                print(f"Model: vocab={vocab_size}, dim={dim}, layers={n_layers}")
                
                # Verify dimensions match
                if vocab_size != self.vocab_size or dim != self.dim:
                    raise ValueError("Model dimensions don't match")
                
                # Load token embeddings
                self.token_embedding = []
                embed_size = vocab_size * dim
                embed_bytes = f.read(embed_size * 4)  # 4 bytes per float32
                
                if len(embed_bytes) != embed_size * 4:
                    raise ValueError("Incomplete embedding data")
                
                # Convert bytes to floats
                embed_floats = struct.unpack(f'{embed_size}f', embed_bytes)
                
                # Reshape into 2D list
                for i in range(vocab_size):
                    row = []
                    for j in range(dim):
                        row.append(embed_floats[i * dim + j])
                    self.token_embedding.append(row)
                
                mem.cleanup()
                
                # Load layer norm weights
                ln1_bytes = f.read(dim * 4)
                ln2_bytes = f.read(dim * 4)
                self.ln1_weight = list(struct.unpack(f'{dim}f', ln1_bytes))
                self.ln2_weight = list(struct.unpack(f'{dim}f', ln2_bytes))
                
                # Load attention weights
                attn_size = dim * dim
                wq_bytes = f.read(attn_size * 4)
                wk_bytes = f.read(attn_size * 4)
                wv_bytes = f.read(attn_size * 4)
                wo_bytes = f.read(attn_size * 4)
                
                self.wq = self._bytes_to_matrix(wq_bytes, dim, dim)
                self.wk = self._bytes_to_matrix(wk_bytes, dim, dim)
                self.wv = self._bytes_to_matrix(wv_bytes, dim, dim)
                self.wo = self._bytes_to_matrix(wo_bytes, dim, dim)
                
                mem.cleanup()
                
                # Load FFN weights
                w1_size = dim * hidden_dim
                w2_size = hidden_dim * dim
                
                w1_bytes = f.read(w1_size * 4)
                w2_bytes = f.read(w2_size * 4)
                
                self.w1 = self._bytes_to_matrix(w1_bytes, dim, hidden_dim)
                self.w2 = self._bytes_to_matrix(w2_bytes, hidden_dim, dim)
                
                print("Model loaded successfully")
                mem.check("After model load")
                return True
                
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _bytes_to_matrix(self, data_bytes, rows, cols):
        """Convert bytes to 2D matrix"""
        floats = struct.unpack(f'{len(data_bytes)//4}f', data_bytes)
        matrix = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(floats[i * cols + j])
            matrix.append(row)
        return matrix
    
    def _layer_norm(self, x, weight):
        """Layer normalization"""
        # Calculate mean
        mean = sum(x) / len(x)
        
        # Calculate variance
        variance = sum((val - mean) ** 2 for val in x) / len(x)
        std = math.sqrt(variance + 1e-6)
        
        # Normalize and scale
        return [(val - mean) / std * weight[i] for i, val in enumerate(x)]
    
    def _matmul(self, a, b_matrix):
        """Matrix multiplication: vector @ matrix"""
        result = []
        for j in range(len(b_matrix[0])):  # columns of b
            val = 0.0
            for i in range(len(a)):  # rows of a / rows of b
                val += a[i] * b_matrix[i][j]
            result.append(val)
        return result
    
    def _softmax(self, x):
        """Softmax activation"""
        max_val = max(x)
        exp_vals = [math.exp(val - max_val) for val in x]
        sum_exp = sum(exp_vals)
        if sum_exp == 0:
            return [1.0 / len(x)] * len(x)
        return [val / sum_exp for val in exp_vals]
    
    def _attention(self, x_seq):
        """Simplified self-attention"""
        seq_len = len(x_seq)
        
        # Compute Q, K, V
        Q = [self._matmul(x, self.wq) for x in x_seq]
        K = [self._matmul(x, self.wk) for x in x_seq]
        V = [self._matmul(x, self.wv) for x in x_seq]
        
        output_seq = []
        
        for i in range(seq_len):
            # Attention scores for position i
            scores = []
            for j in range(i + 1):  # Causal mask
                score = sum(Q[i][k] * K[j][k] for k in range(self.dim))
                score = score / math.sqrt(self.dim)
                scores.append(score)
            
            # Pad with very negative values for future positions
            for j in range(i + 1, seq_len):
                scores.append(-1e9)
            
            # Apply softmax
            attn_weights = self._softmax(scores)
            
            # Weighted sum of values
            output = [0.0] * self.dim
            for j in range(i + 1):  # Only past and current positions
                weight = attn_weights[j]
                for k in range(self.dim):
                    output[k] += weight * V[j][k]
            
            output_seq.append(output)
        
        # Output projection
        projected_seq = []
        for output in output_seq:
            projected = self._matmul(output, self.wo)
            projected_seq.append(projected)
        
        return projected_seq
    
    def _feed_forward(self, x):
        """Feed-forward network"""
        # First linear layer
        hidden = self._matmul(x, self.w1)
        
        # ReLU activation
        hidden = [max(0.0, val) for val in hidden]
        
        # Second linear layer
        output = self._matmul(hidden, self.w2)
        
        return output
    
    def forward(self, tokens):
        """Forward pass through the model"""
        seq_len = len(tokens)
        if seq_len == 0:
            return self.logits
        
        # Token embeddings
        x_seq = []
        for token in tokens:
            if 0 <= token < self.vocab_size:
                embedding = self.token_embedding[token][:]  # Copy
            else:
                embedding = self.token_embedding[3][:]  # <unk>
            x_seq.append(embedding)
        
        # Transformer layer
        # Pre-attention layer norm
        normed_seq = []
        for x in x_seq:
            normed = self._layer_norm(x, self.ln1_weight)
            normed_seq.append(normed)
        
        # Self-attention
        attn_output = self._attention(normed_seq)
        
        # Residual connection
        for i in range(seq_len):
            for j in range(self.dim):
                x_seq[i][j] += attn_output[i][j]
        
        # Pre-FFN layer norm
        normed_seq = []
        for x in x_seq:
            normed = self._layer_norm(x, self.ln2_weight)
            normed_seq.append(normed)
        
        # Feed-forward
        ffn_output = []
        for normed in normed_seq:
            ff_out = self._feed_forward(normed)
            ffn_output.append(ff_out)
        
        # Residual connection
        for i in range(seq_len):
            for j in range(self.dim):
                x_seq[i][j] += ffn_output[i][j]
        
        # Output projection (using transposed embedding weights)
        last_hidden = x_seq[-1]  # Get last position
        
        # Compute logits
        for i in range(self.vocab_size):
            logit = sum(last_hidden[j] * self.token_embedding[i][j] 
                       for j in range(self.dim))
            self.logits[i] = logit
        
        return self.logits
    
    def sample(self, logits, temperature=1.0):
        """Sample next token from logits"""
        if temperature <= 0.0:
            # Greedy sampling
            max_idx = 0
            max_val = logits[0]
            for i in range(1, len(logits)):
                if logits[i] > max_val:
                    max_val = logits[i]
                    max_idx = i
            return max_idx
        
        # Apply temperature
        scaled_logits = [l / temperature for l in logits]
        
        # Softmax
        probs = self._softmax(scaled_logits)
        
        # Sample using built-in random (time-based seed)
        import time
        rand_val = (time.monotonic_ns() % 10000) / 10000.0
        
        cumsum = 0.0
        for i, prob in enumerate(probs):
            cumsum += prob
            if rand_val < cumsum:
                return i
        
        return len(probs) - 1

class TinyLlamaRP2040:
    """Main inference class for RP2040"""
    
    def __init__(self):
        print("=== RP2040 Tiny Transformer Inference ===")
        mem.check("Initialization start")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = TinyTokenizer()
        mem.cleanup()
        
        # Load model
        print("Loading model...")
        self.model = TinyTransformerInference()
        
        if not self.model.load_weights():
            raise RuntimeError("Failed to load model weights")
        
        mem.cleanup()
        print("Model ready for inference!")
        mem.check("Initialization complete")
    
    def generate(self, prompt="hello", max_tokens=MAX_GEN_TOKENS, temperature=0.8):
        """Generate text from prompt"""
        print(f"\nGenerating: '{prompt}'")
        
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        print(f"Input tokens ({len(tokens)}): {tokens}")
        
        if len(tokens) == 0:
            tokens = [1]  # Just start token
        
        start_time = time.monotonic()
        generated_tokens = []
        
        # Generate tokens one by one
        for step in range(max_tokens):
            # Current sequence
            current_tokens = tokens + generated_tokens
            
            # Truncate if too long
            if len(current_tokens) > MAX_SEQ_LEN - 1:
                current_tokens = current_tokens[-(MAX_SEQ_LEN - 1):]
            
            # Forward pass
            logits = self.model.forward(current_tokens)
            
            # Sample next token
            next_token = self.model.sample(logits, temperature)
            
            # Check for end token
            if next_token == 2:  # </s>
                print("Generated end token")
                break
            
            generated_tokens.append(next_token)
            
            # Show progress every few steps
            if step > 0 and step % 5 == 0:
                partial_text = self.tokenizer.decode(current_tokens + [next_token])
                print(f"  Step {step}: {partial_text}")
        
        # Decode full result
        full_tokens = tokens + generated_tokens
        result = self.tokenizer.decode(full_tokens)
        
        # Timing
        elapsed = time.monotonic() - start_time
        tokens_per_sec = len(generated_tokens) / max(elapsed, 0.001)
        
        print(f"\nResult: '{result}'")
        print(f"Generated {len(generated_tokens)} tokens in {elapsed:.2f}s")
        print(f"Speed: {tokens_per_sec:.2f} tokens/sec")
        
        return result
    
    def interactive_mode(self):
        """Simple interactive mode"""
        print("\n=== INTERACTIVE MODE ===")
        print("Enter prompts (or 'quit' to exit)")
        
        while True:
            try:
                # Simple prompt input (RP2040 doesn't have input())
                print("\nEnter prompt:")
                # For demo, we'll cycle through test prompts
                test_prompts = ["hello", "the cat", "good morning", "once upon", "how are you"]
                
                for prompt in test_prompts:
                    print(f"\n> {prompt}")
                    result = self.generate(prompt, max_tokens=8, temperature=0.9)
                    
                    # Brief pause between generations
                    time.sleep(1)
                    mem.cleanup()
                
                break  # Exit after demo
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                break

def benchmark():
    """Benchmark the model performance"""
    print("\n=== BENCHMARK ===")
    
    try:
        llama = TinyLlamaRP2040()
        
        # Test forward pass timing
        test_tokens = [1, 12, 9, 16, 16, 19]  # "hello"
        
        print("Warming up...")
        for _ in range(3):
            llama.model.forward(test_tokens)
        
        print("Benchmarking forward pass...")
        start_time = time.monotonic()
        
        for i in range(10):
            logits = llama.model.forward(test_tokens)
        
        elapsed = time.monotonic() - start_time
        avg_time = elapsed / 10
        
        print(f"Average forward pass: {avg_time*1000:.1f}ms")
        print(f"Estimated max tokens/sec: {1.0/avg_time:.1f}")
        
        mem.check("Benchmark complete")
        
    except Exception as e:
        print(f"Benchmark error: {e}")

def main():
    """Main function"""
    print("Starting RP2040 Tiny Transformer")
    print(f"Free memory: {gc.mem_free()} bytes")
    print(f"CPU frequency: {microcontroller.cpu.frequency // 1000000} MHz")
    
    try:
        # Create inference engine
        llama = TinyLlamaRP2040()
        
        # Run interactive demo
        llama.interactive_mode()
        
        print("\nDemo complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        # Print traceback for debugging
        import traceback
        traceback.print_exception(type(e), e, e.__traceback__)
        
    finally:
        mem.check("Final memory")

# Auto-run when loaded
if __name__ == "__main__":
    main()

# Uncomment to run benchmark instead
# benchmark()

# Simple test function for debugging
def test_basic():
    """Basic functionality test"""
    print("=== BASIC TEST ===")
    
    try:
        # Test tokenizer only
        tokenizer = TinyTokenizer()
        test_text = "hello world"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"'{test_text}' -> {tokens} -> '{decoded}'")
        
        # Test model loading
        model = TinyTransformerInference()
        if model.load_weights():
            print("Model loaded successfully")
            
            # Test forward pass
            logits = model.forward([1, 12, 9, 16])  # "hello"
            print(f"Logits: {logits[:5]}...")  # Show first 5 values
            
            # Test sampling
            next_token = model.sample(logits, temperature=0.8)
            print(f"Next token: {next_token}")
        
    except Exception as e:
        print(f"Test error: {e}")

# Uncomment for basic testing
# test_basic()