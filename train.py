"""
Scalable Transformer Training Script
Test different model sizes to find RP2040 limits

Model Size Presets:
- Tiny (1K):     vocab=64,  dim=8,   layers=1, heads=2  
- Small (5K):    vocab=128, dim=16,  layers=2, heads=4
- Medium (20K):  vocab=256, dim=32,  layers=3, heads=8
- Large (80K):   vocab=512, dim=64,  layers=4, heads=16
- XLarge (300K): vocab=1024,dim=128, layers=6, heads=32

We'll test each size on RP2040 until we hit memory/speed limits
"""

import numpy as np
import struct
import random
import time
import os
from collections import Counter
import json

# Model size presets - we'll test each one
MODEL_CONFIGS = {
    'tiny': {
        'vocab_size': 64,
        'dim': 8,
        'hidden_dim': 32,  # 4x dim
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 32,
        'description': '~1K parameters'
    },
    'small': {
        'vocab_size': 128,
        'dim': 16,
        'hidden_dim': 64,
        'n_layers': 2,
        'n_heads': 4,
        'max_seq_len': 48,
        'description': '~5K parameters'
    },
    'medium': {
        'vocab_size': 256,
        'dim': 32,
        'hidden_dim': 128,
        'n_layers': 3,
        'n_heads': 8,
        'max_seq_len': 64,
        'description': '~20K parameters'
    },
    'large': {
        'vocab_size': 512,
        'dim': 64,
        'hidden_dim': 256,
        'n_layers': 4,
        'n_heads': 16,
        'max_seq_len': 96,
        'description': '~80K parameters'
    },
    'xlarge': {
        'vocab_size': 1024,
        'dim': 128,
        'hidden_dim': 512,
        'n_layers': 6,
        'n_heads': 32,
        'max_seq_len': 128,
        'description': '~300K parameters'
    }
}

class ScalableTokenizer:
    """Tokenizer that adapts to different vocab sizes"""
    
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Build vocabulary scaled to vocab_size"""
        # Special tokens (always first 4)
        self.vocab[0] = "<pad>"
        self.vocab[1] = "<s>"
        self.vocab[2] = "</s>"
        self.vocab[3] = "<unk>"
        self.vocab[4] = " "  # Space is crucial
        
        token_id = 5
        
        # Common letters by frequency
        common_chars = "etaoinshrdlcumwfgypbvkjxqz"
        for char in common_chars:
            if token_id < self.vocab_size:
                self.vocab[token_id] = char
                self.reverse_vocab[char] = token_id
                token_id += 1
        
        # Punctuation and symbols
        punct = ".,!?'-;:()[]{}\"@#$%&*+=/<>\n\t"
        for char in punct:
            if token_id < self.vocab_size:
                self.vocab[token_id] = char
                self.reverse_vocab[char] = token_id
                token_id += 1
        
        # Common bigrams (for larger vocabularies)
        if self.vocab_size > 64:
            bigrams = [" th", " he", " in", " er", " an", " re", " ed", " nd", " ha", " to", 
                      " ou", " it", " is", " en", " as", " at", " es", " on", " hi", " st",
                      "ing", "ion", "ter", "ent", "ity", "ous", "ate", "ive"]
            
            for bigram in bigrams:
                if token_id < self.vocab_size:
                    self.vocab[token_id] = bigram
                    self.reverse_vocab[bigram] = token_id
                    token_id += 1
        
        # Common trigrams (for even larger vocabularies)
        if self.vocab_size > 200:
            trigrams = [" the", " and", " for", " are", " but", " not", " you", " all", 
                       " can", " had", " her", " was", " one", " our", " out", " day",
                       "ing ", "ion ", "tion", "ness", "ment", "able", "ible"]
            
            for trigram in trigrams:
                if token_id < self.vocab_size:
                    self.vocab[token_id] = trigram
                    self.reverse_vocab[trigram] = token_id
                    token_id += 1
        
        # Common words (for largest vocabularies)
        if self.vocab_size > 500:
            words = ["the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
                    "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
                    "how", "man", "new", "now", "old", "see", "two", "way", "who", "boy",
                    "did", "its", "let", "put", "say", "she", "too", "use", "about", "after",
                    "again", "before", "came", "come", "could", "each", "first", "from",
                    "good", "great", "here", "just", "know", "last", "like", "long",
                    "look", "made", "make", "many", "much", "never", "only", "other",
                    "right", "said", "same", "should", "since", "still", "such", "take",
                    "than", "them", "time", "very", "want", "water", "well", "were",
                    "what", "when", "where", "which", "while", "with", "work", "would"]
            
            for word in words:
                if token_id < self.vocab_size:
                    # Add both with and without leading space
                    if token_id < self.vocab_size - 1:
                        self.vocab[token_id] = " " + word
                        self.reverse_vocab[" " + word] = token_id
                        token_id += 1
                    
                    if token_id < self.vocab_size:
                        self.vocab[token_id] = word
                        self.reverse_vocab[word] = token_id
                        token_id += 1
        
        # Fill remaining slots with numbers if needed
        for i in range(10):
            if token_id < self.vocab_size:
                self.vocab[token_id] = str(i)
                self.reverse_vocab[str(i)] = token_id
                token_id += 1
        
        print(f"Built vocabulary: {len(self.vocab)}/{self.vocab_size} tokens")
        print("Sample vocab:", {k: v for k, v in list(self.vocab.items())[:20]})
    
    def encode(self, text):
        """Encode with greedy longest-match tokenization"""
        if not text:
            return [1]
        
        tokens = [1]  # BOS
        text = text.lower()
        i = 0
        
        while i < len(text):
            found = False
            # Try longer matches first (up to 8 chars)
            for length in range(min(8, len(text) - i), 0, -1):
                substr = text[i:i+length]
                if substr in self.reverse_vocab:
                    tokens.append(self.reverse_vocab[substr])
                    i += length
                    found = True
                    break
            
            if not found:
                # Single char or unknown
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
            if token == 1:  # BOS
                continue
            elif token == 2:  # EOS
                break
            elif 0 <= token < len(self.vocab) and token in self.vocab:
                text += self.vocab[token]
        return text
    
    def save_vocab(self, filename):
        """Save vocabulary for RP2040"""
        with open(filename, 'wb') as f:
            f.write(struct.pack('H', self.vocab_size))
            for i in range(self.vocab_size):
                if i in self.vocab:
                    token_str = self.vocab[i].encode('utf-8')
                    token_len = min(len(token_str), 255)  # Max 255 bytes
                    f.write(struct.pack('B', token_len))
                    f.write(token_str[:token_len])
                else:
                    f.write(struct.pack('B', 0))
        print(f"Vocabulary saved to {filename}")

class ScalableTransformer:
    """Transformer that scales with config"""
    
    def __init__(self, config):
        self.config = config
        self.vocab_size = config['vocab_size']
        self.dim = config['dim']
        self.hidden_dim = config['hidden_dim']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.max_seq_len = config['max_seq_len']
        self.head_dim = self.dim // self.n_heads
        
        self._init_weights()
        self._count_parameters()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        print(f"Initializing {self.config['description']} model...")
        
        # Scale initialization based on model size
        embed_scale = 0.1
        attn_scale = 0.02 / np.sqrt(self.n_layers)
        ffn_scale = 0.02 / np.sqrt(self.n_layers)
        
        # Token embeddings
        self.token_embedding = np.random.normal(0, embed_scale, 
                                              (self.vocab_size, self.dim)).astype(np.float32)
        
        # Layer weights
        self.layers = []
        for layer_id in range(self.n_layers):
            layer = {
                # Layer norm
                'ln1_weight': np.ones(self.dim, dtype=np.float32),
                'ln2_weight': np.ones(self.dim, dtype=np.float32),
                
                # Attention
                'wq': np.random.normal(0, attn_scale, (self.dim, self.dim)).astype(np.float32),
                'wk': np.random.normal(0, attn_scale, (self.dim, self.dim)).astype(np.float32),
                'wv': np.random.normal(0, attn_scale, (self.dim, self.dim)).astype(np.float32),
                'wo': np.random.normal(0, attn_scale, (self.dim, self.dim)).astype(np.float32),
                
                # Feed-forward
                'w1': np.random.normal(0, ffn_scale, (self.dim, self.hidden_dim)).astype(np.float32),
                'w2': np.random.normal(0, ffn_scale, (self.hidden_dim, self.dim)).astype(np.float32),
            }
            self.layers.append(layer)
        
        # Final layer norm
        self.final_ln_weight = np.ones(self.dim, dtype=np.float32)
    
    def _count_parameters(self):
        """Count parameters"""
        embedding_params = self.vocab_size * self.dim
        layer_params = self.n_layers * (
            4 * self.dim * self.dim +  # attention: q,k,v,o
            self.dim * self.hidden_dim * 2  # ffn: w1, w2
        )
        total = embedding_params + layer_params
        
        print(f"Parameters: {total:,} ({total/1000:.1f}K)")
        print(f"  Embeddings: {embedding_params:,}")
        print(f"  Layers: {layer_params:,}")
        
        # Estimate memory usage
        memory_mb = total * 4 / (1024 * 1024)  # 4 bytes per float32
        memory_kb = total * 4 / 1024
        print(f"Estimated memory: {memory_kb:.1f}KB ({memory_mb:.2f}MB)")
        
        return total
    
    def layer_norm(self, x, weight, eps=1e-6):
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * weight
    
    def attention(self, x, layer):
        """Multi-head self-attention"""
        seq_len, d_model = x.shape
        
        # Linear projections
        Q = x @ layer['wq']
        K = x @ layer['wk']
        V = x @ layer['wv']
        
        # Multi-head attention (simplified - no head splitting for efficiency)
        scores = Q @ K.T / np.sqrt(d_model)
        
        # Causal mask
        mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
        scores = scores + mask
        
        # Softmax
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / (np.sum(attn_weights, axis=-1, keepdims=True) + 1e-8)
        
        # Apply attention
        out = attn_weights @ V
        
        # Output projection
        return out @ layer['wo']
    
    def feed_forward(self, x, layer):
        """Feed-forward network with SwiGLU-like activation"""
        hidden = x @ layer['w1']
        # SiLU activation: x * sigmoid(x)
        sigmoid_hidden = 1 / (1 + np.exp(-np.clip(hidden, -10, 10)))
        activated = hidden * sigmoid_hidden
        return activated @ layer['w2']
    
    def forward(self, tokens):
        """Forward pass"""
        # Embeddings
        x = self.token_embedding[tokens]
        
        # Transformer layers
        for layer in self.layers:
            # Pre-attention norm
            normed = self.layer_norm(x, layer['ln1_weight'])
            
            # Self-attention with residual
            x = x + self.attention(normed, layer)
            
            # Pre-FFN norm  
            normed = self.layer_norm(x, layer['ln2_weight'])
            
            # FFN with residual
            x = x + self.feed_forward(normed, layer)
        
        # Final norm
        x = self.layer_norm(x, self.final_ln_weight)
        
        # Output projection (last position only)
        logits = x[-1] @ self.token_embedding.T
        
        return logits

class ScalableTrainer:
    """Trainer for different model sizes"""
    
    def __init__(self, model_size='tiny'):
        self.config = MODEL_CONFIGS[model_size]
        self.model_size = model_size
        
        print(f"=== Training {model_size.upper()} model ===")
        print(f"Config: {self.config}")
        
        self.tokenizer = ScalableTokenizer(self.config['vocab_size'])
        self.model = ScalableTransformer(self.config)
        self.training_data = []
    
    def prepare_data(self):
        """Prepare training data"""
        # Expanded training data for larger models
        sentences = [
            "hello world how are you today",
            "the cat sat on the mat in the sun",
            "once upon a time there was a little cat",
            "good morning sunshine how are you doing today",
            "the quick brown fox jumps over the lazy dog",
            "i love you very much my dear friend",
            "hello there friend how are you doing today",
            "the cat likes to play in the garden",
            "good night sleep well my dear friend",
            "how are you today my good friend",
            "the sun is shining bright in the sky",
            "cats and dogs are very good friends",
            "once there was a happy little cat in the garden",
            "hello good morning how are you today",
            "the little cat sat by the warm fire",
            "i want to go to the store today",
            "the weather is very nice today sunshine",
            "can you help me with my homework please",
            "the children are playing in the playground",
            "my mother is cooking dinner in the kitchen",
            "the dog is running fast in the park",
            "we are going to visit our friends tomorrow",
            "the book is on the table by the window",
            "she is reading a story to her children",
            "the birds are singing beautiful songs outside",
            "winter is coming and the leaves are falling",
            "the computer is working very well today",
            "programming is fun when you understand it",
            "artificial intelligence is changing the world rapidly",
            "machine learning helps us solve difficult problems"
        ]
        
        # Create training examples
        self.training_data = []
        for sentence in sentences:
            tokens = self.tokenizer.encode(sentence)
            if len(tokens) > 1:
                for i in range(1, len(tokens)):
                    context = tokens[:i]
                    target = tokens[i]
                    if len(context) <= self.config['max_seq_len']:
                        self.training_data.append((context, target))
        
        print(f"Created {len(self.training_data)} training examples")
    
    def train(self, epochs=50, learning_rate=0.001):
        """Train the model"""
        print(f"Training for {epochs} epochs with lr={learning_rate}")
        
        losses = []
        best_loss = float('inf')
        
        # Adjust learning rate based on model size
        if self.config['dim'] > 32:
            learning_rate *= 0.5  # Lower LR for bigger models
        
        for epoch in range(epochs):
            epoch_loss = 0
            random.shuffle(self.training_data)
            
            for i, (context, target) in enumerate(self.training_data):
                try:
                    # Forward pass
                    logits = self.model.forward(np.array(context))
                    
                    # Compute loss
                    exp_logits = np.exp(logits - np.max(logits))
                    probs = exp_logits / (np.sum(exp_logits) + 1e-8)
                    loss = -np.log(probs[target] + 1e-8)
                    epoch_loss += loss
                    
                    # Simple gradient update on embeddings
                    error = 1.0 - probs[target]
                    
                    # Update target token (increase)
                    if target < self.model.vocab_size:
                        self.model.token_embedding[target] += learning_rate * error * 0.1
                    
                    # Update other tokens (decrease slightly)
                    for j in range(min(100, self.model.vocab_size)):  # Limit updates for speed
                        if j != target and probs[j] > 0.01:  # Only update probable tokens
                            self.model.token_embedding[j] -= learning_rate * probs[j] * 0.01
                
                except Exception as e:
                    print(f"Training error at epoch {epoch}, sample {i}: {e}")
                    continue
            
            avg_loss = epoch_loss / len(self.training_data)
            losses.append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            if epoch % 10 == 0 or epoch < 5:
                print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f} (best: {best_loss:.4f})")
                self.test_generation()
        
        print(f"Training complete! Final loss: {losses[-1]:.4f}")
        return losses
    
    def test_generation(self):
        """Test generation during training"""
        test_prompts = ["hello", "the cat", "good morning"]
        
        for prompt in test_prompts:
            try:
                tokens = self.tokenizer.encode(prompt)
                result_tokens = tokens[:]
                
                for _ in range(6):
                    if len(result_tokens) >= self.config['max_seq_len']:
                        break
                    
                    logits = self.model.forward(np.array(result_tokens))
                    
                    # Sample from top-5
                    top_indices = np.argsort(logits)[-5:]
                    top_logits = logits[top_indices]
                    probs = np.exp(top_logits - np.max(top_logits))
                    probs = probs / np.sum(probs)
                    
                    next_token = np.random.choice(top_indices, p=probs)
                    result_tokens.append(next_token)
                    
                    if next_token == 2:  # EOS
                        break
                
                result_text = self.tokenizer.decode(result_tokens)
                print(f"  '{prompt}' -> '{result_text}'")
                
            except Exception as e:
                print(f"  '{prompt}' -> [generation error: {e}]")
    
    def save_model(self):
        """Save model with size-specific filename"""
        model_filename = f"model_{self.model_size}.bin"
        vocab_filename = f"vocab_{self.model_size}.bin"
        
        print(f"Saving {self.model_size} model...")
        
        # Save model config first
        config_filename = f"config_{self.model_size}.json"
        with open(config_filename, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save model weights
        with open(model_filename, 'wb') as f:
            # Header with config
            header = struct.pack("8I", 
                               self.config['vocab_size'], 
                               self.config['dim'],
                               self.config['hidden_dim'], 
                               self.config['n_layers'],
                               self.config['n_heads'], 
                               self.config['max_seq_len'],
                               0, 0)  # Reserved
            f.write(header)
            
            # Token embeddings
            f.write(self.model.token_embedding.astype(np.float32).tobytes())
            
            # Layer weights
            for layer in self.model.layers:
                f.write(layer['ln1_weight'].astype(np.float32).tobytes())
                f.write(layer['ln2_weight'].astype(np.float32).tobytes())
                f.write(layer['wq'].astype(np.float32).tobytes())
                f.write(layer['wk'].astype(np.float32).tobytes())
                f.write(layer['wv'].astype(np.float32).tobytes())
                f.write(layer['wo'].astype(np.float32).tobytes())
                f.write(layer['w1'].astype(np.float32).tobytes())
                f.write(layer['w2'].astype(np.float32).tobytes())
            
            # Final layer norm
            f.write(self.model.final_ln_weight.astype(np.float32).tobytes())
        
        # Save vocabulary
        self.tokenizer.save_vocab(vocab_filename)
        
        print(f"Files created:")
        print(f"  {model_filename} - model weights")
        print(f"  {vocab_filename} - vocabulary") 
        print(f"  {config_filename} - configuration")

def test_all_sizes():
    """Test training all model sizes"""
    results = {}
    
    for size_name in ['tiny', 'small', 'medium', 'large']:
        print(f"\n{'='*50}")
        print(f"TESTING {size_name.upper()} MODEL")
        print(f"{'='*50}")
        
        try:
            start_time = time.time()
            
            # Create and train model
            trainer = ScalableTrainer(size_name)
            trainer.prepare_data()
            
            # Adjust epochs based on size
            epochs = max(20, 100 // (trainer.config['dim'] // 8))
            losses = trainer.train(epochs=epochs, learning_rate=0.005)
            
            training_time = time.time() - start_time
            
            # Save model
            trainer.save_model()
            
            # Record results
            param_count = trainer.model._count_parameters()
            results[size_name] = {
                'parameters': param_count,
                'training_time': training_time,
                'final_loss': losses[-1],
                'config': trainer.config
            }
            
            print(f"{size_name.upper()} COMPLETE: {param_count:,} params, {training_time:.1f}s training")
            
        except Exception as e:
            print(f"FAILED {size_name}: {e}")
            results[size_name] = {'error': str(e)}
    
    # Print summary
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    
    for size, result in results.items():
        if 'error' in result:
            print(f"{size:8s}: FAILED - {result['error']}")
        else:
            print(f"{size:8s}: {result['parameters']:7,} params, "
                  f"{result['training_time']:5.1f}s, "
                  f"loss={result['final_loss']:.3f}")

def main():
    """Main training function"""
    print("=== Scalable Transformer Training ===")
    print("Testing different model sizes for RP2040")
    
    # Ask which size to train
    print("\nAvailable sizes:")
    for name, config in MODEL_CONFIGS.items():
        print(f"  {name:8s}: {config['description']}")
    
    # For automated testing, train all sizes
    choice = input("\nEnter size (tiny/small/medium/large/xlarge) or 'all': ").strip().lower()
    
    if choice == 'all':
        test_all_sizes()
    elif choice in MODEL_CONFIGS:
        trainer = ScalableTrainer(choice)
        trainer.prepare_data()
        
        # Train with size-appropriate epochs
        epochs = max(50, 200 // max(1, trainer.config['dim'] // 16))
        trainer.train(epochs=epochs, learning_rate=0.005)
        
        trainer.save_model()
        print(f"\n{choice.upper()} model training complete!")
    else:
        print("Invalid choice. Training tiny model as default.")
        trainer = ScalableTrainer('tiny')
        trainer.prepare_data()
        trainer.train(epochs=100, learning_rate=0.01)
        trainer.save_model()

if __name__ == "__main__":
    main()