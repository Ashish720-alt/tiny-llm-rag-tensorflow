# model.py
import tensorflow as tf
from tensorflow.keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation="gelu"),
            layers.Dense(d_model),
        ])
        self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        h = self.ln1(x)
        attn_out = self.attn(h, h, attention_mask=None, training=training)
        x = x + self.drop(attn_out, training=training)
        h2 = self.ln2(x)
        ffn_out = self.ffn(h2, training=training)
        x = x + self.drop(ffn_out, training=training)
        return x

class TinyTransformerLM(tf.keras.Model):
    """
    A very small character-level Transformer LM (~10k parameters).
    Defaults chosen to fit well under 10k.
    """
    def __init__(self, vocab_size: int, max_len: int = 64, d_model: int = 16, num_heads: int = 2, d_ff: int = 32, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model

        self.token_emb = layers.Embedding(vocab_size, d_model)
        self.pos_emb = layers.Embedding(max_len, d_model)

        self.blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.ln_f = layers.LayerNormalization(epsilon=1e-5)
        self.lm_head = layers.Dense(vocab_size, use_bias=True)

    def call(self, x, training=False):
        # x: (B, T)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        tok = self.token_emb(x)  # (B, T, d)
        pos_idx = tf.range(T)[None, :]
        pos = self.pos_emb(pos_idx)  # (1, T, d)
        h = tok + pos
        for blk in self.blocks:
            h = blk(h, training=training)
        h = self.ln_f(h)
        logits = self.lm_head(h)  # (B, T, V)
        return logits

    def generate(self, prompt_ids, max_new_tokens: int, temperature: float = 1.0):
        """
        Greedy sampling with temperature (temperature only for softening logits; still argmax pick).
        prompt_ids: 1D int32 tensor
        """
        x = tf.identity(prompt_ids)[None, :]  # (1, T)
        for _ in range(max_new_tokens):
            # crop to max_len
            x_in = x[:, -self.max_len:]
            logits = self(x_in, training=False)
            next_logits = logits[:, -1, :] / tf.maximum(1e-6, temperature)
            next_id = tf.argmax(next_logits, axis=-1, output_type=tf.int32)  # (1,)
            x = tf.concat([x, tf.expand_dims(next_id, axis=1)], axis=1)
        return tf.squeeze(x, axis=0)

def count_params(model: tf.keras.Model) -> int:
    return int(sum(p.numpy().size for p in model.trainable_variables))
# model.py
import tensorflow as tf
from tensorflow.keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation="gelu"),
            layers.Dense(d_model),
        ])
        self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        h = self.ln1(x)
        attn_out = self.attn(h, h, attention_mask=None, training=training)
        x = x + self.drop(attn_out, training=training)
        h2 = self.ln2(x)
        ffn_out = self.ffn(h2, training=training)
        x = x + self.drop(ffn_out, training=training)
        return x

class TinyTransformerLM(tf.keras.Model):
    """
    A very small character-level Transformer LM (~10k parameters).
    Defaults chosen to fit well under 10k.
    """
    def __init__(self, vocab_size: int, max_len: int = 64, d_model: int = 16, num_heads: int = 2, d_ff: int = 32, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model

        self.token_emb = layers.Embedding(vocab_size, d_model)
        self.pos_emb = layers.Embedding(max_len, d_model)

        self.blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.ln_f = layers.LayerNormalization(epsilon=1e-5)
        self.lm_head = layers.Dense(vocab_size, use_bias=True)

    def call(self, x, training=False):
        # x: (B, T)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        tok = self.token_emb(x)  # (B, T, d)
        pos_idx = tf.range(T)[None, :]
        pos = self.pos_emb(pos_idx)  # (1, T, d)
        h = tok + pos
        for blk in self.blocks:
            h = blk(h, training=training)
        h = self.ln_f(h)
        logits = self.lm_head(h)  # (B, T, V)
        return logits

    def generate(self, prompt_ids, max_new_tokens: int, temperature: float = 1.0):
        """
        Greedy sampling with temperature (temperature only for softening logits; still argmax pick).
        prompt_ids: 1D int32 tensor
        """
        x = tf.identity(prompt_ids)[None, :]  # (1, T)
        for _ in range(max_new_tokens):
            # crop to max_len
            x_in = x[:, -self.max_len:]
            logits = self(x_in, training=False)
            next_logits = logits[:, -1, :] / tf.maximum(1e-6, temperature)
            next_id = tf.argmax(next_logits, axis=-1, output_type=tf.int32)  # (1,)
            x = tf.concat([x, tf.expand_dims(next_id, axis=1)], axis=1)
        return tf.squeeze(x, axis=0)

def count_params(model: tf.keras.Model) -> int:
    return int(sum(p.numpy().size for p in model.trainable_variables))
