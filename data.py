# data.py
import tensorflow as tf
import string
from typing import List, Tuple

# Fixed small character vocabulary (printable ASCII subset)
VOCAB_CHARS = (
    " \n\t" +
    string.ascii_lowercase +
    string.ascii_uppercase +
    string.digits +
    string.punctuation
)
VOCAB = sorted(set(VOCAB_CHARS))
TOK2ID = {ch: i for i, ch in enumerate(VOCAB)}
ID2TOK = {i: ch for ch, i in TOK2ID.items()}

def encode(text: str) -> List[int]:
    return [TOK2ID.get(ch, TOK2ID[" "]) for ch in text]

def decode(ids: List[int]) -> str:
    return "".join(ID2TOK.get(i, " ") for i in ids)

def build_dataset(corpus: str, seq_len: int, batch_size: int, buffer_size: int = 10000) -> tf.data.Dataset:
    ids = tf.constant(encode(corpus), dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices(ids)
    ds = ds.window(seq_len + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(seq_len + 1))
    ds = ds.map(lambda x: (x[:-1], x[1:]), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds

def sample_corpus() -> Tuple[str, List[str]]:
    # Tiny toy corpus and a small RAG document store; feel free to replace with your own data.
    docs = [
        "TensorFlow is a machine learning framework for building and training models.",
        "Keras provides a high-level API for building neural networks in TensorFlow.",
        "Retrieval augmented generation (RAG) prepends retrieved context to the prompt.",
        "Transformers use attention to model long-range dependencies in sequences.",
        "Gradients are computed via automatic differentiation and backpropagation.",
        "Tokenization maps text to integers; decoding maps integers back to text."
    ]
    base_text = (
        "tiny transformer language model demo. "
        "this model learns to predict next characters using attention. "
        "tensorflow and keras make it straightforward to define models. "
        "we train on a very small corpus so the model is tiny. "
        "rag can retrieve helpful context and prepend it to the prompt. "
    )
    # Mix docs into the training text to expose model to some document content/phrases
    corpus = base_text + " ".join(docs)
    return corpus, docs

def vocab_size() -> int:
    return len(VOCAB)
