# data.py
import tensorflow as tf
import string
import json
import os
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

def vocab_size() -> int:
    return len(VOCAB)

def build_dataset(corpus: str, seq_len: int, batch_size: int, buffer_size: int = 10000) -> tf.data.Dataset:
    ids = tf.constant(encode(corpus), dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices(ids)
    ds = ds.window(seq_len + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(seq_len + 1))
    ds = ds.map(lambda x: (x[:-1], x[1:]), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds

# ----- NEW: loaders for MedQuAD JSON files -----
def load_medquad_json(path: str) -> Tuple[str, list]:
    """Load a JSON list of {'question', 'answer'} into (corpus, docs)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"MedQuAD json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = [f"Q: {d['question']}\nA: {d['answer']}" for d in data if d.get("question") and d.get("answer")]
    corpus = " ".join(docs)
    return corpus, docs

# Fallback tiny toy corpus (kept for quick smoke tests)
def sample_corpus() -> Tuple[str, List[str]]:
    docs = [
        "TensorFlow is a machine learning framework for building and training models.",
        "Keras provides a high-level API for building neural networks in TensorFlow.",
        "Retrieval augmented generation (RAG) prepends retrieved context to the prompt.",
        "Transformers use attention to model long-range dependencies in sequences.",
    ]
    base_text = "tiny transformer language model demo for learning character-level modeling. "
    corpus = base_text + " ".join(docs)
    return corpus, docs
