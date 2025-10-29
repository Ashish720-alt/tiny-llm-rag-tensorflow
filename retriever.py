# retriever.py
import tensorflow as tf
from typing import List, Tuple
from data import encode, decode

class SimpleRetriever:
    """
    A minimal retriever that embeds text by averaging token embeddings
    from the LM's token embedding table, and uses cosine similarity.
    """
    def __init__(self, lm_token_embedding_layer, max_len: int = 128):
        self.emb_layer = lm_token_embedding_layer  # Keras Embedding
        self.max_len = max_len
        self.doc_texts: List[str] = []
        self.doc_embs = None  # (N, d)

    @staticmethod
    def _cosine_similarity(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        a = tf.math.l2_normalize(a, axis=-1)
        b = tf.math.l2_normalize(b, axis=-1)
        return tf.matmul(a, b, transpose_b=True)

    def _text_to_emb(self, text: str) -> tf.Tensor:
        ids = encode(text)[: self.max_len]
        if not ids:
            ids = [0]
        ids = tf.constant(ids, dtype=tf.int32)[None, :]  # (1, T)
        emb = self.emb_layer(ids)  # (1, T, d)
        emb = tf.reduce_mean(emb, axis=1)  # (1, d)
        return emb  # (1, d)

    def build(self, documents: List[str]):
        self.doc_texts = documents
        embs = []
        for doc in documents:
            embs.append(self._text_to_emb(doc))
        self.doc_embs = tf.concat(embs, axis=0)  # (N, d)

    def query(self, q: str, k: int = 2) -> List[Tuple[int, float]]:
        if self.doc_embs is None or len(self.doc_texts) == 0:
            return []
        q_emb = self._text_to_emb(q)  # (1, d)
        sims = self._cosine_similarity(q_emb, self.doc_embs)  # (1, N)
        sims = tf.squeeze(sims, axis=0)  # (N,)
        values, indices = tf.math.top_k(sims, k=min(k, tf.shape(sims)[0]))
        return [(int(idx.numpy()), float(val.numpy())) for idx, val in zip(indices, values)]

    def retrieve_text(self, q: str, k: int = 2) -> str:
        matches = self.query(q, k)
        if not matches:
            return ""
        parts = []
        for idx, score in matches:
            parts.append(self.doc_texts[idx])
        return "\n".join(parts)
