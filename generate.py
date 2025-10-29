# generate.py
import tensorflow as tf
from data import encode, decode, sample_corpus, vocab_size
from model import TinyTransformerLM
from retriever import SimpleRetriever

def build_prompt_with_rag(user_query: str, retriever: SimpleRetriever, k: int = 2, max_context_chars: int = 200) -> str:
    ctx = retriever.retrieve_text(user_query, k=k)
    if ctx:
        ctx = ctx[:max_context_chars]
        rag_prompt = f"[CONTEXT]\n{ctx}\n[QUERY]\n{user_query}\n[ANSWER]\n"
    else:
        rag_prompt = f"[QUERY]\n{user_query}\n[ANSWER]\n"
    return rag_prompt

def generate_text(model: TinyTransformerLM, prompt: str, max_new_tokens: int = 128, temperature: float = 1.0) -> str:
    ids = encode(prompt)
    ids = ids[-model.max_len:] if len(ids) > model.max_len else ids
    ids_tf = tf.constant(ids, dtype=tf.int32)
    out_ids = model.generate(ids_tf, max_new_tokens=max_new_tokens, temperature=temperature).numpy().tolist()
    return decode(out_ids)

def main():
    max_len = 64
    model = TinyTransformerLM(vocab_size=vocab_size(), max_len=max_len,
                              d_model=16, num_heads=2, d_ff=32, num_layers=2)
    # Build variables
    _ = model(tf.zeros((1, max_len), dtype=tf.int32))
    model.load_weights("tiny_lm_tf_weights.h5")

    # Build retriever from the trained token embeddings
    _, docs = sample_corpus()
    retriever = SimpleRetriever(model.token_emb, max_len=128)
    retriever.build(docs)

    # Example RAG queries
    queries = [
        "what is keras?",
        "how does rag work?",
        "what is a transformer?"
    ]

    for q in queries:
        prompt = build_prompt_with_rag(q, retriever, k=2)
        out = generate_text(model, prompt, max_new_tokens=120, temperature=1.0)
        print("="*80)
        print("QUERY:", q)
        print(out)

if __name__ == "__main__":
    main()
