import os
import tensorflow as tf
from data import encode, decode, load_medquad_json, vocab_size
from model import TinyTransformerLM
from retriever import SimpleRetriever

# --------------------------------------------------------------------
# Split-weight loader
# --------------------------------------------------------------------
def load_split_weights(model, prefix="medquad_lm_250k_part"):
    """Reassemble split .h5 chunks (<50 MB each) and load them into model."""
    parts = sorted([f for f in os.listdir(".")
                    if f.startswith(prefix) and f.endswith(".h5")])
    if not parts:
        raise FileNotFoundError(f"No split weight files found with prefix '{prefix}'")
    tmp_file = prefix + "_merged.h5"
    with open(tmp_file, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                out.write(f.read())
    model.load_weights(tmp_file)
    os.remove(tmp_file)
    print(f"Loaded weights from {len(parts)} part(s).")

# --------------------------------------------------------------------
# RAG prompt + generation
# --------------------------------------------------------------------
def build_prompt_with_rag(user_query: str, retriever: SimpleRetriever,
                          k: int = 2, max_context_chars: int = 300) -> str:
    ctx = retriever.retrieve_text(user_query, k=k)
    if ctx:
        ctx = ctx[:max_context_chars]
        return f"[CONTEXT]\n{ctx}\n[QUERY]\n{user_query}\n[ANSWER]\n"
    else:
        return f"[QUERY]\n{user_query}\n[ANSWER]\n"

def generate_text(model: TinyTransformerLM, prompt: str,
                  max_new_tokens: int = 150, temperature: float = 1.0) -> str:
    ids = encode(prompt)
    ids = ids[-model.max_len:] if len(ids) > model.max_len else ids
    ids_tf = tf.constant(ids, dtype=tf.int32)
    out_ids = model.generate(ids_tf, max_new_tokens=max_new_tokens,
                             temperature=temperature).numpy().tolist()
    return decode(out_ids)

def main():
    max_len = 128
    model = TinyTransformerLM(vocab_size=vocab_size(), max_len=max_len,
                              d_model=128, num_heads=4, d_ff=512, num_layers=2)
    _ = model(tf.zeros((1, max_len), dtype=tf.int32))

    # --- Load split weights (<50 MB each) ---
    load_split_weights(model)

    # --- Build retriever from MedQuAD train docs ---
    _, docs = load_medquad_json("medquad_train.json")
    retriever = SimpleRetriever(model.token_emb, max_len=128)
    retriever.build(docs)

    queries = [
        "What are the symptoms of diabetes?",
        "How is hypertension treated?",
        "What is asthma?"
    ]

    for q in queries:
        prompt = build_prompt_with_rag(q, retriever, k=2)
        out = generate_text(model, prompt, max_new_tokens=100)
        print("="*80)
        print("QUERY:", q)
        print(out)

if __name__ == "__main__":
    main()
