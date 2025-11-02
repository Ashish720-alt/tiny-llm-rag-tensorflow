# eval_rag.py
import os, json, argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from data import encode, decode, load_medquad_json, vocab_size
from model import TinyTransformerLM
from retriever import SimpleRetriever

# -------------------------------------------------------------
# Utility: load split weights (<50 MB each)
# -------------------------------------------------------------
def load_split_weights(model, prefix="medquad_lm_250k_part"):
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

# -------------------------------------------------------------
# Metrics
# -------------------------------------------------------------
def tokenize_simple(s: str):
    return [t for t in s.lower().split() if t.strip()]

def f1_overlap(pred: str, gold: str):
    from collections import Counter
    p, g = Counter(tokenize_simple(pred)), Counter(tokenize_simple(gold))
    overlap = sum((p & g).values())
    if not overlap:
        return 0.0
    prec, rec = overlap / sum(p.values()), overlap / sum(g.values())
    return 2 * prec * rec / (prec + rec)

def exact_substring(pred: str, gold: str):
    return float(gold.lower() in pred.lower())

# -------------------------------------------------------------
# Prompt builder + generator
# -------------------------------------------------------------
def build_prompt(user_query: str, retriever=None, rag_mode=True,
                 k: int = 2, max_context_chars: int = 600):
    if rag_mode and retriever is not None:
        ctx = retriever.retrieve_text(user_query, k=k)
        if ctx:
            ctx = ctx[:max_context_chars]
            return f"[CONTEXT]\n{ctx}\n[QUERY]\n{user_query}\n[ANSWER]\n"
    return f"[QUERY]\n{user_query}\n[ANSWER]\n"

def generate_text(model, prompt: str, max_new_tokens: int = 150, temperature: float = 1.0):
    ids = encode(prompt)
    ids = ids[-model.max_len:] if len(ids) > model.max_len else ids
    ids_tf = tf.constant(ids, dtype=tf.int32)
    out_ids = model.generate(ids_tf, max_new_tokens=max_new_tokens,
                             temperature=temperature).numpy().tolist()
    return decode(out_ids)

# -------------------------------------------------------------
# Evaluation (used both by CLI and train.py)
# -------------------------------------------------------------
def evaluate_model(model, train_json="medquad_train.json",
                   test_json="medquad_test.json", save_plot=True):
    _, train_docs = load_medquad_json(train_json)
    with open(test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Build retriever for RAG
    retriever = SimpleRetriever(model.token_emb, max_len=128)
    retriever.build(train_docs)

    def run_eval(rag_mode=True):
        f1_scores, em_scores = [], []
        for i, ex in enumerate(test_data[:300]):  # limit to 300 for faster demo
            q, gold = ex.get("question", "").strip(), ex.get("answer", "").strip()
            if not q or not gold:
                continue
            prompt = build_prompt(q, retriever, rag_mode=rag_mode, k=2)
            pred = generate_text(model, prompt, max_new_tokens=150)
            if "[ANSWER]" in pred:
                pred = pred.split("[ANSWER]", 1)[-1]
            pred = pred[:600]
            f1 = f1_overlap(pred, gold)
            em = exact_substring(pred, gold)
            f1_scores.append(f1)
            em_scores.append(em)
        mean_f1 = sum(f1_scores) / max(1, len(f1_scores))
        mean_em = sum(em_scores) / max(1, len(em_scores))
        return mean_f1, mean_em

    # Run both evaluations
    print("→ Evaluating with RAG ON ...")
    f1_rag, em_rag = run_eval(rag_mode=True)
    print("→ Evaluating with RAG OFF ...")
    f1_norag, em_norag = run_eval(rag_mode=False)

    results = {
        "RAG_ON": {"F1": f1_rag, "EM": em_rag},
        "RAG_OFF": {"F1": f1_norag, "EM": em_norag}
    }

    print("\n--- Comparison ---")
    print(f"F1  (RAG ON):  {f1_rag:.4f}")
    print(f"F1  (RAG OFF): {f1_norag:.4f}")
    print(f"EM  (RAG ON):  {em_rag:.4f}")
    print(f"EM  (RAG OFF): {em_norag:.4f}")

    if save_plot:
        plot_comparison(results)

    return results

# -------------------------------------------------------------
# Plotting
# -------------------------------------------------------------
def plot_comparison(results):
    import matplotlib.pyplot as plt
    labels = ["F1", "EM"]
    rag_on = [results["RAG_ON"]["F1"], results["RAG_ON"]["EM"]]
    rag_off = [results["RAG_OFF"]["F1"], results["RAG_OFF"]["EM"]]
    x = range(len(labels))
    plt.figure(figsize=(6, 4))
    width = 0.35
    plt.bar([i - width/2 for i in x], rag_on, width=width, label="RAG ON")
    plt.bar([i + width/2 for i in x], rag_off, width=width, label="RAG OFF")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("RAG vs No-RAG Performance (MedQuAD)")
    plt.legend()
    plt.tight_layout()
    out_path = "rag_comparison_hist.png"
    plt.savefig(out_path)
    plt.close()
    print(f"\nSaved histogram comparison plot to {out_path}")

# -------------------------------------------------------------
# CLI entry
# -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="medquad_train.json")
    ap.add_argument("--test_json", default="medquad_test.json")
    args = ap.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU:", gpus[0].name)
    else:
        print("Using CPU")

    # Build model
    max_len = 128
    model = TinyTransformerLM(vocab_size=vocab_size(), max_len=max_len,
                              d_model=128, num_heads=4, d_ff=512, num_layers=2)
    _ = model(tf.zeros((1, max_len), dtype=tf.int32))

    # Load split weights
    load_split_weights(model)

    evaluate_model(model, train_json=args.train_json,
                   test_json=args.test_json, save_plot=True)

if __name__ == "__main__":
    main()
