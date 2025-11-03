# train.py
import os
import json
import tensorflow as tf
from data import build_dataset, load_medquad_json, vocab_size
from model import TinyTransformerLM, count_params
from eval_rag import evaluate_model, plot_comparison  # <-- reuse our evaluation functions

def save_split_weights(model, prefix="medquad_lm_250k_part", max_mb=50):
    """Save model weights split into multiple parts under <max_mb> MB.

    Keras 3 requires .weights.h5 for save_weights().
    """
    tmp_file = prefix + "_full.weights.h5"   # <- must end with .weights.h5
    model.save_weights(tmp_file)

    with open(tmp_file, "rb") as f:
        data = f.read()
    os.remove(tmp_file)

    chunk_size = max_mb * 1024 * 1024
    num_parts = (len(data) + chunk_size - 1) // chunk_size
    for i in range(num_parts):
        part_path = f"{prefix}{i}.weights.h5"   # keep suffix for clarity
        with open(part_path, "wb") as out:
            out.write(data[i * chunk_size : (i + 1) * chunk_size])
        print(f"Saved {part_path} ({os.path.getsize(part_path)/1e6:.2f} MB)")
    print(f"Split into {num_parts} file(s) under {max_mb} MB each.")


def main():
    # Prefer GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU:", gpus[0].name)
    else:
        print("Using CPU")

    # Hyperparameters (~250K params)
    max_len = 128
    d_model = 128
    num_heads = 4
    d_ff = 512
    num_layers = 2
    batch_size = 128
    epochs = 20 
    lr = 2e-4

    # Load training data
    corpus, _ = load_medquad_json("medquad_train.json")
    ds = build_dataset(corpus=corpus, seq_len=max_len, batch_size=batch_size)

    # Build model
    model = TinyTransformerLM(vocab_size=vocab_size(), max_len=max_len,
                              d_model=d_model, num_heads=num_heads,
                              d_ff=d_ff, num_layers=num_layers)

    # Initialize weights
    for x, _ in ds.take(1):
        _ = model(x)
    print("Trainable parameters:", count_params(model))

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss_fn,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    # Training
    print(f"Starting fine-tuning for {epochs} epochs (RAG OFF)...")
    model.fit(ds, epochs=epochs)

    # Save weights split into <50MB chunks
    save_split_weights(model)

    # ==========================
    # Evaluate after training
    # ==========================
    print("\n==============================")
    print("Running evaluation (RAG ON and OFF)")
    print("==============================")

    results = evaluate_model(model, train_json="medquad_train.json",
                             test_json="medquad_test.json", save_plot=True)

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved detailed results to eval_results.json")

if __name__ == "__main__":
    main()
