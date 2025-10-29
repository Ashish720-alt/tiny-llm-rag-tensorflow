# train.py
import tensorflow as tf
from data import build_dataset, sample_corpus, vocab_size, encode, decode
from model import TinyTransformerLM, count_params

def main():
    # Hyperparameters (kept very small)
    max_len = 64
    d_model = 16
    num_heads = 2
    d_ff = 32
    num_layers = 2
    batch_size = 64
    epochs = 5
    lr = 3e-3

    corpus, _ = sample_corpus()
    ds = build_dataset(corpus=corpus, seq_len=max_len, batch_size=batch_size)

    model = TinyTransformerLM(vocab_size=vocab_size(), max_len=max_len,
                              d_model=d_model, num_heads=num_heads,
                              d_ff=d_ff, num_layers=num_layers)

    # Build model by running one forward pass (to initialize weights)
    for x, y in ds.take(1):
        _ = model(x)

    print("Trainable parameters:", count_params(model))

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.fit(ds, epochs=epochs)

    model.save_weights("tiny_lm_tf_weights.h5")
    print("Saved weights to tiny_lm_tf_weights.h5")

if __name__ == "__main__":
    main()
