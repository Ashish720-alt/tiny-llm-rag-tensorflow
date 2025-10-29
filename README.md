# Tiny TensorFlow LLM with Retrieval-Augmented Generation (RAG)

This project implements a **tiny (~10,000 parameter)** character-level Transformer language model in **TensorFlow/Keras**, along with a **simple Retrieval-Augmented Generation (RAG)** mechanism.  
It is designed for educational purposes — showing how a minimal transformer can be trained from scratch and used with a simple retriever to improve context-aware text generation.

---

## 🧩 Project Structure

tiny-llm-rag/
├── data.py # Dataset utilities, tokenizer, and toy corpus
├── model.py # Tiny Transformer LM (~10k params)
├── retriever.py # Simple cosine-based retriever (for RAG)
├── train.py # Trains the language model
├── generate.py # Generates text with retrieval-augmented context
├── rag_demo.py # Runs training + RAG demo end-to-end
└── README.md # This file

## 2. Install dependencies

pip install tensorflow

## (Optional) For convenience

pip install numpy tqdm


---

## 🚀 How to Run


## 1. Train the Tiny LLM

python train.py

→ Produces a weights file: tiny_lm_tf_weights.h5
## 2. Generate Text with RAG

python generate.py

→ Loads model, retrieves similar docs, and generates contextual text
## 3. Run Full Demo (train + generate)

python rag_demo.py


---

## 🧠 Example Output
================================================================================
QUERY: what is keras?
[CONTEXT]
Keras provides a high-level API for building neural networks in TensorFlow.
TensorFlow is a machine learning framework for building and training models.
[QUERY]
what is keras?
[ANSWER]
keras is a high level api built on tensorflow for neural network design and training.


---

## 🧮 Model Summary



Embedding dimension : 16
Attention heads : 2
Feed-forward size : 32
Transformer layers : 2
Total parameters : ~10,000
Vocabulary : Printable ASCII (~95 chars)
Sequence length : 64


---

## 📘 RAG Description



The retriever embeds documents by averaging token embeddings from the model’s
embedding table. At generation time:

Retrieves top-k most similar documents to the user query using cosine similarity.

Prepends them to the model input as [CONTEXT].

The model then generates [ANSWER] text conditioned on the retrieved context.


---

## 📄 License



MIT License — free for research and educational use.


---

## ✏️ Notes



• This is an educational demonstration, not a production LLM.
• Runs quickly on CPU — no external data or internet required.
• The RAG mechanism is intentionally minimal (no vector DBs or APIs).
