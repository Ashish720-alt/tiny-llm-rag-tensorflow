# 1. Tiny Bio-LLM with RAG (250K params, MedQuAD fine-tuned)

This project implements a **250,000-parameter character-level Transformer** in TensorFlow/Keras, fine-tuned on the **MedQuAD medical QA dataset**, with an integrated **Retrieval-Augmented Generation (RAG)** pipeline.

## 2. Run Code: First trains on 90% of MedQuad (train) without RAG, then on 10 % of MedQuad (test), compare RAG vs no-RAG results.
python main.py

# 3. Architecture
Embedding size: 128
Layers: 2
Attention heads: 4
FFN size: 512
Parameters: ~250K
Dataset: MedQuAD (medical question answering)


### 4. Optional: To create a different train/test split of MedQuAD

If you want to regenerate a new split of MedQuad (for example 90/10 or 80/20), run the following commands:

```bash
# 1. Clone the MedQuAD dataset
git clone https://github.com/abachaa/MedQuAD.git

# 2. Convert all MedQuAD XML QA pairs into a single JSON file
python medquad_prepare.py --medquad_dir MedQuAD --out_json medquad_all.json

# 3. Create a new train/test split (adjust test_size as needed)
python medquad_split.py \
  --in_json medquad_all.json \
  --train_out medquad_train.json \
  --test_out medquad_test.json \
  --test_size 0.1 \
  --seed 42

# 4. Now as before you can run the main training + evaluation pipeline
python main.py
