# rag_demo.py
# Optional script to quickly (re)train then run RAG generation.
import subprocess
import sys
import os

def run(cmd):
    print("+", " ".join(cmd))
    subprocess.check_call([sys.executable] + cmd)

def main():
    # Train
    run(["train.py"])
    # Generate with RAG
    run(["generate.py"])

if __name__ == "__main__":
    main()
