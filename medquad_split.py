# medquad_split.py
import json, argparse, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", default="medquad_all.json")
    ap.add_argument("--train_out", default="medquad_train.json")
    ap.add_argument("--test_out", default="medquad_test.json")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(args.seed)
    random.shuffle(data)

    n = len(data)
    n_test = int(round(n * args.test_size))
    test_data = data[:n_test]
    train_data = data[n_test:]

    with open(args.train_out, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(args.test_out, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train_data)}  Test: {len(test_data)}  (Total: {n})")

if __name__ == "__main__":
    main()
