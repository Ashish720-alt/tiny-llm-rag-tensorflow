# medquad_prepare.py  (XML-compatible)
import os, json, argparse, glob
from lxml import etree
from tqdm import tqdm

def extract_pairs(medquad_dir: str):
    xml_files = glob.glob(os.path.join(medquad_dir, "**", "*.xml"), recursive=True)
    pairs = []

    for xf in tqdm(xml_files, desc="Parsing XML files"):
        try:
            tree = etree.parse(xf)
        except Exception:
            continue
        root = tree.getroot()
        for qa in root.findall(".//QAPair"):
            q_el = qa.find("Question")
            a_el = qa.find("Answer")
            if q_el is not None and a_el is not None:
                q = (q_el.text or "").strip()
                a = (a_el.text or "").strip()
                if q and a:
                    pairs.append({"question": q, "answer": a})
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medquad_dir", default="MedQuAD", help="Path to MedQuAD root folder")
    ap.add_argument("--out_json", default="medquad_all.json")
    args = ap.parse_args()

    pairs = extract_pairs(args.medquad_dir)
    if not pairs:
        raise SystemExit("No Q/A pairs found. Ensure you cloned the full MedQuAD repo with XML files.")
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(pairs)} QA pairs to {args.out_json}")

if __name__ == "__main__":
    main()
