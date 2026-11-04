import argparse
import json
import random
import re
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm import tqdm

from .data import (
    PROCESSED_DIR,
    RAW_DIR,
    clean_text,
    download_fever_file,
    iter_wiki_pages,
    load_claims,
    normalize_page_id,
    parse_lines,
    save_claims,
    title_from_page_id,
)

TOTAL_WIKI_PAGES = 5_416_537


def claim_ngrams(claims, max_n=3):
    grams = set()
    for c in claims:
        tokens = re.findall(r"\w+", c["claim"].lower())
        for n in range(1, max_n + 1):
            for i in range(len(tokens) - n + 1):
                gram = tokens[i:i + n]
                if n == 1:
                    if gram[0] in ENGLISH_STOP_WORDS or len(gram[0]) < 3:
                        continue
                grams.add(" ".join(gram))
    return grams


def base_title(title):
    title = re.sub(r"\s*\(.*\)\s*$", "", title)
    return " ".join(re.findall(r"\w+", title.lower()))


def build_subset_corpus(claims, zip_path, n_random=20000, seed=42, max_sentences=None):
    gold_pages = {page for c in claims for ev in c["evidence_sets"] for page, _ in ev}
    grams = claim_ngrams(claims)
    rng = random.Random(seed)
    random_prob = n_random / TOTAL_WIKI_PAGES

    corpus = []
    stats = {"gold": 0, "hard_negative": 0, "random": 0}

    for page in tqdm(iter_wiki_pages(zip_path), total=TOTAL_WIKI_PAGES, desc="Scan wiki"):
        page_id = normalize_page_id(page["id"])
        title = title_from_page_id(page_id)

        if page_id in gold_pages:
            source = "gold"
        elif base_title(title) in grams:
            source = "hard_negative"
        elif rng.random() < random_prob:
            source = "random"
        else:
            continue

        sentences = parse_lines(page["lines"])
        if max_sentences:
            sentences = sentences[:max_sentences]
        if not sentences:
            continue

        corpus.append({
            "id": page_id,
            "title": title,
            "sentences": [[line_id, clean_text(text)] for line_id, text in sentences],
            "source": source,
        })
        stats[source] += 1

    missing = gold_pages - {p["id"] for p in corpus if p["source"] == "gold"}
    return corpus, stats, missing


def save_corpus(corpus, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for page in corpus:
            f.write(json.dumps(page, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="shared_task_dev.jsonl",
                        help="file claim FEVER yang dipakai (default: labelled dev set)")
    parser.add_argument("--n-claims", type=int, default=500)
    parser.add_argument("--n-random", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(PROCESSED_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    claims_path = download_fever_file(args.split)
    zip_path = RAW_DIR / "wiki-pages.zip"
    if not zip_path.exists():
        zip_path = download_fever_file("wiki-pages.zip")

    claims = load_claims(claims_path, n=args.n_claims, seed=args.seed, include_nei=False)
    print(f"Jumlah claim sampel: {len(claims)}")

    corpus, stats, missing = build_subset_corpus(claims, zip_path, n_random=args.n_random, seed=args.seed)
    print(f"Jumlah halaman di corpus: {len(corpus)} -> {stats}")
    if missing:
        print(f"Warning: {len(missing)} gold page tidak ketemu di dump, contoh: {list(missing)[:5]}")

    save_claims(claims, out_dir / "claims_sample.jsonl")
    save_corpus(corpus, out_dir / "wiki_subset.jsonl")
    print(f"Tersimpan di {out_dir}")


if __name__ == "__main__":
    main()
