import argparse
import random
import re
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm import tqdm

from ..data_prep.download import Downloader
from ..utils import JsonlFile
from .data import PROCESSED_DIR, RAW_DIR, FeverClaims, FeverText, WikiDump

TOTAL_WIKI_PAGES = 5_416_537


class SubsetCorpusBuilder:
    """Wikipedia subset for a set of claims: gold pages + hard negatives (title matches a claim n-gram) + random pages."""

    def __init__(self, claims, n_random=20000, seed=42, max_sentences=None):
        self.claims = claims
        self.n_random = n_random
        self.seed = seed
        self.max_sentences = max_sentences

    @staticmethod
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

    @staticmethod
    def base_title(title):
        title = re.sub(r"\s*\(.*\)\s*$", "", title)
        return " ".join(re.findall(r"\w+", title.lower()))

    def build(self, zip_path):
        gold_pages = {page for c in self.claims for ev in c["evidence_sets"] for page, _ in ev}
        grams = self.claim_ngrams(self.claims)
        rng = random.Random(self.seed)
        random_prob = self.n_random / TOTAL_WIKI_PAGES

        corpus = []
        stats = {"gold": 0, "hard_negative": 0, "random": 0}

        for page in tqdm(WikiDump(zip_path).iter_pages(), total=TOTAL_WIKI_PAGES, desc="Scan wiki"):
            page_id = FeverText.normalize_page_id(page["id"])
            title = FeverText.title_from_page_id(page_id)

            if page_id in gold_pages:
                source = "gold"
            elif self.base_title(title) in grams:
                source = "hard_negative"
            elif rng.random() < random_prob:
                source = "random"
            else:
                continue

            sentences = WikiDump.parse_lines(page["lines"])
            if self.max_sentences:
                sentences = sentences[:self.max_sentences]
            if not sentences:
                continue

            corpus.append({
                "id": page_id,
                "title": title,
                "sentences": [[line_id, FeverText.clean(text)] for line_id, text in sentences],
                "source": source,
            })
            stats[source] += 1

        missing = gold_pages - {p["id"] for p in corpus if p["source"] == "gold"}
        return corpus, stats, missing

    @staticmethod
    def save(corpus, path):
        JsonlFile.write(corpus, path)


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

    claims_path = Downloader.fever(args.split)
    zip_path = RAW_DIR / "wiki-pages.zip"
    if not zip_path.exists():
        zip_path = Downloader.fever("wiki-pages.zip")

    claims = FeverClaims.load(claims_path, n=args.n_claims, seed=args.seed, include_nei=False)
    print(f"Jumlah claim sampel: {len(claims)}")

    builder = SubsetCorpusBuilder(claims, n_random=args.n_random, seed=args.seed)
    corpus, stats, missing = builder.build(zip_path)
    print(f"Jumlah halaman di corpus: {len(corpus)} -> {stats}")
    if missing:
        print(f"Warning: {len(missing)} gold page tidak ketemu di dump, contoh: {list(missing)[:5]}")

    FeverClaims.save(claims, out_dir / "claims_sample.jsonl")
    builder.save(corpus, out_dir / "wiki_subset.jsonl")
    print(f"Tersimpan di {out_dir}")


if __name__ == "__main__":
    main()
