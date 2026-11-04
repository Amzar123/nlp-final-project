import json
import random
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

FEVER_BASE_URL = "https://fever.ai/download/fever"

PTB_TOKENS = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LCB-": "{",
    "-RCB-": "}",
    "-COLON-": ":",
    "``": '"',
    "''": '"',
}


def download_fever_file(filename, dest_dir=RAW_DIR):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.exists():
        return dest

    url = f"{FEVER_BASE_URL}/{filename}"
    print(f"Download {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def normalize_page_id(page_id):
    return unicodedata.normalize("NFC", page_id)


def clean_text(text):
    for token, char in PTB_TOKENS.items():
        text = text.replace(token, char)
    text = text.replace("( ", "(").replace(" )", ")")
    return " ".join(text.split())


def title_from_page_id(page_id):
    return clean_text(page_id.replace("_", " "))


def load_claims(path, n=None, seed=42, include_nei=False):
    claims = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["label"] == "NOT ENOUGH INFO" and not include_nei:
                continue

            evidence_sets = []
            for ev_set in row["evidence"]:
                sentences = []
                for _, _, page_id, line_id in ev_set:
                    if page_id is None:
                        continue
                    sentences.append((normalize_page_id(page_id), int(line_id)))
                sentences = tuple(sorted(set(sentences)))
                if sentences and sentences not in evidence_sets:
                    evidence_sets.append(sentences)

            claims.append({
                "id": row["id"],
                "claim": row["claim"],
                "label": row["label"],
                "evidence_sets": [list(s) for s in evidence_sets],
            })

    if n is not None and n < len(claims):
        rng = random.Random(seed)
        claims = rng.sample(claims, n)
    return claims


def save_claims(claims, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def read_claims_cache(path):
    claims = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            c["evidence_sets"] = [[tuple(s) for s in ev] for ev in c["evidence_sets"]]
            claims.append(c)
    return claims


def iter_wiki_pages(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(
            n for n in zf.namelist()
            if n.endswith(".jsonl") and not n.startswith("__MACOSX")
        )
        for name in names:
            with zf.open(name) as f:
                for raw_line in f:
                    page = json.loads(raw_line)
                    if not page["id"]:
                        continue
                    yield page


def parse_lines(lines_field):
    sentences = []
    for row in lines_field.split("\n"):
        parts = row.split("\t")
        if len(parts) < 2 or not parts[1].strip():
            continue
        try:
            line_id = int(parts[0])
        except ValueError:
            continue
        sentences.append((line_id, parts[1]))
    return sentences
