import json
import random
import unicodedata
import zipfile

from ..paths import DATA_DIR, PROCESSED_DIR, PROJECT_DIR, RAW_DIR  # noqa: F401  (re-exported)
from ..utils import JsonlFile

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

class FeverText:
    """FEVER stores text as PTB tokens (-LRB-, ``) and page ids with underscores."""

    @staticmethod
    def normalize_page_id(page_id):
        return unicodedata.normalize("NFC", page_id)

    @staticmethod
    def clean(text):
        for token, char in PTB_TOKENS.items():
            text = text.replace(token, char)
        text = text.replace("( ", "(").replace(" )", ")")
        return " ".join(text.split())

    @classmethod
    def title_from_page_id(cls, page_id):
        return cls.clean(page_id.replace("_", " "))

class FeverClaims:
    """Claim files: raw FEVER jsonl (load) and our own cache of sampled claims (save / read_cache)."""

    @staticmethod
    def load(path, n=None, seed=42, include_nei=False):
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
                        sentences.append((FeverText.normalize_page_id(page_id), int(line_id)))
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

    @staticmethod
    def save(claims, path):
        JsonlFile.write(claims, path)

    @staticmethod
    def read_cache(path):
        claims = JsonlFile.read(path)
        for c in claims:
            c["evidence_sets"] = [[tuple(s) for s in ev] for ev in c["evidence_sets"]]
        return claims
    
class WikiDump:
    """FEVER wiki-pages.zip: many jsonl files, one page per line."""

    def __init__(self, zip_path):
        self.zip_path = zip_path

    def iter_pages(self):
        with zipfile.ZipFile(self.zip_path) as zf:
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

    @staticmethod
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
