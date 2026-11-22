# Panduan Kode: Fact-Checking Pipeline

Dokumen ini menjelaskan **gambaran besar** project: alur data, peran setiap file, dan cara
semuanya tersambung. Detail eksperimen dan hasil per task ada di README masing-masing:
[Task 1](../src/task1_claim_detection/README.md) ·
[Task 2](../src/task2_evidence_retrieval/README.md) ·
[Task 3](../src/task3_verification/README.md).

---

## 1. Gambaran besar

Input berupa teks mentah (misalnya transkrip pidato), output berupa putusan untuk setiap klaim faktual di dalamnya.

```
teks mentah
  │
  ▼
┌─────────────────────────────── Task 1: Claim Detection ───────────────────────────────┐
│ spaCy pecah jadi kalimat → RoBERTa (fine-tuned ClaimBuster) → P(claim) per kalimat    │
│ simpan kalimat dengan P(claim) ≥ threshold (0.60, hasil tuning di validation)          │
└────────────────────────────────────────────────────────────────────────────────────────┘
  │  daftar klaim, mis. "The Eiffel Tower is located in Berlin."
  ▼
┌──────────────────────────── Task 2: Evidence Retrieval ───────────────────────────────┐
│ [1] BM25 (judul ×2 + isi)          → top-5 halaman Wikipedia                           │
│ [2] bi-encoder all-MiniLM-L6-v2    → top-30 kalimat kandidat (cosine similarity)       │
│ [3] cross-encoder ms-marco-MiniLM  → top-5 kalimat evidence                            │
└────────────────────────────────────────────────────────────────────────────────────────┘
  │  5 kalimat evidence + skor
  ▼
┌──────────────────────────── Task 3: NLI Verification ─────────────────────────────────┐
│ DeBERTa-v3 NLI: premise = "<judul>: <kalimat evidence>", hypothesis = klaim            │
│ 5 pasangan per kalimat + 1 pasangan gabungan (semua evidence disambung)                │
│ support = max P(entail), refute = max P(contradict)                                    │
│ max(support, refute) < 0.5 → NOT ENOUGH INFO, selain itu yang lebih besar menang       │
└────────────────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
SUPPORTS / REFUTES / NOT ENOUGH INFO (+ confidence + evidence)
```

Semua glue ada di [src/pipeline.py](../src/pipeline.py), class `FactCheckPipeline`.

---

## 2. Peta file

```
src/
├── paths.py                  ← lokasi folder bersama (data/, models/, outputs/)
├── pipeline.py               ← ENTRY POINT utama: Task 1 → 2 → 3
├── run_test_cases.py         ← jalankan tests/test_cases.json lewat pipeline
│
├── data_prep/                ← persiapan data (dijalankan sekali)
│   ├── download.py           ← download FEVER + ClaimBuster (dengan cek file terpotong)
│   └── build_task1_data.py   ← ClaimBuster CSV → task1_{train,val,test}.parquet
│
├── task1_claim_detection/
│   ├── train.py              ← fine-tune RoBERTa, simpan model + threshold.json
│   ├── baseline.py           ← pembanding TF-IDF + LogReg
│   ├── infer.py              ← ★ ClaimDetector (dipakai pipeline)
│   └── domain_gap.py         ← berapa % klaim FEVER yang lolos Task 1
│
├── task2_evidence_retrieval/
│   ├── data.py               ← baca FEVER (claim jsonl, wiki-pages.zip), bersihkan token PTB
│   ├── build_corpus.py       ← bikin subset Wikipedia → wiki_subset.jsonl
│   ├── bm25.py               ← implementasi BM25 pakai sparse matrix
│   ├── retriever.py          ← Evidence dataclass + EvidenceRetriever (tahap [1] & [2])
│   ├── reranker.py           ← CrossEncoderReranker (tahap [3])
│   ├── pipeline.py           ← ★ EvidencePipeline (dipakai pipeline)
│   └── evaluate.py           ← recall / hit@1 / MRR sebelum vs sesudah rerank
│
└── task3_verification/
    ├── nli.py                ← ★ NLIVerifier + aggregate() (dipakai pipeline)
    └── evaluate.py           ← accuracy, macro F1, FEVER score, sweep threshold

tests/test_cases.json         ← test case manual (claim detection + verifikasi)
notebooks/                    ← notebook Colab untuk training Task 1
data/  models/  outputs/      ← hasil generate, tidak masuk git
```

Yang ditandai ★ adalah tiga class yang dirangkai oleh `FactCheckPipeline`. File lain berfungsi
untuk **menyiapkan** (data, training) atau **mengukur** (evaluate).

---

## 3. Urutan setup dari nol

Semua perintah dijalankan dari root project dengan format `python -m src.<modul>`.
Kalau pakai `python src/xxx.py` langsung, relative import-nya error.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# --- Task 1 ---
python -m src.data_prep.download --no-wiki         # ClaimBuster + FEVER claims → data/raw/
python -m src.data_prep.build_task1_data           # → data/processed/task1_*.parquet
python -m src.task1_claim_detection.train          # → models/task1_roberta-base_label/model/
#   (lebih cepat di Colab, lihat notebooks/task1_colab.ipynb, lalu unzip hasilnya ke root)

# --- Task 2 ---
python -m src.task2_evidence_retrieval.build_corpus  # download wiki-pages.zip (1.7 GB) + shared_task_dev.jsonl
                                                     # → data/processed/wiki_subset.jsonl + claims_sample.jsonl

# --- Task 3 ---
# tidak perlu training, model NLI di-download otomatis dari HuggingFace saat pertama dipakai

# --- Coba ---
python -m src.pipeline "Good evening everyone. The Eiffel Tower is located in Berlin."
python -m src.run_test_cases
```

| File yang dibutuhkan | Dibuat oleh | Dipakai oleh |
|---|---|---|
| `data/raw/claimbuster/*.csv` | `data_prep.download` | `build_task1_data` |
| `data/processed/task1_*.parquet`, `task1_meta.json` | `build_task1_data` | `task1.train`, `task1.baseline` |
| `models/task1_roberta-base_label/model/` | `task1.train` | `ClaimDetector` |
| `data/raw/wiki-pages.zip`, `shared_task_dev.jsonl` | `task2.build_corpus` (via `task2/data.py`) | `build_corpus`, `task3.evaluate` |
| `data/processed/wiki_subset.jsonl` | `task2.build_corpus` | `EvidencePipeline` |
| `data/processed/claims_sample.jsonl` | `task2.build_corpus` | `task2.evaluate`, `task3.evaluate` |

> Catatan: ada **dua downloader**. `data_prep/download.py` (lebih aman, ada cek integritas,
> mengunduh `paper_dev`/`train`/ClaimBuster) dan `download_fever_file` di
> `task2_evidence_retrieval/data.py` (sederhana, dipakai `build_corpus` untuk `shared_task_dev.jsonl`).
> Keduanya menyimpan ke `data/raw/`.

---

## 4. Class-class utama

### 4.1 `ClaimDetector`: [task1_claim_detection/infer.py](../src/task1_claim_detection/infer.py)

```python
detector = ClaimDetector()              # model dir default + threshold dari threshold.json
detector.split_sentences(text)          # → ["kalimat 1", "kalimat 2", ...]  (spaCy, di-load lazy)
detector.score(["kal 1", "kal 2"])      # → np.array([0.99, 0.01])  = P(claim)
detector.detect(text)                   # → [{"index", "sentence", "p_claim", "is_claim"}, ...]  (hanya klaim)
detector.detect(text, return_all=True)  # semua kalimat
```

- Label 1 = **factual** (ClaimBuster CFS + UFS), bukan "check-worthy". Alasannya: mapping check-worthy
  membuang 83% klaim FEVER (lihat README Task 1).
- Threshold dibaca dari `model/threshold.json` (0.60). Kalau file tidak ada, dipakai 0.5.

### 4.2 `EvidencePipeline`: [task2_evidence_retrieval/pipeline.py](../src/task2_evidence_retrieval/pipeline.py)

```python
pipe = EvidencePipeline.from_corpus_file()   # load wiki_subset.jsonl, build BM25, load cross-encoder
pipe.run(claim, top_k=5)                     # → list of dict Evidence (urut paling relevan)
```

Isinya:

- `EvidenceRetriever` ([retriever.py](../src/task2_evidence_retrieval/retriever.py))
  - `retrieve_documents(claim, k)`: BM25 pada 2 index (judul dan isi), skor = `2 × judul + isi`
  - `candidate_sentences(claim, doc_k)`: semua kalimat dari top `doc_k` halaman
  - `score_sentences(...)`: cosine similarity bi-encoder, disimpan di `retrieval_score`
  - `retrieve(claim, top_k=30, doc_k=5)`: gabungan ketiga langkah di atas
- `CrossEncoderReranker` ([reranker.py](../src/task2_evidence_retrieval/reranker.py)): skor pasangan
  `(claim, "judul: kalimat")`, disimpan di `rerank_score`, lalu diurutkan ulang
- `BM25` ([bm25.py](../src/task2_evidence_retrieval/bm25.py)): BM25 klasik di atas `CountVectorizer`.
  Bobot TF sudah dihitung saat `fit`, jadi query cukup satu perkalian sparse matrix.

Bentuk satu evidence:

```json
{"page_id": "Eiffel_Tower", "line_id": 0, "title": "Eiffel Tower",
 "text": "The Eiffel Tower is a wrought iron lattice tower ... in Paris, France.",
 "doc_score": 41.2, "retrieval_score": 0.71, "rerank_score": 6.3}
```

### 4.3 `NLIVerifier`: [task3_verification/nli.py](../src/task3_verification/nli.py)

```python
verifier = NLIVerifier()                     # DeBERTa-v3-base-mnli-fever-anli, threshold 0.5
verifier.score(claim, evidence)              # → (sentence_probs (n,3), concat_probs (3,) atau None)
verifier.verify(claim, evidence)             # → dict hasil lengkap (lihat bawah)
aggregate(sentence_probs, concat_probs, t)   # → (label, scores), fungsi murni tanpa model
```

- Kolom probabilitas selalu berurutan `(entailment, neutral, contradiction)` = `NLI_KEYS`.
  Urutan label setiap checkpoint berbeda, jadi `__init__` membacanya dari `config.id2label`.
- `aggregate` dipisah dari model supaya `evaluate.py` bisa mencoba banyak threshold tanpa
  menjalankan ulang NLI.
- `evidence` boleh berupa list dict (output Task 2) atau list string biasa.

### 4.4 `FactCheckPipeline`: [pipeline.py](../src/pipeline.py)

```python
from src.pipeline import FactCheckPipeline

pipe = FactCheckPipeline.load()                         # load ketiga model
pipe = FactCheckPipeline.load(use_claim_detection=False) # tanpa Task 1: input dianggap satu klaim
results = pipe.check(raw_text)                          # Task 1 → (2 → 3) per klaim
one = pipe.verify_claim("Some claim.")                  # Task 2 → 3 saja
```

> Pakai `.load()`, jangan `FactCheckPipeline()` kosong. Constructor-nya hanya menerima
> komponen yang sudah jadi (berguna untuk mocking/test). Kalau kosong, `evidence_pipeline`
> dan `verifier` bernilai `None` sehingga `check()` crash.

Alur `check()`:

```
check(text)
 ├─ claim_detector ada?  ya → detector.detect(text)       → [{sentence, p_claim}, ...]
 │                       tidak → [{sentence: text, p_claim: None}]
 └─ untuk setiap klaim → verify_claim(sentence, p_claim)
                           ├─ evidence_pipeline.run(claim, top_k)   (Task 2)
                           └─ verifier.verify(claim, evidence)      (Task 3)
```

Output per klaim:

```json
{
  "claim": "The Eiffel Tower is located in Berlin.",
  "check_worthy_confidence": 0.998,
  "verdict": "REFUTES",
  "verdict_confidence": 0.97,
  "scores": {"SUPPORTS": 0.01, "REFUTES": 0.97, "NOT ENOUGH INFO": 0.03},
  "evidence": [
    {"page_id": "...", "line_id": 0, "title": "...", "text": "...",
     "doc_score": 0, "retrieval_score": 0, "rerank_score": 0,
     "nli": {"entailment": 0.01, "neutral": 0.02, "contradiction": 0.97},
     "nli_label": "contradiction"}
  ]
}
```

`check_worthy_confidence` adalah `p_claim` dari Task 1 (`None` jika claim detection dimatikan).

---

## 5. Script evaluasi & testing

| Perintah | Mengukur | Output |
|---|---|---|
| `python -m src.task1_claim_detection.baseline` | TF-IDF+LogReg di split Task 1 | `models/task1_tfidf-logreg/metrics.json` |
| `python -m src.task1_claim_detection.domain_gap` | % klaim FEVER yang lolos Task 1 (≈94.5%) | `models/.../domain_gap_paper_dev.json` |
| `python -m src.task2_evidence_retrieval.evaluate` | doc recall, evidence recall@5, hit@1, MRR | `outputs/task2/` |
| `python -m src.task3_verification.evaluate` | accuracy, macro F1, FEVER score | `outputs/task3/metrics_retrieved.json` |
| `python -m src.task3_verification.evaluate --oracle-evidence` | sama, tapi pakai gold evidence (batas atas NLI) | `outputs/task3/metrics_oracle.json` |
| `python -m src.run_test_cases` | test case manual di `tests/test_cases.json` | `outputs/test_cases/results.json` |

Semua script punya `--limit N` atau opsi sejenis untuk tes cepat. Cek dengan `--help`.

Format `tests/test_cases.json`:

```json
{
  "claim_detection": [{"text": "Good evening everyone.", "expected": "not_claim", "category": "greeting"}],
  "verification":    [{"claim": "Mogadishu is a book.", "expected": "REFUTES", "note": "opsional"}]
}
```

`run_test_cases.py` memanggil `verify_claim` langsung (pipeline di-load dengan
`use_claim_detection=False`), lalu menghitung `p_claim` secara terpisah. Akibatnya, klaim yang
`p_claim`-nya rendah **tetap** diverifikasi dan tidak dibuang seperti di `check()`.

---

## 6. Hal yang sering membingungkan

- **Corpus hanya subset.** `wiki_subset.jsonl` berisi ~31 ribu halaman (gold page dari 500 klaim
  sampel + hard negative + acak), bukan seluruh Wikipedia. Klaim bebas di luar sampel sering
  tidak menemukan evidence, sehingga hasilnya `NOT ENOUGH INFO` atau melenceng. Yang salah di sini
  corpus-nya, bukan model NLI-nya.
- **Tiga skor berbeda di Task 2.** `doc_score` (BM25, skala bebas), `retrieval_score` (cosine,
  -1..1), dan `rerank_score` (logit cross-encoder, bisa negatif). Ketiganya tidak bisa dibandingkan.
  Urutan final ditentukan oleh `rerank_score`.
- **Dua threshold berbeda.** Task 1: threshold P(claim) (0.60, dari `threshold.json`). Task 3:
  threshold NEI (0.5, `--threshold` di `pipeline.py`).
- **Token PTB.** Teks Wikipedia FEVER memakai token seperti `-LRB-` untuk `(`.
  `clean_text()` di `task2/data.py` mengubahnya kembali. Fungsi ini juga dipakai Task 3 sebelum
  memasukkan teks ke NLI.
- **Lazy loading.** spaCy (Task 1) dan bi-encoder (Task 2) baru di-load saat pertama dipakai,
  jadi panggilan pertama lebih lambat.
- **Device.** Semua komponen otomatis memilih `cuda` → `mps` (Mac) → `cpu`.
