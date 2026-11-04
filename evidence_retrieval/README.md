# Evidence Retrieval & Reranking

Modul ini mencari kalimat evidence dari Wikipedia untuk sebuah claim, lalu
mengurutkan ulang (rerank) supaya kalimat yang paling relevan ada di atas.
Output-nya dipakai oleh modul stance detection / NLI.

## Alur

```
claim
  -> [1] Document retrieval : BM25 (judul + isi halaman)       -> top-5 halaman
  -> [2] Sentence retrieval : bi-encoder all-MiniLM-L6-v2      -> top-30 kalimat
  -> [3] Reranking          : cross-encoder ms-marco-MiniLM-L-6-v2 -> top-5 kalimat
```

| File | Isi |
|---|---|
| `data.py` | download & baca file FEVER (claim + wiki-pages.zip) |
| `build_corpus.py` | bikin subset corpus Wikipedia untuk eksperimen |
| `bm25.py` | implementasi BM25 pakai sparse matrix |
| `retriever.py` | tahap [1] dan [2] |
| `reranker.py` | tahap [3] |
| `pipeline.py` | gabungan semuanya, ini yang dipanggil saat integrasi |
| `evaluate.py` | evaluasi di FEVER (sebelum vs sesudah rerank) |
| `sample_output.json` | contoh output untuk mocking di modul lain |

## Dataset

Dataset HuggingFace [`fever/fever`](https://huggingface.co/datasets/fever/fever)
isinya cuma script `fever.py` yang download file dari
[fever.ai](https://fever.ai/dataset/fever.html). Library `datasets` versi baru
sudah tidak bisa menjalankan dataset berbasis script, jadi kita download langsung
file yang sama dari `https://fever.ai/download/fever/`:

- `shared_task_dev.jsonl` (4 MB): claim dev berlabel, dipakai untuk evaluasi
- `wiki-pages.zip` (1.7 GB): dump Wikipedia 5.4 juta halaman, dibaca langsung dari zip tanpa di-extract

Semua file disimpan di `data/` (sudah di-`.gitignore`).

### Subset corpus

Index 5.4 juta halaman terlalu berat untuk laptop, jadi `build_corpus.py`
membuat subset berisi:

1. **gold pages**: halaman evidence dari claim sampel
2. **hard negatives**: halaman yang judulnya muncul di claim
   (contoh: claim soal "Soul Food" ikut menarik "Soul Food (TV series)", "Soul Food (soundtrack)", dll)
3. **random pages**: halaman acak sebagai noise

Hal yang perlu diingat:
- Hard negatives bergantung pada claim yang di-sample, jadi evaluasi harus
  pakai `claims_sample.jsonl` yang dibuat bersamaan dengan corpus-nya.
- Angka recall di subset ini **lebih optimis** dibanding kalau pakai full corpus,
  karena jumlah pengecohnya jauh lebih sedikit.
- Claim `NOT ENOUGH INFO` tidak dipakai untuk evaluasi karena tidak punya evidence.

## Cara menjalankan

```bash
pip install -r requirements.txt

# 1. bikin subset corpus, cukup sekali
#    (download wiki-pages.zip ~20 menit tergantung internet, scan-nya sendiri ~1-2 menit)
python -m evidence_retrieval.build_corpus --n-claims 500 --n-random 20000

# 2. evaluasi
python -m evidence_retrieval.evaluate --limit 50   # tes cepat
python -m evidence_retrieval.evaluate              # semua 500 claim

# 3. coba claim sendiri
python -m evidence_retrieval.pipeline "The principal photography of The Disaster Artist (film) started on the 9th."
```

## Pemakaian di kode (untuk integrasi)

```python
from evidence_retrieval.pipeline import EvidencePipeline

pipe = EvidencePipeline.from_corpus_file()   # default: data/processed/wiki_subset.jsonl
evidence = pipe.run("The principal photography of The Disaster Artist (film) started on the 9th.", top_k=5)

for ev in evidence:
    print(ev["title"], ev["line_id"], ev["text"], ev["rerank_score"])
```

Format output `run()` (list of dict, urut dari paling relevan):

```json
{
  "page_id": "The_Disaster_Artist_-LRB-film-RRB-",
  "line_id": 5,
  "title": "The Disaster Artist (film)",
  "text": "Principal photography of the film began on December 8 , 2015 .",
  "doc_score": 55.47,
  "retrieval_score": 0.87,
  "rerank_score": 7.65
}
```

Retriever tidak terikat ke FEVER. Corpus lain bisa dipakai asalkan formatnya
list of dict `{"id", "title", "sentences": [[line_id, text], ...]}`:

```python
from evidence_retrieval.retriever import EvidenceRetriever
from evidence_retrieval.reranker import CrossEncoderReranker
from evidence_retrieval.pipeline import EvidencePipeline

pipe = EvidencePipeline(EvidenceRetriever(my_pages), CrossEncoderReranker())
```

## Metrik evaluasi

- **Doc recall@k**: semua halaman di salah satu evidence set masuk top-k dokumen BM25
- **Candidate recall**: evidence set lengkap ada di 30 kandidat kalimat (batas atas reranker)
- **Evidence recall@5**: metrik resmi FEVER, minimal satu evidence set lengkap ada di top-5 kalimat
- **Hit@1**: kalimat ranking pertama adalah gold evidence
- **MRR**: rata-rata 1/ranking gold evidence pertama

## Hasil (500 claim dari `shared_task_dev`, seed 42)

Subset corpus: 30.997 halaman (476 gold, 10.808 hard negative, 19.713 random).
Jalan di MacBook (MPS), sekitar 0.2 detik per claim.

| Tahap | Metrik | Sebelum rerank | Sesudah rerank |
|---|---|---|---|
| BM25 | Doc recall@1 | 0.646 | |
| BM25 | Doc recall@5 | 0.900 | |
| BM25 | Doc recall@10 | 0.922 | |
| Kandidat (top-30) | Candidate recall | 0.900 | |
| Kalimat | Evidence recall@5 | 0.816 | **0.884** |
| Kalimat | Hit@1 | 0.568 | **0.796** |
| Kalimat | MRR | 0.706 | **0.863** |

Catatan: dari top-5 halaman rata-rata ada ~39 kalimat per claim (66% claim punya
lebih dari 30), lalu bi-encoder memotongnya jadi 30 kandidat. Candidate recall sama
dengan doc recall@5, artinya pemotongan ini tidak pernah membuang gold evidence.

Reranker menaikkan Hit@1 sekitar 23 poin. Evidence recall@5 sesudah rerank (0.884)
sudah dekat dengan batas atasnya (0.900), jadi bottleneck sekarang ada di tahap
document retrieval. Ingat, angka ini di subset corpus, jadi lebih optimis
dibanding full Wikipedia.

## Batasan

- Corpus hanya berisi halaman yang terkait dengan 500 claim sampel + halaman acak.
  Claim di luar sampel kemungkinan besar **tidak akan ketemu evidence-nya**.
  Contoh: "Telemundo is a English-language television network." tidak ada di sampel,
  jadi halaman Telemundo tidak ada di corpus dan hasilnya meleset. Untuk demo dengan
  claim bebas, perlu corpus yang lebih besar (misalnya subset dengan `--n-claims`
  lebih banyak, atau full wiki + index yang lebih efisien seperti Pyserini/FAISS).
- Model yang dipakai berbahasa Inggris (sama seperti FEVER). Kalau claim dari modul
  claim classification berbahasa Indonesia, perlu diterjemahkan dulu atau ganti
  ke model multilingual (mis. `paraphrase-multilingual-MiniLM-L12-v2`).
- Cross-encoder `ms-marco` dilatih untuk relevansi query-passage, bukan khusus
  fact-checking. Fine-tuning di data train FEVER bisa jadi pengembangan selanjutnya.
