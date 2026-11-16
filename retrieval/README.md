# Grupa 3: Retrieval

Za svakog kandidata vraća top k semantički najsličnijih oglasa kao `RetrievalResult`.

```
retrieval/
  text_builder.py   NormalizedProfile -> tekst za encoder (ESCO labele)
  index.py          JobIndex: gradi, sprema i učitava FAISS indeks oglasa
  retrieve.py       retrieve(): kandidat -> RetrievalResult + provjera formata
  io_utils.py       učitavanje profila iz .json / .jsonl / mape
  __main__.py       komandna linija
  examples/         primjeri NormalizedProfile (2 kandidata, 7 oglasa)
```

## Pokretanje (iz korijena repoa)

```
pip install faiss-cpu sentence-transformers

python -m retrieval build --jobs retrieval/examples/jobs.json --index faiss_index
python -m retrieval query --candidates retrieval/examples/candidates.json --index faiss_index --k 50 --out results.json
```

`build` se pokreće jednom (i kad se promijene oglasi ili model), `query` koliko god puta.
Opcije: `--encoder <model>` za drugi Sentence Transformer, `--raw-terms` za sirove pojmove umjesto ESCO labela.
Mapa `faiss_index/` je u `.gitignore` i ne commita se.

## Iz Pythona (za Grupu 4)

```python
from retrieval import JobIndex, retrieve
from retrieval.io_utils import load_profiles

job_index = JobIndex.load("faiss_index")
candidate = load_profiles("retrieval/examples/candidates.json")[0]
result = retrieve(candidate, job_index, k=50)
```

## Odluke

- Vještine i zanimanja: `esco_concept.preferred_label`, a za UNMAPPED izvorni `source_term`.
- Lokacija, remote mode i jezici nisu u tekstu za embedding: to su tvrdi uvjeti koje provjerava Grupa 4.
- `k` u izlazu je stvarni broj vraćenih oglasa (manji od traženog ako je oglasa manje).
