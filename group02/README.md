# Grupa 2: normalizacija vještina i zanimanja (ESCO)

Grupa 1 iz CV-a ili oglasa izvuče vještine onako kako ih je čovjek napisao ("Python", "Pythona",
"Pyhton", "Excel"...). Mi svaki takav pojam povežemo s konceptom iz ESCO-a, EU rječnika vještina i
zanimanja. Tako kandidat i oglas dobiju isti ESCO URI za istu vještinu i Grupa 4 ih može usporediti.

Ulaz je `CandidateProfile` ili `JobProfile`, a izlaz `NormalizedProfile` (shema v1.0,
`schemas/project_schemas.json`). Originalni profil ostaje netaknut pod `profile`, a mi dodajemo
`normalized`. Na primjer, za vještinu "Excel" iz oglasa:

```json
{
  "source_term": "Excel",
  "normalized_label": "Excel",
  "esco_concept": {"uri": "http://data.europa.eu/esco/skill/1973c966-f236-40c9-b2d4-5d71a89019be", "preferred_label": "use spreadsheets software"},
  "mapping": {"status": "RELATED", "method": "embedding_similarity", "confidence": 0.96},
  "modality": "required",
  "evidence": {"text": "Advanced Excel skills are required.", "source_field": "job_text", "char_start": 210, "char_end": 245}
}
```

## Datoteke

| Datoteka | Što radi |
|---|---|
| `normalize.py` | glavni program: čita profile, normalizira vještine i zanimanja, sprema rezultat |
| `esco_loader.py` | učitava ESCO CSV u rječnik, varijante pojma, exact/alias lookup, fuzzy, spaja sve u `map_term()` |
| `esco_embeddings.py` | embedding ESCO naziva (jednom, spremi se na disk) i traženje po značenju |
| `evaluate.py` | mjeri točnost na gold setu |
| `eval/gold_terms.json` | 89 ručno označenih pojmova |
| `eval/results.md` | rezultati evaluacije i zaključci |
| `normalization_demo/` | primjeri ulaza (kandidat i oglas) |

## Instalacija

Potreban je Python 3.11 (s 3.9 kod drugih grupa ne radi). Iz roota repoa:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r group02/requirements.txt
```

Za NVIDIA grafičku (vektori se grade ~5x brže) nakon toga zamijeni torch verzijom s CUDA podrškom:

```powershell
pip install "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126
```

Kod sam bira grafičku ako je dostupna, inače radi na procesoru (ispiše `Embedding model running on cuda` ili `cpu`).

## ESCO podaci

ESCO v1.2.1, engleski, CSV (https://esco.ec.europa.eu/en/use-esco/download). Trebaju nam samo
`skills_en.csv` i `occupations_en.csv`, i stavljaju se u `data/esco/` u rootu repoa.
`data/` je u `.gitignore`, baza se ne commita.

## Izgradnja vektora (jednom)

```powershell
python group02/esco_embeddings.py --skills data/esco/skills_en.csv --occupations data/esco/occupations_en.csv
```

Svaki službeni naziv i sinonim iz ESCO-a (oko 100 000 za vještine, 33 000 za zanimanja) pretvori se u
vektor modelom `sentence-transformers/all-MiniLM-L6-v2` i spremi u `data/esco/embeddings/`.
Na GTX 1650 Ti traje oko 50 s, na procesoru nekoliko minuta. Ovo ne treba pokretati ručno: ako vektora
nema, ili se promijenio CSV ili model, `normalize.py` ih sam izgradi.

## Pokretanje

Iz roota repoa:

```powershell
# jedan profil
python group02/normalize.py --input group02/normalization_demo/candidate.json --skills data/esco/skills_en.csv --occupations data/esco/occupations_en.csv --output data/output/candidate_normalized.json

# vise profila: .jsonl (kako ih sprema Grupa 1), .json s listom ili mapa s .json datotekama
python group02/normalize.py --input output/cv_profiles.jsonl --skills data/esco/skills_en.csv --occupations data/esco/occupations_en.csv --output data/output/candidates_normalized.jsonl
```

- `--output` s nastavkom `.jsonl` sprema profil po retku (to čita i Grupa 3), a `.json` jedan objekt ili listu.
- `--occupations` nije obavezan, ali bez njega zanimanja ostaju UNMAPPED.
- `--no-embeddings` preskače embedding (samo exact i fuzzy). Brže je jer se ne učitava model.

Na kraju se ispiše koliko je pojmova dobilo koji status i kojom metodom.

### Iz Pythona (npr. Grupa 4)

```python
import sys
sys.path.insert(0, "group02")

from esco_embeddings import load_embeddings
from esco_loader import load_esco_index
from normalize import normalize_profile

skills = load_esco_index("data/esco/skills_en.csv")
occupations = load_esco_index("data/esco/occupations_en.csv")
skills_emb = load_embeddings("data/esco/skills_en.csv")
occupations_emb = load_embeddings("data/esco/occupations_en.csv")

result = normalize_profile(profile, skills, occupations, skills_emb, occupations_emb)
```

ESCO i model učitaj jednom pa zovi `normalize_profile` za svaki profil.

## Kako radi

Svaki pojam prolazi kroz korake redom i staje na prvom koji uspije:

| korak | primjer | status / metoda | confidence |
|---|---|---|---|
| 1. exact lookup (službeni naziv) | "SQL" -> SQL | EXACT / `exact_lookup` | 1.0 |
| 1. alias lookup (altLabels, hiddenLabels) | "Python" -> Python (computer programming) | EXACT / `alias_lookup` | 1.0 |
| 2. fuzzy (RapidFuzz) | "Pyhton" -> Python (computer programming) | RELATED / `fuzzy_match` | sličnost / 100 |
| 3. embedding (MiniLM) | "Excel" -> use spreadsheets software | RELATED / `embedding_similarity` | kosinusna sličnost |
| ništa nije prošlo | "Kafka" | UNMAPPED / `none` | 0.0 |

Detalji:

- **Varijante pojma.** Prije exact lookupa probaju se i: bez crtica ("machine-learning"), bez nastavka
  iza crtice ("SQL-a"), bez padežnog nastavka ("Pythona", "Javi") i, samo kod zanimanja, bez riječi
  o razini ("Junior", "Intern", "(m/f)"). Original se uvijek proba prvi.
- **Dvosmisleni pojmovi.** Ako isti naziv ima više ESCO koncepata, a samo je jedan od njih službeni
  naziv, uzima se on ("SQL"). Ako se ne može odlučiti, ide se na sljedeći korak.
- **Fuzzy** preskače pojmove kraće od 4 slova ("C", "R", "Go") i traži isto prvo slovo. Bez toga
  "HTML" postane "PHTML" (PHP).
- **Embedding** uspoređuje pojam sa svim ESCO nazivima i uzme najsličniji, ako je iznad praga.
  Računa se top-5, ali u izlaz ide samo prvi (shema ima jedan `esco_concept`).
- **`normalized_label`**: kod exact i fuzzy je to naziv kako piše u ESCO-u ("Pyhton" -> "Python"),
  a kod embeddinga i UNMAPPED izvorni pojam ("FastAPI"), kao u primjeru iz Ideje projekta.
  Službeni ESCO naziv je uvijek u `esco_concept.preferred_label`.
- **`null` iz Grupe 1** (npr. `target_roles: null`) ne ruši program, tada jednostavno nema zanimanja.

## Pragovi

| | prag | gdje |
|---|---|---|
| fuzzy, vještine | 83 | `FUZZY_CUTOFF` u `esco_loader.py` |
| fuzzy, zanimanja | 92 | `FUZZY_CUTOFF_OCCUPATION` u `esco_loader.py` |
| embedding | 0.78 | `EMBEDDING_CUTOFF` u `esco_embeddings.py` |

Odabrani su na gold setu (`eval/results.md`). Ukratko: "Pyhton" ima točno 83.3 pa je fuzzy prag 83;
zanimanja su duže fraze pa im treba strože ("ML engineer" bi inače postao "metal engineer");
kod embeddinga smeće ide do 0.76 (Docker -> "dock operations"), a dobri pogoci su uglavnom od 0.78.

## Rezultati

Na gold setu od 89 pojmova:

| | mapirano | točno od mapiranih |
|---|---|---|
| vještine | 49/67 | 94% |
| zanimanja | 14/22 | 93% |
| ukupno | 63/89 | 94% |

Kad se pojam mapira, gotovo uvijek je točno; sustav radije ostavi UNMAPPED nego da pogodi krivo.
Cijela analiza, pragovi i usporedba varijanti embeddinga su u `eval/results.md`.

```powershell
python group02/evaluate.py --skills data/esco/skills_en.csv --occupations data/esco/occupations_en.csv
```

## Ograničenja

- **Imena alata.** MiniLM je mali model i ne zna što su Docker, FastAPI ili Kafka. Riječ razbije na
  komade ("dock" + "er") i traži po zvuku, pa takvi pojmovi uglavnom ostaju UNMAPPED.
- **Isti naziv, drugo značenje.** "Spark" se mapira na ESCO "SPARK" (Java framework, ne Apache Spark),
  "transformers" na "electrical machines", "Rust" na uklanjanje hrđe.
- **Zanimanja.** Točan odgovor je skoro uvijek u top-5, ali ne uvijek prvi; rerank (cross-encoder ili
  LLM) bi tu najviše pomogao.
- **Gold set je mali** i pragovi su podešeni na njemu, pa su brojke optimistične.
- Pokrivenost na pravim oglasima (HF tech jobs) još nije izmjerena. Kad dataset bude lokalno,
  vještine u `.txt` (jedna po retku) i `evaluate.py ... --terms putanja.txt`.

