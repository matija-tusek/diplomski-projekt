# Grupa 1: demo ekstrakcije (Ollama + Qwen3-4B-Instruct)

Ovaj demo pokazuje osnovnu ideju naše grupe: **lokalni LLM čita sirovi tekst (životopis ili
oglas za posao) i vraća strogo strukturirani JSON**. Uz svaki izvučeni podatak vraća i **dokaz
(evidence)**, odnosno rečenicu iz originalnog teksta iz koje je podatak izvučen.

Izlaz je pojednostavljena verzija objekata `CandidateProfile` i `JobProfile` iz zajedničke
sheme v1.0 (`project_schemas.json`).

---

## 1. Kako koristiti

### 1.1 Instalacija (jednom)

Potrebno: Windows, Python 3.10+ i oko 3 GB slobodnog prostora za model.

```powershell
# 1) Ollama: pokreće LLM-ove lokalno (radi u pozadini na http://localhost:11434)
winget install --id Ollama.Ollama -e
#    ako winget zapne, preuzmi instalaciju s https://ollama.com/download

# 2) Model Qwen3-4B-Instruct (Q4, ~2.5 GB)
ollama pull qwen3:4b-instruct

# 3) Python okruženje (u mapi extraction_demo)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Provjera da sve radi:

```powershell
ollama list        # u popisu mora biti qwen3:4b-instruct
```

### 1.2 Pokretanje

```powershell
# životopis -> CandidateProfile
.\.venv\Scripts\python.exe extract.py cv samples/cv.txt

# oglas -> JobProfile
.\.venv\Scripts\python.exe extract.py job samples/job.txt
```

Skripta ispisuje JSON, sprema ga u `output/<ime_datoteke>.json` i na kraju ispisuje
provjeru dokaza:

```
Evidence found verbatim in source: 12/13
Saved to output\cv.json
```

### 1.3 Vlastiti tekst

Spremi životopis ili oglas kao običnu `.txt` datoteku (UTF-8) i proslijedi njezinu putanju:

```powershell
.\.venv\Scripts\python.exe extract.py job moji_oglasi/oglas1.txt
```

Prvi argument je `cv` ili `job` i određuje koja se shema i koji prompt koriste.

### 1.4 Česti problemi

| Problem | Rješenje |
|---------|----------|
| `Failed to connect to Ollama` | Ollama ne radi. Pokreni aplikaciju Ollama iz Start izbornika ili `ollama serve`. |
| `model "qwen3:4b-instruct" not found` | Pokreni `ollama pull qwen3:4b-instruct`. |
| `ollama` nije prepoznata naredba | Otvori novi terminal (PATH se osvježi tek nakon instalacije) ili koristi `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`. |
| Prvi poziv traje dugo (~1,5 min) | Normalno: model se tada učitava u memoriju. Sljedeći pozivi traju ~15 s. |

---

## 2. Kako radi

```
 tekst (.txt) ──► prompt + JSON shema ──► Ollama / Qwen3-4B ──► JSON
                                                                 │
                       Pydantic validacija ◄─────────────────────┘
                               │
                 add_offsets(): traži svaki citat u tekstu ──► char_start / char_end
                               │
                         output/<ime>.json + "Evidence found: X/Y"
```

### Korak 1: Shema se definira Pydantic modelima

U `extract.py` je izgled izlaza opisan Python klasama (`CandidateProfile`, `JobProfile`,
`Skill`, `Language`, `Education`...). Na primjer:

```python
class JobSkill(Skill):          # name + evidence
    modality: Modality          # "required" | "preferred" | "unspecified"
```

Dozvoljene vrijednosti (enumi iz zajedničke sheme) zadane su kao `Literal[...]`, a polja koja
mogu nedostajati kao `X | None`. Pydantic iz toga automatski generira **JSON Schemu**
(`Model.model_json_schema()`).

### Korak 2: Ollama prisiljava model da poštuje shemu (structured output)

```python
ollama.chat(
    model="qwen3:4b-instruct",
    messages=[{"role": "system", "content": prompt},
              {"role": "user", "content": text}],
    format=schema.model_json_schema(),   # <- ovdje predajemo shemu
    options={"temperature": 0},          # deterministički izlaz
)
```

Kad se preda `format`, Ollama pretvara JSON shemu u gramatiku. Pri generiranju svakog tokena
dopušteni su samo tokeni koji ne krše tu gramatiku. Zato model **fizički ne može** vratiti
neispravan JSON, izostaviti obavezno polje ili napisati npr. `"modality": "mandatory"`. To ne
znači da je sadržaj točan, samo da je oblik ispravan.

`temperature: 0` znači da model uvijek bira najvjerojatniji token, pa isti ulaz daje (gotovo)
isti izlaz. To je važno za testiranje.

### Korak 3: Prompt

Sistemski prompt daje pravila koja shema ne može izraziti:

- izvuci samo ono što piše u tekstu; ako nešto ne piše, stavi `null` i **nikad ne pogađaj**
  (pravilo iz sheme: *unknown is not false*);
- `evidence.text` mora biti **doslovno prepisan** iz dokumenta;
- vještine se pišu kako su navedene (npr. "Python", "Docker"), bez normalizacije.
  Normalizacija i ESCO su posao Grupe 2.
- (samo oglasi) `required` za "must / required / strong", `preferred` za
  "a plus / an advantage / nice to have", a inače `unspecified`.

### Korak 4: Validacija

`schema.model_validate_json(...)` parsira odgovor i još jednom provjerava da odgovara modelu.
Ako ne odgovara, dobijemo jasnu grešku umjesto pokvarenih podataka koji bi tek kasnije
srušili Grupu 2.

### Korak 5: Pozicije dokaza računa kod, a ne LLM

Shema traži `char_start` i `char_end` za svaki dokaz. **LLM-ovi ne znaju pouzdano brojati
znakove**, pa od modela tražimo samo citat, a pozicije računa funkcija `add_offsets()`:

```python
start = source.find(ev["text"])
ev["char_start"] = start if start >= 0 else None
ev["char_end"]   = start + len(ev["text"]) if start >= 0 else None
```

Ako se citat **ne nađe doslovno** u tekstu, pozicije su `null`. To je znak da je model
parafrazirao ili izmislio dokaz. Na kraju skripta ispisuje koliko je citata pronađeno
(`12/13`), što je prva, gruba mjera točnosti ekstrakcije.

---

## 3. Što smo dobili na primjerima (30. 10. 2026.)

**Oglas (`samples/job.txt`):** 7/7 dokaza pronađeno. Svi modaliteti su točni: Python i SQL
`required`, Docker, AWS i njemački `preferred`. Razina njemačkog je ispravno `null` jer je oglas
ne navodi. **Propušteno:** scikit-learn (iz "pandas and scikit-learn" izvučen je samo pandas).

**Životopis (`samples/cv.txt`):** 12/13 dokaza pronađeno. Provjera je uhvatila stvarnu
grešku: model je napisao "German (A-2)" umjesto "(A2)". **Pogrešno:** `field` = "Master's"
umjesto "Computer Science". **Propušteno:** vještine koje se spominju samo u opisu iskustva
(FastAPI, C).

**Brzina (RTX 3050, 4 GB VRAM):** model ne stane cijeli u VRAM (34 % CPU / 66 % GPU). Prvi
poziv traje ~87 s zajedno s učitavanjem, a sljedeći ~16 s.

Iz ovoga slijede sljedeći koraci:

- dodati u prompt "izvuci vještine i iz opisa iskustva";
- dodati opise polja u shemu (`Field(description=...)`, npr. da je *field* područje studija,
  a ne razina);
- napraviti mali ručno označeni skup (gold set) i mjeriti preciznost i odziv po polju.

---

## 4. Datoteke

| Datoteka | Sadržaj |
|----------|---------|
| `extract.py` | sheme (Pydantic), promptovi, poziv Ollami, računanje pozicija dokaza |
| `samples/cv.txt` | izmišljeni životopis studentice za testiranje |
| `samples/job.txt` | izmišljeni oglas s mješavinom required / preferred / nenavedenih uvjeta |
| `requirements.txt` | Python paketi: `ollama`, `pydantic` |
| `output/` | generirani JSON-ovi (ne ide u git) |
| `README.md` | kratke upute na engleskom |
