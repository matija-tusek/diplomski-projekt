# Evaluacija normalizacije (Grupa 2)

Gold set: `eval/gold_terms.json`, 89 pojmova (67 vjestina, 22 zanimanja).
Za svaki pojam su rucno upisani ESCO koncepti koje prihvacamo kao tocne. Prazna lista znaci da u ESCO-u
nema nicega dobrog i da je tocan odgovor UNMAPPED.

Izvori pojmova: 45 vjestina iz gold seta Grupe 1 (ukljucujuci hrvatske oblike), pojmovi iz primjera
Grupe 3, tipfeleri, opisne fraze i nekoliko zanimanja.

Pokretanje (iz roota repoa):

```
python group02/evaluate.py --skills data/esco/skills_en.csv --occupations data/esco/occupations_en.csv
```

## Rezultati

Postavke: exact/alias lookup, fuzzy (prag 83, za zanimanja 92), embedding all-MiniLM-L6-v2 nad
sluzbenim nazivima i sinonimima (prag 0.78).

| | mapirano | tocno od mapiranih | ukupno tocno |
|---|---|---|---|
| vjestine | 49/67 | 46/49 (94%) | 49/67 (73%) |
| zanimanja | 14/22 | 13/14 (93%) | 13/22 (59%) |
| **sve** | **63/89** | **59/63 (94%)** | **62/89 (70%)** |

"Ukupno tocno" broji i pojmove koji su ispravno ostali UNMAPPED.

Po metodi:

| metoda | broj | tocno | krivo |
|---|---|---|---|
| exact_lookup | 18 | 17 | 1 |
| alias_lookup | 20 | 19 | 1 |
| fuzzy_match | 9 | 9 | 0 |
| embedding_similarity | 16 | 14 | 2 |
| none (UNMAPPED) | 26 | 3 ispravno | 23 promasaja |

## Odabir pragova

Embedding:

| prag | mapirano | preciznost | ukupno tocno |
|---|---|---|---|
| 0.70 | 73/89 | 88% | 74% |
| 0.75 | 68/89 | 90% | 72% |
| **0.78** | 63/89 | **94%** | 70% |
| 0.80 | 62/89 | 94% | 69% |
| 0.85 | 58/89 | 95% | 65% |

Fuzzy (vjestine):

| prag | mapirano | preciznost | ukupno tocno |
|---|---|---|---|
| 80 | 64/89 | 92% | 70% |
| **83** | 63/89 | **94%** | 70% |
| 85 | 62/89 | 94% | 69% |
| 88 | 62/89 | 92% | 67% |

Odabrano: embedding 0.78, fuzzy 83 (za zanimanja 92).
Nizi embedding prag mapira vise pojmova, ali je vecina dodatnih pogodaka kriva. Kriv ESCO koncept
je gori od UNMAPPED, jer se kod UNMAPPED cuva izvorni pojam i Grupe 3 i 4 koriste njega.

## Odabir onoga sto se embeda

| varijanta | mapirano | preciznost | ukupno tocno | recall@1 / @5 vjestine | recall@1 / @5 zanimanja |
|---|---|---|---|---|---|
| samo sluzbeni naziv | 57/89 | 95% | 64% | 56% / 61% | 82% / 91% |
| **sluzbeni naziv + sinonimi** | 63/89 | 94% | 70% | 59% / 72% | 73% / 100% |
| naziv + sinonimi + opis | 63/89 | 94% | 70% | 58% / 72% | 82% / 100% |

recall@k gleda samo embedding (bez exact i fuzzy): je li tocan odgovor prvi, odnosno barem u top-5.

Odabrano: sluzbeni naziv + sinonimi. Sinonimi donose 6 tocnih pogodaka vise. Opisi ne mijenjaju
konacni rezultat, a cache je veci i gradi se dulje. Ocekivali smo da ce opisi pomoci za Docker i
Kubernetes (opis koncepta "manage ICT virtualisation environments" ih spominje), ali jedna rijec u
dugom opisu se izgubi i model i dalje bira kratke nazive koji zvuce slicno ("dock operations").

## Greske

Krivo mapirano (4):
- Spark -> SPARK (exact): u ESCO-u je to Java framework, a ne Apache Spark
- transformers -> electrical machines (alias): u ML kontekstu se misli na modele
- Rust -> remove rust from motor vehicles (embedding): programski jezik vs hrda
- Backend developer -> user interface developer (embedding): to je frontend

Promaseno (23):
- imena alata koja MiniLM ne poznaje: Docker, Kubernetes, FastAPI, Terraform, pandas, Hugging Face,
  Jest, Firebase. Model ih razbija na komade ("dock" + "er") i trazi po zvuku, ne po znacenju.
- kratki ili viseznacni pojmovi: C, Go
- zanimanja s kraticama ili bez jasnog naziva posla: QA tester, ML engineer, Machine Learning Intern,
  NLP Engineer Intern, Product Owner, Full Stack Developer. Tocan odgovor je uglavnom u top-5, ali
  ispod praga.

## Zakljucci

1. Kad se pojam mapira, gotovo uvijek je tocno (94%). Sustav radije ostavi UNMAPPED nego da pogodi krivo.
2. Exact i fuzzy su pouzdani. Greske exact lookupa (Spark, transformers) dolaze od istog naziva s
   drugim znacenjem u ESCO-u.
3. Embedding dobro radi za opisne fraze (Excel, HTML, "unit testing", "data visualization"),
   a lose za imena proizvoda.
4. Kod zanimanja je tocan odgovor u top-5 u 100% slucajeva, a prvi u 73%. Rerank top-5 kandidata
   (cross-encoder ili LLM) bi tu najvise pomogao.
5. Kod vjestina je tocan odgovor u top-5 u 72% slucajeva, pa rerank pomaze manje. Za imena alata bi
   trebao jaci model ili kratki rucni popis sinonima za najcesce alate.

