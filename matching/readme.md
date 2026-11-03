# Pokretanje
Pokreće se iz terminala u direktoriju diplomski-projekt naredbom: python -m matching.recommend

# Trenutno stanje
Trenutno je napisan samo osnovni kostur (pipeline) koji se sastoji od računanja featura, računanja konačnog rezultata i stvaranja opisa llm-om.
Zasad se koristi samo semantic_similarity za konacni score. Čitaju se mock podaci u mapi mock_data.

# Treba napraviti
Potrebno je napisati metode koje sadrže TODO (cijelo računanje featura i scora), dodati filtriranje po hard constraintovima, prilagoditi recommend datoteku, optimizirati prompt za llm, ...
Opis na hrvatskom jeziku miješa se sa srpskim jezikom.


## Prijedlog za tim
Dodati `matched_required_skills` i `matched_preferred_skills` u `features` (uz `missing_*`) kako bi LLM mogao navesti vještine koje kandidat ima. Zahtijeva promjenu sheme, pa treba dogovor.