# Poetski atlas – AI aplikacija

Ovo je prototip aplikacije za semi-automatizovano kreiranje poetskog atlasa.
Aplikacija omogućava:

- unos pesme ili proznog odlomka;
- izbor LLM provajdera i modela;
- klik na **Kreiraj poetski atlas**;
- prikaz poetskih elemenata: prostor, vreme, likovi/glasovi, motivi, umetnički postupci i atmosfera;
- fotorealističnu ilustraciju i njenu interpretaciju (samo OpenAI / GPT);
- animaciju kao animirani GIF i njenu interpretaciju (samo OpenAI / GPT);
- JSON izlaz pogodan za dalje čuvanje, korekciju ili poređenje modela.

## Pokretanje lokalno

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
python app.py
```

Zatim otvorite lokalni Gradio link koji se pojavi u terminalu.

## API ključevi

Aplikacija može da radi u `Demo/offline` režimu bez ključeva. Za prave modele dodajte odgovarajući ključ u `.env`:

- `OPENAI_API_KEY` za OpenAI / GPT;
- `ANTHROPIC_API_KEY` za Anthropic / Claude;
- `GOOGLE_API_KEY` ili `GEMINI_API_KEY` za Gemini;
- `PERPLEXITY_API_KEY` za Perplexity;
- `XAI_API_KEY` za Grok / xAI;
- `OLLAMA_URL` za lokalni Meta/Llama režim preko Ollama.

## Hugging Face Spaces

1. Napravite novi Space sa SDK opcijom **Gradio**.
2. Uploadujte fajlove `app.py`, `requirements.txt` i po želji `.env.example`.
3. U **Settings → Secrets** dodajte API ključeve.
4. Space će automatski pokrenuti `app.py`.

## Napomena o ilustraciji i animaciji

Ilustraciju i animaciju generiše samo OpenAI / GPT, modelom za slike GPT Image 2.6 (`gpt-image-2.6-sunburst`; drugi model, npr. brži `gpt-image-2.6-flare`, može se zadati promenljivom `OPENAI_IMAGE_MODEL` u `.env`). Ilustracija je fotorealistična slika (JPEG), a animacija animirani GIF koji se ponavlja. Obe nastaju samo iz elemenata teksta, a zatim ih izabrani GPT model interpretira na osnovu same slike, odnosno kadrova animacije.

Anthropic / Claude ne generiše slike, pa za njega aplikacija daje samo elemente teksta i njihovu interpretaciju; kartice Ilustracija i Animacija to i navode. To je ograničenje tog provajdera.

Slike se čuvaju u JSON-u atlasa (base64), pa je sačuvani atlas veći nego ranije. Atlasi sačuvani ranije, sa SVG crtežima, i dalje se mogu učitati i prikazati.

## Ideja proširenja

Za naučnu verziju preporučuje se dodavanje:

- uporednog prikaza više modela za isti tekst;
- ručne validacije i korekcije elemenata;
- izvoza u CSV/JSON/RDF;
- čuvanja rezultata po tekstu, modelu i datumu;
- posebnog modula za arhaizme i nestandardne jezičke oblike.
