# Poetic Atlas – AI Application

This is a prototype application for the semi-automated creation of a poetic atlas.
The application provides:

- input of a poem or a prose passage;
- a choice of LLM provider and model;
- a click on **Create Poetic Atlas**;
- a display of the poetic elements: space, time, characters/voices, motifs, artistic devices and atmosphere;
- a photorealistic illustration and its interpretation (OpenAI / GPT only);
- an animation as an animated GIF and its interpretation (OpenAI / GPT only);
- JSON output suitable for further storage, correction or comparison of models.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
python app.py
```

Then open the local Gradio link that appears in the terminal.

## API keys

The application can run in `Demo/offline` mode without keys. For real models, add the corresponding key to `.env`:

- `OPENAI_API_KEY` for OpenAI / GPT;
- `ANTHROPIC_API_KEY` for Anthropic / Claude;
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` for Gemini;
- `PERPLEXITY_API_KEY` for Perplexity;
- `XAI_API_KEY` for Grok / xAI;
- `OLLAMA_URL` for the local Meta/Llama mode via Ollama.

## Hugging Face Spaces

1. Create a new Space with the **Gradio** SDK option.
2. Upload the files `app.py`, `requirements.txt` and, optionally, `.env.example`.
3. Add the API keys under **Settings → Secrets**.
4. The Space will run `app.py` automatically.

## Note on the illustration and the animation

The illustration and the animation are generated only by OpenAI / GPT, with the image model GPT Image 2.6 (`gpt-image-2.6-sunburst`; another model, e.g. the faster `gpt-image-2.6-flare`, can be set with the `OPENAI_IMAGE_MODEL` variable in `.env`). The illustration is a photorealistic image (JPEG), and the animation is a looping animated GIF. Both are created from the elements of the text alone, and the selected GPT model then interprets them from the image itself, or from the frames of the animation.

Anthropic / Claude does not generate images, so with it the application provides only the elements of the text and their interpretation; the Illustration and Animation tabs say so. This is a limitation of that provider.

The images are stored in the atlas JSON (base64), so a saved atlas is larger than before. Atlases saved earlier, with SVG drawings, can still be loaded and displayed.

## Ideas for extension

For a scholarly version, it is recommended to add:

- a side-by-side comparison of several models for the same text;
- manual validation and correction of the elements;
- export to CSV/JSON/RDF;
- storage of results by text, model and date;
- a dedicated module for archaisms and non-standard linguistic forms.
