"""Poetic Atlas – Gradio application.

The application lets the user enter a literary text, choose an LLM provider/model,
and create a structured poetic atlas in three steps: the basic elements of the
text and their interpretation; an illustration and an animation drawn from those
elements; and the interpretations of the finished illustration and animation.

The illustration (a photorealistic image) and the animation (an animated GIF)
are generated only with OpenAI, by its image model (GPT Image 2.6). The other
providers identify and interpret the elements of the text only.

Supported modes:
- Demo/offline: works without API keys, for checking the interface.
- OpenAI/GPT: uses OPENAI_API_KEY (also for the illustration and the animation).
- Anthropic/Claude: uses ANTHROPIC_API_KEY; no illustration or animation.
- Gemini: uses GOOGLE_API_KEY or GEMINI_API_KEY.
- Perplexity: uses PERPLEXITY_API_KEY.
- Grok/xAI: uses XAI_API_KEY.
- Meta/Llama via Ollama: uses a local Ollama server.
"""

from __future__ import annotations

import base64
import copy
import html
import io
import json
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import requests
from dotenv import load_dotenv

# override=True: .env values take precedence over environment variables that
# are already set (e.g. an old OPENAI_API_KEY from the shell or HF Secrets).
load_dotenv(override=True)

warnings.filterwarnings(
    "ignore",
    message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*",
)


APP_TITLE = "Poetic Atlas – AI Application"
DEFAULT_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.3"))
MAX_TEXT_CHARS = 14000


# Languages the atlas can be written in: the name shown in the UI, and how the
# language is named in the instructions to the model.
LANGUAGES = {
    "English": "English",
    "Serbian": "Serbian, in the Latin script",
}

ELEMENT_KEYS = (
    "space",
    "time",
    "characters_or_voices",
    "motifs",
    "artistic_procedures",
    "atmosphere",
)

# Headings of the atlas in each of those languages.
HEADINGS = {
    "English": {
        "elements": "Basic elements",
        "space": "Space",
        "time": "Time",
        "characters_or_voices": "Characters",
        "motifs": "Main motifs",
        "artistic_procedures": "Artistic devices",
        "atmosphere": "Dominant atmosphere",
        "elements_interpretation": "Interpretation of the basic elements",
        "illustration": "Interpretation of the illustration",
        "animation": "Interpretation of the animation",
    },
    "Serbian": {
        "elements": "Osnovni elementi",
        "space": "Prostor",
        "time": "Vreme",
        "characters_or_voices": "Likovi",
        "motifs": "Glavni motivi",
        "artistic_procedures": "Umetnički postupci",
        "atmosphere": "Dominantna atmosfera",
        "elements_interpretation": "Interpretacija osnovnih elemenata",
        "illustration": "Interpretacija ilustracije",
        "animation": "Interpretacija animacije",
    },
}

# Messages of the results area (status, tab names, buttons, notices) in each of
# those languages.
MESSAGES = {
    "English": {
        "tab_elements": "Basic elements",
        "tab_illustration": "Illustration",
        "tab_animation": "Animation",
        "tab_json": "JSON",
        "json_label": "Structured result",
        "regenerate_illustration": "Regenerate illustration",
        "regenerate_animation": "Regenerate animation",
        "enter_text": "Enter a text to analyze.",
        "no_text": "No text entered.",
        "error": "Error creating the atlas.",
        "error_help": (
            "An error occurred: {err}\n\n"
            "Please check:\n"
            "- that the appropriate provider is selected;\n"
            "- that the model name is entered correctly;\n"
            "- that the API key has been added to the .env file or Hugging Face Secrets;\n"
            "- that the model returns valid JSON."
        ),
        "step_0": "Identifying the elements of the text…",
        "step_1": (
            "Step 1 of 3 done: the elements are identified. "
            "Generating the illustration and the animation…"
        ),
        "step_2": (
            "Step 2 of 3 done: the illustration and the animation are generated. "
            "Writing their interpretations…"
        ),
        "done": "The atlas was created successfully.",
        "done_partly": (
            "The elements were identified, but not everything could be generated "
            "(see the Illustration and Animation tabs)."
        ),
        "done_no_drawings": (
            "The atlas was created: the elements are identified. {provider} does not "
            "generate images, so the illustration and the animation are left out."
        ),
        "truncated": " Note: only the first {n} characters of the text were analyzed.",
        "generating_illustration": "Generating the illustration…",
        "generating_animation": "Generating the animation…",
        "writing": "Writing the interpretation…",
        "drawn_illustration": "Illustration generated by {model}.",
        "drawn_animation": "Animation generated by {model}.",
        "failed_illustration": "The illustration was not generated: {err}",
        "failed_animation": "The animation was not generated: {err}",
        "missing_illustration": "The image model did not return the illustration.",
        "missing_animation": "The image model did not return the animation.",
        "unsupported_illustration": (
            "{provider} does not generate images, so the atlas has no illustration. "
            "Choose OpenAI / GPT to generate it."
        ),
        "unsupported_animation": (
            "{provider} does not generate images, so the atlas has no animation. "
            "Choose OpenAI / GPT to generate it."
        ),
        "failed_interpretation": "The interpretation was not generated: {err}",
        "no_atlas_selected": "Choose an existing atlas to load.",
        "loaded": "Loaded the existing atlas: {name}.",
        "load_error": "The atlas could not be loaded: {err}",
    },
    "Serbian": {
        "tab_elements": "Osnovni elementi",
        "tab_illustration": "Ilustracija",
        "tab_animation": "Animacija",
        "tab_json": "JSON",
        "json_label": "Strukturisani rezultat",
        "regenerate_illustration": "Ponovo generiši ilustraciju",
        "regenerate_animation": "Ponovo generiši animaciju",
        "enter_text": "Unesite tekst za analizu.",
        "no_text": "Nema unetog teksta.",
        "error": "Greška pri kreiranju atlasa.",
        "error_help": (
            "Došlo je do greške: {err}\n\n"
            "Proverite:\n"
            "- da li je izabran odgovarajući provajder;\n"
            "- da li je upisan ispravan naziv modela;\n"
            "- da li je API ključ dodat u .env fajl ili Hugging Face Secrets;\n"
            "- da li model vraća validan JSON."
        ),
        "step_0": "Određuju se elementi teksta…",
        "step_1": (
            "Korak 1 od 3 je završen: elementi su određeni. "
            "Generišu se ilustracija i animacija…"
        ),
        "step_2": (
            "Korak 2 od 3 je završen: ilustracija i animacija su generisane. "
            "Pišu se njihove interpretacije…"
        ),
        "done": "Atlas je uspešno kreiran.",
        "done_partly": (
            "Elementi su određeni, ali nije sve moglo da se generiše "
            "(pogledajte kartice Ilustracija i Animacija)."
        ),
        "done_no_drawings": (
            "Atlas je kreiran: elementi su određeni. {provider} ne generiše slike, "
            "pa ilustracija i animacija nisu generisane."
        ),
        "truncated": " Napomena: analizirano je samo prvih {n} znakova teksta.",
        "generating_illustration": "Ilustracija se generiše…",
        "generating_animation": "Animacija se generiše…",
        "writing": "Interpretacija se piše…",
        "drawn_illustration": "Ilustraciju je generisao model {model}.",
        "drawn_animation": "Animaciju je generisao model {model}.",
        "failed_illustration": "Ilustracija nije generisana: {err}",
        "failed_animation": "Animacija nije generisana: {err}",
        "missing_illustration": "Model za slike nije vratio ilustraciju.",
        "missing_animation": "Model za slike nije vratio animaciju.",
        "unsupported_illustration": (
            "{provider} ne generiše slike, pa atlas nema ilustraciju. "
            "Izaberite OpenAI / GPT da biste je generisali."
        ),
        "unsupported_animation": (
            "{provider} ne generiše slike, pa atlas nema animaciju. "
            "Izaberite OpenAI / GPT da biste je generisali."
        ),
        "failed_interpretation": "Interpretacija nije generisana: {err}",
        "no_atlas_selected": "Izaberite postojeći atlas za učitavanje.",
        "loaded": "Učitan je postojeći atlas: {name}.",
        "load_error": "Atlas nije mogao da se učita: {err}",
    },
}

PROVIDER_DEFAULT_MODELS = {
    # "Demo/offline": "demo-poetic-atlas",
    "OpenAI / GPT": "gpt-6.1-sol",
    "Anthropic / Claude": "claude-opus-5-5",
    # "Gemini": "gemini-2.5-flash",
    # "Perplexity": "sonar",
    # "Grok / xAI": "grok-4",
    # "Meta/Llama via Ollama": "llama3.1",
}

# Models offered in the model dropdown. Providers not listed here offer only
# their default model; any other model name can still be typed in.
PROVIDER_MODELS = {
    "OpenAI / GPT": [
        "gpt-6.1-sol",
        "gpt-6-astra",
        "gpt-6-luna",
    ],
    "Anthropic / Claude": [
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-4-5",
        "claude-fable-5-1",
    ],
}

# The illustration and the animation are generated only by OpenAI's image
# model; Anthropic has no image generation, so with Claude (and the other
# providers) the atlas holds only the elements of the text.
IMAGE_PROVIDERS = ("OpenAI / GPT",)

# GPT Image 2.6 comes as Sunburst ("for detailed visual work") and Flare ("for
# fast iteration"); Sunburst suits photorealistic pictures.
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2.6-sunburst")

# Image API settings of each section. The illustration is a photorealistic
# still (JPEG keeps the saved atlas small); the animation is an animated GIF of
# GIF_FRAME_COUNT frames, GIF_FRAME_MS each, which loops. A GIF frame size must
# be a multiple of 16 (1088x608 is about 16:9), and all its frames together must
# fit the pixel budget of one request, so keep frames x width x height below
# about 8 million pixels when changing them.
GIF_FRAME_COUNT = 9
GIF_FRAME_MS = 400

IMAGE_SETTINGS: Dict[str, Dict[str, Any]] = {
    "illustration": {
        "size": "1024x1024",
        "quality": "high",
        "output_format": "jpeg",
        "output_compression": 90,
    },
    "animation": {
        "size": "1088x608",
        "quality": "medium",
        "output_format": "gif",
        # Not yet parameters of the openai SDK, so sent as extra fields.
        "extra_body": {
            "frame_count": GIF_FRAME_COUNT,
            "frame_duration_ms": GIF_FRAME_MS,
        },
    },
}

IMAGE_MIME_TYPES = {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp", "gif": "image/gif"}

# Generating an image, and a GIF above all, can take a few minutes.
IMAGE_TIMEOUT_SECONDS = 600

CUSTOM_CSS = """
* {
    font-family: Arial, Helvetica, "DejaVu Sans", "Noto Sans", sans-serif !important;
    letter-spacing: normal !important;
}

body,
.gradio-container,
.gradio-container * {
    font-family: Arial, Helvetica, "DejaVu Sans", "Noto Sans", sans-serif !important;
    letter-spacing: normal !important;
}

h1, h2, h3, h4, h5, h6,
p, span, div, label,
textarea, input, button, select {
    font-family: Arial, Helvetica, "DejaVu Sans", "Noto Sans", sans-serif !important;
    letter-spacing: normal !important;
}

textarea,
input {
    font-size: 16px !important;
    line-height: 1.45 !important;
}

.markdown,
.prose,
.gr-markdown {
    font-family: Arial, Helvetica, "DejaVu Sans", "Noto Sans", sans-serif !important;
    letter-spacing: normal !important;
    line-height: 1.5 !important;
}

button {
    font-weight: 600 !important;
}
"""

SYSTEM_PROMPT = """
You are a careful assistant for digital humanities and literary interpretation.
You help create a poetic atlas of a literary text, step by step: the elements of
the text, an illustration and an animation drawn from those elements, and their
interpretations. Work analytically, but do not invent external biographical or
historical claims that are not evident from the text. You may suggest possible
interpretive directions, but clearly mark them as possibilities.
""".strip()

USER_PROMPT_TEMPLATE = """
Identify the elements of the following literary text. This is the first step of its
poetic atlas.

TEXT:
{text}

Identify the space, time, characters or voices, main motifs, artistic devices, and
dominant atmosphere of the text, and then write a brief interpretation of the
elements you identified.

In the next steps the illustration and the animation of the atlas are drawn from
these elements alone, without the text, so each element should be stated fully
enough to be understood on its own.

Return only valid JSON without Markdown formatting, in the following structure:
{
  "elements": {
    "space": ["..."],
    "time": ["..."],
    "characters_or_voices": ["..."],
    "motifs": ["..."],
    "artistic_procedures": ["..."],
    "atmosphere": ["..."]
  },
  "elements_interpretation": "brief interpretation of the identified elements"
}
""".strip()

ILLUSTRATION_PROMPT_TEMPLATE = """
A photorealistic illustration for the poetic atlas of a literary text, created from
the elements identified in that text (below), without the text itself.

ELEMENTS OF THE TEXT (JSON):
{elements}

Show one real-looking scene, as if photographed: natural light, real materials and
textures, a believable lens and depth of field. Let the space, the time, the
characters or voices, the motifs and the atmosphere of the text decide what is in
the scene, how it is framed and lit, and its colors. Render the artistic devices
(such as contrast, repetition or symbolism) through composition and light, not
through visible signs. The picture contains no written text, letters, captions or
signatures.
""".strip()

ANIMATION_PROMPT_TEMPLATE = """
A photorealistic animated GIF for the poetic atlas of a literary text, created from
the elements identified in that text (below), without the text itself.

ELEMENTS OF THE TEXT (JSON):
{elements}

Show one real-looking scene, as if filmed with a steady camera: natural light, real
materials and textures. Let the space, the time, the characters or voices and the
motifs of the text decide what is in the scene, and let the motion carry its
atmosphere: what moves (light, weather, water, smoke, figures, objects) and how,
slowly or restlessly, while the rest of the scene stays still. The animation loops
seamlessly: the last frame leads back into the first. The frames contain no written
text, letters, captions or signatures.
""".strip()

INTERPRETATION_PROMPT_TEMPLATE = """
The {section} attached below was generated for the poetic atlas of a literary text,
from the elements identified in that text. Write an interpretation of this
{section}: what it shows, and how that renders the elements of the text.

ELEMENTS OF THE TEXT (JSON):
{elements}

{attached}

Interpret the {section} as a viewer sees it, including how it moves if it is
animated. Do not mention the image model, the file format or other technical
details. Return only the interpretation, as one or two paragraphs of plain prose,
without a title or heading.
""".strip()

ATTACHED_ILLUSTRATION = "The illustration is attached as an image."

ATTACHED_ANIMATION = (
    "The animation is attached as its {n} frames, in order. Each frame is shown "
    "for {ms} ms, and the sequence loops."
)


def _compact_text(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TEXT TRUNCATED DUE TO LENGTH.]"


def in_language(prompt: str, language: str | None) -> str:
    """Adds the language of the answer to a prompt."""
    name = LANGUAGES.get(language, LANGUAGES["English"])
    return (
        f"{prompt}\n\nAny text you write for the reader should be in {name}; "
        "JSON keys stay unchanged."
    )


def build_prompt(text: str, language: str | None = None) -> str:
    # Note: the template contains literal { } from the JSON structure, so
    # str.format must NOT be used (it would try to interpret those curly braces
    # as replacement fields). That is why only the explicit {text} placeholder
    # is replaced.
    return in_language(USER_PROMPT_TEMPLATE.replace("{text}", _compact_text(text)), language)


def extract_json(raw: str) -> Dict[str, Any]:
    """Tries to extract JSON from the model response."""
    if not raw:
        raise ValueError("The model returned no response.")

    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("The model response is not valid JSON.")


def list_md(items: Any) -> str:
    if not items:
        return "- —"
    if isinstance(items, str):
        return f"- {items}"
    return "\n".join(f"- {x}" for x in items)


def text_md(value: Any) -> str:
    return str(value) if value else "—"


def atlas_headings(atlas: Dict[str, Any]) -> Dict[str, str]:
    """The headings in the language the atlas was written in."""
    return HEADINGS.get((atlas or {}).get("language"), HEADINGS["English"])


def messages(language: str | None) -> Dict[str, str]:
    """The messages of the results area in the given language."""
    return MESSAGES.get(language, MESSAGES["English"])


def format_basic_elements(atlas: Dict[str, Any], provider: str, model: str) -> str:
    elements = atlas.get("elements", {}) or {}
    headings = atlas_headings(atlas)

    # A loaded atlas may not say which model created it.
    lines = [f"**Model:** {provider} / `{model}`", ""] if model else []
    lines.append(f"## {headings['elements']}")
    for key in ELEMENT_KEYS:
        lines += ["", f"### {headings[key]}", list_md(elements.get(key))]
    lines += [
        "",
        f"## {headings['elements_interpretation']}",
        text_md(atlas.get("elements_interpretation")),
    ]

    return "\n".join(lines)


def demo_response(text: str) -> Dict[str, Any]:
    """A simple offline response so the interface can be tested without an API key."""
    low = text.lower()
    is_poetry = len([ln for ln in text.splitlines() if ln.strip()]) > 1

    motifs: List[str] = []

    if any(w in low for w in ["ljubav", "volim", "srce", "draga", "nje", "ona"]):
        motifs.append("love and the impossibility of a complete encounter")

    if any(w in low for w in ["san", "budim", "noć", "spava"]):
        motifs.append("dream, awakening, and the boundary between the inner and outer world")

    if any(w in low for w in ["smrt", "grob", "pepeo", "kraj"]):
        motifs.append("death, loss, and transience")

    if any(w in low for w in ["grad", "ulica", "trg", "kuća", "prozor"]):
        motifs.append("urban space as a symbol of a psychological state")

    if not motifs:
        motifs = [
            "inner tension",
            "search for meaning",
            "the relationship between voice and world",
        ]

    procedures = ["metaphoricity", "symbolization", "contrast"]

    if is_poetry:
        procedures += ["rhythmic repetition", "lyricization of the utterance"]

    if any(w in low for w in ["az", "vse", "žizn", "bivstvu", "čeda"]):
        procedures.append("archaization of language and a hybrid linguistic register")

    return {
        "elements": {
            "space": [
                "the space is not necessarily explicitly defined; it can be read as both outer and inner space"
            ],
            "time": [
                "fluid time of the text",
                "interweaving of moment, memory, and expectation",
            ],
            "characters_or_voices": [
                "lyric/narrative voice",
                "possible figure of the other",
            ],
            "motifs": motifs,
            "artistic_procedures": procedures,
            "atmosphere": ["melancholic", "introspective", "tense"],
        },
        "elements_interpretation": (
            "In demo mode the application does not use a real LLM; it builds the atlas "
            "heuristically. The result is intended only for checking the application flow."
        ),
    }


def openai_client(**options):
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from the .env file or HF Secrets.")
    # Diagnostics: which key the application actually uses (masked).
    print(f"Using OPENAI_API_KEY: {api_key[:7]}...{api_key[-4:]}")
    return OpenAI(api_key=api_key, **options)


def call_openai(prompt: str, model: str, images: List[str] | None = None) -> str:
    """Asks an OpenAI model; images (data URIs) are attached after the prompt."""
    print(f"Calling OpenAI model: {model}")
    client = openai_client()

    # GPT-5 and newer (and the o-series) are reasoning models that reject a
    # custom temperature; only the older GPT-4 / GPT-3.5 families accept it.
    extra = {}
    if model.startswith(("gpt-4", "gpt-3.5")):
        extra["temperature"] = DEFAULT_TEMPERATURE

    content: Any = prompt
    if images:
        content = [{"type": "input_text", "text": prompt}] + [
            {"type": "input_image", "image_url": image} for image in images
        ]

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        **extra,
    )

    return response.output_text


def generate_openai_image(section: str, prompt: str) -> str:
    """Generates the picture of one section with OpenAI's image model.

    Returns it as a data URI, or "" if the model returned no image.
    """
    print(f"Calling OpenAI image model: {OPENAI_IMAGE_MODEL} ({section})")
    client = openai_client(timeout=IMAGE_TIMEOUT_SECONDS)
    settings = IMAGE_SETTINGS[section]

    result = client.images.generate(model=OPENAI_IMAGE_MODEL, prompt=prompt, n=1, **settings)

    image = (result.data or [None])[0]
    if not image or not image.b64_json:
        return ""
    return f"data:{IMAGE_MIME_TYPES[settings['output_format']]};base64,{image.b64_json}"


def call_anthropic(prompt: str, model: str) -> str:
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is missing from the .env file or HF Secrets.")

    client = anthropic.Anthropic(api_key=api_key)

    # No temperature or thinking settings here: the newest Claude models reject
    # a custom temperature and decide on their own how much to think.
    # max_tokens has to cover both that thinking and the JSON answer.
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to process this text.")

    return "".join(block.text for block in response.content if block.type == "text")


def call_openai_compatible(
    prompt: str,
    model: str,
    api_key_env: str,
    base_url: str,
) -> str:
    from openai import OpenAI

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is missing from the .env file or HF Secrets.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=DEFAULT_TEMPERATURE,
    )

    return response.choices[0].message.content or ""


def call_gemini(prompt: str, model: str) -> str:
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is missing from the .env file or HF Secrets.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
        config={"temperature": DEFAULT_TEMPERATURE},
    )

    return response.text or ""


def call_ollama(prompt: str, model: str) -> str:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": DEFAULT_TEMPERATURE},
    }

    response = requests.post(url, json=payload, timeout=180)
    response.raise_for_status()

    data = response.json()
    return data.get("message", {}).get("content", "")


def call_model(provider: str, model: str, prompt: str) -> str:
    if provider == "OpenAI / GPT":
        return call_openai(prompt, model)

    if provider == "Anthropic / Claude":
        return call_anthropic(prompt, model)

    if provider == "Gemini":
        return call_gemini(prompt, model)

    if provider == "Perplexity":
        return call_openai_compatible(
            prompt,
            model,
            api_key_env="PERPLEXITY_API_KEY",
            base_url=os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai"),
        )

    if provider == "Grok / xAI":
        return call_openai_compatible(
            prompt,
            model,
            api_key_env="XAI_API_KEY",
            base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
        )

    if provider == "Meta/Llama via Ollama":
        return call_ollama(prompt, model)

    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# ILLUSTRATION AND ANIMATION: both are generated by OpenAI's image model, from
# the elements of the text alone: the illustration as a photorealistic image,
# the animation as an animated GIF. Each of them is interpreted by the selected
# language model only after it exists, from the picture itself. Other
# providers (Anthropic among them) generate no images.
#
# Atlases saved earlier hold SVG drawings ("svg") instead of images ("image");
# they are still shown, in a sandboxed frame.
# ---------------------------------------------------------------------------

SECTIONS = ("illustration", "animation")

DRAWING_PROMPTS = {
    "illustration": ILLUSTRATION_PROMPT_TEMPLATE,
    "animation": ANIMATION_PROMPT_TEMPLATE,
}

IMAGE_SIZES = {
    "illustration": "width:100%;max-width:640px;height:auto",
    # The GIF is at most half as wide as the browser window.
    "animation": "width:100%;max-width:50vw;height:auto",
}

# For the SVG drawings of older atlases.
FRAME_SIZES = {
    "illustration": "width:100%;max-width:640px;aspect-ratio:1/1",
    "animation": "width:100%;aspect-ratio:16/9",
}

# Frames of the animation are sent to the language model at most this large.
FRAME_MAX_PIXELS = 768


def has_drawing(data: Dict[str, Any] | None) -> bool:
    """Whether a section holds a picture: an image, or the SVG of an older atlas."""
    return bool(data) and ("image" in data or "svg" in data)


def notice_html(message: str) -> str:
    return f"<p><em>{html.escape(message)}</em></p>"


def caption_html(caption: str) -> str:
    return f'<p style="font-size:0.85em;opacity:0.7;">{html.escape(caption)}</p>'


def image_html(src: str, caption: str, size_css: str) -> str:
    """An image generated for the atlas (a data URI), with its caption."""
    return (
        f'<img src="{html.escape(src)}" alt="" '
        f'style="{size_css};display:block;border-radius:12px;">' + caption_html(caption)
    )


def gif_frames(data_uri: str) -> Tuple[List[str], int]:
    """The frames of an animated GIF (as JPEG data URIs) and the duration of a frame in ms.

    Vision models see only the first frame of a GIF, so the animation is
    interpreted from all of its frames, in order.
    """
    from PIL import Image, ImageSequence

    raw = base64.b64decode(data_uri.split(",", 1)[1])
    frames = []
    with Image.open(io.BytesIO(raw)) as gif:
        duration = int(gif.info.get("duration") or GIF_FRAME_MS)
        for frame in ImageSequence.Iterator(gif):
            picture = frame.convert("RGB")
            picture.thumbnail((FRAME_MAX_PIXELS, FRAME_MAX_PIXELS))
            buffer = io.BytesIO()
            picture.save(buffer, format="JPEG", quality=85)
            frames.append("data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode())
    return frames, duration


def svg_frame(svg: str, caption: str, size_css: str) -> str:
    """Wraps the SVG of an older atlas in a sandboxed frame: no scripts, no network access."""
    page = (
        "<!doctype html><html><head>"
        '<meta http-equiv="Content-Security-Policy" '
        "content=\"default-src 'none'; style-src 'unsafe-inline'; img-src data:\">"
        "<style>html,body{margin:0;height:100%;background:#111}"
        "svg{display:block;width:100%;height:100%}</style>"
        f"</head><body>{svg}</body></html>"
    )
    return (
        f'<iframe sandbox srcdoc="{html.escape(page)}" '
        f'style="{size_css};border:0;border-radius:12px;"></iframe>' + caption_html(caption)
    )


def in_parallel(fn, items) -> list:
    """Runs fn(item) for all items at the same time and returns the results in order."""
    with ThreadPoolExecutor(max_workers=max(1, len(items))) as pool:
        return list(pool.map(fn, items))


def elements_json(atlas: Dict[str, Any]) -> str:
    return json.dumps(atlas.get("elements") or {}, ensure_ascii=False, indent=2)


def draw_section(atlas: Dict[str, Any], provider: str, model: str, section: str) -> Dict[str, Any]:
    """Generates the picture of one section from the elements of the text.

    Only the image providers (OpenAI) generate pictures; for the others the
    section records that it has none.
    """
    msg = messages(atlas.get("language"))
    if provider not in IMAGE_PROVIDERS:
        return {"error": msg[f"unsupported_{section}"].format(provider=provider)}

    # The picture holds no text, so its prompt does not name a language.
    prompt = DRAWING_PROMPTS[section].replace("{elements}", elements_json(atlas))
    try:
        image = generate_openai_image(section, prompt)
    except Exception as err:  # noqa: BLE001
        return {"error": msg[f"failed_{section}"].format(err=err)}

    if not image:
        return {"error": msg[f"missing_{section}"]}
    return {"image": image, "model": OPENAI_IMAGE_MODEL}


def interpret_section(atlas: Dict[str, Any], provider: str, model: str, section: str) -> str:
    """Asks the selected model to interpret the finished picture of one section.

    The model sees the picture itself; pictures exist only with OpenAI, so it
    is always an OpenAI model.
    """
    msg = messages(atlas.get("language"))
    try:
        image = atlas[section]["image"]
        if section == "animation":
            images, duration = gif_frames(image)
            attached = ATTACHED_ANIMATION.format(n=len(images), ms=duration)
        else:
            images, attached = [image], ATTACHED_ILLUSTRATION
        prompt = in_language(
            INTERPRETATION_PROMPT_TEMPLATE.replace("{section}", section)
            .replace("{elements}", elements_json(atlas))
            .replace("{attached}", attached),
            atlas.get("language"),
        )
        interpretation = call_openai(prompt, model, images=images).strip()
    except Exception as err:  # noqa: BLE001
        return msg["failed_interpretation"].format(err=err)
    # Some models still put a Markdown heading on top; the tab has its own.
    return re.sub(r"\A#+[^\n]*\n+", "", interpretation)


def section_view(atlas: Dict[str, Any], section: str) -> str:
    """HTML for the picture area of one section, from what the atlas holds for it."""
    data = (atlas or {}).get(section)
    if not data:
        return ""
    if not has_drawing(data):
        return notice_html(data.get("error", ""))
    caption = messages(atlas.get("language"))[f"drawn_{section}"].format(
        model=data.get("model", "")
    )
    if "image" in data:
        return image_html(data["image"], caption, IMAGE_SIZES[section])
    return svg_frame(data["svg"], caption, FRAME_SIZES[section])


def section_text(atlas: Dict[str, Any], section: str) -> str:
    """Markdown for the interpretation of one section; it exists only once the picture does."""
    data = (atlas or {}).get(section) or {}
    if not has_drawing(data):
        return ""
    if "interpretation" not in data:
        return f"*{messages(atlas.get('language'))['writing']}*"
    return f"## {atlas_headings(atlas)[section]}\n\n{data['interpretation']}"


def section_outputs(atlas: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """(picture, text) of the illustration and then of the animation, for the UI."""
    return tuple(
        part
        for section in SECTIONS
        for part in (section_view(atlas, section), section_text(atlas, section))
    )


def tabs_state(ready: bool) -> Tuple[Any, Any, Any, Any]:
    """Updates for the tab strip and for the Illustration, Animation and JSON tabs.

    Those three tabs can be opened only once the result is ready; until then the
    first tab is shown.
    """
    strip = gr.update() if ready else gr.update(selected="elements")
    return (strip, *(gr.update(interactive=ready) for _ in range(3)))


def create_poetic_atlas(text: str, provider: str, model: str, language: str = "English"):
    """Handler: builds the atlas in three steps and reports each one in the status.

    Yields (elements, atlas, status, illustration, its text, animation, its text),
    followed by the four updates of tabs_state().
    """
    msg = messages(language)
    locked = tabs_state(False)
    if not text or not text.strip():
        yield (msg["enter_text"], {}, msg["no_text"], *section_outputs({}), *locked)
        return

    model = (model or "").strip() or PROVIDER_DEFAULT_MODELS.get(provider, "")
    yield ("", {}, msg["step_0"], *section_outputs({}), *locked)

    # Step 1: the elements of the text and their interpretation.
    try:
        if provider == "Demo/offline":
            atlas = demo_response(text)
        else:
            atlas = extract_json(call_model(provider, model, build_prompt(text, language)))
        # The atlas records who created it, in which language, and from which
        # text, so that a saved atlas can be shown again with its inputs. The
        # later steps and the "Regenerate" buttons write in the same language.
        record = {
            "provider": provider,
            "model": model,
            "language": language if language in LANGUAGES else "English",
            "text": text.strip(),
        }
        atlas = {**record, **{k: v for k, v in atlas.items() if k not in record}}
        elements_md = format_basic_elements(atlas, provider, model)
    except Exception as err:  # noqa: BLE001
        help_msg = msg["error_help"].format(err=err)
        yield (help_msg, {}, msg["error"], *section_outputs({}), *locked)
        return

    # Only the image providers go through steps 2 and 3; for the others the
    # sections only record that the provider generates no pictures.
    can_draw = provider in IMAGE_PROVIDERS

    # Each step yields a copy of the atlas, because the following steps keep
    # adding to it.
    if can_draw:
        yield (elements_md, copy.deepcopy(atlas), msg["step_1"], *section_outputs(atlas), *locked)

    # Step 2: the illustration and the animation, generated from the elements alone.
    drawings = in_parallel(lambda s: draw_section(atlas, provider, model, s), SECTIONS)
    atlas.update(zip(SECTIONS, drawings))
    if can_draw:
        yield (elements_md, copy.deepcopy(atlas), msg["step_2"], *section_outputs(atlas), *locked)

    # Step 3: the interpretations, written only now that the pictures exist.
    drawn = [s for s in SECTIONS if has_drawing(atlas[s])]
    interpretations = in_parallel(
        lambda s: interpret_section(atlas, provider, model, s), drawn
    )
    for section, interpretation in zip(drawn, interpretations):
        atlas[section]["interpretation"] = interpretation

    if not can_draw:
        status_msg = msg["done_no_drawings"].format(provider=provider)
    elif len(drawn) == len(SECTIONS):
        status_msg = msg["done"]
    else:
        status_msg = msg["done_partly"]
    if len(text.strip()) > MAX_TEXT_CHARS:
        status_msg += msg["truncated"].format(n=MAX_TEXT_CHARS)

    # Only now, with the result complete, can the other tabs be opened.
    yield (elements_md, atlas, status_msg, *section_outputs(atlas), *tabs_state(True))


def regenerate_section(atlas: Dict[str, Any], provider: str, model: str, section: str):
    """Redraws one section from the elements of the current atlas, then interprets it.

    Yields (atlas, picture, text) for that section.
    """
    atlas = atlas or {}
    if not atlas.get("elements"):
        yield atlas, "", ""
        return

    model = (model or "").strip() or PROVIDER_DEFAULT_MODELS.get(provider, "")
    generating = messages(atlas.get("language"))[f"generating_{section}"]
    yield copy.deepcopy(atlas), notice_html(generating), ""

    atlas[section] = draw_section(atlas, provider, model, section)
    yield copy.deepcopy(atlas), section_view(atlas, section), section_text(atlas, section)

    if has_drawing(atlas[section]):
        atlas[section]["interpretation"] = interpret_section(atlas, provider, model, section)
        yield atlas, section_view(atlas, section), section_text(atlas, section)


def regenerate_illustration(atlas: Dict[str, Any], provider: str, model: str):
    yield from regenerate_section(atlas, provider, model, "illustration")


def regenerate_animation(atlas: Dict[str, Any], provider: str, model: str):
    yield from regenerate_section(atlas, provider, model, "animation")


def atlas_provider_model(atlas: Dict[str, Any]) -> Tuple[str, str]:
    """The provider and the model that created a saved atlas, or ("", "") if unknown.

    Newer atlases record them at the top ("provider", "model"); older ones only
    in their drawings, e.g. "OpenAI / GPT / gpt-6.1-sol".
    """
    if atlas.get("provider") and atlas.get("model"):
        return str(atlas["provider"]), str(atlas["model"])
    for section in SECTIONS:
        drawn_by = ((atlas.get(section) or {}).get("model") or "").strip()
        if " / " in drawn_by:
            provider, model = drawn_by.rsplit(" / ", 1)
            return provider, model
    return "", ""


def load_existing_atlas(file_path: str | None, language: str):
    """Handler: shows a saved atlas (the JSON of the JSON tab) in the results area.

    Returns the same outputs as create_poetic_atlas, followed by the language,
    the analyzed text, the provider and the model of the atlas, which are put
    back into the input fields.
    """
    def only_status(message: str) -> tuple:
        # Everything else stays as it is.
        return (gr.update(), gr.update(), message, *(gr.update() for _ in range(12)))

    msg = messages(language)
    if not file_path:
        return only_status(msg["no_atlas_selected"])

    path = Path(file_path)
    try:
        atlas = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(atlas, dict) or not isinstance(atlas.get("elements"), dict):
            raise ValueError("the file has no \"elements\" of the atlas.")
    except Exception as err:  # noqa: BLE001
        return only_status(msg["load_error"].format(err=err))

    atlas["language"] = atlas.get("language") if atlas.get("language") in LANGUAGES else "English"
    provider_name, model_name = atlas_provider_model(atlas)
    elements_md = format_basic_elements(atlas, provider_name, model_name)
    status_msg = messages(atlas["language"])["loaded"].format(name=path.stem)

    # The provider dropdown accepts only the providers offered in it; the model
    # dropdown accepts any model name.
    if provider_name in PROVIDER_DEFAULT_MODELS:
        provider_update = gr.update(value=provider_name)
        model_update = gr.update(choices=model_choices(provider_name), value=model_name)
    else:
        provider_update = model_update = gr.update()

    return (
        elements_md,
        atlas,
        status_msg,
        *section_outputs(atlas),
        gr.update(selected="elements"),
        *(gr.update(interactive=True) for _ in range(3)),
        gr.update(value=atlas["language"]),
        # Older atlases do not hold the text; the field is then emptied, so that
        # it never shows a text that does not belong to the loaded atlas.
        gr.update(value=atlas.get("text") or ""),
        provider_update,
        model_update,
    )


def model_choices(provider: str) -> List[str]:
    return PROVIDER_MODELS.get(provider, [PROVIDER_DEFAULT_MODELS.get(provider, "")])


def update_model(provider: str) -> "gr.update":
    return gr.update(
        choices=model_choices(provider),
        value=PROVIDER_DEFAULT_MODELS.get(provider, ""),
    )


def update_language(language: str):
    """Puts the tab names and the buttons of the results area in the selected language."""
    msg = messages(language)
    return (
        gr.update(label=msg["tab_elements"]),
        gr.update(label=msg["tab_illustration"]),
        gr.update(label=msg["tab_animation"]),
        gr.update(label=msg["tab_json"]),
        gr.update(value=msg["regenerate_illustration"]),
        gr.update(value=msg["regenerate_animation"]),
        gr.update(label=msg["json_label"]),
    )


# Shows the app in dark mode, whatever the system theme. Gradio 6 ignores the
# ?__theme=dark URL parameter (its theme_mode, "system" by default, takes
# precedence) and marks dark mode with the "dark" class on <body>, which it
# also resets whenever the system theme changes. So the class is set here and
# kept. ?__theme=light still opens the light theme.
#
# Gradio 6 adds this code to the page as a <script>, after the app has loaded;
# it does not call it as a function, so the function calls itself.
DARK_MODE_JS = """
(() => {
    if (new URLSearchParams(window.location.search).get("__theme") === "light") return;
    const keepDark = () => {
        if (!document.body.classList.contains("dark")) document.body.classList.add("dark");
    };
    keepDark();
    new MutationObserver(keepDark).observe(document.body, {
        attributes: true,
        attributeFilter: ["class"],
    });
})();
"""

APP_THEME = gr.themes.Soft(
    font=[
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Noto Sans",
        "sans-serif",
    ],
    font_mono=[
        "Consolas",
        "DejaVu Sans Mono",
        "monospace",
    ],
)


with gr.Blocks(title=APP_TITLE) as demo:
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            gr.Markdown(
                f"""
                # {APP_TITLE}
                Enter a prose or lyric text, choose a provider and model, and click
                **Create Poetic Atlas**. The application returns the basic elements of the
                text (space, time, characters, main motifs, artistic devices, and dominant
                atmosphere). With **OpenAI / GPT**, the **Illustration** tab shows a
                photorealistic image and the **Animation** tab an animated GIF, both
                generated by `{OPENAI_IMAGE_MODEL}` from the elements of the text and then
                interpreted. **Anthropic / Claude** does not generate images, so with it the
                atlas holds only the elements and their interpretation. On the **JSON** tab, a structured
                JSON with all the elements of the atlas, including the analyzed text, is
                also available. An atlas saved earlier as JSON can be shown again with
                **Load existing atlas**.
                """
            )

        # Top right: opens a saved atlas (the JSON of the JSON tab) and shows it.
        with gr.Column(scale=1, min_width=200):
            load_btn = gr.UploadButton(
                "Load existing atlas",
                file_types=[".json"],
                file_count="single",
                variant="secondary",
            )

    with gr.Row():
        with gr.Column(scale=1):
            provider = gr.Dropdown(
                choices=list(PROVIDER_DEFAULT_MODELS.keys()),
                value="OpenAI / GPT",
                label="LLM provider",
            )

            model = gr.Dropdown(
                choices=model_choices("OpenAI / GPT"),
                value=PROVIDER_DEFAULT_MODELS["OpenAI / GPT"],
                label="Model",
                allow_custom_value=True,
            )

            language = gr.Radio(
                choices=list(LANGUAGES),
                value="English",
                label="Response language",
            )

        with gr.Column(scale=2):
            text = gr.Textbox(
                label="Text to analyze",
                lines=18,
                placeholder="Enter a poem or a prose passage here...",
                # Literary texts are full of words a spell checker does not know.
                html_attributes=gr.InputHTMLAttributes(spellcheck=False),
            )

            create_btn = gr.Button(
                "Create Poetic Atlas",
                variant="primary",
            )

    status = gr.Textbox(
        label="Status",
        interactive=False,
    )

    # The results area starts in English and follows the selected language.
    msg = messages("English")

    with gr.Tabs() as tabs:
        with gr.Tab(msg["tab_elements"], id="elements") as elements_tab:
            basic_elements_output = gr.Markdown()

        # These three tabs can be opened only once the result is ready.
        with gr.Tab(msg["tab_illustration"], interactive=False) as illustration_tab:
            illustration_view = gr.HTML()
            gen_illustration_btn = gr.Button(
                msg["regenerate_illustration"],
                variant="secondary",
            )
            illustration_output = gr.Markdown()

        with gr.Tab(msg["tab_animation"], interactive=False) as animation_tab:
            animation_view = gr.HTML()
            gen_animation_btn = gr.Button(
                msg["regenerate_animation"],
                variant="secondary",
            )
            animation_text_output = gr.Markdown()

        with gr.Tab(msg["tab_json"], interactive=False) as json_tab:
            json_output = gr.JSON(label=msg["json_label"])

    # .input, not .change: only a provider chosen by the user resets the model.
    # A loaded atlas sets the provider and its own model together.
    provider.input(
        fn=update_model,
        inputs=provider,
        outputs=model,
    )

    language.change(
        fn=update_language,
        inputs=language,
        outputs=[
            elements_tab,
            illustration_tab,
            animation_tab,
            json_tab,
            gen_illustration_btn,
            gen_animation_btn,
            json_output,
        ],
    )

    create_btn.click(
        fn=create_poetic_atlas,
        inputs=[text, provider, model, language],
        outputs=[
            basic_elements_output,
            json_output,
            status,
            illustration_view,
            illustration_output,
            animation_view,
            animation_text_output,
            tabs,
            illustration_tab,
            animation_tab,
            json_tab,
        ],
    )

    # A saved atlas is shown just like a newly created one, and its text,
    # provider, model and language go back into the input fields (the language
    # also sets the tabs and buttons, through language.change).
    load_btn.upload(
        fn=load_existing_atlas,
        inputs=[load_btn, language],
        outputs=[
            basic_elements_output,
            json_output,
            status,
            illustration_view,
            illustration_output,
            animation_view,
            animation_text_output,
            tabs,
            illustration_tab,
            animation_tab,
            json_tab,
            language,
            text,
            provider,
            model,
        ],
    )

    # The "Regenerate" buttons redraw from the elements of the already created
    # atlas (json_output) and then interpret the new drawing.
    gen_illustration_btn.click(
        fn=regenerate_illustration,
        inputs=[json_output, provider, model],
        outputs=[json_output, illustration_view, illustration_output],
    )

    gen_animation_btn.click(
        fn=regenerate_animation,
        inputs=[json_output, provider, model],
        outputs=[json_output, animation_view, animation_text_output],
    )


if __name__ == "__main__":
    # Gradio 6 takes the theme and CSS in launch(), not in the Blocks constructor.
    demo.launch(show_error=True, theme=APP_THEME, css=CUSTOM_CSS, js=DARK_MODE_JS)