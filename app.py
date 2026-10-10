"""Poetic Atlas – Gradio application.

The application lets the user enter a literary text, choose an LLM provider/model,
and create a structured poetic atlas in three steps: the basic elements of the
text and their interpretation; an illustration and an animation drawn from those
elements; and the interpretations of the finished illustration and animation.

Supported modes:
- Demo/offline: works without API keys, for checking the interface.
- OpenAI/GPT: uses OPENAI_API_KEY.
- Anthropic/Claude: uses ANTHROPIC_API_KEY.
- Gemini: uses GOOGLE_API_KEY or GEMINI_API_KEY.
- Perplexity: uses PERPLEXITY_API_KEY.
- Grok/xAI: uses XAI_API_KEY.
- Meta/Llama via Ollama: uses a local Ollama server.
"""

from __future__ import annotations

import copy
import html
import json
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
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
            "Drawing the illustration and the animation…"
        ),
        "step_2": (
            "Step 2 of 3 done: the illustration and the animation are drawn. "
            "Writing their interpretations…"
        ),
        "done": "The atlas was created successfully.",
        "done_partly": (
            "The elements were identified, but not everything could be drawn "
            "(see the Illustration and Animation tabs)."
        ),
        "truncated": " Note: only the first {n} characters of the text were analyzed.",
        "generating_illustration": "Generating the illustration…",
        "generating_animation": "Generating the animation…",
        "writing": "Writing the interpretation…",
        "drawn_illustration": "Illustration drawn by {model}.",
        "drawn_animation": "Animation drawn by {model}.",
        "failed_illustration": "The illustration was not generated: {err}",
        "failed_animation": "The animation was not generated: {err}",
        "missing_illustration": "The selected model did not return the illustration.",
        "missing_animation": "The selected model did not return the animation.",
        "failed_interpretation": "The interpretation was not generated: {err}",
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
            "Crtaju se ilustracija i animacija…"
        ),
        "step_2": (
            "Korak 2 od 3 je završen: ilustracija i animacija su nacrtane. "
            "Pišu se njihove interpretacije…"
        ),
        "done": "Atlas je uspešno kreiran.",
        "done_partly": (
            "Elementi su određeni, ali nije sve moglo da se nacrta "
            "(pogledajte kartice Ilustracija i Animacija)."
        ),
        "truncated": " Napomena: analizirano je samo prvih {n} znakova teksta.",
        "generating_illustration": "Ilustracija se generiše…",
        "generating_animation": "Animacija se generiše…",
        "writing": "Interpretacija se piše…",
        "drawn_illustration": "Ilustraciju je nacrtao model {model}.",
        "drawn_animation": "Animaciju je nacrtao model {model}.",
        "failed_illustration": "Ilustracija nije generisana: {err}",
        "failed_animation": "Animacija nije generisana: {err}",
        "missing_illustration": "Izabrani model nije vratio ilustraciju.",
        "missing_animation": "Izabrani model nije vratio animaciju.",
        "failed_interpretation": "Interpretacija nije generisana: {err}",
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
The elements below were identified in a literary text for its poetic atlas. Draw
the illustration of the atlas from these elements, as a single self-contained SVG
picture.

ELEMENTS OF THE TEXT (JSON):
{elements}

The SVG is displayed in a sandboxed frame of a web page, where scripts cannot run
and nothing can be loaded from the network, so everything the picture needs has to
be drawn inside the SVG itself.

Use viewBox="0 0 800 800". Make it a still picture, without animation: a stylized,
expressive composition whose shapes and colors carry the space, the motifs and the
atmosphere of the text; avoid putting explanatory text in the picture.

Return only the SVG markup, from <svg to </svg>.
""".strip()

ANIMATION_PROMPT_TEMPLATE = """
The elements below were identified in a literary text for its poetic atlas. Create
the animation of the atlas from these elements, as a single self-contained animated
SVG.

ELEMENTS OF THE TEXT (JSON):
{elements}

The SVG is displayed in a sandboxed frame of a web page, where scripts cannot run
and nothing can be loaded from the network. All movement therefore has to come
from SMIL elements (<animate>, <animateTransform>, <animateMotion>) or from CSS
animations in a <style> element, and everything the picture needs has to be drawn
inside the SVG itself.

Use viewBox="0 0 800 450". Let the animation unfold in a few scenes that follow one
another, and loop the whole sequence. Show each scene as a stylized, expressive
picture whose colors and movement carry the atmosphere of the text; avoid putting
explanatory text on screen.

Return only the SVG markup, from <svg to </svg>.
""".strip()

INTERPRETATION_PROMPT_TEMPLATE = """
The {section} below was drawn for the poetic atlas of a literary text, as SVG, from
the elements identified in that text. Write an interpretation of this {section}:
what it shows, and how that renders the elements of the text.

ELEMENTS OF THE TEXT (JSON):
{elements}

THE DRAWING (SVG source):
{svg}

Interpret the drawing as a viewer sees it, including how it moves if it is animated.
Leave the code out of it: do not mention SVG, markup or other technical details.
Return only the interpretation, as one or two paragraphs of plain prose, without a
title or heading.
""".strip()


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
        "JSON keys and SVG markup stay unchanged."
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

    lines = [
        f"**Model:** {provider} / `{model}`",
        "",
        f"## {headings['elements']}",
    ]
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


def call_openai(prompt: str, model: str) -> str:
    from openai import OpenAI

    print(f"Calling OpenAI model: {model}")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from the .env file or HF Secrets.")
    # Diagnostics: which key the application actually uses (masked).
    print(f"Using OPENAI_API_KEY: {api_key[:7]}...{api_key[-4:]}")

    client = OpenAI(api_key=api_key)

    # GPT-5 and newer (and the o-series) are reasoning models that reject a
    # custom temperature; only the older GPT-4 / GPT-3.5 families accept it.
    extra = {}
    if model.startswith(("gpt-4", "gpt-3.5")):
        extra["temperature"] = DEFAULT_TEMPERATURE

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        **extra,
    )

    return response.output_text


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
# ILLUSTRATION AND ANIMATION: both are drawn by the selected language model
# itself, as SVG (a still picture and an animated one), from the elements of
# the text alone. Each of them is interpreted only after it has been drawn.
# No separate image or video model is used.
# ---------------------------------------------------------------------------

SECTIONS = ("illustration", "animation")

DRAWING_PROMPTS = {
    "illustration": ILLUSTRATION_PROMPT_TEMPLATE,
    "animation": ANIMATION_PROMPT_TEMPLATE,
}

FRAME_SIZES = {
    "illustration": "width:100%;max-width:640px;aspect-ratio:1/1",
    "animation": "width:100%;aspect-ratio:16/9",
}


def extract_svg(raw: str) -> str:
    """Returns the <svg>...</svg> markup from the model response, or "" if there is none."""
    match = re.search(r"<svg\b.*</svg>", raw or "", re.DOTALL | re.IGNORECASE)
    return match.group(0) if match else ""


def notice_html(message: str) -> str:
    return f"<p><em>{html.escape(message)}</em></p>"


def svg_frame(svg: str, caption: str, size_css: str) -> str:
    """Wraps a model-drawn SVG in a sandboxed frame: no scripts, no network access."""
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
        f'style="{size_css};border:0;border-radius:12px;"></iframe>'
        f'<p style="font-size:0.85em;opacity:0.7;">{html.escape(caption)}</p>'
    )


def in_parallel(fn, items) -> list:
    """Runs fn(item) for all items at the same time and returns the results in order."""
    with ThreadPoolExecutor(max_workers=max(1, len(items))) as pool:
        return list(pool.map(fn, items))


def elements_json(atlas: Dict[str, Any]) -> str:
    return json.dumps(atlas.get("elements") or {}, ensure_ascii=False, indent=2)


def draw_section(atlas: Dict[str, Any], provider: str, model: str, section: str) -> Dict[str, Any]:
    """Asks the selected model to draw one section from the elements of the text, as SVG."""
    prompt = in_language(
        DRAWING_PROMPTS[section].replace("{elements}", elements_json(atlas)),
        atlas.get("language"),
    )
    msg = messages(atlas.get("language"))
    try:
        svg = extract_svg(call_model(provider, model, prompt))
    except Exception as err:  # noqa: BLE001
        return {"error": msg[f"failed_{section}"].format(err=err)}

    if not svg:
        return {"error": msg[f"missing_{section}"]}
    return {"svg": svg, "model": f"{provider} / {model}"}


def interpret_section(atlas: Dict[str, Any], provider: str, model: str, section: str) -> str:
    """Asks the selected model to interpret the finished drawing of one section."""
    prompt = in_language(
        INTERPRETATION_PROMPT_TEMPLATE.replace("{section}", section)
        .replace("{elements}", elements_json(atlas))
        .replace("{svg}", atlas[section]["svg"]),
        atlas.get("language"),
    )
    try:
        interpretation = call_model(provider, model, prompt).strip()
    except Exception as err:  # noqa: BLE001
        return messages(atlas.get("language"))["failed_interpretation"].format(err=err)
    # Some models still put a Markdown heading on top; the tab has its own.
    return re.sub(r"\A#+[^\n]*\n+", "", interpretation)


def section_view(atlas: Dict[str, Any], section: str) -> str:
    """HTML for the picture area of one section, from what the atlas holds for it."""
    data = (atlas or {}).get(section)
    if not data:
        return ""
    if "svg" not in data:
        return notice_html(data.get("error", ""))
    caption = messages(atlas.get("language"))[f"drawn_{section}"]
    return svg_frame(
        data["svg"],
        caption.format(model=data.get("model", "")),
        FRAME_SIZES[section],
    )


def section_text(atlas: Dict[str, Any], section: str) -> str:
    """Markdown for the interpretation of one section; it exists only once the drawing does."""
    data = (atlas or {}).get(section) or {}
    if "svg" not in data:
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
        # The later steps and the "Regenerate" buttons write in the same language.
        atlas["language"] = language if language in LANGUAGES else "English"
        elements_md = format_basic_elements(atlas, provider, model)
    except Exception as err:  # noqa: BLE001
        help_msg = msg["error_help"].format(err=err)
        yield (help_msg, {}, msg["error"], *section_outputs({}), *locked)
        return

    # Each step yields a copy of the atlas, because the following steps keep
    # adding to it.
    yield (elements_md, copy.deepcopy(atlas), msg["step_1"], *section_outputs(atlas), *locked)

    # Step 2: the illustration and the animation, drawn from the elements alone.
    drawings = in_parallel(lambda s: draw_section(atlas, provider, model, s), SECTIONS)
    atlas.update(zip(SECTIONS, drawings))
    yield (elements_md, copy.deepcopy(atlas), msg["step_2"], *section_outputs(atlas), *locked)

    # Step 3: the interpretations, written only now that the drawings exist.
    drawn = [s for s in SECTIONS if "svg" in atlas[s]]
    interpretations = in_parallel(
        lambda s: interpret_section(atlas, provider, model, s), drawn
    )
    for section, interpretation in zip(drawn, interpretations):
        atlas[section]["interpretation"] = interpretation

    status_msg = msg["done"] if len(drawn) == len(SECTIONS) else msg["done_partly"]
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

    if "svg" in atlas[section]:
        atlas[section]["interpretation"] = interpret_section(atlas, provider, model, section)
        yield atlas, section_view(atlas, section), section_text(atlas, section)


def regenerate_illustration(atlas: Dict[str, Any], provider: str, model: str):
    yield from regenerate_section(atlas, provider, model, "illustration")


def regenerate_animation(atlas: Dict[str, Any], provider: str, model: str):
    yield from regenerate_section(atlas, provider, model, "animation")


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
    gr.Markdown(
        f"""
        # {APP_TITLE}
        Enter a prose or lyric text, choose a provider and model, and click
        **Create Poetic Atlas**. The application returns the basic elements of the
        text (space, time, characters, main motifs, artistic devices, and dominant
        atmosphere). On the **Illustration** tab, the model automatically generates
        an image based on the text. On the **Animation** tab, the model automatically
        generates an animation based on the text. On the **JSON** tab, a structured
        JSON with all the elements of the atlas is also available.
        """
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

    provider.change(
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
    demo.launch(show_error=True, theme=APP_THEME, css=CUSTOM_CSS)