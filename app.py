"""
Llama Académico: asistente universitario para alumnos de la UTSLRC.

Pipeline completo:
  1. RAG      -> embeddings (all-MiniLM-L6-v2) + similitud coseno sobre conocimiento.txt
  2. Tono     -> personalidad "ajustada con LoRA" (simulada con system prompt + few-shot
                 tomados del mismo dataset que usa train_lora.py con trl.SFTTrainer)
  3. LLM      -> Groq API (openai/gpt-oss-20b o familia Llama)
  4. Métricas -> tiempo de respuesta (s) y tokens consumidos por consulta
"""

from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

KNOWLEDGE_PATH = BASE_DIR / "conocimiento.txt"
TONE_DATA_PATH = BASE_DIR / "datos_tono.jsonl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_RELEVANCE = 0.30  # por debajo de este coseno se avisa al modelo que el contexto es débil
HISTORY_TURNS = 6  # mensajes previos que se reenvían al modelo

MODELS = ["openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

TONES = {
    "Coordinador académico (formal)": {
        "key": "formal",
        "persona": (
            "Eres el Coordinador Académico virtual de la UTSLRC. Hablas con un tono formal, "
            "institucional y cordial: trato de «usted», sin coloquialismos ni emojis, frases "
            "completas y estructura clara (primero la respuesta directa, luego el detalle o los "
            "pasos). Cierras con una sola frase breve ofreciendo más ayuda."
        ),
    },
    "Compañero estudiante (amigable)": {
        "key": "amigable",
        "persona": (
            "Eres Llama, un compañero estudiante de la UTSLRC que ya pasó por todos estos "
            "trámites. Hablas cercano, ligero y alentador: tuteo, lenguaje natural de estudiante, "
            "frases cortas y máximo un emoji por respuesta. Das primero la respuesta directa y, "
            "si aplica, un consejo práctico."
        ),
    },
}

RULES = (
    "Reglas de contenido:\n"
    "1. Responde únicamente con la información del CONTEXTO RECUPERADO.\n"
    "2. No inventes fechas, montos, requisitos ni correos. Si el contexto no contiene la "
    "respuesta, dilo con claridad y sugiere acudir a Servicios Escolares.\n"
    "3. Al dar una fecha o regla, menciona de qué sección sale (por ejemplo, «Reglamento de "
    "evaluación»).\n"
    "4. Responde siempre en español y con concisión (máximo 150 palabras, salvo que pidan pasos)."
)

EXAMPLE_QUESTIONS = [
    "¿Hasta cuándo puedo reinscribirme en enero-abril 2027?",
    "¿Cuál es la calificación mínima para aprobar?",
    "¿Qué pasa si tengo menos del 80 % de asistencia?",
    "¿A qué hora abre la biblioteca los sábados?",
]


# --------------------------------------------------------------------------------------
# Configuración de página y estilos
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Llama Académico", page_icon="🦙", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Source+Sans+3:wght@400;600&display=swap');
.stApp { font-family: 'Source Sans 3', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.hero {
    padding: 1.5rem 1.7rem; border-radius: 18px; margin-bottom: 1.1rem; color: #fff;
    background: linear-gradient(120deg, #14213D 0%, #1F4E5F 58%, #2F6F5E 100%);
}
.hero-title {
    font-family: 'Bricolage Grotesque', 'Source Sans 3', system-ui, sans-serif;
    font-size: 2.1rem; font-weight: 700; line-height: 1.15; margin: 0; color: #fff;
}
.hero-sub { margin: .35rem 0 0; font-size: 1.05rem; color: rgba(255,255,255,.84); max-width: 62ch; }
.hero-badges { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem; }
.badge {
    padding: .2rem .7rem; border-radius: 999px; font-size: .84rem;
    background: rgba(255,255,255,.13); border: 1px solid rgba(255,255,255,.28);
}
.perf { display: flex; flex-wrap: wrap; gap: .55rem; margin: .7rem 0 .3rem; }
.chip {
    display: flex; flex-direction: column; min-width: 104px; padding: .4rem .75rem;
    border-radius: 12px; border: 1px solid rgba(128,128,128,.32); background: rgba(128,128,128,.07);
}
.chip.main { border-color: #F2A541; background: rgba(242,165,65,.13); }
.chip-label { font-size: .74rem; opacity: .72; }
.chip-value { font-size: 1.1rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.chip.main .chip-value { font-size: 1.3rem; }
.intro {
    padding: 1rem 1.2rem; border-radius: 14px; margin-bottom: .8rem;
    border: 1px solid rgba(128,128,128,.3); background: rgba(128,128,128,.06);
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# RAG: fragmentación, embeddings y similitud coseno
# --------------------------------------------------------------------------------------
def parse_chunks(text: str) -> list[dict]:
    """Cada '### Título' bajo un '## Categoría' se convierte en un fragmento indexable."""
    chunks: list[dict] = []
    categoria, titulo, buf = "", None, []

    def flush() -> None:
        if titulo and buf:
            cuerpo = " ".join(l.strip() for l in buf if l.strip())
            if cuerpo:
                chunks.append(
                    {"id": len(chunks) + 1, "categoria": categoria, "titulo": titulo, "texto": cuerpo}
                )

    for line in text.splitlines():
        if line.startswith("### "):
            flush()
            titulo, buf = line[4:].strip(), []
        elif line.startswith("## "):
            flush()
            categoria, titulo, buf = line[3:].strip(), None, []
        elif line.startswith("#"):
            continue
        else:
            buf.append(line)
    flush()
    return chunks


@st.cache_resource(show_spinner="Cargando modelo de embeddings (solo la primera vez)…")
def load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Indexando la base de conocimiento…")
def build_index(file_mtime: float):
    """Devuelve (fragmentos, matriz de embeddings). Se reconstruye si el .txt cambia."""
    chunks = parse_chunks(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
    textos = [f"{c['categoria']}. {c['titulo']}. {c['texto']}" for c in chunks]
    embeddings = load_embedder().encode(textos, convert_to_numpy=True)
    return chunks, embeddings


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """cos(q, d) = (q · d) / (‖q‖ ‖d‖) para todos los fragmentos a la vez."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return m @ q


def retrieve(question: str, chunks: list[dict], embeddings: np.ndarray, top_k: int) -> list[dict]:
    q_vec = load_embedder().encode([question], convert_to_numpy=True)[0]
    sims = cosine_similarity(q_vec, embeddings)
    best = np.argsort(-sims)[:top_k]
    return [{**chunks[i], "score": float(sims[i])} for i in best]


# --------------------------------------------------------------------------------------
# Tono "ajustado con LoRA" (simulado) y construcción del prompt
# --------------------------------------------------------------------------------------
@st.cache_data
def load_tone_examples(tone_key: str, file_mtime: float) -> list[tuple[str, str]]:
    """Lee los pares pregunta/respuesta del dataset de SFT para el tono elegido."""
    examples: list[tuple[str, str]] = []
    if not TONE_DATA_PATH.exists():
        return examples
    for line in TONE_DATA_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("tone") != tone_key:
            continue
        msgs = row["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
        examples.append((user, assistant))
    return examples


def lora_config_for(tone_key: str) -> dict:
    return {
        "adaptador": f"llama-academico-lora-{tone_key}-v1",
        "entrenamiento (train_lora.py)": {
            "trainer": "trl.SFTTrainer",
            "modelo_base": "Qwen/Qwen2.5-0.5B-Instruct",
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": "all-linear",
            "dataset": "datos_tono.jsonl",
        },
        "inferencia en esta app": "simulada: system prompt de personalidad + ejemplos few-shot del mismo dataset",
    }


def build_system_prompt(
    tone_label: str,
    use_adapter: bool,
    use_rag: bool,
    sources: list[dict],
    fewshot: list[tuple[str, str]],
) -> str:
    parts: list[str] = []
    if use_adapter:
        parts.append(TONES[tone_label]["persona"])
        if fewshot:
            ejemplos = "\n\n".join(f"Alumno: {u}\nRespuesta: {a}" for u, a in fewshot)
            parts.append(
                "Estilo aprendido en el ajuste (imita solo la forma de hablar; los datos "
                "salen siempre del CONTEXTO RECUPERADO, nunca de estos ejemplos):\n\n" + ejemplos
            )
    else:
        parts.append("Eres un asistente virtual de la UTSLRC. Responde de forma clara y neutral.")

    if use_rag and sources:
        parts.append(RULES)
        bloques = [
            f"[Fuente {i} | {s['categoria']} > {s['titulo']} | similitud {s['score']:.2f}]\n{s['texto']}"
            for i, s in enumerate(sources, start=1)
        ]
        contexto = "CONTEXTO RECUPERADO:\n\n" + "\n\n".join(bloques)
        if sources[0]["score"] < MIN_RELEVANCE:
            contexto += (
                "\n\nAviso: la similitud del contexto es baja; probablemente no responde la "
                "pregunta. Si es así, dilo en lugar de adivinar."
            )
        parts.append(contexto)
    else:
        parts.append(
            "El modo RAG está desactivado: no tienes documentos institucionales. No inventes "
            "fechas, montos ni reglas específicas de la UTSLRC; si no conoces el dato exacto, "
            "dilo y sugiere acudir a Servicios Escolares. Responde en español y con concisión."
        )
    return "\n\n".join(parts)


# --------------------------------------------------------------------------------------
# Groq
# --------------------------------------------------------------------------------------
def get_api_key(sidebar_value: str) -> str:
    if sidebar_value.strip():
        return sidebar_value.strip()
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"])
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "").strip()


def call_groq(api_key: str, model: str, messages: list[dict], temperature: float, max_tokens: int):
    from groq import Groq

    client = Groq(api_key=api_key)
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model.startswith("openai/gpt-oss"):
        kwargs["extra_body"] = {"reasoning_effort": "low"}  # respuestas más rápidas

    t0 = time.perf_counter()
    response = client.chat.completions.create(**kwargs)
    llm_seconds = time.perf_counter() - t0

    content = (response.choices[0].message.content or "").strip()
    usage = response.usage
    return content, llm_seconds, usage


def answer(question: str, cfg: dict, api_key: str, history: list[dict]) -> dict:
    """Ejecuta el pipeline completo y devuelve el texto junto con sus métricas."""
    t_start = time.perf_counter()

    sources: list[dict] = []
    retrieval_s = 0.0
    if cfg["use_rag"]:
        chunks, embeddings = build_index(KNOWLEDGE_PATH.stat().st_mtime)
        query = question
        previous_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        if len(question.split()) < 5 and previous_user:  # preguntas cortas de seguimiento
            query = f"{previous_user} {question}"
        t0 = time.perf_counter()
        sources = retrieve(query, chunks, embeddings, cfg["top_k"])
        retrieval_s = time.perf_counter() - t0

    fewshot: list[tuple[str, str]] = []
    if cfg["use_adapter"]:
        ex = load_tone_examples(TONES[cfg["tone"]]["key"], TONE_DATA_PATH.stat().st_mtime)
        fewshot = (ex[:2] + ex[-1:]) if len(ex) >= 3 else ex

    system_prompt = build_system_prompt(cfg["tone"], cfg["use_adapter"], cfg["use_rag"], sources, fewshot)
    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history[-HISTORY_TURNS:]]
    messages.append({"role": "user", "content": question})

    content, llm_s, usage = call_groq(api_key, cfg["model"], messages, cfg["temperature"], cfg["max_tokens"])
    total_s = time.perf_counter() - t_start

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)

    if not content:
        content = (
            "El modelo no devolvió texto. Sube el máximo de tokens de salida en la barra lateral "
            "o prueba con otro modelo."
        )

    return {
        "content": content,
        "meta": {
            "total_s": total_s,
            "llm_s": llm_s,
            "retrieval_s": retrieval_s,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_s": (completion_tokens / llm_s) if llm_s > 0 else 0.0,
            "model": cfg["model"],
            "tone": cfg["tone"] if cfg["use_adapter"] else "Neutral (sin adaptador)",
            "use_rag": cfg["use_rag"],
            "sources": sources,
        },
    }


# --------------------------------------------------------------------------------------
# Componentes de interfaz
# --------------------------------------------------------------------------------------
def render_metrics(meta: dict) -> None:
    chips = [
        ("main", "Tiempo de respuesta", f"{meta['total_s']:.2f} s"),
        ("main", "Tokens totales", f"{meta['total_tokens']:,}"),
        ("", "Prompt / respuesta", f"{meta['prompt_tokens']:,} / {meta['completion_tokens']:,}"),
        ("", "Solo el LLM", f"{meta['llm_s']:.2f} s"),
        ("", "Recuperación RAG", f"{meta['retrieval_s'] * 1000:.0f} ms" if meta["use_rag"] else "apagado"),
        ("", "Velocidad", f"{meta['tokens_per_s']:.0f} tok/s"),
    ]
    html_chips = "".join(
        f'<div class="chip {cls}"><span class="chip-label">{label}</span>'
        f'<span class="chip-value">{html.escape(value)}</span></div>'
        for cls, label, value in chips
    )
    st.markdown(f'<div class="perf">{html_chips}</div>', unsafe_allow_html=True)
    st.caption(f"Modelo: {meta['model']}. Tono: {meta['tone']}.")


def render_sources(meta: dict) -> None:
    if not meta["use_rag"] or not meta["sources"]:
        return
    with st.expander("Fragmentos recuperados por el RAG"):
        for i, s in enumerate(meta["sources"], start=1):
            st.markdown(f"**{i}. {s['categoria']} > {s['titulo']}** (similitud coseno {s['score']:.3f})")
            st.progress(min(max(s["score"], 0.0), 1.0))
            st.caption(s["texto"])


def render_assistant_extras(meta: dict) -> None:
    render_metrics(meta)
    render_sources(meta)


# --------------------------------------------------------------------------------------
# Estado y barra lateral
# --------------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stats" not in st.session_state:
    st.session_state.stats = {"queries": 0, "time": 0.0, "tokens": 0}

with st.sidebar:
    st.markdown("### Configuración")
    api_key_input = st.text_input(
        "Groq API key",
        type="password",
        placeholder="gsk_… (o usa .env / secrets)",
        help="Si lo dejas vacío se usa GROQ_API_KEY del archivo .env, de st.secrets o del entorno.",
    )
    model_choice = st.selectbox("Modelo", MODELS)
    custom_model = st.text_input("Otro modelo de Groq (opcional)", placeholder="p. ej. openai/gpt-oss-120b")
    tone_label = st.radio("Tono de respuesta", list(TONES))
    use_adapter = st.toggle("Adaptador de tono LoRA (simulado)", value=True)
    use_rag = st.toggle("RAG con base de conocimiento", value=True)
    top_k = st.slider("Fragmentos a recuperar (top-k)", 1, 5, 3)
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.3, 0.05)
    max_tokens = st.slider("Máximo de tokens de salida", 256, 4096, 1500, 128)

    with st.expander("Configuración del adaptador LoRA"):
        st.json(lora_config_for(TONES[tone_label]["key"]))

    if st.button("Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.stats = {"queries": 0, "time": 0.0, "tokens": 0}
        st.rerun()

    stats_box = st.container()

cfg = {
    "model": custom_model.strip() or model_choice,
    "tone": tone_label,
    "use_adapter": use_adapter,
    "use_rag": use_rag,
    "top_k": top_k,
    "temperature": temperature,
    "max_tokens": max_tokens,
}
api_key = get_api_key(api_key_input)

# --------------------------------------------------------------------------------------
# Página principal
# --------------------------------------------------------------------------------------
badges = [
    f"Modelo: {html.escape(cfg['model'])}",
    f"Tono: {html.escape(tone_label) if use_adapter else 'neutral'}",
    "RAG activo" if use_rag else "RAG apagado",
]
st.markdown(
    f"""
<div class="hero">
  <div class="hero-title">🦙 Llama Académico</div>
  <p class="hero-sub">Resuelve tus dudas sobre inscripciones, evaluación y horarios de la UTSLRC, a cualquier hora del día.</p>
  <div class="hero-badges">{''.join(f'<span class="badge">{b}</span>' for b in badges)}</div>
</div>
""",
    unsafe_allow_html=True,
)

if use_rag:
    build_index(KNOWLEDGE_PATH.stat().st_mtime)  # precarga: la 1ª consulta no incluye la carga del modelo

if not st.session_state.messages:
    st.markdown(
        '<div class="intro">Escribe tu duda o prueba con una de estas preguntas. Cada respuesta '
        "muestra su tiempo de respuesta, los tokens consumidos y los fragmentos del reglamento "
        "que usó.</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(question, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending = question

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🦙" if m["role"] == "assistant" else None):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_assistant_extras(m["meta"])

user_input = st.chat_input("Pregunta sobre fechas, reglas de evaluación u horarios…")
if not user_input and "pending" in st.session_state:
    user_input = st.session_state.pop("pending")

if user_input:
    if not api_key:
        st.error("Falta la API key de Groq. Pégala en la barra lateral o defínela como GROQ_API_KEY en .env.")
    else:
        history = list(st.session_state.messages)
        with st.chat_message("user"):
            st.markdown(user_input)
        result = None
        with st.chat_message("assistant", avatar="🦙"):
            try:
                with st.spinner("Buscando en el reglamento y generando la respuesta…"):
                    result = answer(user_input, cfg, api_key, history)
            except Exception as exc:  # errores de red, clave inválida, modelo no disponible…
                st.error(f"No se pudo obtener la respuesta: {exc}")
            if result:
                st.markdown(result["content"])
                render_assistant_extras(result["meta"])
        if result:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append(
                {"role": "assistant", "content": result["content"], "meta": result["meta"]}
            )
            s = st.session_state.stats
            s["queries"] += 1
            s["time"] += result["meta"]["total_s"]
            s["tokens"] += result["meta"]["total_tokens"]

with stats_box:
    s = st.session_state.stats
    st.markdown("### Sesión")
    c1, c2 = st.columns(2)
    c1.metric("Consultas", s["queries"])
    c2.metric("Tokens", f"{s['tokens']:,}")
    avg = s["time"] / s["queries"] if s["queries"] else 0.0
    st.metric("Tiempo promedio", f"{avg:.2f} s")

st.caption("La base de conocimiento es un documento simulado para el hackathon; no son fechas ni reglas oficiales.")
