# 🦙 Llama Académico

Asistente universitario 24/7 para alumnos de la UTSLRC. Resuelve dudas sobre fechas de inscripción, reglas de evaluación y horarios, con el tono de un coordinador formal o de un compañero amigable.

## Cómo correrlo

Requiere Python 3.10 o superior y una API key gratuita de Groq (https://console.groq.com/keys).

```bash
# 1. Descomprime el ZIP y entra a la carpeta
cd llama_academico

# 2. (Recomendado) entorno virtual
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Dependencias
pip install -r requirements.txt

# 4. API key: copia .env.example como .env y pega tu clave
#    (también puedes pegarla en la barra lateral de la app)
cp .env.example .env               # Windows: copy .env.example .env

# 5. Lanza la app
streamlit run app.py
```

La primera consulta con RAG descarga el modelo `all-MiniLM-L6-v2` (unos 90 MB); después queda en caché.

## Estructura

| Archivo | Para qué sirve |
|---|---|
| `app.py` | App Streamlit: RAG, llamada a Groq, tono y métricas |
| `conocimiento.txt` | Base de conocimiento simulada (calendario, evaluación, horarios, servicios) |
| `datos_tono.jsonl` | Pares pregunta/respuesta en tono formal y amigable (dataset de SFT y few-shot) |
| `train_lora.py` | Entrenamiento real del adaptador LoRA con `trl.SFTTrainer` (opcional) |
| `requirements.txt` | Dependencias de la app |
| `requirements-train.txt` | Dependencias extra, solo para entrenar |

## Cómo cubre cada requisito

1. **Integración con Groq.** La clave se lee de la barra lateral, de `st.secrets` o de `GROQ_API_KEY` en `.env`; nunca queda en el código. Modelos: `openai/gpt-oss-20b`, `llama-3.3-70b-versatile` y `llama-3.1-8b-instant`, más un campo para escribir cualquier otro modelo de Groq.
2. **RAG.** `conocimiento.txt` se divide por secciones (`###`), se vectoriza con `sentence-transformers/all-MiniLM-L6-v2`, se calcula la similitud coseno entre la pregunta y cada fragmento (`cosine_similarity` en `app.py`) y los `top-k` mejores se inyectan en el system prompt. Cada respuesta muestra los fragmentos usados con su puntaje.
3. **Tono ajustado con LoRA.** `train_lora.py` entrena un adaptador con `LoraConfig` + `SFTTrainer` sobre `datos_tono.jsonl`. En la app, el adaptador se **simula**: el system prompt adopta la personalidad entrenada y se añaden como few-shot ejemplos del mismo dataset. El interruptor "Adaptador de tono LoRA" permite comparar con y sin él.
4. **Medición de desempeño.** Cada respuesta muestra el tiempo total en segundos (recuperación + LLM), el tiempo solo del LLM, los tokens totales (prompt y respuesta, tal como los reporta Groq) y la velocidad en tokens por segundo. La barra lateral acumula consultas, tokens y tiempo promedio de la sesión.

## Nota para presentar el proyecto

Groq no sirve pesos LoRA propios en cuentas estándar, hasta donde sé, por eso la inferencia usa el tono simulado. Conviene decirlo abiertamente al jurado: el entrenamiento es real y reproducible con `train_lora.py`, y el adaptador resultante se puede cargar con `peft` si más adelante sirves el modelo base por tu cuenta (por ejemplo con `transformers` o vLLM).

## Ideas para mejorar el resultado

- `all-MiniLM-L6-v2` está optimizado para inglés. Para mejor recuperación en español, cambia `EMBED_MODEL` en `app.py` por `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Sustituye el contenido de `conocimiento.txt` por el reglamento real; respeta el formato `## Categoría` / `### Título`.
- Amplía `datos_tono.jsonl` antes de entrenar: con 6 ejemplos por tono solo sirve como demostración.

> La base de conocimiento incluida es ficticia (fechas, cuotas y reglas inventadas para el hackathon).
