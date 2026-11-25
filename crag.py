import warnings
warnings.filterwarnings("ignore")

import json
import os
from datetime import date
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness
from tavily import TavilyClient

# Reutilizamos tu conexión y modelo de embeddings de la Sesión 5
from conexion import conectar, embedding

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
modelo_llm = os.getenv("MODEL")
base_url = os.getenv("BASE_URL")

client = OpenAI(base_url=base_url, api_key=api_key)


# NEON 
def recuperar_neon(pregunta: str, k: int = 3) -> list[dict]:
    q = embedding(pregunta)
    conn = conectar()
    filas = conn.execute(
        """
        SELECT fuente, texto, embedding <=> %s AS distancia
        FROM rag_documentos
        ORDER BY embedding <=> %s
        LIMIT %s;
        """,
        (q, q, k),
    ).fetchall()
    conn.close()
    return [{"fuente": f[0], "texto": f[1], "distancia": float(f[2])} for f in filas]


# EVALUACIÓN DE RELEVANCIA DE LOS DOCUMENTOS RECUPERADOS

EVALUAR = """Eres un evaluador. Del 1 al 10, ¿cuánto sirve el FRAGMENTO para responder a la PREGUNTA?
10 = la responde por completo · 7 = la responde casi entera · 5 = habla del tema pero no la responde ·
2 = apenas relacionado · 1 = nada que ver.
Responde SOLO con JSON válido: {{"nota": <entero del 1 al 10>, "motivo": "<una frase corta>"}}

PREGUNTA: {pregunta}
FRAGMENTO: {texto}"""

# >= 7 correcto · 4-6 ambiguo · <= 3 incorrecto
CORRECTO, AMBIGUO = 0.7, 0.4  


def puntuar(pregunta: str, chunk: dict) -> dict:
    resp = client.chat.completions.create(
        model=modelo_llm,
        temperature=0,
        messages=[{"role": "user", "content": EVALUAR.format(pregunta=pregunta, texto=chunk["texto"])}],
    ).choices[0].message.content.strip()

    resp_limpia = resp.replace("```json", "").replace("```", "").strip()
    try:
        r = json.loads(resp_limpia)
        p = float(r["nota"]) / 10 if "nota" in r else float(r.get("puntuacion", 0))
        motivo = r.get("motivo", "")
    except (ValueError, TypeError, KeyError):
        p, motivo = 0.0, "(el evaluador no devolvió un JSON válido)"

    clase = "correcto" if p >= CORRECTO else "ambiguo" if p >= AMBIGUO else "incorrecto"
    return {**chunk, "puntuacion": p, "clase": clase, "motivo": motivo}


def decidir(evaluados: list[dict]) -> str:
    clases = {c["clase"] for c in evaluados}
    return "correcto" if "correcto" in clases else "ambiguo" if "ambiguo" in clases else "incorrecto"


# FALLBACK: REFINAR, REESCRIBIR Y BUSCAR EN LA WEB

REFINAR = """Del siguiente texto, copia SOLO las frases que ayudan a responder la pregunta, tal cual, sin añadir nada.
Si ninguna ayuda, responde exactamente: NADA

PREGUNTA: {pregunta}
TEXTO: {texto}"""


def refinar(pregunta: str, chunks: list[dict]) -> list[dict]:
    refinados = []
    for c in chunks:
        util = client.chat.completions.create(
            model=modelo_llm,
            temperature=0,
            messages=[{"role": "user", "content": REFINAR.format(pregunta=pregunta, texto=c["texto"])}],
        ).choices[0].message.content.strip()
        if util and not util.upper().startswith("NADA"):
            refinados.append({**c, "texto": util})
    return refinados


REESCRIBIR = """Convierte la pregunta en una buena consulta para un buscador web: palabras clave concretas, en español,
sin signos de interrogación, añadiendo el país si hace falta (España) y el año si el dato cambia con el tiempo.
 Devuelve SOLO la consulta.

PREGUNTA: {pregunta}
CONSULTA:"""


def reescribir(pregunta: str) -> str:
    consulta = client.chat.completions.create(
        model=modelo_llm,
        temperature=0,
        messages=[
            {"role": "user", "content": REESCRIBIR.format(pregunta=pregunta, hoy=date.today().strftime("%d/%m/%Y"))}
        ],
    ).choices[0].message.content.strip().strip('"')
    if not consulta or len(consulta) > 150 or any(x in consulta.lower() for x in ("lo siento", "no puedo")):
        return pregunta
    return consulta


def buscar_web(consulta: str, n: int = 3) -> list[dict]:
    """Busca en internet usando la API de Tavily como fallback."""
    try:
        tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        respuesta = tavily.search(query=consulta, max_results=n)
        return [
            {"fuente": r["url"], "texto": f"{r['title']}. {r['content']}"}
            for r in respuesta.get("results", [])
        ]
    except Exception as e:
        print(f"Error en Tavily: {e}")
        return []


#  GENERACIÓN CON CONTROL DE CONTEXTO VACÍO

def generar(pregunta: str, chunks: list[dict]) -> str:
    if not chunks:
        return "No lo sé: no aparece en los documentos ni en la búsqueda web."

    contexto = "\n\n---\n\n".join([f"[Fuente: {c['fuente']}]\n{c['texto']}" for c in chunks])
    return client.chat.completions.create(
        model=modelo_llm,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente estricto de consulta documental. "
                    "Responde utilizando ÚNICAMENTE la información explícita en el CONTEXTO.\n"
                    "1. Si la respuesta no está en el contexto, responde exactamente: 'No lo sé'.\n"
                    "2. No deduzcas ni inventes datos externos.\n"
                    "3. Cita siempre la fuente entre corchetes al final."
                ),
            },
            {"role": "user", "content": f"CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"},
        ],
    ).choices[0].message.content.strip()


# FLUJO RAG (ANTES) VS CRAG "Corrective rag" (DESPUÉS)

def ejecutar_rag_basico(pregunta: str) -> tuple[str, list[dict]]:
    chunks = recuperar_neon(pregunta, k=3)
    return generar(pregunta, chunks), chunks


def ejecutar_crag(pregunta: str) -> dict:
    # Recuperar y evaluar relevancia
    recuperados = recuperar_neon(pregunta, k=3)
    evaluados = [puntuar(pregunta, c) for c in recuperados]
    ruta = decidir(evaluados)
    consulta_web = ""
    autocorreccion = False

    # Enrutamiento según la evaluación
    if ruta == "correcto":
        contexto_final = [c for c in evaluados if c["clase"] == "correcto"]
        respuesta = generar(pregunta, contexto_final)

        # Red de seguridad: si el evaluador se equivocó y el generador dice "No lo sé", vamos a la web
        if respuesta.startswith("No lo sé"):
            autocorreccion = True
            consulta_web = reescribir(pregunta)
            web_chunks = [puntuar(pregunta, w) for w in buscar_web(consulta_web)]
            contexto_final = [w for w in web_chunks if w["clase"] != "incorrecto"]
            respuesta = generar(pregunta, contexto_final)

    elif ruta == "ambiguo":
        refinados = refinar(pregunta, [c for c in evaluados if c["clase"] == "ambiguo"])
        consulta_web = reescribir(pregunta)
        web_chunks = [puntuar(pregunta, w) for w in buscar_web(consulta_web)]
        contexto_final = refinados + [w for w in web_chunks if w["clase"] != "incorrecto"]
        respuesta = generar(pregunta, contexto_final)

    else:  # ruta == "incorrecto" -> Fallback web directo
        consulta_web = reescribir(pregunta)
        web_chunks = [puntuar(pregunta, w) for w in buscar_web(consulta_web)]
        contexto_final = [w for w in web_chunks if w["clase"] != "incorrecto"]
        respuesta = generar(pregunta, contexto_final)

    return {
        "evaluados": evaluados,
        "ruta": ruta,
        "autocorreccion": autocorreccion,
        "consulta_web": consulta_web,
        "contexto": contexto_final,
        "respuesta": respuesta,
    }


#  COMPARATIVA Y MEDICIÓN FAITHFULNESS CON RAGAS

if __name__ == "__main__":
    preguntas_prueba = [
        # CORRECTO (está en normativa TFG)
        "¿Cuánto tiempo dura la exposición del TFG en la Fase I y qué penalización se aplica si dura menos de 14 minutos?",
        # AMBIGUO (está en el manual de limpieza, pero formulada de manera más cotidiana)
        "¿Qué hago si me cae el producto de limpieza en los ojos y a quién llamo?",
        # INCORRECTO -> Fallback Tavily (no está en documentos)
        "¿Cuál es el salario mínimo interprofesional en España?",
    ]

    juez = LangchainLLMWrapper(
        ChatOpenAI(model=modelo_llm, api_key=api_key, base_url=base_url, temperature=0)
    )
    config = RunConfig(timeout=300, max_workers=1, max_retries=2)
    f = lambda v: "NaN" if v != v else f"{v:.2f}"

    scores_basico, scores_crag = [], []

    for i, pregunta in enumerate(preguntas_prueba, 1):
        print("\n" + "=" * 90)
        print(f"PREGUNTA {i}: {pregunta}")
        print("=" * 90)

        # ANTES: RAG Básico
        resp_basico, ctx_basico = ejecutar_rag_basico(pregunta)

        # DESPUÉS: CRAG(Evaluador + Fallback Tavily)
        res_crag = ejecutar_crag(pregunta)

        # MEDIR FAITHFULNESS CON RAGAS
        def medir(resp, ctx):
            ds = EvaluationDataset.from_list([{
                "user_input": pregunta,
                "response": resp,
                "retrieved_contexts": [c["texto"] for c in ctx] or ["(sin contexto)"],
            }])
            ev = evaluate(ds, metrics=[Faithfulness()], llm=juez, run_config=config, show_progress=False)
            return float(ev.to_pandas()["faithfulness"].tolist().pop(0))

        s_basico = medir(resp_basico, ctx_basico)
        s_crag = medir(res_crag["respuesta"], res_crag["contexto"])
        if s_basico == s_basico: scores_basico.append(s_basico)
        if s_crag == s_crag: scores_crag.append(s_crag)

        print(f"\n[ANTES - RAG]  Faithfulness: {f(s_basico)}")
        print(f"  Respuesta: {resp_basico}")

        print(f"\n[DESPUÉS - CRAG + Tavily] Ruta: {res_crag['ruta'].upper()} | Faithfulness: {f(s_crag)}")
        if res_crag["consulta_web"]:
            print(f"  Consulta Tavily: '{res_crag['consulta_web']}'")
        print(f"  Fuentes:   {[c['fuente'] for c in res_crag['contexto']]}")
        print(f"  Respuesta: {res_crag['respuesta']}")

    # MEDIA FINAL
    media_b = sum(scores_basico) / len(scores_basico) if scores_basico else 0.0
    media_c = sum(scores_crag) / len(scores_crag) if scores_crag else 0.0
    print("\n" + "=" * 80)
    print(f"MEDIA GLOBAL FAITHFULNESS -> RAG Básico: {media_b:.2f}  |  CRAG: {media_c:.2f}")
    print("=" * 90)
