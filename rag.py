# OBJETIVO: Recuperar los fragmentos más relevantes de Neon y pasarlos a OpenRouter
# para responder SOLO con esa información o decir 'No lo sé'.

import os
import sys
from openai import OpenAI
from conexion import conectar, embedding

api_key = os.getenv("OPENAI_API_KEY")
modelo_llm = os.getenv("MODEL")
url = os.getenv("BASE_URL")

# Apuntamos a OpenRouter porque tu clave empieza por sk-or-v1-
client = OpenAI(
    base_url= url,
    api_key=api_key,
)


def recuperar_contexto(pregunta: str, limite: int = 3):
    """
     Convierte la pregunta en vector y recupera los 'limite' fragmentos
    más parecidos usando distancia coseno (<=>).
    """
    q = embedding(pregunta)
    conn = conectar()
    resultados = conn.execute("""
        SELECT fuente, texto, embedding <=> %s AS distancia
        FROM rag_documentos
        ORDER BY embedding <=> %s
        LIMIT %s;
    """, (q, q, limite)).fetchall()
    conn.close()
    return resultados


def preguntar_rag(pregunta: str):
    print(f"\n{'='*65}")
    print(f"PREGUNTA: {pregunta}")

    # Recuperación de resultados
    resultados = recuperar_contexto(pregunta, limite=3)

    print("\n--- FRAGMENTOS RECUPERADOS DE NEON ---")
    bloque_contexto = []
    for i, (fuente, texto, distancia) in enumerate(resultados, 1):
        similitud = 1 - distancia
        resumen = texto.replace("\n", " ")[:85]
        print(f"{i}. [{fuente}] distancia {distancia:.3f} · similitud {similitud:.3f} | {resumen}...")
        bloque_contexto.append(f"[Fuente: {fuente}]\n{texto}")

    contexto = "\n\n---\n\n".join(bloque_contexto)

    # Pasar contexto al modelo con restricción estricta
    respuesta = client.chat.completions.create(
        model=modelo_llm,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente estricto de consulta documental. "
                    "Responde a la pregunta del usuario utilizando ÚNICAMENTE la información "
                    "que aparece explícitamente en el CONTEXTO proporcionado.\n"
                    "Reglas obligatorias:\n"
                    "1. Si la respuesta no está en el contexto, responde exactamente: 'No lo sé'.\n"
                    "2. No deduzcas, no inventes y no utilices conocimiento externo.\n"
                    "3. Sé claro y directo."
                )
            },
            {
                "role": "user",
                "content": f"CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"
            }
        ]
    )

    print(f"\nRESPUESTA:\n> {respuesta.choices[0].message.content}")
    # Repite el símbolo = 65 veces seguidas para dibujar una línea separadora en la pantalla
    print(f"{'='*65}")
    


if __name__ == "__main__":
    if len(sys.argv) > 1:
        preguntar_rag(sys.argv)
    else:
        # =========================================================
        # DOCUMENTO 1: PDF Manual de Limpieza y Desinfección
        # =========================================================
        # SÍ respondible:
        preguntar_rag("¿Cuál es el teléfono del Servicio Médico de Atención Toxicológica y qué hacer si salpica un producto en los ojos?")
        # NO respondible ('No lo sé'):
        preguntar_rag("¿Cuánto cuesta una botella de quitagrasas KH-7 en el supermercado?")

        # =========================================================
        # DOCUMENTO 2: TXT juego CRONOS-9
        # =========================================================
        # SÍ respondible:
        preguntar_rag("En Cronos-9, ¿cómo se determina el jugador inicial y qué recursos iniciales recibe cada jugador?")
        # NO respondible ('No lo sé'):
        preguntar_rag("¿Cuál es el precio de venta al público del juego Cronos-9 y cuántas expansiones tiene?")

        # =========================================================
        # DOCUMENTO 3: TXT Normativa TFG 
        # =========================================================
        # SÍ respondible:
        preguntar_rag("¿Cuánto tiempo dura la exposición del TFG en la Fase I y qué penalización se aplica si dura menos de 14 minutos?")

        # NO respondible ('No lo sé'):
        preguntar_rag("¿Qué día exacto y a qué hora se celebra la defensa pública de Alejandro Valdivia Mendoza?")
