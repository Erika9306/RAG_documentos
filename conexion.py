# OBJETIVO: Guardar las constantes y abrir la conexión segura hacia Neon,
# además de cargar el modelo de embeddings una sola vez.


import os  
from dotenv import load_dotenv
# Permite a Python hablar con PostgreSQL (Neon)
import psycopg
# Enseña a Python a enviar y recibir listas de números (vectores).
from pgvector.psycopg import register_vector

load_dotenv()

# CONSTANTES COMPARTIDAS EN TODO EL PROYECTO:
# 384 es la cantidad de números que genera el modelo elegido para cada frase.
DIMENSIONES = 384
# El modelo gratuito de Hugging Face que entiende español y otros idiomas.
MODELO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def conectar(registrar_vector: bool = True, autocommit: bool = False) -> psycopg.Connection:
    """
    Abre una conexión a Neon y activa el soporte de vectores si ya existe.
    - 'autocommit=True' guarda cada cambio al instante.
    - 'autocommit=False' permite agrupar operaciones en transacciones seguras (Pooler + RLS).
    """
    url = os.getenv("NEON_POSTGRE")
    if not url:
        raise ValueError("ERROR: Falta la variable 'NEON_POSTGRE' en tu archivo .env")

    conn = psycopg.connect(url, autocommit=autocommit)

    # Para pasar listas/arrays de Python como columnas vector
    if registrar_vector:
        register_vector(conn)
    return conn


# _modelo se usa en Python para crear un espacio vacío reservado antes de cargar un objeto pesado,
# técnica conocida como carga perezosa (lazy loading) o patrón Singleton.
# RESERVAMOS UNA CAJA VACÍA EN MEMORIA:
# - El guion bajo (_) indica que es una variable interna de este archivo.
# - 'None' significa que todavía está vacía (aún no hemos cargado el modelo pesado).
# - Así, cuando ejecutemos 'setup_db.py', el programa no perderá tiempo cargando la IA.
_modelo = None


def embedding(textos):
    """
    Un embedding es la traducción de una palabra o frase a una lista de números (llamada vector)
    para que una computadora pueda entender su significado y medir el parecido entre textos.
    """
    # 'global' significa que si no existe crear una "caja" nueva y si existe sobreescribirla.
    # Así el modelo queda guardado en memoria para todo el programa sin recargarse.
    global _modelo
    if _modelo is None:
        from sentence_transformers import SentenceTransformer
        _modelo = SentenceTransformer(MODELO)

    # normalize_embeddings=True ajusta los vectores a longitud 1 para que
    # el cálculo de similitud coseno sea exacto y muy rápido.
    return _modelo.encode(textos, normalize_embeddings=True)
