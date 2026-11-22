# OBJETIVO: Leer los documentos de la carpeta 'documents/', trocearlos (chunking),
# generar embeddings con el modelo local y guardarlos en Neon.

from pathlib import Path
from pypdf import PdfReader
from conexion import MODELO, conectar, embedding


def leer_documentos(carpeta: str = "documents") -> list[tuple[str, str]]:
    """Lee todos los archivos .txt y .pdf dentro de la carpeta indicada."""
    ruta_docs = Path(__file__).parent / carpeta
    if not ruta_docs.exists():
        raise FileNotFoundError(f"No existe la carpeta '{carpeta}/'.")

    documentos = []
    for archivo in sorted(ruta_docs.iterdir()):
        texto = ""
        if archivo.suffix.lower() == ".txt":
            texto = archivo.read_text(encoding="utf-8")
        elif archivo.suffix.lower() == ".pdf":
            lector = PdfReader(archivo)
            texto = "\n".join([pagina.extract_text() or "" for pagina in lector.pages])

        if texto.strip():
            documentos.append((archivo.name, texto))

    return documentos


def trocear(texto: str, tamano: int = 400, solapamiento: int = 50) -> list[str]:
    """Divide un texto largo en fragmentos (chunks de 400) con solapamiento de 50 para no perder contexto."""
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + tamano
        fragmento = texto[inicio:fin].strip()
        if fragmento:
            chunks.append(fragmento)
        inicio += tamano - solapamiento
    return chunks


def main():
    conn = conectar()

    # Vaciamos la tabla antes de empezar para evitar duplicados
    with conn.transaction():
        conn.execute("TRUNCATE rag_documentos RESTART IDENTITY;")

    docs = leer_documentos("documents")
    if not docs:
        raise ValueError("La carpeta 'documents/' está vacía.")

    total_chunks = 0
    for nombre_archivo, contenido in docs:
        fragmentos = trocear(contenido)
        vectores = embedding(fragmentos)
        print(f"Modelo {MODELO} [{nombre_archivo}]: {len(vectores)} fragmentos vectorizados")

        with conn.transaction():
            for frag, emb in zip(fragmentos, vectores):
                conn.execute(
                    "INSERT INTO rag_documentos (fuente, texto, embedding) VALUES (%s, %s, %s);",
                    (nombre_archivo, frag, emb)
                )
        total_chunks += len(fragmentos)

    total_filas = conn.execute("SELECT count(*) FROM rag_documentos;").fetchone()[0]
    print(f"\nListo: {total_filas} fragmentos de {len(docs)} documentos guardados en Neon.")
    conn.close()


if __name__ == "__main__":
    main()
