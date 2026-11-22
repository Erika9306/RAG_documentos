# OBJETIVO: Leer el archivo 'esquema.sql', ejecutarlo en Neon y comprobar
# que la extensión, las columnas y la versión estén listas.

from pathlib import Path
from conexion import conectar

# Conectamos con registrar_vector=False (porque la extensión aún no existe)
# y autocommit=True para que Neon guarde todos los cambios al instante.
conn = conectar(registrar_vector=False, autocommit=True)

ruta_sql = Path(__file__).parent / "esquema.sql"

with open(ruta_sql, "r", encoding="utf-8") as archivo_sql:
    sql_completo = archivo_sql.read()

# Ejecutamos todas las órdenes SQL que leímos del archivo
conn.execute(sql_completo)

# COMPROBAR VERSIÓN DE PGVECTOR (DIAGNÓSTICO):
version = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';").fetchone()[0]
print(f"Extensión vector activa: PGVECTOR {version}")

# MOSTRAR EN PANTALLA LAS COLUMNAS DE LA TABLA CREADA:
print("\nTabla 'rag_documentos' creada:")
columnas = conn.execute("""
    SELECT column_name AS nombre, data_type AS tipo
    FROM information_schema.columns 
    WHERE table_name = 'rag_documentos'
    ORDER BY ordinal_position;
""").fetchall()

for nombre, tipo in columnas:
    print(f"  - {nombre}: {tipo}")

# MOSTRAR VERSIÓN DE POSTGRESQL EN NEON:
servidor = conn.execute("SHOW server_version;").fetchone()[0]
print(f"\nConectado a PostgreSQL {servidor}")

conn.close()
