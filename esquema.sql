-- OBJETIVO: Definir la estructura de la base de datos en Neon
-- (pgvector, tabla 'rag_documentos' e índice HNSW).

-- ACTIVAR PGVECTOR:
CREATE EXTENSION IF NOT EXISTS vector;

-- CREAR TABLA 'rag_documentos':
DROP TABLE IF EXISTS rag_documentos;

CREATE TABLE rag_documentos (
    id         SERIAL PRIMARY KEY,
    fuente     TEXT NOT NULL,
    texto      TEXT NOT NULL,
    embedding  VECTOR(384) NOT NULL
);

-- ÍNDICE HNSW PARA BÚSQUEDA RÁPIDA POR SIMILITUD COSENO:
CREATE INDEX rag_embedding_hnsw ON rag_documentos USING hnsw (embedding vector_cosine_ops);
