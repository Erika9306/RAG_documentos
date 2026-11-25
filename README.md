# Sistema RAG Mínimo con PGVector (Neon)

Sistema de **RAG** desarrollado en Python sobre **PostgreSQL + PGVector (Neon)**. Indexa una colección de documentos (`.pdf` y `.txt`) mediante embeddings multilingües locales y responde preguntas utilizando **únicamente** la información recuperada de dichos documentos, respondiendo `"No lo sé"` cuando el dato solicitado no aparece en ellos.

Este proyecto reutiliza la infraestructura de conexión a Neon (`conexion.py`), el modelo local de embeddings (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensiones) y la estructura de índices vectoriales `HNSW` desarrollados en la **Sesión 4 (PGVector)**, ampliándolos con un pipeline de *chunking* documental (`embeddings.py`) y generación aumentada (`rag.py`).


---

### Documentos
1. **`ManualDeLimpieza_290620_B.pdf`**
2. **`juego.txt`**
3. **`manual.txt`**

---

## Instalación y Ejecución

### 1. Crear entorno virtual e instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Configurar variables de entorno
Crea un archivo `.env` a partir de `.env.example` con tus credenciales (incluyendo `TAVILY_API_KEY` para la extensión de Corrective RAG).

### 3. Ejecutar el pipeline completo
```bash
python setup_db.py     
python embeddings.py   
python rag.py         
python crag.py         
```

---

## Ejemplos de Pregunta-Respuesta (Entregable)

### Ejemplo 1: Pregunta respondible (información presente en los documentos)
* **Pregunta:**
  > *¿Cuál es el teléfono del Servicio Médico de Atención Toxicológica y qué hacer si salpica un producto en los ojos?*
* **Fragmento recuperado:** `ManualDeLimpieza_290620_B.pdf` (distancia `0.366` · similitud `0.634`)
* **Respuesta:**
  > El teléfono es el **91 562 04 20**. Si el producto salpica en los ojos, hay que **lavarlos inmediatamente con abundante agua** y acudir al oftalmólogo si fuera necesario, llevando si es posible la etiqueta o el envase del producto.

![Captura de ejemplo respondido](img/pregunta1_respondido.png)

---

### Ejemplo 2: Pregunta NO respondible (información ausente en los documentos)
* **Pregunta:**
  > *¿Cuánto cuesta una botella de quitagrasas KH-7 en el supermercado?*
* **Fragmento recuperado:** `ManualDeLimpieza_290620_B.pdf` (distancia `0.633` · similitud `0.367`)
* **Respuesta:**
  > No lo sé

![Captura de ejemplo sin respuesta](img/pregunta2_no.png)

---

### Pruebas adicionales verificadas sobre los otros documentos

* **Documento 2 (`juego.txt` — Respondible):** *En Cronos-9, ¿cómo se determina el jugador inicial y qué recursos iniciales recibe cada jugador?* -> **Respuesta:** El jugador que haya reparado o cambiado la pila de un reloj más recientemente recibe la Llave Maestra de Cuerda y comienza la partida. Por orden de turno, los demás reciben: 2.º, 1 cubo de Latón; 3.º, 1 cubo de Acero Templado; 4.º, 1 cubo de Acero Templado y 1 Ficha de Éter Líquido.

  ![Captura de prueba Cronos-9 respondida](img/pregunta3.png)

* **Documento 2 (`juego.txt` — Sin datos):** *¿Cuál es el precio de venta al público del juego Cronos-9 y cuántas expansiones tiene?* -> **Respuesta:** `No lo sé`

  ![Captura de prueba Cronos-9 sin respuesta](img/pregunta4_no.png)

* **Documento 3 (`manual.txt` — Respondible):** *¿Cuánto tiempo dura la exposición del TFG en la Fase I y qué penalización se aplica si dura menos de 14 minutos?* -> **Respuesta:** La exposición de la Fase I tiene un máximo de 20 minutos. Si dura menos de 14 minutos, se aplica una penalización de 0,75 puntos por falta de rigor expositivo.

  ![Captura de prueba manual TFG respondida](img/pregunta5.png)

* **Documento 3 (`manual.txt` — Sin datos):** *¿Qué día exacto y a qué hora se celebra la defensa pública de Alejandro Valdivia Mendoza?* -> **Respuesta:** `No lo sé`

  ![Captura de prueba manual TFG sin respuesta](img/pregunta6_no.png)


### Estrategia de Troceado (*Chunking*) y Recuperación
* **Chunking con solapamiento:** Los documentos se dividen en fragmentos de **400 caracteres** con un **solapamiento (*overlap*) de 50 caracteres** (`embeddings.py`). Esto evita que frases o datos clave en los bordes de un corte pierdan su contexto semántico.
* **Recuperación Top-3:** Ante cada pregunta, `rag.py` recupera los 3 fragmentos con menor distancia coseno (`<=>`) en Neon y los inyecta íntegros en el contexto del modelo (en consola se muestra únicamente una vista previa de los primeros 85 caracteres de cada fragmento).

---

## Extensión: Evaluación con CRAG y Evaluación con RAGAS

Como ampliación al RAG mínimo anterior, se ha implementado una arquitectura de **Corrective RAG (`crag.py`)** que añade control de calidad sobre los fragmentos recuperados y evaluación automática mediante la métrica **`Faithfulness` de RAGAS**.

### Mejoras incorporadas
* **Evaluación de relevancia (`puntuar` y `decidir`):** Un juez LLM puntúa de 0 a 1 cada fragmento recuperado de Neon y clasifica la consulta en tres rutas: `CORRECTO` (filtra solo los fragmentos útiles), `AMBIGUO` (extrae frases clave con `refinar` y complementa con la web) o `INCORRECTO` (descarta los documentos locales y activa el respaldo web).
* **Estrategia de fallback web con Tavily (`reescribir` y `buscar_web`):** Cuando la documentación interna no basta o no contiene la respuesta, el sistema reescribe la consulta para buscadores y consulta la API de **Tavily**, filtrando también los resultados web antes de generar la respuesta final.

---

### Pruebas verificadas por ruta (Antes vs. Después)

#### Prueba 1 — Ruta `AMBIGUO` (`manual.txt` — Normativa TFG)
* **Pregunta:**
  > *¿Cuánto tiempo dura la exposición del TFG en la Fase I y qué penalización se aplica si dura menos de 14 minutos?*
* **Antes (RAG Básico — `Faithfulness: 1.00`):**
  > La exposición de la Fase I tiene un máximo de 20 minutos. Si dura menos de 14 minutos, se aplica una penalización de 0,75 puntos. `[manual.txt]`
* **Después (Corrective RAG — Ruta `AMBIGUO` | `Faithfulness: 1.00`):**
  > La exposición dura como máximo 20 minutos. Si dura menos de 14 minutos, se aplica una penalización de 0,75 puntos por falta de rigor expositivo. `[Fuente: manual.txt]`

![Captura de prueba ruta AMBIGUO](img/crag1.png)

---

#### Prueba 2 — Ruta `CORRECTO` (`ManualDeLimpieza_290620_B.pdf`)
* **Pregunta:**
  > *¿Qué hago si me cae el producto de limpieza en los ojos y a quién llamo?*
* **Antes (RAG Básico — `Faithfulness: 1.00`):**
  > Lávate inmediatamente los ojos con abundante agua y acude al oftalmólogo si fuera necesario, llevando la etiqueta o el envase del producto si es posible. El contexto no indica a quién llamar. `[ManualDeLimpieza_290620_B.pdf]`
* **Después (Corrective RAG — Ruta `CORRECTO` | `Faithfulness: 1.00`):**
  > Lávate inmediatamente los ojos con abundante agua. Si fuera necesario, acude al oftalmólogo y lleva, si es posible, la etiqueta o el envase del producto. `[Fuente: ManualDeLimpieza_290620_B.pdf]`

![Captura de prueba ruta CORRECTO](img/crag2.png)

---

#### Prueba 3 — Ruta `INCORRECTO` → Fallback Web con Tavily (Dato externo)
* **Pregunta:**
  > *¿Cuál es el salario mínimo interprofesional en España?*
* **Antes (RAG Básico — `Faithfulness: 0.00`):**
  > No lo sé `[Fuente: ManualDeLimpieza_290620_B.pdf; manual.txt]`
* **Después (Corrective RAG — Ruta `INCORRECTO` + Tavily | `Faithfulness: 1.00`):**
  > En 2026, el Salario Mínimo Interprofesional es de 1.221 euros al mes en 14 pagas, equivalentes a 17.094 euros brutos anuales para una jornada completa. `[Fuente: La Moncloa]`

![Captura de prueba ruta INCORRECTO](img/crag3.png)

---

### Tabla Comparativa Final: RAG Básico vs. Corrective RAG

| Criterio de comparación | Antes: RAG Básico (`rag.py`) | Después: Corrective RAG (`crag.py`) |
| :--- | :--- | :--- |
| **Flujo de ejecución** | Recuperar top-3 en Neon → Generar respuesta directamente. | Recuperar en Neon → **Evaluar relevancia** → Enrutar (`CORRECTO`, `AMBIGUO`, `INCORRECTO`) → Generar. |
| **Control de calidad del contexto** | **Ninguno.** Confía ciegamente en la distancia vectorial. | **Evaluador LLM** (nota 0–1 y motivo por chunk) + refinado de ruido. |
| **Plan B (Estrategia de Fallback)** | **No tiene.** Si el dato no está en los PDFs, responde *"No lo sé"*. | **Reescritura de consulta + Búsqueda web (Tavily)** filtrada también por el evaluador. |
| **Pregunta 1 (Normativa TFG)** | Respuesta correcta sin filtrado de contexto.<br>**Faithfulness:** `1.00` | **[Ruta `AMBIGUO`]** Filtra frases útiles y añade el motivo exacto.<br>**Faithfulness:** **`1.00`** |
| **Pregunta 2 (Manual Limpieza)** | Respuesta correcta con 3 chunks mezclados.<br>**Faithfulness:** `1.00` | **[Ruta `CORRECTO`]** Descarta los 2 chunks irrelevantes.<br>**Faithfulness:** **`1.00`** |
| **Pregunta 3 (Dato externo)** | **Falla:** *"No lo sé"*<br>**Faithfulness:** `0.00` | **[Ruta `INCORRECTO` → Tavily]** Responde **1.221 €/mes** citando fuente oficial.<br>**Faithfulness:** **`1.00`** |
| **Media Global Faithfulness (RAGAS)** | **`0.67`** | **`1.00` (+0.33 de mejora)** |


### Conclusión Final

### Conclusión Final

Pasar de un **RAG básico (`rag.py`)** a un **Corrective RAG (`crag.py`)** demuestra que la clave para mejorar las respuestas está en **evaluar y filtrar el contexto antes de generar**:

* **Menos ruido en el prompt:** El evaluador revisa cada fragmento de Neon para quedarse solo con los útiles (`CORRECTO`), refinar la información parcial (`AMBIGUO`) o descartar lo que no sirve (`INCORRECTO`).
* **Plan B automático (Fallback):** Cuando la respuesta no está en los documentos locales, el sistema reescribe la consulta y busca en internet con **Tavily**, evitando el bloqueo en *"No lo sé"*.
* **Mejora demostrada con RAGAS:** La métrica **`Faithfulness`** sube de una media de **`0.67`** (RAG básico) a **`1.00`** (CRAG), asegurando respuestas fieles a las fuentes y sin alucinaciones.
