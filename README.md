Detección de cambios semánticos en el discurso legislativo sobre salud de Argentina mediante procesamiento de lenguaje natural en pequeños datos abiertos
=========================================================================================================================================================

Repositorio de investigación asociado a la tesis de Maestría en Explotación de
Datos y Descubrimiento del Conocimiento de la Universidad de Buenos Aires.

 

Sobre el proyecto
-----------------

Este repositorio contiene código, recursos y documentación asociados al estudio
de los **cambios semánticos en el discurso legislativo sobre salud en
Argentina**, a partir del análisis de iniciativas legislativas de la Cámara de
Diputados de la Nación.

La investigación aborda el desafío de detectar cambios semánticos en **corpus
legislativos pequeños**, utilizando técnicas de Procesamiento de Lenguaje
Natural (NLP) y *word embeddings* en un marco reproducible basado en Datos
Abiertos.

El estudio analiza **2.848 iniciativas legislativas sobre salud**,
correspondientes a los períodos parlamentarios **127 a 141**, entre **2009 y
2024**, organizadas en tres tramos temporales comparativos.

El trabajo parte de una pregunta central:

>   **¿Cómo detectar cambios semánticos en el discurso legislativo sobre salud
>   cuando se dispone de pequeños conjuntos de datos abiertos?**

 

La información institucional de la defensa y el resumen de la investigación
pueden consultarse en:

[Defensa de tesis – Maestría en Data Mining –
UBA](https://datamining.dc.uba.ar/datamining/2026/06/12/defensa-de-tesis-amalia-guaymas-17-06-2026-10-30-hs/)

La defensa también se encuentra disponible en YouTube:

[Ver defensa de tesis](https://www.youtube.com/live/rzYQvVy7PPs)

 

 

Problema de investigación
-------------------------

El lenguaje legislativo refleja transformaciones en las prioridades políticas,
los valores sociales y las formas de conceptualizar problemas públicos.

En el campo de la salud, identificar modificaciones en el significado y las
asociaciones de determinados términos permite explorar cómo evolucionan los
marcos conceptuales presentes en la producción legislativa.

Sin embargo, los métodos de detección de cambio semántico suelen desarrollarse
sobre grandes corpus. Esta investigación aborda el problema desde una
perspectiva de **pequeños datos abiertos**, donde la escasez de observaciones
constituye una condición metodológica central.

 

Objetivo
--------

Detectar cambios semánticos en títulos de proyectos de ley sobre salud generados
en la Cámara de Diputados de la Nación Argentina entre 2009 y 2024 utilizando
PLN y datos abiertos

 

### Objetivos específicos

1. Construir un conjunto de datos de iniciativas legislativas de ley sobre salud
corintervinientes al período parlamentario 127 al 141 (años 2009–2024).

2. Identificar y evaluar métodos DCS mediante word embedding en corpus reducidos
para períodos temporales determinados.

3. Analizar términos cambiantes en relación con tendencias discursivas
legislativas de salud pública.

4. Analizar términos específicos e iniciativas legislativas mediante un caso de
uso sobre prevención de la muerte súbita.

 

📊 Datos
-------

El corpus utilizado en la investigación corresponde a iniciativas legislativas
de la Cámara de Diputados de la Nación Argentina.

 

\* Período analizado: 2009–2024

\* Períodos parlamentarios: 127–141

\* Iniciativas legislativas totales: 92.535

\* Iniciativas relacionadas con salud: 8.244

\* Corpus principal de análisis: 2.848

\* Tramos temporales: 3

 

Los datos utilizados corresponden a información legislativa de acceso público.

Por razones de tamaño, procedencia y/o condiciones de distribución, no todos los
datos originales necesariamente forman parte de este repositorio. El código
permite documentar y reproducir las etapas de procesamiento a partir de los
datos disponibles.

 

 Metodología
------------

Este estudio adopta un enfoque metodológico mixto que combina los principios
centrados en el usuario de Design Thinking con la estructura analítica de la
metodología Cross-Industry Standard Process for Data Mining (CRISP-DM). Esta
integración se justifica por la naturaleza dual del proyecto: constituye una
iniciativa de innovación social, orientada a generar valor público a partir de
datos abiertos legislativos, y de manera simultánea, un proyecto de ciencia de
datos que requiere un proceso sistemático, reproducible y validable.

 

 

Estructura del repositorio
--------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
iamas_iniciativasleg_salud/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   └── nlp/
│   └── notebooks/
│
├── papers/
│   └── documentos principales
│
└── ...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>   La estructura del repositorio se encuentra actualmente en proceso de
>   reorganización y depuración para facilitar su uso académico y reproducible.

 

Documentación y publicaciones
-----------------------------

### Tesis de Maestría

**Guaymás Canavire, Amalia.**

*Detección de cambios semánticos en el discurso legislativo sobre salud de
Argentina mediante procesamiento de lenguaje natural en pequeños datos
abiertos.*

Universidad de Buenos Aires — Facultad de Ciencias Exactas y Naturales.

 

### 55.º JAIIO — 2026

Parte de los resultados de esta investigación fueron presentados en las **55.º
Jornadas Argentinas de Informática (JAIIO 2026)**.

📄 **Trabajo presentado:**  
[Consultar publicación en las 55
JAIIO](https://55jaiio.sadio.org.ar/wp-content/uploads/2026/07/436.pdf)

La publicación presenta parte del trabajo desarrollado en torno al análisis
computacional del discurso legislativo y la detección de cambios semánticos.

 

Tecnologías
-----------

El proyecto utiliza principalmente:

-   Python

-   NumPy

-   Pandas

-   SciPy

-   Scikit-learn

-   Gensim

-   NLTK

-   spaCy

-   Matplotlib

-   Seaborn

-   Plotly

-   Jupyter

Las versiones y dependencias necesarias se encuentran en:

`requirements.txt`

⚙️ Instalación
-------------

Clonar el repositorio:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
git clone https://github.com/AmaliaGuaymasCanavire/iamas_iniciativasleg_salud.git
cd iamas_iniciativasleg_salud
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Crear un entorno virtual:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python -m venv .venv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Activarlo en Windows:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.venv\Scripts\activate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Activarlo en Linux/macOS:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
source .venv/bin/activate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instalar las dependencias:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pip install -r requirements.txt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

Licencia y uso
--------------

Este repositorio tiene fines académicos y de investigación.

Antes de reutilizar datos, documentos o materiales de terceros incluidos en el
repositorio, se deben verificar sus condiciones de uso y distribución.
