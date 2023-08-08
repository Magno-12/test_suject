# Trabajo de Cálculos con FastAPI

Este proyecto tiene como objetivo recibir y analizar múltiples archivos Excel que contienen datos de señales y etiquetas. Los análisis incluyen regresión lineal, regresión polinómica, ANOVA, LDA y más.

## Características

- Recepción de múltiples archivos a través de una API RESTful.
- Análisis y procesamiento de datos.
- Respuestas estructuradas con resultados del análisis.

## Requisitos

- Python 3.9+
- Un entorno virtual (opcional, pero recomendado).

## Configuración del Entorno

- Clonar el repositorio

(Opcional) Crear un entorno virtual:

bash
Copy code
python -m venv env
## Activar el entorno virtual:

- source env/bin/activate  # En Linux/macOS
O, si estás en Windows:
- .\env\Scripts\activate
- 
## Instalar las dependencias:

- pip install -r requirements.txt
## Cómo correr el proyecto
### Con el entorno virtual activo:
- uvicorn main:app --reload
El servidor se iniciará, generalmente en http://127.0.0.1:8000/. Puedes acceder a esa URL en tu navegador para ver la documentación de la API gracias a FastAPI.

## Uso
### Subir archivos:
- Haz una solicitud POST a /upload/{subject_number}/ adjuntando los archivos requeridos.

El flujo de la API funciona de la siguiente manera:

1. Un usuario hace una solicitud POST a "/upload/{subject_number}/" con los archivos adjuntos necesarios.
2. Los archivos se leen y se procesan.
3. Se realiza el análisis sobre los datos proporcionados.
4. Los resultados del análisis se devuelven inmediatamente en la respuesta.
