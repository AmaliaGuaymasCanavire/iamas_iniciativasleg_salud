"""
Módulo para limpieza y normalización de texto en español, pensado para textos legislativos.

Incluye funciones para:
- Normalización unicode y eliminación de acentos
- Tokenización y lematización
- Eliminación de stopwords, puntuación y palabras cortas
- Generación y eliminación de n-gramas frecuentes es analizado previamente en notebook

Requiere:
- spaCy
- textacy.preprocessing 
"""

import multiprocessing
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import pickle
import os
from collections import Counter
from itertools import islice
import logging
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
import textacy
import textacy.preprocessing as tprep



### CONFIGURACION
load_dotenv() # Cargar las variables de entorno del archivo .env
RESULTADOS_DIR = os.getenv("DIR_DATOS_PROCESADOS") # Acceder a las variables de entorno
MODELO_NLP = spacy.load("es_core_news_md") # Cargar modelo de spaCy para español . Se reemplazo es_core_news_md es_core_news_md



def limpiar_texto_basico(texto):
    """
    Aplica una serie de transformaciones básicas para limpiar el texto.

    Parámetros:
        texto (str): Texto de entrada.

    Retorna:
        str: Texto limpio y normalizado.
    """
    try:
        texto = texto.lower() # Pasar a minúscula
        texto = tprep.remove.accents(texto) # Eliminar los acentos de cualquier carácter Unicode acentuado en el texto, ya sea reemplazándolos con equivalentes ASCII o eliminándolos por completo. 
        texto = tprep.normalize.unicode(texto) # Normalizar carácter Unicode del texto en formas canónicas. 
        texto = texto.replace('.', '') # Por tema de leyes y números 
        texto = tprep.remove.punctuation(texto) # Eliminar la puntuación del texto reemplazando todas las instancias de puntuación (o un subconjunto de la misma especificada por solamente) con espacios en blanco. 
        texto = tprep.normalize.quotation_marks(texto) # Reemplazar comillas simples y dobles a solo los equivalentes básicos de ASCII. 
        texto = tprep.normalize.whitespace(texto) # Reemplazar espacios respetando la división entre palabras 
        texto = tprep.remove.brackets(texto) # Reemplazar  {}, square [], and/or round ()
        texto = tprep.replace.numbers(texto, "numero") # Reemplazar los números por la palabra 'numero'  
        
    except Exception as e:
        print(f"Error al limpiar el texto: {e}")
        texto = ""
    return texto


def obtener_tokens(texto):
    """
    Tokeniza el texto y devuelve una lista de tokens no vacíos.
    Aplica una serie de transformaciones básicas para limpiar el texto.

    Parámetros:
        texto (str): Texto de entrada.

    Retorna:
        list: lista de token Spacy

    """
    doc = MODELO_NLP(texto)
    lista = []
    try: 
        lista = [token for token in doc if len(token.text.strip()) > 0]
    except Exception as e:
        print(f"Error al limpiar el texto: {e}")
   
    return lista


def eliminar_puntuacion(tokens):
    """
    Elimina los signos de puntuación de una lista de tokens.

    Parámetros:
        tokens (str): lista de palabras

    Retorna:
        list: lista de palabras


    """
    doc = spacy.tokens.doc.Doc(MODELO_NLP.vocab, words=tokens)
    return [token.text for token in doc if not token.is_punct]


def eliminar_palabras_cortas(tokens, minimo_caracteres):
    """
    Elimina las palabras con menos caracteres que el mínimo especificado.

    Parámetros:
        tokens (str): lista de tokens
        minimo_caracteres (int):  cantidad mínima de caracteres en una palabra

    Retorna:
        str: string 

    """
    return " ".join([token.text for token in tokens if len(token.text) > minimo_caracteres])


def eliminar_stopwords(tokens):
    """
    Elimina las stopwords de una lista de tokens.
    
    Parámetros:
        tokens (str): lista de tokens
    Retorna:
        list: lista de tokens

    """
    return [token for token in tokens if not token.is_stop]


def a_minusculas(tokens):
    """
    Convierte todos los tokens a minúsculas.
    """
    return [token.lower() for token in tokens]


def lematizar(texto, etiquetas_permitidas=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """
    Lematiza el texto, conservando solo las palabras con ciertas etiquetas gramaticales.
    Retorna un string con los lemas.

    Parámetros:
        texto (str): texto
        etiquetas_permitidas (list): Lista de POS a considerar
    Retorna:
        str: string con lemas

    """
    doc = MODELO_NLP(texto)
    lemas = " ".join([
        token.lemma_ if token.lemma_ not in ['-PRON-'] else ''
        for token in doc if token.pos_ in etiquetas_permitidas
    ])
    print(lemas)
    return lemas


def normalizar(texto):
    """
    Función principal que aplica el pipeline completo de normalización.
    Parámetros:
        texto (str): texto
        
    Retorna:
        list: lista de palabras
    """
    try: 
        #print("TEXTO: ", texto)
        #texto_lema = lematizar(texto) # lematizar
        #print(" ... lematizado ...")
        #print(texto_lema)
        tokens = obtener_tokens(texto) # Tokenizar
        #print(" ... tokenizado ...")
        tokens = eliminar_stopwords(tokens) # Eliminar stopword
        #print(" ... sin stop word  ...")
        tokens = eliminar_palabras_cortas(tokens, minimo_caracteres=2) # eliminar palabras raras 
        #print(" ... sin palabras menores a 2 letras  ...")
        print(tokens)
    except Exception as e:
        print(f"Error al limpiar el texto: {e}")
        texto = ""
    
    
    return tokens


def obtener_ngrams(texto, n):
    """
    Genera n-gramas de un texto, filtrando solo palabras alfabéticas y no stopwords.
    """
    doc = MODELO_NLP(texto)
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return list(zip(*[tokens[i:] for i in range(n)]))


def eliminar_ngrams_frecuentes(texto, ngrams_a_eliminar):
    """
    Elimina los n-gramas especificados de un texto.

    Retorna el texto limpio.
    """
    doc = MODELO_NLP(texto)
    texto_limpio = texto
    for ngram in ngrams_a_eliminar:
        texto_limpio = texto_limpio.replace(ngram, "")
    return " ".join(texto_limpio.split())


def obtener_ngrama(serie_texto,n):
    """
    Función principal que aplica el pipeline completo de normalización.
    Parámetros:
        serie_texto (serie): serie de textos
        
    Retorna:
        resultado: lista de gramas
    """

   
    resultado = []
    for doc in MODELO_NLP.pipe(iter(serie_texto), batch_size = 1000, n_process=-1):
        tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
        resultado.append(list(zip(*[tokens[i:] for i in range(n)])))
    return resultado


def normalizar2(serie_texto):
    """
    Función principal que aplica el pipeline completo de normalización.
    Parámetros:
        serie_texto (serie): serie de textos
        
    Retorna:
        list: lista de textos normalizado
    """

    etiquetas_permitidas = ['NOUN', 'ADJ', 'VERB', 'ADV']
    minimo_caracteres = 2
    resultado = []
    STOP_WORDS.add('numero')

    for doc in MODELO_NLP.pipe(iter(serie_texto), batch_size = 1000, n_process=-1):
        texto = []  
        for token in doc:
            if token.lemma_ not in ['-PRON-']:
                token_lem = token.lemma_
            else:
                token_lem = ""
            if len(token_lem.strip()) > 0:
                if not (token_lem in STOP_WORDS):
                    if len(token_lem) > minimo_caracteres:
                        texto.append(token_lem)
        texto_unido =  "".join(texto)      
        print(texto_unido)
        resultado.append(texto_unido.strip())
    return resultado

    
if __name__ == "__main__":
    with open(RESULTADOS_DIR + 'proyecto_2009_2024_LIMPIO2_df.pkl', 'rb') as file:
        proyecto_2009_2024_df = pickle.load(file)
    
    print("DataFrame original:", proyecto_2009_2024_df.shape) # Leer objeto base -- 97738, 22
    print(proyecto_2009_2024_df.head())

    texto_df = proyecto_2009_2024_df.loc[ # Seleccionar los textos - titulos IL de ley para 2009 a 2024 
        proyecto_2009_2024_df['Tipo'] == 'LEY' ,
          ['Proyecto.ID','Título']
        ]
    print("DataFrame texto_df:", texto_df.shape) # 35179, 2
    print(texto_df.info())
   
    texto_df['Título procesado'] = texto_df['Título'].copy() # Limpiar texto legislativo
    texto_df['Título procesado'] = texto_df['Título procesado'].apply(limpiar_texto_basico) # caracteres especiales
    print("Limpieza básica de texto exitosa!!")
    
    texto_df['Título normalizado'] = texto_df['Título procesado'].copy() 
    texto_df['Título normalizado'] = normalizar2(texto_df['Título normalizado']) # Normalizar
    texto_df['Título normalizado'] = texto_df['Título normalizado'].apply(tprep.remove.accents)
    print("Normalización de texto exitosa!!")

    # Controlar
    print("Cantidad de IL sin título normalizado:", texto_df[texto_df['Título normalizado']==""].shape)

    with open(RESULTADOS_DIR + 'texto_normalizado_2009_2024_df.pkl', 'wb') as file: # Guardar objeto
        pickle.dump(texto_df, file)
    print("Se guardo el objeto de texto normalizado")



    

    
    # Procesar la columna "texto_original"
    #df_procesado = procesar_dataframe(df, 'texto_original')
    
    #print("\nDataFrame procesado:")
    #print(df_procesado)
