"""
"""

# Importar librerías
from dotenv import load_dotenv
from cade.cade import CADE

import pandas as pd
import numpy as np
import pickle
import os
import re
import random as rn
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from gensim.models import Word2Vec
from scipy.spatial.distance import cosine, euclidean
import itertools
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
tqdm.pandas()

# Configuración
load_dotenv() # Cargar las variables de entorno del archivo .env
BASE_DIR =  os.getenv("DIR_BASE")
RESULTADOS_DIR = os.getenv("DIR_DATOS_PROCESADOS") # Acceder a las variables de entorno
pd.set_option('display.max_colwidth', None)

def pares_primer_paso(lista_items):
    return [(lista_items[i],lista_items[i+1]) for i in range(len(lista_items)-1)]

def entrenar_modelos_periodos(iter, df, m_dir, tv= 50, v= 10, mf= 2, w= 4, s= 1 , e= 50, se= 1):

    primer = True
    corpus_base = None
    periodo_base = ""
    # Entrenar modelos de Word2Vec para cada período
    for anios5, corpus in tqdm(zip(df['Periodo_5anios'], df['corpus'])):
        print("Período:", anios5)
        print("Tamaño corpus:", len(corpus))
        if primer:
            corpus_base = corpus
            periodo_base = anios5
            primer = False
        else:
            # Concatenar las dos columnas 'corpus'
            corpus_total = pd.concat([df['corpus'], df2['corpus']], ignore_index=True)

            # Si querés que sea una lista de textos (útil para NLP)
            corpus_list = corpus_total.tolist()
        


            

        base_embed = alin_embed
        alin_embed.save(a_dir + "alin_" + str(periodo)+'_'+str(iter)+'.mdl')
        print("Modelo alineado guardado")

        print(se)
        modelo = entrenar_word2vec(corpus, size=tv, window=v, min_count=mf, w=w, s=s, e=e, seed=se)
        modelo.save(m_dir + "Word2Vec_" + str(anios5)+'_'+str(iter)+".mdl")
# https://github.com/williamleif/histwords/blob/master/vecanalysis/seq_procrustes.py

def alinear_compass_periodos(iter, per_lista, m_dir, a_dir):

    primer = True
    base_embed = None

    for periodo in tqdm(per_lista):
        print("Cargando período:", periodo)
  
        if primer:
            alin_embed = per_embed
            primer = False
        else:
            alin_embed =  smart_procrustes_align_gensim(
            base_embed, per_embed
            )

        base_embed = alin_embed
        alin_embed.save(a_dir + "alin_" + str(periodo)+'_'+str(iter)+'.mdl')
        print("Modelo alineado guardado")


if __name__ == "__main__":

    # Config modelo https://radimrehurek.com/gensim/models/word2vec.html
    tam_vector = 50  # Dimensionalidad de los vectores de palabras.
    ventana = 10  # ventana ( int , opcional ): distancia máxima entre la palabra actual y la palabra prevista dentro de una oración.
    min_frec = 2 #  Ignora todas las palabras con una frecuencia total menor que esta.
    w = 4  # utilice estos muchos subprocesos de trabajo para entrenar el modelo (=entrenamiento más rápido con máquinas de múltiples núcleos).
    s = 1 # Algoritmo de entrenamiento: 1 para skip-gram; de lo contrario, CBOW.
    e = 50 # Número de iteraciones (épocas) en el corpus. (Anteriormente: iter )
    semilla = 1 # seed ( int , opcional ): Semilla para el generador de números aleatorios. 
    iteracion = 10

    # Crear directorio para algoritmo de cambio semántico
    modelo_dir =  RESULTADOS_DIR+ './archivos_out/modelos/estabilidad_compass/'
    if not os.path.exists(modelo_dir):
        print('Creando directorio de modelo...')
        os.makedirs(modelo_dir)

    # Crear directorio modelo alineado
    alin_dir = modelo_dir +'alineado/'
    if not os.path.exists(alin_dir):
        print('Creando directorio de modelo alineado...')
        os.makedirs(alin_dir)


    # Cargar corpus
    with open(RESULTADOS_DIR + 'periodo_5anios_df.pkl', 'rb') as file: 
        periodo_5anios_df = pickle.load(file)
    print(periodo_5anios_df.info())

    # Definir períodos y pares a comparar
    anios5_lista = sorted(periodo_5anios_df['Periodo_5anios'].to_list())
    anios5_pares = pares_primer_paso(anios5_lista)
    print('Lista de períodos de 5 años:', anios5_lista)
    print('Pares de de períodos de 5 años:',anios5_pares)

    iter_periodos_lista = []
    for i in range(iteracion):
    
        np.random.seed(i)
        rn.seed(i)
        semilla = i

        print('********************************************************')
        print('Número de repetición ', str(i))
         
        print('Entrenando modelos por período de 5 años...')
        entrenar_modelos_periodos(i, periodo_5anios_df, modelo_dir, tv= tam_vector, v= ventana, mf= min_frec, w= w, s= s , e= e, se= semilla)

        print('Alinear por Compasss  Compass-aligned Distributional Embedding ..')
        alinear_compass_periodos(i,anios5_lista,modelo_dir,alin_dir)

        print('Común vocabulario por pares de períodos alineados...')
        iter_periodos_lista = comun_vocabulario_iter(i,anios5_pares,alin_dir,iter_periodos_lista)
        

    iter_periodos_df = pd.DataFrame(
        iter_periodos_lista,
        columns = ['iteracion', 'par_periodo', 'palabra', 
        'similaridad_semantica', 'cantidad_palabras_comun',
        'top10_vecindad_t1','top10_vecindad_t2'
        ]
    )
    print(iter_periodos_df.describe())

    iter_periodos_df = iter_periodos_df.sort_values('similaridad_semantica')

    print('Palabras con la menor similitud de coseno / el mayor cambio')
    print(iter_periodos_df.head(20))

    print('Palabras con la mayor similitud de coseno / menor cambio')
    print(iter_periodos_df.tail(20))

    iter_periodos_df.to_csv(RESULTADOS_DIR+'/archivos_out/estabilidad_compass'+'_iter'+str(iteracion)+
                            '_tam'+str(tam_vector)+'.csv', 
                            index=False)
    


    anios5_pares

    topn_dict = {}
    X = []
    Y = []

    k=[10,50,100,250,500,750,1000]

    for n in k:
        
        for iter in range(iteracion):
            subdf = iter_periodos_df.loc[(iter_periodos_df.iteracion==iter)]
            subdf = subdf.sort_values('similaridad_semantica', ascending=True).reset_index(drop=True)
            topn_dict[iter] = subdf.head(n).palabra.to_list()
        
        topn_list_of_lists = [val for key, val in topn_dict.items()]

        interseccion = len(set(topn_list_of_lists[0]).intersection(*topn_list_of_lists))

        Y.append(interseccion/n)
        X.append(n)

    fig = plt.figure(figsize=(15, 8))

    fig.set_size_inches(20, 10)
    plt.scatter(X,Y)
    plt.plot(X,Y)
    plt.gca().tick_params(axis='both', which='major', labelsize=15)
    plt.ylim(0,1.)
    plt.xlabel('k', fontsize=18)
    plt.ylabel('Interseccion@k', fontsize=18)
    plt.title('Estabilidad por Compass', fontsize=20)
    plt.savefig(RESULTADOS_DIR+'/archivos_out/estabilidad_compass'+'_iter'+str(iteracion)+
                            '_tam'+str(tam_vector)+'.png', dpi=200,  bbox_inches='tight')

    

