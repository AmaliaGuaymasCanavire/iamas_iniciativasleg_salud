"""
"""
# Importar libreria
from dotenv import load_dotenv
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
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
tqdm.pandas()

# Configurar
load_dotenv() # Cargar las variables de entorno del archivo .env
BASE_DIR =  os.getenv("DIR_BASE")
RESULTADOS_DIR = os.getenv("DIR_DATOS_PROCESADOS") # Acceder a las variables de entorno
pd.set_option('display.max_colwidth', None)


def compute_cosine_similarity(model1,model2,word):
    vector1 = model1.wv[word].reshape(1,-1)
    vector2 = model2.wv[word].reshape(1,-1)
    return(cosine_similarity(X=vector1, Y=vector2)[0][0])

def entrenar_word2vec(sentencias, size=50, window=10, min_count=2,  w=4, s=1, e=50, seed=1):
    return Word2Vec(sentencias, vector_size=size, window=window, min_count=min_count, workers=w, sg= s, epochs=e,seed=seed)

def pares_primer_paso(lista_items):
    return [(lista_items[i],lista_items[i+1]) for i in range(len(lista_items)-1)]

def entrenar_modelos_periodos(iter, df, m_dir, tv= 50, v= 10, mf= 2, w= 4, s= 1 , e= 50, se= 1):

    # Entrenar modelos de Word2Vec para cada período
    for anios5, corpus in tqdm(zip(df['Periodo_5anios'], df['corpus'])):
        print("Período:", anios5)
        print("Tamaño corpus:", len(corpus))
        print(se)
        modelo = entrenar_word2vec(corpus, size=tv, window=v, min_count=mf, w=w, s=s, e=e, seed=se)
        modelo.save(m_dir + "Word2Vec_" + str(anios5)+'_'+str(iter)+".mdl")


def palabras_elegibles(m1, m2, palabras_top_m1, palabras_menosFrec_m1,
                   palabras_top_m2, palabras_menosFrec_m2):
    ''' 
    Recopila palabras aptas para el cálculo del desplazamiento semántico a partir de la intersección de los vocabularios que cumplen umbrales de frecuencia específicos
    '''
    
    m1_vocab = [key for key, value in m1.wv.key_to_index.items() if key != ' ']
    m2_vocab = [key for key, value in m2.wv.key_to_index.items() if key != ' ']

    interseccion = set(m1_vocab).intersection(set(m2_vocab))
    # Palabras muy frecuentes - top
    top_frec = set(palabras_top_m1 + palabras_top_m2)
    # Palabras menos frecuentes en base a umbral, ejemplo media
    menos_frec = set(palabras_menosFrec_m1 + palabras_menosFrec_m2)

    # Palabras limpias para buscar cambios de uso
    final_list = [w for w in interseccion if
                  w not in top_frec and w not in menos_frec and w != ' ']
   
    print("Cantidad Final lista de palabras: ",len(final_list))

    return m1_vocab, m2_vocab, final_list

def vecinos_elegibles(vocab1, vocab2, menos_frec_union):
    ''' 
    Recopila palabras que son vecinos elegibles de las palabras estudiadas para el cambio semántico.
    Las vecinos elegibles deben estar en ambos vocabularios modelo y 
    deben aparecer más de determinadas veces en cada corpus, para ello se tiene en cuenta los menos frecuentes.
    '''

    interseccion_vocabs = list(set(vocab1) & set(vocab2))
    # se considera palabras repetidas en general
    vecinos_plausibles = [w for w in interseccion_vocabs if
                           w not in menos_frec_union and w != ' ']

    return vecinos_plausibles

def recolectar_vecinos_elegibles(palabra, m, vecinos_plausibles, topn_vecinos):
    c = 0
    out = []
    for w, s in m.wv.most_similar(positive=[palabra], topn=topn_vecinos):
        if w in vecinos_plausibles:
            out.append(w)
            c += 1
        if c == topn_vecinos:
            break

    return (out)

def calcular_vecindades(m1,m2,anio_par,top_palabras, palabras_menosFrec,palabras_menosFrecFull) :
 
    m1_vocab, m2_vocab, final_list = palabras_elegibles(

        m1, m2, # embeddings por período consecutivo
        top_palabras[anio_par[0]], # top N palabras de embedding 1
        palabras_menosFrec[anio_par[0]], # palabras con frec < umbral en embedding 1
        top_palabras[anio_par[1]], # top 5 palabras de embedding 2
        palabras_menosFrec[anio_par[1]] # palabras con frec < umbral en embedding 2
        )
    
    palabras_menos_union = set(palabras_menosFrecFull[anio_par[0]]+palabras_menosFrecFull[anio_par[1]])
    
    vecinos = vecinos_elegibles(m1_vocab, m2_vocab, palabras_menos_union) # vecinos elegibles
    
    return final_list, vecinos

def vecindades_periodosSec (per_lista,m_dir,iter,top_palabras, palabras_menosFrec,palabras_menosFrecFull):

    primer = True
    base_embed = None
    anio_par = (0,0)

    final_lista_dic = {}
    vecinos_dic = {} 

    for periodo in per_lista:
        print("Cargando período:", periodo)
        per_embed = Word2Vec.load(m_dir + "Word2Vec_" + str(periodo)+'_'+str(iter)+'.mdl')
        
        if primer:
            primer = False
            anio_par = (0, periodo)
    
        else:
           anio_par = (anio_par[1],periodo)
           final_lista, vecinos = calcular_vecindades(base_embed,per_embed,anio_par,top_palabras, palabras_menosFrec,palabras_menosFrecFull)
           final_lista_dic[anio_par] = final_lista
           vecinos_dic[anio_par] = vecinos

        print(" vecindades calculadas ", anio_par)
        base_embed = per_embed
        
    
    return final_lista_dic, vecinos_dic

if __name__ == "__main__":

    umbral_top = 5
    umbral_frec_menos = 5
    umbral_frec_menos_full  = 9 
    topn_vecinos = 100

    # Config modelo https://radimrehurek.com/gensim/models/word2vec.html
    tam_vector = 50 # Dimensionalidad de los vectores de palabras.
    ventana = 10 # ventana ( int , opcional ): distancia máxima entre la palabra actual y la palabra prevista dentro de una oración.
    min_frec = 2 #  Ignora todas las palabras con una frecuencia total menor que esta.
    w = 4  # utilice estos muchos subprocesos de trabajo para entrenar el modelo (=entrenamiento más rápido con máquinas de múltiples núcleos).
    s = 1 # Algoritmo de entrenamiento: 1 para skip-gram; de lo contrario, CBOW.
    e = 50 # Número de iteraciones (épocas) en el corpus. (Anteriormente: iter )
    semilla = 1 # seed ( int , opcional ): Semilla para el generador de números aleatorios. 
    iteracion = 10

    # Crear directorio para algoritmo de cambio semántico
    modelo_dir =  RESULTADOS_DIR+ './archivos_out/modelos/estabilidad_NN/'
    if not os.path.exists(modelo_dir):
        print('Creando directorio de modelo...')
        os.makedirs(modelo_dir)

    # Cargar corpus
    with open(RESULTADOS_DIR + 'periodo_5anios_df.pkl', 'rb') as file: 
        periodo_5anios_df = pickle.load(file)
    print(periodo_5anios_df.info())

    # Definir períodos y pares a comparar
    anios5_lista = sorted(periodo_5anios_df['Periodo_5anios'].to_list())
    anios5_pares = pares_primer_paso(anios5_lista)
    print('Lista de períodos de 5 años:', anios5_lista)
    print('Pares de de períodos de 5 años:',anios5_pares)


    #Recopilar palabras para el análisis de cambio semántico que cumplan con los umbrales
    #Contar la frecuencia de palabras por década
    frec_anio_dic = {}
    top_palabras_dic = {}
    palabras_menosFrec_dic = {}
    palabras_menosFrecFull_dic = {}

    for anio in anios5_lista:
        df = pd.read_csv(RESULTADOS_DIR+'/archivos_out/frec_para_datos_limpios_por_desplaz_semantico_anios5_'+str(anio)+'.csv')
        print(anio)

        print(df.Frecuencia.describe().apply(lambda x: format(x, 'f')))
        df = df.sort_values('Frecuencia', ascending=False)
        frec_anio_dic[anio] = df

        top_palabras_dic[anio] = df.Palabra.head(umbral_top).to_list()
        palabras_menosFrec_dic[anio] = df.loc[df.Frecuencia < umbral_frec_menos].Palabra.to_list()
        palabras_menosFrecFull_dic[anio] = df.loc[df.Frecuencia < umbral_frec_menos_full].Palabra.to_list()

    error_lista = []
    iter_periodos_lista = []
    for i in range(iteracion):
    
        np.random.seed(i)
        rn.seed(i)

        semilla = i
        print('********************************************************')
        print('Número de repetición ', str(i))
           
        print('Entrenando modelos por período de 5 años...')
        entrenar_modelos_periodos(i, periodo_5anios_df, modelo_dir, tv= tam_vector, v= ventana, mf= min_frec, w= w, s= s , e= e, se= semilla)
  
        print('Calculando vecindades para pares de períodos', str(i))
    
        final_lista_dic, vecinos_dic = vecindades_periodosSec (
            anios5_lista, modelo_dir, i,
            top_palabras_dic, palabras_menosFrec_dic,palabras_menosFrecFull_dic
            )

        for anio_par, palabras_lista in final_lista_dic.items():
            m1 = Word2Vec.load(modelo_dir+ "Word2Vec_" + str(anio_par[0])+'_'+str(i)+'.mdl')
            m2 = Word2Vec.load(modelo_dir+ "Word2Vec_" + str(anio_par[1])+'_'+str(i)+'.mdl')

            for palabra in tqdm(palabras_lista):
    
                # unión de vecinas en dos puntos en el tiempo
                
                neighbors_t1 = recolectar_vecinos_elegibles(palabra, m1, vecinos_dic[anio_par], topn_vecinos)
                neighbors_t2 = recolectar_vecinos_elegibles(palabra, m2, vecinos_dic[anio_par], topn_vecinos)

                #if len(neighbors_t1)<topn_vecinos or len(neighbors_t2)<topn_vecinos:
                #    error_lista.append([palabra, len(neighbors_t1), len(neighbors_t2)])

                neighbors = set(neighbors_t1).intersection(set(neighbors_t2)) 
                score = -len(neighbors)
                iter_periodos_lista.append([i, str(anio_par), palabra, score, len(neighbors), neighbors_t1, neighbors_t2])
            

    iter_periodos_df = pd.DataFrame(
        iter_periodos_lista,
        columns = ['iteracion', 'par_periodo', 'palabra', 
        'similaridad_semantica', 'cantidad_palabras_comun',
        'topn_vecindad_t1','topn_vecindad_t2'
        ]
    )
    print(iter_periodos_df.describe())

    iter_periodos_df = iter_periodos_df.sort_values('similaridad_semantica', ascending =False)
    
    print('Palabras con mayor score / mayor cambio')
    print(iter_periodos_df.head(20))                                                    

    print('Palabras con menor score / menor cambio')
    print(iter_periodos_df.tail(20))



    iter_periodos_df.to_csv(RESULTADOS_DIR+'/archivos_out/estabilidad_NN'+'_iter'+str(iteracion)+
                            '_tam'+str(tam_vector)+'.csv', 
                            index=False)

    topn_dict = {}
    X = []
    Y = []

    k=[10,50,100,250,500,750,1000]

    for n in k:
        
        for iter in range(iteracion):
            subdf = iter_periodos_df.loc[(iter_periodos_df.iteracion==iter)]
            subdf = subdf.sort_values('similaridad_semantica', ascending=False).reset_index(drop=True)
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
    plt.title('Estabilidad por NN', fontsize=20)
    plt.savefig(RESULTADOS_DIR+'/archivos_out/estabilidad_NN'+'_iter'+str(iteracion)+
                            '_tam'+str(tam_vector)+'.png', dpi=200,  bbox_inches='tight')

    

    



