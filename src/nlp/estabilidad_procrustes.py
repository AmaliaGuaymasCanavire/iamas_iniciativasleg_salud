"""
"""

# Importar librerías
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

# Configuración
load_dotenv() # Cargar las variables de entorno del archivo .env
BASE_DIR =  os.getenv("DIR_BASE")
RESULTADOS_DIR = os.getenv("DIR_DATOS_PROCESADOS") # Acceder a las variables de entorno
pd.set_option('display.max_colwidth', None)


def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """
    Guión original: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
    Procrustes alinea dos modelos word2vec de gensim (para permitir la comparación entre la misma palabra en diferentes modelos).
    Código transferido de HistWords <https://github.com/williamleif/histwords> por William Hamilton <wleif@stanford.edu>.

    Primero, intersecta los vocabularios (consulta la documentación de `intersection_align_gensim`).
    Luego, realiza la alineación en el modelo `other_embed`.
    Reemplaza las matrices numpy `syn0` y `syn0norm` del modelo `other_embed` con la versión alineada.
    Devuelve `other_embed`.
    Si `words` está definido, interseca el vocabulario de los dos modelos con el vocabulario en `words` (consulta la documentación de `intersection_align_gensim`).    """

    # Parche de Richard So [https://twitter.com/richardjeanso] (¡gracias!) para actualizar este código para la nueva versión de gensim
    # base_embed.init_sims(replace=True)
    # other_embed.init_sims(replace=True)

    # asegúrese de que el vocabulario y los índices estén alineados
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

    # rellenando los vectores normalizados
    in_base_embed.wv.fill_norms(force=True)
    in_other_embed.wv.fill_norms(force=True)

    # obtener las matrices de incrustación (normalizadas)
    base_vecs = in_base_embed.wv.get_normed_vectors()  
    other_vecs = in_other_embed.wv.get_normed_vectors()

    # solo un producto escalar de la matriz con numpy
    m = other_vecs.T.dot(base_vecs) 
    
    # Método SVD de numpy
    u, _, v = np.linalg.svd(m)
    
    # otra operación de matriz
    ortho = u.dot(v) 

    # Reemplazar la matriz original con una modificada, es decir, multiplicando la matriz de incrustación por "orto
    other_embed.wv.vectors = (other_embed.wv.vectors).dot(ortho)    
    
    return other_embed

def intersection_align_gensim(m1, m2, words=None):
    """
    Intersecta dos modelos de gensim word2vec, m1 y m2.
    Solo se conserva el vocabulario compartido.
    Si se establece 'words' (como lista o conjunto), el vocabulario también se interseca con esta lista.
    Los índices se reorganizan de 0 a N en orden descendente de frecuencia (=suma de los recuentos de m1 y m2).
    Estos índices corresponden a los nuevos objetos syn0 y syn0norm en ambos modelos de gensim:
    -- de modo que la fila 0 de m1.syn0 corresponderá a la misma palabra que la fila 0 de m2.syn0.
    -- Puedes encontrar el índice de cualquier palabra en la lista .index2word: model.index2word.index(word) => 2.
    El diccionario .vocab también se actualiza para cada modelo, conservando el recuento pero actualizando el índice.
    """

    # Obtenga el vocabulario para cada modelo
    vocab_m1 = set(m1.wv.index_to_key)
    vocab_m2 = set(m2.wv.index_to_key)

    # Encuentra el vocabulario común
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # Si no es necesaria la alineación porque el vocabulario es idéntico...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1,m2)

    # De lo contrario, ordenar por frecuencia (sumada para ambos)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w, "count"), reverse=True)
   

    # Luego, para cada modelo...
    for m in [m1, m2]:
        # Reemplace la antigua matriz syn0norm por una nueva (con vocabulario común)
        indices = [m.wv.key_to_index[w] for w in common_vocab]
        old_arr = m.wv.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.vectors = new_arr

        # Reemplazar el antiguo diccionario de vocabulario por uno nuevo (con vocabulario común)
        # y el antiguo index2word por uno nuevo
        new_key_to_index = {}
        new_index_to_key = []
        for new_index, key in enumerate(common_vocab):
            new_key_to_index[key] = new_index
            new_index_to_key.append(key)
        m.wv.key_to_index = new_key_to_index
        m.wv.index_to_key = new_index_to_key
        
        print(len(m.wv.key_to_index), len(m.wv.vectors))
        
    return (m1,m2)

def compute_cosine_similarity(model1,model2,word):
    vector1 = model1.wv[word].reshape(1,-1)
    vector2 = model2.wv[word].reshape(1,-1)
    return(cosine_similarity(X=vector1, Y=vector2)[0][0])

def entrenar_word2vec(sentencias, size=50, window=10, min_count=2,  w=4, s=1, e=50, seed=1):
    return Word2Vec(sentencias, vector_size=size, window=window, min_count=min_count, workers=w, sg= s, epochs=e, seed=seed)

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

def alinear_periodos_viejo(per_lista,per_pares,m_dir,a_dir,iter,periodo_sim_list):

    for j in range(1, len(per_lista)):

        periodo_base = per_lista[j-1]
        periodo_actual = per_lista[j]
        
        embed_base = Word2Vec.load(m_dir+ "Word2Vec_" + str(periodo_base)+'_'+str(iter)+'.mdl')
        embed_actual = Word2Vec.load(m_dir+ "Word2Vec_" + str(periodo_actual)+'_'+str(iter)+'.mdl')
        
        embed_actual_alin = smart_procrustes_align_gensim(
            embed_base, embed_actual)
        embed_actual_alin.save(a_dir+"alin_" + str(periodo_actual)+'_'+str(iter)+'.mdl')

        m1 = embed_base
        m2 = embed_actual_alin
        
        comun_vocab = list(set(m1.wv.key_to_index).intersection(set(m2.wv.key_to_index)))
        print('Tamaño de vocabulario común... ', str(len(comun_vocab)))
        print('Computando similaridad entre palabras ....')

        for palabra in tqdm(comun_vocab):

            cos_sim = compute_cosine_similarity(m1, m2, palabra)
            palabras_masSim_period0 = m1.wv.most_similar(positive=[palabra], topn=10)
            palabras_masSim_period1 = m2.wv.most_similar(positive=[palabra], topn=10)
            periodo_sim_list.append([iter, per_pares[j-1], palabra, cos_sim,
                                        len(comun_vocab),
                                        palabras_masSim_period0, palabras_masSim_period1])

    print(datetime.datetime.now())    
    
    return periodo_sim_list

# https://github.com/williamleif/histwords/blob/master/vecanalysis/seq_procrustes.py
def alinear_periodos(iter, per_lista, m_dir, a_dir):

    primer = True
    base_embed = None

    for periodo in per_lista:
        print("Cargando período:", periodo)
        
        per_embed = Word2Vec.load(m_dir+ "Word2Vec_" + str(periodo)+'_'+str(iter)+'.mdl')
        print("Alineando período:", periodo)
    
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


def comun_vocabulario_iter_viejo(per_lista,anios5_pares,a_dir,iter,periodo_sim_list):
    
    alin_modelos_lista = [Word2Vec.load(a_dir+"alin_" + str(periodo)+'_'+str(iter)+'.mdl') for periodo in per_lista]
    
    primer = True
    for alin_model in alin_modelos_lista:
        vocab = set(alin_model.wv.key_to_index)

        if primer:
            comun_vocab = vocab
            primer = False
        else: 
            comun_vocab = comun_vocab.intersection(vocab)
    
    print('Tamaño de vocabulario común... ', str(len(comun_vocab)))
    print('Computando similaridad entre palabras ....')
    
    for palabra in tqdm(comun_vocab): 
        
        primer = True
        j=0
        m1 = None

        for alin_model in alin_modelos_lista:

            print(palabra)
            #print("indice:", alin_model.wv.key_to_index[palabra])
            m2 = alin_model
            
            if primer:
                primer = False
            else:
                print(len(m1.wv[palabra]),len(m2.wv[palabra]))
             
                cos_sim = compute_cosine_similarity(m1, m2, palabra)
                palabras_masSim_period0 = m1.wv.most_similar(positive=[palabra], topn=10)
                palabras_masSim_period1 = m2.wv.most_similar(positive=[palabra], topn=10)
                periodo_sim_list.append([iter, anios5_pares[j], palabra, cos_sim,
                                            len(comun_vocab),
                                            palabras_masSim_period0, palabras_masSim_period1])
                j = j+1
            
            m1 = m2
        

    print(datetime.datetime.now())    
    
    return periodo_sim_list

def comun_vocabulario_iter(iter,anios5_pares,a_dir,periodo_sim_list):
    
    for anio5_par in anios5_pares:
        print("Identificando similaridades de períodos:",anio5_par )
        m1 = Word2Vec.load(a_dir+"alin_" + str(anio5_par[0])+'_'+str(iter)+'.mdl')
        m2 = Word2Vec.load(a_dir+"alin_" + str(anio5_par[1])+'_'+str(iter)+'.mdl') 
        comun_vocab = list(set(m1.wv.key_to_index).intersection(set(m2.wv.key_to_index)))
    
        for palabra in tqdm(comun_vocab): 
            cos_sim = compute_cosine_similarity(m1, m2, palabra)
            palabras_masSim_period0 = m1.wv.most_similar(positive=[palabra], topn=10)
            palabras_masSim_period1 = m2.wv.most_similar(positive=[palabra], topn=10)
            periodo_sim_list.append([iter, anio5_par, palabra, cos_sim, len(comun_vocab),
                                                palabras_masSim_period0, palabras_masSim_period1])

    print(datetime.datetime.now())    
    
    return periodo_sim_list


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
    modelo_dir =  RESULTADOS_DIR+ './archivos_out/modelos/estabilidad_procrustes/'
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

        print('Alineado modelos...')
        alinear_periodos(i,anios5_lista,modelo_dir,alin_dir)

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

    iter_periodos_df.to_csv(RESULTADOS_DIR+'/archivos_out/estabilidad_procrustes'+'_iter'+str(iteracion)+
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
    plt.title('Estabilidad por Procrustes', fontsize=20)
    plt.savefig(RESULTADOS_DIR+'/archivos_out/estabilidad_procrustes'+'_iter'+str(iteracion)+
                            '_tam'+str(tam_vector)+'.png', dpi=200,  bbox_inches='tight')

    

    



