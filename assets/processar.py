
from pymongo import MongoClient # pip install pymongo
import pandas as pd # pip install nltk
import re
from bs4 import BeautifulSoup # pip install bs4
import nltk # pip install nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from unidecode import unidecode # pip install unidecode
import string 
from collections import Counter
import gensim # pip install gensim
from gensim.models import word2vec
import random 
import os
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import matplotlib.pyplot as plt
from gensim import corpora
from sklearn.decomposition import PCA
#import seaborn as sns # pip install seaborn
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans
import spacy # python -m spacy download pt_core_news_sm
from spacy.lang.pt.examples import sentences 
from sklearn.model_selection import GridSearchCV
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from numpy import array, linspace, float64
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

nltk.download('stopwords')
stopwords = stopwords.words('portuguese')

# #####################################
# carregar do banco
# #####################################

print('Carregando notícias do banco...')

noticias_dict = {}
with MongoClient() as client: # MongoClient(host="localhost", port=27017)
    db = client.trabalho_puc
    noticias_db = db.noticias
    noticias_ls = list(noticias_db.find({})) # Se necessário, trabalhar com o cursor em vez de carregar tudo pra mem

noticias_df = pd.json_normalize(noticias_ls)
noticias_df = noticias_df.reset_index(drop=True)
noticias_df = noticias_df.sort_values('data_publ')

# #####################################
# preprocessar
# #####################################

print('Preprocessando - Substituindo missing values...')
noticias_df = noticias_df.fillna('')

print('Preprocessando - Concatenando título, resumo e conteúdo...')
noticias_df['conteudo_preproc'] = noticias_df['titulo'] # + ' ' + noticias_df['resumo'] + ' ' + noticias_df['conteudo']

#for index, row in noticias_df.iterrows():
#    if (not bool(BeautifulSoup(row['resumo'], "html.parser").find())):
#        noticias_df.at[index, 'conteudo_preproc'] = row['conteudo_preproc'] + ' ' + row['resumo']

print('Preprocessando - removendo espaços extras...')
noticias_df['conteudo_preproc'] = [re.sub(' +', ' ', noticia) for noticia in noticias_df['conteudo_preproc']]

#print('Preprocessando - removendo tags html...')
#noticias_df['conteudo_preproc'] = [BeautifulSoup(noticia, features='html.parser').get_text() for noticia in noticias_df['conteudo_preproc']]

#print('Preprocessando - removendo URLs...')
#noticias_df['conteudo_preproc'] = [re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", '', noticia, flags=re.MULTILINE) for noticia in noticias_df['conteudo_preproc']]

print('Preprocessando - tokenizando...')
nltk.download('punkt')
nltk.download('corpus')
noticias_df['conteudo_preproc'] = [word_tokenize(noticia) for noticia in noticias_df['conteudo_preproc']]
#noticias_df['conteudo_preproc'] = [noticia.split() for noticia in noticias_df['conteudo_preproc']]

print('Preprocessando - convertendo para lower...')
noticias_df['conteudo_preproc'] = [[palavra.lower() for palavra in noticia] for noticia in noticias_df['conteudo_preproc']]

print('Preprocessando - removendo números...')
noticias_df['conteudo_preproc'] = [[''.join([ch for ch in palavra if not ch.isdigit()]) for palavra in noticia] for noticia in noticias_df['conteudo_preproc']]

print('Preprocessando - removendo pontuação...')
noticias_df['conteudo_preproc'] = [[''.join([ch for ch in palavra if not ch in string.punctuation]) for palavra in noticia] for noticia in noticias_df['conteudo_preproc']]
noticias_df['conteudo_preproc'] = [[re.sub(r"\w+…|…", "", palavra) for palavra in noticia] for noticia in noticias_df['conteudo_preproc']] # Por algum motivo, a linha acima não remove

print('Preprocessando - removendo stopwords...')
noticias_df['conteudo_preproc'] = [[palavra for palavra in noticia if not palavra in stopwords] for noticia in noticias_df['conteudo_preproc']]

print('Preprocessando - removendo acentos...')
noticias_df['conteudo_preproc'] = [[unidecode(palavra) for palavra in noticia] for noticia in noticias_df['conteudo_preproc']]

print('Preprocessando - removendo palavras com tamanho menor q 2')
noticias_df['conteudo_preproc'] = [[palavra for palavra in noticia if len(palavra) > 1] for noticia in noticias_df['conteudo_preproc']]

print('Preprocessando - Removendo matérias com mesmo título...')
noticias_df['conteudo_preproc_str'] = [' '.join(noticia) for noticia in noticias_df['conteudo_preproc']]
noticias_df = noticias_df.drop_duplicates(subset='conteudo_preproc_str', keep="first")
del(noticias_df['conteudo_preproc_str'])
noticias_df = noticias_df.reset_index(drop=True)

#noticias_df.to_csv('teste.csv', sep='|')

#todas_palavras = [palavra for noticia in noticias_df['conteudo_preproc'] for palavra in noticia]

#Counter = Counter(todas_palavras)
#print(Counter.most_common(30))

#ngrams_5 = ngrams(todas_palavras, 4)
#Counter = Counter(ngrams_5)
#print(Counter.most_common(1000))


# #####################################
# vetorizar
# #####################################

# Outros seriam FastText, GloVe
# Sent2Vec, SkipThoughts - sophisticated strategies that take into account the order of words to encode the sentence

def vectorize(list_of_docs, wv):
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(wv.vector_size)
        vectors = []
        for token in tokens:
            if token in wv:
                try:
                    vectors.append(wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features

print('Vetorizando - carregando embeddings...')
#https://drive.google.com/file/d/1f5sNZcV8LDam4zxbHnkm472r3r8D_UpX/view?usp=sharing
#http://nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc
path = 'ptwiki_20180420_100d.txt.bz2'
model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)

#print('Vetorizando - gerando vetores...')
noticias_df['conteudo_vetorizado'] = vectorize(noticias_df['conteudo_preproc'], wv=model)



# Carregar embeddings do Spacy
#nlp = spacy.load("pt_core_news_sm")
#noticias_df['conteudo_vetorizado'] = [nlp(titulo).vector for titulo in noticias_df['titulo']]

#SEED = 42
#random.seed(SEED)
#os.environ["PYTHONHASHSEED"] = str(SEED)
#np.random.seed(SEED)

#print('Vetorizando - gerando embeddings...')
#model = word2vec.Word2Vec(sentences=noticias_df['conteudo_preproc'], vector_size=100, workers=1, seed=SEED, sg=1)

#print('Vetorizando - gerando vetores...')
#noticias_df['conteudo_vetorizado'] = vectorize(noticias_df['conteudo_preproc'], wv=model.wv)


# #####################################

#Visualizar pontos
#pca = PCA(n_components=2)
#pca_result = pca.fit_transform(list(noticias_df['conteudo_vetorizado']))
#plt.scatter(pca_result[:,0], pca_result[:,1])
#plt.show()

# #####################################
# run dbscan
# #####################################

#print('Experimentando valores de epsilon...')
#n_classes_por_eps = {}
#avg_por_eps = {}
#stddev_por_eps = {}
#n_alek_out_por_eps = {}
#n_nao_alek_label_alek_nao_out_por_eps = {}
#
##razao_por_eps = {}
##razao2_por_eps = {}
##silhouette_por_eps = {}
#for i in tqdm(np.arange(0.01, 0.085, 0.002)):
#    dbscan = DBSCAN(eps=i, min_samples=3, metric='cosine').fit(list(noticias_df['conteudo_vetorizado']))
#    classes_tams = Counter(dbscan.labels_)
#    noticias_df['teste'] = dbscan.labels_
#    classes_tams_nao_out = {k: v for k, v in classes_tams.items() if k > -1}
#
#    n_classes = len(classes_tams_nao_out.keys())
#    avg = sum(list(classes_tams_nao_out.values()))/n_classes
#    stddev = np.std(list(classes_tams_nao_out.values()))
#    #razao = n_classes/stddev
#    #razao2 = sum(list(classes_tams_nao_out.values()))*n_classes/stddev
#    #if (len(np.unique(dbscan.labels_)) > 1):
#    #    silhouette_por_eps.update({i: silhouette_score(list(noticias_df['conteudo_vetorizado']), dbscan.labels_, metric = 'cosine')})
#    #else:
#    #    silhouette_por_eps.update({i: -1})
#    n_alek_out = noticias_df.loc[(noticias_df['teste'] == -1) & (noticias_df['titulo'].str.contains('leksan'))].shape[0]
#    alek_nao_out = noticias_df.loc[(noticias_df['teste'] != -1) & (noticias_df['titulo'].str.contains('leksan'))]
#    labels_alek_nao_out = alek_nao_out['teste'].unique()
#    n_nao_alek_label_alek_nao_out = noticias_df.loc[(noticias_df['teste'].isin([labels_alek_nao_out])) & (~noticias_df['titulo'].str.contains('leksan'))].shape[0]
#
#    n_classes_por_eps.update({i: n_classes})
#    avg_por_eps.update({i: avg})
#    stddev_por_eps.update({i: stddev})
#    n_alek_out_por_eps.update({i: n_alek_out})
#    n_nao_alek_label_alek_nao_out_por_eps.update({i: n_nao_alek_label_alek_nao_out})
#    #razao_por_eps.update({i: razao})
#    #razao2_por_eps.update({i: razao2})
    

#figure, axis = plt.subplots(2, 3)
#axis[0, 0].plot(list(n_classes_por_eps.keys()), list(n_classes_por_eps.values()))
#axis[0, 0].set_title("Num classes")
#axis[0, 1].plot(list(avg_por_eps.keys()), list(avg_por_eps.values()))
#axis[0, 1].set_title("Avg sem contar a -1")
#axis[0, 2].plot(list(stddev_por_eps.keys()), list(stddev_por_eps.values()))
#axis[0, 2].set_title("Std dev")
#axis[1, 0].plot(list(n_alek_out_por_eps.keys()), list(n_alek_out_por_eps.values()))
#axis[1, 0].set_title("Aleksandro outliers")
#axis[1, 1].plot(list(n_nao_alek_label_alek_nao_out_por_eps.keys()), list(n_nao_alek_label_alek_nao_out_por_eps.values()))
#axis[1, 1].set_title("Intrusos label Alek")
#plt.show()

print('Experimentando valores de epsilon e min_samples')
params = []
for eps in tqdm(np.arange(0.01, 0.095, 0.002)):
    for min_samples in [3]: # [2, 3, 4, 5]
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(list(noticias_df['conteudo_vetorizado']))
        classes_e_seus_tamanhos = Counter(dbscan.labels_)
        num_outliers = classes_e_seus_tamanhos[-1]
        tamanho_da_maior_classe = max({k: v for k, v in classes_e_seus_tamanhos.items() if k > -1}.values())
        tamanhos_das_classes = list(classes_e_seus_tamanhos.values())
        stddev = np.std(tamanhos_das_classes)
        num_classes = len(tamanhos_das_classes)
        avg = sum(tamanhos_das_classes)/num_classes
        
        params.append({'eps': eps, 'min_samples': min_samples, 'avg': avg, 'stddev': stddev, 'num_classes': num_classes, 'num_outliers': num_outliers, 'tamanho_da_maior_classe': tamanho_da_maior_classe})
melhor_eps = max(params, key=lambda x:x['num_classes'])['eps']
melhor_min_samples = max(params, key=lambda x:x['num_classes'])['min_samples']


#Experimentando usando silhouette - não funciona
#ii = np.arange(0.001, 1, 0.002)
#scores = []
#for i in tqdm(ii):
#  dbscan = DBSCAN(eps=i, min_samples=2, metric='cosine').fit(list(noticias_df['conteudo_vetorizado']))
#  labels = dbscan.labels_
#  if (len(np.unique(dbscan.labels_)) > 1):
#    scores.append(silhouette_score(list(noticias_df['conteudo_vetorizado']), labels, metric = 'euclidean'))
#  else:
#    scores.append(0)
#plt.plot(ii, scores)
#plt.show()



#Experimentando com random search
#print('Grid search')
#def scorer_num_classes(estimator, X):
#    estimator.fit(X)
#    print('foi')
#    return len(np.unique(estimator.labels_))
#cross_validation = [(slice(None), slice(None))]
#param_dict = {'eps': np.arange(0.080, 0.099, 0.001), 'min_samples': [5, 4, 3, 2]}
#random_search = GridSearchCV(estimator=DBSCAN(), param_grid=param_dict, 
#                  scoring=scorer_num_classes, cv=cross_validation)
#res = random_search.fit(list(noticias_df['conteudo_vetorizado']))



print('Clusterizando...')
dbscan = DBSCAN(eps=melhor_eps, min_samples=melhor_min_samples, metric='cosine').fit(list(noticias_df['conteudo_vetorizado']))
noticias_df['cluster'] = dbscan.labels_
noticias_df = noticias_df.loc[noticias_df['cluster'] > -1] # removendo outliers
noticias_df = noticias_df.reset_index(drop=True)


para_salvar = noticias_df[['cluster', 'titulo']].copy()
para_salvar.sort_values('cluster', inplace=True)
para_salvar.to_csv('salvando.csv', index=False)

#Mostrar gráfico clusterizado - não funciona
#pca = PCA(n_components=2)
#pca_result = pca.fit_transform(list(noticias_df['conteudo_vetorizado']))
#cores = dict.fromkeys(noticias_df['cluster'].unique())
#for cor in cores.keys():
#    cores[cor] = (random.random(), random.random(), random.random())
#for i, p in tqdm(enumerate(pca_result)):
#    plt.scatter(p[0], p[1], c=[cores[noticias_df['cluster'][i]]])
#plt.show()


# #####################################
# Rodando KMeans
# #####################################

#Experimentando diferentes valores de k
#ks = range(1000, 3000, 50)
#scores = []
#for k in tqdm(ks):
#  kmeans = KMeans(n_clusters = k).fit(list(noticias_df['conteudo_vetorizado']))
#  labels = kmeans.labels_
#  scores.append(silhouette_score(list(noticias_df['conteudo_vetorizado']), labels, metric = 'euclidean'))
#plt.plot(ks, scores)
#plt.show()

# #####################################
# Montando os clusters
# #####################################

#Com spacy
nlp = spacy.load("pt_core_news_sm", ) # disable=["tok2vec", "parser", "attribute_ruler", "lemmatizer", "tagger"] 
def gerar_tags(noticias):
    titulos = noticias['titulo']
    titulos = '. '.join(titulos)
    pos_tags_validos = ['NOUN', 'PROPN'] #ADJ
    importantes = [token.text.capitalize() for token in nlp(titulos) if token.pos_ in pos_tags_validos]
    tags = [common[0] for common in Counter(importantes).most_common(5) if common[1] >= 2]
    return tags

def gerar_subclusters(noticias, centro_do_cluster):
    datas = noticias['data_publ']
    datas_as_seconds = array([(d-datas[0]).total_seconds() for d in datas])
    eixo_x = np.linspace(start = datas_as_seconds[0], stop = datas_as_seconds[-1], num=100, dtype = float64)
    eixo_x = np.sort(np.unique(np.concatenate([eixo_x, datas_as_seconds])))
    datas_reshaped = datas_as_seconds.reshape(-1, 1)
    kde = KernelDensity(bandwidth=(datas_as_seconds[-1]-datas_as_seconds[0])/20, kernel='gaussian').fit(datas_reshaped)
    eixo_x = eixo_x.reshape(-1, 1)
    eixo_y = kde.score_samples(eixo_x)
    #plt.plot(eixo_x, eixo_y)
    #plt.show()

    minimos_locais = argrelextrema(eixo_y, np.less)[0] # Tem também picos = argrelextrema(eixo_y, np.greater)[0]

    pendentes = datas_as_seconds.view()
    subclusters = []
    for minimo_local in minimos_locais:
        subcluster = pendentes[pendentes < eixo_x[minimo_local]]
        subcluster[::-1].sort() # Ordena decrescente as datas do subcluster. Sintaxe horrível.
        subclusters.append(subcluster)
        pendentes = array([dt for dt in pendentes if dt not in subcluster]) # Vai eliminando dentre os pendentes os já processados
    subclusters.append(pendentes)
    subclusters = [[datas[0] + timedelta(seconds=plus_secs) for plus_secs in subcluster] for subcluster in subclusters]
    subclusters = sorted(subclusters, key=lambda subcluster: subcluster[0], reverse=True)
    subclusters = [[noticias.loc[noticias['data_publ'] == data] for data in subcluster] for subcluster in subclusters]
    #subclusters = [sorted(subcluster, key=lambda noticia: cosine_distances(noticia['conteudo_vetorizado'].values[0].reshape(1, -1), centro_do_cluster.reshape(1, -1))) for subcluster in subclusters]
    subclusters = [[noticia.loc[:,['id', 'fonte', 'titulo', 'data_publ', 'href', 'img']].to_dict('records')[0] for noticia in subcluster] for subcluster in subclusters]
    return subclusters

clusters = []  
for cluster_label in tqdm(noticias_df['cluster'].unique()):
    noticias_do_cluster = noticias_df.loc[noticias_df['cluster'] == cluster_label]
    noticias_do_cluster = noticias_do_cluster.reset_index()
    centro_do_cluster = np.asarray(noticias_do_cluster['conteudo_vetorizado']).mean(axis=0)
    clusters.append({
        'id': int(cluster_label), 
        'tags': gerar_tags(noticias_do_cluster), 
        'subclusters': gerar_subclusters(noticias_do_cluster, centro_do_cluster)
    })

# Assumindo que os subclusters já estão ordenados por data decrescente:
clusters = sorted(clusters, key=lambda cluster: cluster['subclusters'][0][0]['data_publ'], reverse=True)

# Salvar os clusters no banco

with MongoClient() as client: # MongoClient(host="localhost", port=27017)
    db = client.trabalho_puc
    clusters_collection = db['clusters']
    clusters_collection.delete_many({})
    clusters_collection.insert_many(clusters)
#client.close()


'''


# #####################################
# baixar notícias para um dataframe
# #####################################

KEY = '70a72c4570cf44579b4daa7a3a6e6382'
api = NewsApiClient(api_key=KEY)
sources = api.get_sources()
fontes_br = [j['id'] for j in sources['sources'] if j['country'] == 'br']

noticias_dict = api.get_everything(from_param= '2022-05-01', sources=','.join(fontes_br))
print(len(noticias_dict['articles']))
noticias_df = pd.json_normalize(noticias_dict, record_path = ['articles'])
noticias_df.drop(columns = ['author', 'source.id'], inplace = True)
noticias_df.rename(columns = {'source.name': 'fonte', 'title': 'titulo', 'description': 'sumario', 'urlToImage': 'imagem', 'publishedAt': 'data', 'content': 'conteudo'}, inplace = True)



# low case
#for index, noticia in noticias_df.iterrows():
    #conteudo_key = next(key for key in ['conteudo', 'resumo', 'titulo'] if key in noticia and type(noticia[key]) == str and len(noticia[key]) > 0)
#    noticias_df.iloc[index] = noticia[conteudo_key].lower()


teste_dict = {}
for noticia in noticias_df['conteudo_preproc']:
    for palavra in noticia:
        if (palavra not in vocabulario):
            if (not palavra in teste_dict):
                teste_dict[palavra] = 0
            teste_dict[palavra] = teste_dict[palavra] + 1
{k: v for k, v in sorted(teste_dict.items(), key=lambda item: item[1], reverse=False)}
print(teste_dict)

'''