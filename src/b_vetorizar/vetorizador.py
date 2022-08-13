
# -*- coding: utf-8 -*-

import os
import json
from pymongo import MongoClient
import tensorflow as tf
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
import pandas as pd
import gensim
from sklearn.decomposition import PCA
import os
import urllib.request
import numpy as np
from _common import util

class Vetorizador:

	def __init__(self):
		self.model_use = None
		self.model_sbert = None
		self.model_word2vec = None

	def carregar_noticias_nao_vetorizadas(self):
		with MongoClient(host="localhost", port=27017) as client:
			db = client.trabalho_puc
			noticias_ls = list(db.noticias.find({
				'$and': [
					{ 
						'conteudo_vetorizado' : { 
							'$exists': False 
						} 
					},
					{ 
						'ignorar' : { 
							'$ne': True
						} 
					},
				]
			}))
			noticias_df = pd.json_normalize(noticias_ls)
			noticias_df = noticias_df.reset_index(drop=True)
			noticias_df = noticias_df.sort_values('data_publ')
			if 'ignorar' not in noticias_df:
				noticias_df['ignorar'] = False
			return noticias_df

	def carregar_configs(self):
		with open(util.full_path('config.json', __file__)) as json_file:
			return json.load(json_file)

	def executar(self):
		noticias_df = self.carregar_noticias_nao_vetorizadas()
		configs_dict = self.carregar_configs()
		self.vetorizar(noticias_df, **configs_dict)
		util.armazenar_todas(noticias_df)

	def vetorizar(self, df, usar_texto_limpo = True, modelo = 'SBERT', incluir_resumo = False, reduzir_dimensionalidade = False ):
		if (modelo == 'USE'):
			self.vetorizar_use(df, usar_texto_limpo=usar_texto_limpo, incluir_resumo=incluir_resumo)
		elif (modelo == 'SBERT'):
			self.vetorizar_sbert(df, usar_texto_limpo=usar_texto_limpo, incluir_resumo=incluir_resumo)
		elif (modelo == 'WORD2VEC'):
			self.vetorizar_word2vec(df, usar_texto_limpo=usar_texto_limpo, incluir_resumo=incluir_resumo)
		else:
			raise ValueError("Modelo de linguagem invalido")
		if reduzir_dimensionalidade:
			reduzir_dimensionalidade(df)

	def vetorizar_tirando_a_media(self, list_of_docs, wv):
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

	def gerar_conteudo(self, df, modo='limpo', incluir_resumo = False):
		_modo = ('_' if len(modo) > 0 else '') + modo
		if not incluir_resumo:
			return df['titulo' + _modo]
		return df['titulo' + _modo] + '' + df['resumo' + _modo]

	def carregar_sbert(self):
		print('Baixando e carregando modelo SBERT')
		# https://www.sbert.net/docs/pretrained_models.html
		self.model_sbert = SentenceTransformer('distiluse-base-multilingual-cased-v1') 

	def vetorizar_sbert(self, df, usar_texto_limpo=True, incluir_resumo = False):
		if self.model_sbert == None:
			self.carregar_sbert()
		print('Vetorizando com SBERT')
		conteudo_df = self.gerar_conteudo(df, modo = ('limpo_str' if usar_texto_limpo else ''), incluir_resumo=incluir_resumo)
		embeddings = self.model_sbert.encode(conteudo_df)
		df['conteudo_vetorizado'] = embeddings.tolist()

	def carregar_use(self):
		print('Baixando e carregando modelo Google USE')
		module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
		self.model_use = hub.load(module_url) 

	def vetorizar_use(self, df, usar_texto_limpo=True, incluir_resumo = False):
		if self.model_use == None:
			self.carregar_use()
		print('Vetorizando com Google USE')
		conteudo_df = self.gerar_conteudo(df, modo = ('limpo_str' if usar_texto_limpo else ''), incluir_resumo=incluir_resumo)
		embeddings = self.model_use(conteudo_df)
		df['conteudo_vetorizado'] = embeddings.numpy().tolist()

	def carregar_word2vec(self):
		#http://nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc
		print('Baixando modelo Word2Vec')
		filename = 'cbow_s100.txt.bz2'
		if not os.path.isfile(filename):
			urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1n6D6XRkdKaDLucK_iNi5aVf3AlD5O7SP&confirm=t', filename)
		print('Carregando modelo Word2Vec')
		self.model_word2vec = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=False)

	def vetorizar_word2vec(self, df, usar_texto_limpo=True, incluir_resumo = False):
		if (not usar_texto_limpo):
			raise ValueError('Combinacao invalida. word2vec requer limpeza')
		if self.model_word2vec == None:
			self.carregar_word2vec()
		print('Vetorizando com Word2Vec')
		conteudo_df = self.gerar_conteudo(df, modo = 'limpo', incluir_resumo=incluir_resumo)
		df['conteudo_vetorizado'] = self.vetorizar_tirando_a_media(conteudo_df, wv=self.model_word2vec)

	def reduzir_dimensionalidade(self, df, n_components=2):
		print('Reduzindo dimensionalidade')
		pca = PCA(n_components=n_components)
		pca_result = pca.fit_transform(list(df['conteudo_vetorizado']))
		df['conteudo_vetorizado'] = pca_result.tolist()
		return pca_result

if __name__ == "__main__":
	Vetorizador().executar()

