
# -*- coding: utf-8 -*-

import json
from pymongo import MongoClient
import re
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import string
from unidecode import unidecode
from _common import util

class Limpador:

	def __init__(self):
		nltk.download('stopwords')
		self.stops = stopwords.words('portuguese')

	def carregar_noticias_nao_limpas(self):
		with MongoClient(host="localhost", port=27017) as client:
			db = client.trabalho_puc
			noticias_ls = list(db.noticias.find({
				'$and': [
					{ 
						'titulo_limpo' : { 
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

	def executar(self):
		print('\nIniciando limpeza do texto...')
		noticias_df = self.carregar_noticias_nao_limpas()
		print(str(len(noticias_df)) + ' noticias serão limpas...')
		self.limpar(noticias_df)
		util.armazenar_todas(noticias_df)

	def limpar_se_necessario(self, df, remover_conteudo_inutil = True):
		if 'titulo_limpo' not in df:
			self.limpar(df, remover_conteudo_inutil=remover_conteudo_inutil)

	def limpar(self, df, remover_conteudo_inutil = True):
		self.limpar_coluna(df, 'titulo', remover_duplicados=True, remover_conteudo_inutil=remover_conteudo_inutil)
		self.limpar_coluna(df, 'resumo', remover_conteudo_inutil=remover_conteudo_inutil)

	def limpar_coluna(self, df, coluna, remover_duplicados = False, remover_conteudo_inutil = False):
		
		print('Limpando ' + coluna)

		coluna_nova = coluna + '_limpo'
		coluna_nova_str = coluna_nova + '_str'

		df[coluna_nova] = df[coluna]
		

		if remover_conteudo_inutil:
			with open(util.full_path('remover_conteudo.json', __file__), encoding='utf-8') as json_file:
				remover_conteudo_json = json.load(json_file)
		
			self.remover_conteudo(df, coluna, remover_conteudo_json['remover_noticia'][coluna]['antes_de_limpar'], noticia_inteira=True)
			self.remover_conteudo(df, coluna, remover_conteudo_json['remover_trecho'][coluna]['antes_de_limpar'], noticia_inteira=False)
		
		# Removendo espaços extras...
		df[coluna_nova] = [re.sub(' +', ' ', reg) for reg in df[coluna_nova]]

		# Removendo hifen para evitar que palavras como segunda-feira sejam separadas:
		df[coluna_nova] = [reg.replace('-', '') for reg in df[coluna_nova]]

		# Removendo tags html...
		regex_html = re.compile('/<\/?[a-z][\s\S]*>/i.test()')
		df[coluna_nova] = [BeautifulSoup(reg, features='html.parser').get_text() if regex_html.search(reg) else reg for reg in df[coluna_nova]]

		# Removendo URLs...
		df[coluna_nova] = [re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", '', reg, flags=re.MULTILINE) for reg in df[coluna_nova]]

		# Tokenizando...
		df[coluna_nova] = [wordpunct_tokenize(reg) for reg in df[coluna_nova]]

		# Convertendo para lower...
		df[coluna_nova] = [[palavra.lower() for palavra in reg] for reg in df[coluna_nova]]

		# Removendo números...
		df[coluna_nova] = [[''.join([ch for ch in palavra if not ch.isdigit()]) for palavra in reg] for reg in df[coluna_nova]]

		# Removendo pontuação...
		df[coluna_nova] = [[''.join([ch for ch in palavra if not ch in string.punctuation]) for palavra in reg] for reg in df[coluna_nova]]
		df[coluna_nova] = [[re.sub(r"\w+…|…", "", palavra) for palavra in reg] for reg in df[coluna_nova]] # Por algum motivo, a linha acima não remove

		# Removendo stopwords...
		df[coluna_nova] = [[palavra for palavra in reg if not palavra in self.stops] for reg in df[coluna_nova]]

		# Removendo acentos...
		df[coluna_nova] = [[unidecode(palavra) for palavra in reg] for reg in df[coluna_nova]]

		# Removendo palavras com tamanho menor q 2
		df[coluna_nova] = [[palavra for palavra in reg if len(palavra) > 1] for reg in df[coluna_nova]]
		
		df[coluna_nova_str] = [' '.join(reg) for reg in df[coluna_nova]]
	
		if remover_conteudo_inutil:
			self.remover_conteudo(df, coluna_nova_str, remover_conteudo_json['remover_noticia'][coluna]['depois_de_limpar'], noticia_inteira=True)
			self.remover_conteudo(df, coluna_nova_str, remover_conteudo_json['remover_trecho'][coluna]['depois_de_limpar'], noticia_inteira=False)
		
		if remover_duplicados:
			duplicados_df = df.duplicated(subset='titulo_limpo_str', keep="first")
			df['ignorar'] = df['ignorar'] | duplicados_df
						
	def remover_conteudo(self, df, coluna, criterios, limpo = False, noticia_inteira = False):
		
		nome_coluna = coluna
		
		if limpo:
			nome_coluna = coluna + 'limpo_str'
			
		for criterio in criterios:
			if noticia_inteira:
				df.loc[df[nome_coluna].str.contains(criterio), 'ignorar'] = True
			else:
				df[nome_coluna] = df[nome_coluna].str.replace(criterio,'', regex=True, case=False)


if __name__ == "__main__":
    Limpador().executar()