
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
import os

def full_path(arq_str):
   dir = os.path.dirname(os.path.abspath(__file__))
   full = os.path.join(dir, arq_str)
   return full

def carregar_noticias_nao_preparadas():
	with MongoClient(host="localhost", port=27017) as client:
		db = client.trabalho_puc
		noticias_ls = list(db.noticias.find({
			'$and': [
				{ 
					'titulo_preparado' : { 
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

def armazenar(noticias_df):
	with MongoClient(host="localhost", port=27017) as client: #mongodb
		db = client.trabalho_puc
		noticias_db = db.noticias
		noticias_dict_ls = noticias_df.to_dict('records')
		for noticia_dict in noticias_dict_ls:
			id_entry = {'id': noticia_dict['id']}
			noticias_db.replace_one(id_entry, noticia_dict)

def executar():
	print('\nIniciando preparação do texto...')
	noticias_df = carregar_noticias_nao_preparadas()
	print(str(len(noticias_df)) + ' noticias serão preparadas...')
	preparar(noticias_df)
	armazenar(noticias_df)

def preparar(df):
	preparar_coluna(df, 'titulo', remover_duplicados=True)
	preparar_coluna(df, 'resumo')

# Setup do nltk
nltk.download('stopwords')
stops = stopwords.words('portuguese')

def preparar_coluna(df, coluna, remover_duplicados = False):
	
	coluna_nova = coluna + '_preparado'
	coluna_nova_str = coluna_nova + '_str'

	df[coluna_nova] = df[coluna]
	

	with open(full_path('./remover_conteudo.json'), encoding='utf-8') as json_file:
		remover_conteudo_json = json.load(json_file)
	
	remover_conteudo(df, coluna, remover_conteudo_json['noticia'][coluna]['antes_de_preparar'], noticia_inteira=True)
	remover_conteudo(df, coluna, remover_conteudo_json['trecho'][coluna]['antes_de_preparar'], noticia_inteira=False)
	
	# Removendo espaços extras...
	df[coluna_nova] = [re.sub(' +', ' ', reg) for reg in df[coluna_nova]]

	# Removendo hifen para evitar que palavras como segunda-feira sejam separadas:
	df[coluna_nova] = [reg.replace('-', '') for reg in df[coluna_nova]]

	# Removendo tags html...
	df[coluna_nova] = [BeautifulSoup(reg, features='html.parser').get_text() for reg in df[coluna_nova]]

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
	df[coluna_nova] = [[palavra for palavra in reg if not palavra in stops] for reg in df[coluna_nova]]

	# Removendo acentos...
	df[coluna_nova] = [[unidecode(palavra) for palavra in reg] for reg in df[coluna_nova]]

	# Removendo palavras com tamanho menor q 2
	df[coluna_nova] = [[palavra for palavra in reg if len(palavra) > 1] for reg in df[coluna_nova]]
	
	df[coluna_nova_str] = [' '.join(reg) for reg in df[coluna_nova]]
  
	remover_conteudo(df, coluna_nova_str, remover_conteudo_json['noticia'][coluna]['depois_de_preparar'], noticia_inteira=True)
	remover_conteudo(df, coluna_nova_str, remover_conteudo_json['trecho'][coluna]['depois_de_preparar'], noticia_inteira=False)
	
	if remover_duplicados:
		duplicados_df = df.duplicated(subset='titulo_preparado_str', keep="first")
		df['ignorar'] = df['ignorar'] | duplicados_df
					
def remover_conteudo(df, coluna, criterios, preparado = False, noticia_inteira = False):
	
	nome_coluna = coluna
	
	if preparado:
		nome_coluna = coluna + 'preparado_str'
		
	for criterio in criterios:
		if noticia_inteira:
			df.loc[df[nome_coluna].str.contains(criterio), 'ignorar'] = True
		else:
			df[nome_coluna] = df[nome_coluna].str.replace(criterio,'', regex=True, case=False)


if __name__ == "__main__":
    executar()