import os
from pymongo import MongoClient
import pandas as pd
import json

def full_path(nome_arq_a_carregar, arq_base=__file__):
   dir = os.path.dirname(os.path.abspath(arq_base))
   full = os.path.join(dir, nome_arq_a_carregar)
   return full

def carregar_configs():
   with open(full_path('config.json'), encoding='utf-8') as json_file:
      return json.load(json_file)

def config(nome):
   return carregar_configs()[nome]

def host_mongodb():
   return config('host_mongodb')

def carregar_todas_as_noticias():
	with MongoClient(host=host_mongodb(), port=27017) as client:
		db = client.trabalho_puc
		noticias_ls = list(db.noticias.find({}))
		noticias_df = pd.json_normalize(noticias_ls)
		noticias_df = noticias_df.reset_index(drop=True)
		noticias_df = noticias_df.sort_values('data_publ')
		if 'ignorar' not in noticias_df:
			noticias_df['ignorar'] = False
		return noticias_df

def armazenar_todas(noticias_df):
	with MongoClient(host="localhost", port=27017) as client: #mongodb
		db = client.trabalho_puc
		noticias_db = db.noticias
		noticias_dict_ls = noticias_df.to_dict('records')
		for noticia_dict in noticias_dict_ls:
			id_entry = {'id': noticia_dict['id']}
			noticias_db.replace_one(id_entry, noticia_dict)