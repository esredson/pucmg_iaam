
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk import ngrams
import re
from bs4 import BeautifulSoup
from unidecode import unidecode
import string
from pymongo import MongoClient
import pandas as pd

def carregar_noticias():
    with MongoClient(host="mongodb", port=27017) as client:
        db = client.trabalho_puc
        noticias_db = db.noticias
        noticias_ls = list(noticias_db.find({})) # Se necessário, trabalhar com o cursor em vez de carregar tudo pra mem

    noticias_df = pd.json_normalize(noticias_ls)
    noticias_df = noticias_df.reset_index(drop=True)
    noticias_df = noticias_df.sort_values('data_publ')
    return noticias_df

def remover_registros_contendo_regex(df, coluna, str):
    df.drop(df[df[coluna].str.contains(str)].index, inplace=True)
    df = df.reset_index(drop=True)
    
def remover_trechos_contendo_regex(df, coluna, str):
    df[coluna] = df[coluna].str.replace(str,'', regex=True, case=False)
    preparar(df, coluna)

# Setup do nltk
nltk.download('stopwords')
stops = stopwords.words('portuguese')

def preparar(df, coluna, force = False, remover_duplicados = False):
   
    coluna_nova = coluna + '_prep'

    if coluna_nova in df and not force:
        return coluna_nova
    
    df[coluna_nova] = df[coluna]

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

    # Removendo registros com mesmo título...
    if remover_duplicados:
        df[coluna_nova + '_str'] = [' '.join(reg) for reg in df[coluna_nova]]
        df = df.drop_duplicates(subset=coluna_nova + '_str', keep="first")
        df = df.reset_index(drop=True)

def remover_registros_contendo_str_apos_preparacao(df, coluna, ngram_as_str):
    coluna_nova = preparar(df, coluna)
    df.drop(df[df[coluna_nova].str.contains(ngram_as_str)].index, inplace=True)
    df = df.reset_index(drop=True)