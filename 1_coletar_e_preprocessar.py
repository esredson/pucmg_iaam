import schedule # pip install schedule
import pandas as pd
import feedparser # pip install feedparser
from tqdm import tqdm
from pymongo import MongoClient # pip install pymongo
from dateutil import parser

def carregar_fontes():
    return pd.read_csv('assets/fontes.csv')

def normalizar_id(noticia):
    return noticia['id'] if 'id' in noticia else noticia['link']

def normalizar_idioma(str_data_pt):
    str_data_en = str_data_pt
    for pt, en in {'Fev': 'Feb', 'Abr': 'Apr','Mai': 'May','Ago': 'Aug','Set': 'Sep','Out': 'Oct','': '','Dez': 'Dec'}.items():
        str_data_en = str_data_en.replace(pt, en)
    for pt, en in {'Seg': 'Mon', 'Ter': 'Tue', 'Qua': 'Wed', 'Qui': 'Thu', 'Sex': 'Fri', 'Sáb': 'Sat', 'Sab': 'Sat', 'Dom': 'Sun'}.items():
        str_data_en = str_data_en.replace(pt, en)
    return str_data_en

def normalizar_timezone(dt):
    return

def normalizar_data_publ(str_data):
    data = normalizar_idioma(str_data)
    data = normalizar_timezone(data)
    return data

def normalizar_img(noticia_baixada):
    if 'imagem-destaque' in noticia_baixada: # Agência Brasil etc
        return noticia_baixada['imagem-destaque']
    elif 'mediaurl' in noticia_baixada: # R7 etc
        return noticia_baixada['mediaurl']
    elif 'media_content' in noticia_baixada and len(noticia_baixada['media_content']) > 0 and 'url' in noticia_baixada['media_content'][0]: # G1 etc
        return noticia_baixada['media_content'][0]['url']

def preprocessar(nome_fonte, noticia_baixada):
    noticia = {}
    noticia['id'] = normalizar_id(noticia_baixada)
    noticia['fonte'] = nome_fonte
    noticia['titulo'] = noticia_baixada['title']
    noticia['resumo'] = noticia_baixada['summary']
    noticia['link'] = noticia_baixada['link']
    noticia['data_publ'] = normalizar_data_publ(noticia_baixada['published'])
    noticia['img'] = normalizar_img(noticia_baixada)
    return noticia

def baixar_por_fonte(fonte):
    NewsFeed = feedparser.parse(fonte['url'])
    return NewsFeed.entries

def baixar_e_preprocessar_por_fonte(fonte):
    noticias = baixar_por_fonte(fonte)
    return [preprocessar(fonte['nome'], noticia) for noticia in noticias]

def baixar_e_preprocessar_ultimas_noticias(fontes_df):
    return [noticia for noticia in (baixar_e_preprocessar_por_fonte(fonte) for fonte in tqdm(fontes_df))]

def armazenar(noticias_dict):
    with MongoClient() as client: # MongoClient(host="localhost", port=27017)
        db = client.trabalho_puc
        noticias_db = db.noticias
        for noticia in noticias_dict:
            id_entry = {'id': noticia['id']}
            noticias_db.replace_one(id_entry, noticia, upsert=True)
    #client.close() não necessário devido ao with
    return

def executar():
    fontes_df = carregar_fontes()
    noticias = baixar_e_preprocessar_ultimas_noticias(fontes_df)
    armazenar(noticias)

schedule.every(10).minutes.do(executar)




