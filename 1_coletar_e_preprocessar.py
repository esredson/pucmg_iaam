# -*- coding: utf-8 -*-
import schedule # pip install schedule
import pandas as pd
import feedparser # pip install feedparser
from tqdm import tqdm
from pymongo import MongoClient # pip install pymongo
from datetime import datetime
from pytz import timezone
import time

def carregar_fontes():
    df = pd.read_csv('assets/fontes.csv', sep='|', header=0)
    return df.to_dict('records')

def normalizar_id(noticia):
    id = noticia['id'] if 'id' in noticia else noticia['link']
    if (not id):
        raise ValueError("Id não informado")
    return id

def normalizar_titulo(noticia):
    titulo = noticia['title']
    if (not titulo):
        raise ValueError("Titulo não informado")
    return titulo

def normalizar_resumo(noticia):
    resumo = noticia['summary']
    if (resumo is None): # Por enquanto, aceitando resumo vazio
        raise ValueError("Resumo não informado")
    return resumo

def normalizar_link(noticia):
    link = noticia['link']
    if (not link):
        raise ValueError("Link não informado")
    return link

def normalizar_idioma(str_data_pt):
    str_data_en = str_data_pt
    for pt, en in {'Fev': 'Feb', 'Abr': 'Apr','Mai': 'May','Ago': 'Aug','Set': 'Sep','Out': 'Oct','': '','Dez': 'Dec'}.items():
        str_data_en = str_data_en.replace(pt, en)
    for pt, en in {'Seg': 'Mon', 'Ter': 'Tue', 'Qua': 'Wed', 'Qui': 'Thu', 'Sex': 'Fri', 'Sáb': 'Sat', 'Sab': 'Sat', 'Dom': 'Sun'}.items():
        str_data_en = str_data_en.replace(pt, en)
    return str_data_en

def converter_str_para_data(str):
    formatos = [
        '%a, %d %b %Y %H:%M:%S %Z', # Sun, 29 May 2022 09:53:09 GMT
        '%a, %d %b %Y %H:%M:%S %z', # Sun, 29 May 2022 19:53:27 -0000 e -0300
        '%Y-%m-%dT%H:%M:%S%z', # 2022-05-29T17:00:01-03:00
        '%a, %d %b %Y %H:%M:%S %Z', # Dom, 29 Mai 2022 17:04:00 -0300  
    ]
    def _converter(str, formato):
        try:
            return datetime.strptime(str, formato)
        except ValueError:
            return None
    
    dt = None
    for formato in formatos:
        dt = _converter(str, formato) 
        if (dt):
            break
    if (not dt):
        raise ValueError("Data " + str + " inválida")

    return dt

def normalizar_timezone(dt):
    if (dt.tzinfo == None):
        dt = dt.replace(tzinfo=timezone('UTC'))
    return dt.astimezone(timezone('America/Sao_Paulo'))

def normalizar_data_publ(noticia):
    str_dt = noticia['published']
    if (not str_dt):
        raise ValueError("Data de publicação não informada")
    str_dt_en = normalizar_idioma(str_dt)
    dt = converter_str_para_data(str_dt_en)
    dt = normalizar_timezone(dt)
    return dt

def normalizar_img(noticia_baixada):
    if 'imagem-destaque' in noticia_baixada: # Agência Brasil etc
        return noticia_baixada['imagem-destaque']
    elif 'mediaurl' in noticia_baixada: # R7 etc
        return noticia_baixada['mediaurl']
    elif 'media_content' in noticia_baixada and len(noticia_baixada['media_content']) > 0 and 'url' in noticia_baixada['media_content'][0]: # G1 etc
        return noticia_baixada['media_content'][0]['url']
    else:
        return ''

def preprocessar(nome_fonte, noticia_baixada):
    noticia = {}
    noticia['id'] = normalizar_id(noticia_baixada)
    noticia['fonte'] = nome_fonte
    noticia['titulo'] = normalizar_titulo(noticia_baixada)
    noticia['resumo'] = normalizar_resumo(noticia_baixada)
    noticia['link'] = normalizar_link(noticia_baixada)
    noticia['data_publ'] = normalizar_data_publ(noticia_baixada)
    noticia['img'] = normalizar_img(noticia_baixada)
    return noticia

def baixar_por_fonte(fonte):
    NewsFeed = feedparser.parse(fonte['url'])
    return NewsFeed.entries

def baixar_e_preprocessar_por_fonte(fonte):
    noticias = baixar_por_fonte(fonte)
    return [preprocessar(fonte['nome'], noticia) for noticia in noticias]

def baixar_e_preprocessar_ultimas_noticias(fontes):
    noticias_por_fonte = [baixar_e_preprocessar_por_fonte(fonte) for fonte in tqdm(fontes)]
    return [noticia for fonte in noticias_por_fonte for noticia in fonte] # flattening

def armazenar(noticias):
    with MongoClient() as client: # MongoClient(host="localhost", port=27017)
        db = client.trabalho_puc
        noticias_db = db.noticias
        for noticia in noticias:
            id_entry = {'id': noticia['id']}
            noticias_db.replace_one(id_entry, noticia, upsert=True)
    #client.close() não necessário devido ao with
    return

def executar():
    print('\nIniciando nova coleta...')
    fontes = carregar_fontes()
    noticias = baixar_e_preprocessar_ultimas_noticias(fontes)
    armazenar(noticias)
    print(str(len(noticias)) + ' notícias coletadas, preprocessadas e armazenadas.\n')

schedule.every(1).minutes.do(executar)
while True:
    schedule.run_pending()
    time.sleep(1)


