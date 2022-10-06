# -*- coding: utf-8 -*-

import pandas as pd
import feedparser
from pymongo import MongoClient
from datetime import datetime
from pytz import timezone
from _common import util

class ColetorPreprocessador:

    def carregar_fontes(self):
        df = pd.read_csv(util.full_path('fontes.csv', __file__), sep='|', header=0)
        return df.to_dict('records')

    def normalizar_id(self, noticia):
        id = noticia['id'] if 'id' in noticia else noticia['link']
        if (not id):
            raise ValueError("Id não informado")
        return id

    def normalizar_titulo(self, noticia):
        titulo = noticia['title']
        if (not titulo):
            raise ValueError("Titulo não informado")
        return titulo

    def normalizar_resumo(self, noticia):
        resumo = noticia['summary']
        if (resumo is None): # Por enquanto, aceitando resumo vazio
            raise ValueError("Resumo não informado")
        return resumo

    def normalizar_link(self, noticia):
        link = noticia['link']
        if (not link):
            raise ValueError("Link não informado")
        return link

    def normalizar_idioma(self, str_data_pt):
        str_data_en = str_data_pt
        for pt, en in {'Fev': 'Feb', 'Abr': 'Apr','Mai': 'May','Ago': 'Aug','Set': 'Sep','Out': 'Oct','': '','Dez': 'Dec'}.items():
            str_data_en = str_data_en.replace(pt, en)
        for pt, en in {'Seg': 'Mon', 'Ter': 'Tue', 'Qua': 'Wed', 'Qui': 'Thu', 'Sex': 'Fri', 'Sáb': 'Sat', 'Sab': 'Sat', 'Dom': 'Sun'}.items():
            str_data_en = str_data_en.replace(pt, en)
        return str_data_en

    def converter_str_para_data(self, str):
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

    def normalizar_timezone(self, dt):
        if (dt.tzinfo == None):
            dt = dt.replace(tzinfo=timezone('UTC'))
        return dt.astimezone(timezone('America/Sao_Paulo'))

    def normalizar_data_publ(self, noticia):
        str_dt = noticia['published']
        if (not str_dt):
            raise ValueError("Data de publicação não informada")
        str_dt_en = self.normalizar_idioma(str_dt)
        dt = self.converter_str_para_data(str_dt_en)
        dt = self.normalizar_timezone(dt)
        return dt

    def normalizar_img(self, noticia_baixada):
        if 'imagem-destaque' in noticia_baixada: # Agência Brasil etc
            return noticia_baixada['imagem-destaque']
        elif 'mediaurl' in noticia_baixada: # R7 etc
            return noticia_baixada['mediaurl']
        elif 'media_content' in noticia_baixada and len(noticia_baixada['media_content']) > 0 and 'url' in noticia_baixada['media_content'][0]: # G1 etc
            return noticia_baixada['media_content'][0]['url']
        else:
            return ''

    def preprocessar(self, nome_fonte, noticia_baixada):
        noticia = {}
        noticia['id'] = self.normalizar_id(noticia_baixada)
        noticia['fonte'] = nome_fonte
        noticia['titulo'] = self.normalizar_titulo(noticia_baixada)
        noticia['resumo'] = self.normalizar_resumo(noticia_baixada)
        noticia['link'] = self.normalizar_link(noticia_baixada)
        noticia['data_publ'] = self.normalizar_data_publ(noticia_baixada)
        noticia['img'] = self.normalizar_img(noticia_baixada)
        return noticia

    def baixar_por_fonte(self, fonte):
        NewsFeed = feedparser.parse(fonte['url'])
        return NewsFeed.entries

    def baixar_e_preprocessar_por_fonte(self, fonte):
        noticias = self.baixar_por_fonte(fonte)
        return [self.preprocessar(fonte['nome'], noticia) for noticia in noticias]

    def baixar_e_preprocessar_ultimas_noticias(self, fontes):
        noticias_por_fonte = [self.baixar_e_preprocessar_por_fonte(fonte) for fonte in fontes]
        return [noticia for fonte in noticias_por_fonte for noticia in fonte] # flattening

    def armazenar(self, noticias):
        armazenadas = 0
        with MongoClient(host=util.host_mongodb(), port=27017) as client:
            db = client.trabalho_puc
            noticias_db = db.noticias
            for noticia in noticias:
                if noticias_db.count_documents({'id': noticia['id']}) == 0:
                    noticias_db.insert_one(noticia)
                    armazenadas+=1
        return armazenadas

    def executar(self):
        print('\nColetando e preprocessando')
        fontes = self.carregar_fontes()
        noticias = self.baixar_e_preprocessar_ultimas_noticias(fontes)
        qtd_armazenadas = self.armazenar(noticias)
        print(str(len(noticias)) + ' notícias coletadas e preprocessadas. ' + str(qtd_armazenadas) + ' armazenadas\n')

if __name__ == "__main__":
    ColetorPreprocessador().executar()


