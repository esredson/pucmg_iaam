
import feedparser # pip install feedparser
import locale
import time

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

fonte = 'http://agenciabrasil.ebc.com.br/rss/politica/feed.xml'
NewsFeed = feedparser.parse(fonte)
noticias_dict = NewsFeed.entries
a = 2
#noticias_dict['data'] = noticias_dict[]

formatos = [
    { 'formato': '%a, %d %b %Y %H:%M:%S %Z', 'locale': 'en_US.utf8' }, # Sun, 29 May 2022 09:53:09 GMT
    { 'formato': '%a, %d %b %Y %H:%M:%S %z', 'locale': 'en_US.utf8' }, # Sun, 29 May 2022 19:53:27 -0000
    { 'formato': '%Y-%m-%dT%H:%M:%S%z', 'locale': 'en_US.utf8' }, # 2022-05-29T17:00:01-03:00
    'pt_BR.utf8'
]