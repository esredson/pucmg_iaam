FROM mongo
COPY assets/importar_noticias.sh /docker-entrypoint-initdb.d/importar_noticias.sh
COPY assets/noticias.json /docker-entrypoint-initdb.d/noticias.json

#TODO: usar um container separado para o Python usando a imagem oficial do Python
#CMD mkdir /scripts_python
COPY 1_coletar_e_preprocessar.py /scripts_python/1_coletar_e_preprocessar.py
COPY assets/preparar_python.sh /scripts_python/preparar_python.sh
#CMD /scripts_python/preparar_python.sh