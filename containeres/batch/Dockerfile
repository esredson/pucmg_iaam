FROM python

#WORKDIR /app

# Instalar bibliotecas
#RUN pip install pipreqs
#RUN pipreqs .
#RUN pip install -r ./requirements.txt

RUN pip install -r /app/requirements.txt
RUN pip install -r /core/requirements.txt

CMD ["python", "-u", "/core/a_coletar_preprocessar.py"]
