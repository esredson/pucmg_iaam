FROM python

WORKDIR /app

COPY ./coletar_e_preprocessar.py .
COPY ./fontes.csv .

# Instalar bibliotecas
RUN pip install pipreqs
RUN pipreqs .
RUN pip install -r ./requirements.txt

CMD ["python", "-u", "./coletar_e_preprocessar.py"]
