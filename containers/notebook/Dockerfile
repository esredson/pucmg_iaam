FROM jupyter/base-notebook

COPY ./src/ /app/notebook/
COPY ./pyproject.toml /app/notebook/
COPY ./poetry.lock /app/notebook/

WORKDIR /app/notebook
ENV PYTHONPATH=${PYTHONPATH}:${PWD}

RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install
