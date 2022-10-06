from fastapi import FastAPI  
from pymongo import MongoClient
from _common import util

app = FastAPI()   
@app.get("/") 
async def main_route():     
    print('Carregando todos os clusters')
    with MongoClient(host=util.host_mongodb(), port=27017) as client:
        db = client.trabalho_puc
        clusters_ls = list(db.clusters.find({}, {'_id': 0}))
    return {"assuntos": clusters_ls}