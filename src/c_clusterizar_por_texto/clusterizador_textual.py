from sklearn.cluster import DBSCAN
from _common import util
import json
import numpy as np

class ClusterizadorTextual:

    def __init__(self):
        self.a = 0

    def carregar_configs(self):
        with open(util.full_path('config_clusterizacao_textual.json', __file__)) as json_file:
            return json.load(json_file)

    def clusterizar(self, df, epsilon = None, min_samples = None, metric=None):
        configs_dict = self.carregar_configs()
        if (epsilon == None):
            epsilon = float(configs_dict['epsilon'])
        if (min_samples == None):
            min_samples = int(configs_dict['min_samples'])
        if (metric == None):
            metric = configs_dict['metric']

        conteudo_ls = list(df.loc[df['ignorar'] == False]['conteudo_vetorizado'])
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric=metric).fit(conteudo_ls)
        df['assunto'] = None
        df.loc[df['ignorar'] == False, 'assunto'] = dbscan.labels_

    def executar(self):
        print('\nIniciando clusterização por texto')
        noticias_df = util.carregar_todas_as_noticias()
        print(str(len(noticias_df)) + ' noticias serão clusterizadas')
        noticias_df['conteudo_vetorizado'] = [np.array(reg) for reg in noticias_df['conteudo_vetorizado']]
        self.clusterizar(noticias_df)
        #noticias_df['conteudo_vetorizado'] = [reg.tolist() for reg in noticias_df['conteudo_vetorizado']]
        #util.armazenar_todas(noticias_df)
        return noticias_df

if __name__ == "__main__":
	ClusterizadorTextual().executar()