from sklearn.cluster import DBSCAN
class ClusterizadorTextual:

    def __init__(self):
        self.a = 0

    def clusterizar(self, df, epsilon = 0.06, min_samples = 2, metric='cosine'):
        conteudo_ls = list(df.loc[df['ignorar'] == False]['conteudo_vetorizado'])
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric=metric).fit(conteudo_ls)
        df['assunto'] = None
        df.loc[df['ignorar'] == False, 'assunto'] = dbscan.labels_

if __name__ == "__main__":
	ClusterizadorTextual().executar()