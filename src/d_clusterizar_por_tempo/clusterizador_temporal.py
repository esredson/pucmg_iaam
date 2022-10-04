import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from datetime import timedelta

class ClusterizadorTemporal:

    KERNEL_DEFAULT = 'gaussian'
    DENOMINADOR_BANDWIDTH_DEFAULT = 55

    def __init__(self):
        self.a = 0

    def clusterizar(self, df, denominador_bandwidth = DENOMINADOR_BANDWIDTH_DEFAULT, kernel = KERNEL_DEFAULT):
        clusters = []  
        for cluster_label in df['assunto'].unique():
            if (cluster_label == None or cluster_label == -1):
                continue
            noticias_do_cluster = df.loc[df['assunto'] == cluster_label]
            noticias_do_cluster = noticias_do_cluster.reset_index()
            clusters.append({
                'id': int(cluster_label), 
                'subclusters': self.gerar_subclusters(noticias_do_cluster, denominador_bandwidth, kernel)
            })
        return clusters

    def gerar_subclusters(self, subconjunto_noticias, denominador_bandwidth = DENOMINADOR_BANDWIDTH_DEFAULT, kernel = KERNEL_DEFAULT):
        datas = subconjunto_noticias['data_publ']
        datas_as_seconds = np.array([(d-datas[0]).total_seconds() for d in datas])
        resultado_kde = self.aplicar_kde(datas_as_seconds, denominador_bandwidth, kernel)
        minimos_locais = resultado_kde['minimos_locais']
        eixo_x = resultado_kde['eixo_x']

        # Cada mínimo local encontrado pelo KDE delimitará um subcluster
        datas_ainda_nao_processadas = datas_as_seconds.view()
        subclusters = []
        num_total_noticias = len(subconjunto_noticias)
        for minimo_local in minimos_locais: 
            datas_deste_subcluster = datas_ainda_nao_processadas[datas_ainda_nao_processadas < eixo_x[minimo_local]] # Coloca no subcluster as datas anteriores àquele ponto mínimo
            if (len(datas_deste_subcluster) == 0): # Isso porque às vezes o argrelextrema pode retornar dois pontos mínimos muito próximos um do outro
                continue
            subcluster = {'evento': datas_deste_subcluster[-1], 'importancia': len(datas_deste_subcluster)/num_total_noticias} # A notícia que representará o subcluster será a mais nova. A importância é a qtd de notícias que a escolhida está representando.
            subclusters.append(subcluster)
            datas_ainda_nao_processadas = np.array([dt for dt in datas_ainda_nao_processadas if dt not in datas_deste_subcluster]) # Vai eliminando dentre os datas_ainda_nao_processadas os já processados
        subclusters.append({'evento': datas_ainda_nao_processadas[-1], 'importancia': len(datas_ainda_nao_processadas)/num_total_noticias}) #Cria um último cluster com as que estão depois do último ponto mínimo
        
        # Convertendo de volta segundos pra datas:
        for subcluster in subclusters:
            subcluster['evento'] = datas[0] + timedelta(seconds=subcluster['evento'])

        # Ordena os subclusters por ordem inversa de data:
        subclusters = sorted(subclusters, key=lambda subcluster: subcluster['evento'], reverse=True)

        # Substituindo cada data nos subclusters pela notícia correspondente:
        for subcluster in subclusters:
            subcluster['evento'] = subconjunto_noticias.loc[subconjunto_noticias['data_publ'] == subcluster['evento']]

        # Convertendo cada notícia do formato de linha de dataframe para o de dicionário
        for subcluster in subclusters:
            subcluster['evento'] = subcluster['evento'].loc[:,['id', 'fonte', 'titulo', 'resumo', 'data_publ', 'link', 'img']].to_dict('records')[0]
                
        return subclusters

    def aplicar_kde(self, samples, denominador_bandwidth = DENOMINADOR_BANDWIDTH_DEFAULT, kernel = KERNEL_DEFAULT):
        eixo_x = np.linspace(start = samples[0], stop = samples[-1], num=100, dtype = np.float64) # Cria o eixo x do gráfico com 100 posições distribuídas da primeira data até a última data
        eixo_x = np.sort(np.unique(np.concatenate([eixo_x, samples]))) # Inclui no eixo x os próprios valores das datas
        samples_reshaped = samples.reshape(-1, 1)
        kde = KernelDensity(bandwidth = (samples[-1]-samples[0]) / denominador_bandwidth, kernel=kernel).fit(samples_reshaped)
        eixo_x = eixo_x.reshape(-1, 1)
        eixo_y = kde.score_samples(eixo_x)

        minimos_locais = argrelextrema(eixo_y, np.less_equal)[0]
        maximos_locais = argrelextrema(eixo_y, np.greater_equal)[0]

        return {'minimos_locais': minimos_locais, 'maximos_locais': maximos_locais, 'eixo_x': eixo_x, 'eixo_y': eixo_y}


if __name__ == "__main__":
	ClusterizadorTemporal().executar()