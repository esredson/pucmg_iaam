import argparse
import schedule
import time
from a_coletar_preprocessar import coletor_preprocessador
from a_coletar_preprocessar.coletor_preprocessador import ColetorPreprocessador
from b_limpar.limpador import Limpador
from b_vetorizar.vetorizador import Vetorizador
from c_clusterizar_por_texto.clusterizador_textual import ClusterizadorTextual
from d_clusterizar_por_tempo.clusterizador_temporal import ClusterizadorTemporal

coletor_preprocessador = ColetorPreprocessador()
limpador = Limpador()
vetorizador = Vetorizador()
clusterizador_textual = ClusterizadorTextual()
clusterizador_temporal = ClusterizadorTemporal()

def executar(baixar=False):
    print('\nIniciando nova passagem pelo ciclo')
    if (baixar):
        coletor_preprocessador.executar()
    limpador.executar()
    vetorizador.executar()
    clusterizador_temporal.executar(clusterizador_textual.executar()) # Pra evitar ter de salvar todas as notícias toda vez
    print('Terminando passagem pelo ciclo')

parser=argparse.ArgumentParser()
parser.add_argument("--baixar_continuamente", action='store_true',
    help="Indica se será feito o download de novas notícias em intervalos regulares")
args=parser.parse_args()

if (not args.baixar_continuamente):
    executar()
    exit()

print("\nIniciando fluxo contínuo\n")
schedule.every(5).minutes.do(executar, True)
while True:
    schedule.run_pending()
    time.sleep(1)