import schedule
import time
import sys
sys.path.append('/core') # funciona apenas dentro do container
import a_coletar_e_preprocessar.py
import b1_preparar_texto.py
import b2_vetorizar.py

def executar():
    a_coletar_e_preprocessar.executar()
    b1_preparar_texto.executar()
    #b2_vetorizar.executar()

print("\nIniciando agendamento...\n")
schedule.every(1).minutes.do(executar)
while True:
    schedule.run_pending()
    time.sleep(1)