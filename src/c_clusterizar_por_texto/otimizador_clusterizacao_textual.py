
from collections import Counter
from hyperopt import fmin, tpe, STATUS_OK, Trials, partial, hp, space_eval
from _common import util
import pandas as pd

class OtimizadorClusterizacaoTextual:

    def __init__(self, vetorizador, clusterizador, df):
        self.vetorizador = vetorizador
        self.clusterizador = clusterizador
        self.df = df

    def objetivo(self, params):
        self.clusterizador.clusterizar(self.df, epsilon = params['epsilon'], min_samples=None, metric=params['metric'])
        return {'loss': -len(Counter(self.df['assunto']).items()), 'status': STATUS_OK}

    def busca_bayesiana(self, espaco_busca, max_avaliacoes=100):
        tentativas = Trials()
        objetivo_fmin = partial(self.objetivo)
        funcao_fmin = fmin(objetivo_fmin, space = espaco_busca, algo = tpe.suggest, max_evals = max_avaliacoes, trials = tentativas)
        resultado = space_eval(espaco_busca, funcao_fmin)
        return {'melhores_valores': resultado, 'maior_num_classes': tentativas.best_trial['result']['loss']*-1}

    def executar(self,
        usar_texto_limpo = [True, False],
        incluir_resumo = [True, False],
        modelo_linguagem = ['USE', 'SBERT', 'WORD2VEC'],
        epsilon = [0.01, 2.00],
        metric=['cosine'],
        random_state = util.config('random_state')
        ):

        # Atualmente, o Hyperopt está sendo chamado muitas vezes pra otimizar só um hiperparâmetro.
        # Pendência: usar melhor os recursos da ferramenta pra fazer essa otimização.

        resultado = []
        espaco_busca = {
            'epsilon': hp.uniform('epsilon', epsilon[0], epsilon[1]),
            'metric': hp.choice('metric', metric),
            'random_state': random_state
        }
        for m in modelo_linguagem:
            for l in usar_texto_limpo:
                for r in incluir_resumo:
                    try:
                        self.vetorizador.vetorizar(self.df, usar_texto_limpo = l, modelo = m, incluir_resumo = r)
                    except ValueError:
                        continue
                    resultado_bayesiana = self.busca_bayesiana(espaco_busca=espaco_busca, max_avaliacoes=100)
                    resultado.append({
                        'modelo_linguagem': m, 
                        'usar_texto_limpo': l, 
                        'incluir_resumo': r,
                        'melhor_epsilon': resultado_bayesiana['melhores_valores']['epsilon'],
                        'melhor_metric': resultado_bayesiana['melhores_valores']['metric'],
                        'maior_num_classes': resultado_bayesiana['maior_num_classes']})
        return pd.DataFrame.from_records(resultado)

if __name__ == "__main__":
	OtimizadorClusterizacaoTextual().executar()