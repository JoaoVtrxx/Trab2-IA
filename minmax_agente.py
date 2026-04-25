# Agente Min-Max com Poda Alfa-Beta para o jogo Othello.

# Função de avaliação heurística:
#     Avaliar(s) = w1·f1(s) + w2·f2(s) + w3·f3(s) + w4·f4(s)

# Onde:
#     f1(s) = Diferença de peças normalizada
#             f1 = (minhas - adversárias) / (minhas + adversárias)
#             Peso w1 = 1.0  -> métrica base, sempre relevante

#     f2(s) = Mobilidade relativa (número de jogadas disponíveis)
#             f2 = (mob_minha - mob_adv) / (mob_minha + mob_adv + ε)
#             Peso w2 = 3.0  -> mobilidade é crítica no meio-jogo;
#                              ter mais opções evita posições forçadas

#     f3(s) = Controle de cantos
#             cantos: peso altíssimo (capturam definitivamente)
#             Peso w3 = 5.0  -> cantos são irreversíveis e dominantes;
#                              literatura de Othello confirma superioridade

#     f4(s) = Controle de bordas
#             bordas: peso médio (estáveis, difíceis de reverter)
#             Peso w4 = 2.0  -> bordas são importantes mas não tanto quanto cantos

# Justificativa dos pesos:
#     A hierarquia w3 > w2 > w4 > w1 reflete a teoria clássica do Othello:
#     (1) Cantos determinam o resultado na maioria dos casos.
#     (2) Mobilidade controla o ritmo do jogo.
#     (3) Bordas dão estabilidade.
#     (4) Diferença bruta de peças é enganosa no início (sacrifícios são táticos).

import math
import time
from typing import Tuple, Optional

from othello_jogo import Othello

# Pesos da função de avaliação (ver início do módulo)
W_DIFF_PECAS    = 1.0   # w1
W_MOBILIDADE    = 3.0   # w2
W_CANTOS        = 5.0   # w3
W_BORDAS        = 2.0   # w4


class MinMaxAgente:
    # Parâmetros
    # ----------
    # jogador : int
    #     Identidade do agente (1 ou -1).
    # profundidade : int
    #     Profundidade máxima de busca (3 a 5 recomendado).
    # verboso : bool
    #     Se True, imprime a árvore de decisão.

    def __init__(self, jogador: int, profundidade: int = 4, verboso: bool = False):
        assert 3 <= profundidade # Profundidade deve ser maior ou igual a 3
        self.jogador  = jogador
        self.profundidade   = profundidade
        self.verboso = verboso

        # Estatísticas internas
        self.nos_avaliados = 0
        self.contagem_podas     = 0
        self.tempo_do_ultimo_movimento  = 0.0

    def escolhe_movimento(self, jogo: Othello) -> Optional[Tuple[int, int]]:
        # Escolhe a melhor jogada para o estado atual do jogo, com base na árvore.
        # Retorna None se não houver jogadas disponíveis (passa a vez).
        # Trata o nó inicial (raiz), sendo ele um de max.

        start = time.time()
        self.nos_avaliados = 0
        self.contagem_podas     = 0

        movimentos_validos = jogo.movimentos_validos(self.jogador)
        if not movimentos_validos:
            return None

        melhor_movimento  = None
        melhor_valor = -math.inf
        alfa = -math.inf
        beta  =  math.inf

        if self.verboso:
            print(f"\n{'='*60}")
            print(f"  MinMax (profundidade={self.profundidade}) - Jogador {self.jogador}")
            print(f"  Jogadas válidas: {movimentos_validos}")
            print(f"{'='*60}")

        for movimento in movimentos_validos:
            filho = jogo.clonar()
            filho.executa_jogada(movimento[0], movimento[1], self.jogador)
            filho.troca_jogador()

            valor = self.minimax(filho, self.profundidade - 1, # O nível do raiz é nó de MAX (jogo atual, em que o jogador precisa jogar), 
                                  alfa, beta,                  # o próximo é nó de MIN, com profundidade já ajustada para profundidade - 1.
                                  maximizando=False,
                                  level=1, movimento_label=str(movimento))

            if self.verboso:
                print(f"  [Raiz -> {movimento}] valor propagado = {valor:.3f}")

            if valor > melhor_valor:
                melhor_valor = valor
                melhor_movimento  = movimento

            alfa = max(alfa, melhor_valor)

        self.tempo_do_ultimo_movimento = time.time() - start

        if self.verboso:
            print(f"\n   Melhor jogada: {melhor_movimento}  (valor={melhor_valor:.3f})")
            print(f"  Nós avaliados : {self.nos_avaliados}")
            print(f"  Podas Alfa-Beta     : {self.contagem_podas}")
            print(f"  Tempo         : {self.tempo_do_ultimo_movimento:.4f}s")

        return melhor_movimento

    def minimax(self, jogo: Othello, profundidade: int,
                 alfa: float, beta: float,
                 maximizando: bool, level: int = 0,
                 movimento_label: str = "") -> float:
        # Busca recursiva Min-Max com Poda Alfa-Beta.

        # Parâmetros
        # ----------
        # jogo            : estado atual do jogo
        # profundidade    : profundidade restante
        # alfa            : melhor valor garantido para MAX
        # beta            : melhor valor garantido para MIN
        # maximizando     : True se é o turno do agente (MAX)
        # level           : nível atual (para verboso)
        # movimento_label : rótulo para exibição na árvore verboso

        jogador_atual = self.jogador if maximizando else -self.jogador

        # Caso base: profundidade zero (chegou na folha) ou jogo terminou (sem mais jogadas)
        if profundidade == 0 or jogo.verifica_fim():
            self.nos_avaliados += 1
            return self.avaliacao_posicao(jogo)

        movimentos_validos = jogo.movimentos_validos(jogador_atual)

        # Sem jogadas
        if not movimentos_validos:
            filho = jogo.clonar()
            filho.troca_jogador()
            return self.minimax(filho, profundidade - 1,
                                 alfa, beta, not maximizando, # Inverte o maximizando
                                 level + 1, "PASS")

        identar = "  " * level

        # ----- Nó MAX
        if maximizando:
            valor_maximo = -math.inf
            for movimento in movimentos_validos:
                filho = jogo.clonar()
                filho.executa_jogada(movimento[0], movimento[1], jogador_atual)
                filho.troca_jogador()

                valor = self.minimax(filho, profundidade - 1,
                                    alfa, beta, False,
                                    level + 1, str(movimento))

                if self.verboso:
                    print(f"{identar}  [MAX nível={level} movimento={movimento}] val={valor:.3f}  Alfa={alfa:.3f}  Beta={beta:.3f}")

                valor_maximo = max(valor_maximo, valor)
                alfa   = max(alfa, valor_maximo)

                # Poda Beta
                if alfa >= beta:
                    self.contagem_podas += 1
                    if self.verboso:
                        print(f"{identar} PODA Beta em {movimento}: Alfa({alfa:.3f}) ≥ Beta({beta:.3f})")
                    break

            return valor_maximo

        # ----- Nó MIN
        else:
            valor_minimo = math.inf
            for movimento in movimentos_validos:
                filho = jogo.clonar()
                filho.executa_jogada(movimento[0], movimento[1], jogador_atual)
                filho.troca_jogador()

                valor = self.minimax(filho, profundidade - 1,
                                    alfa, beta, True,
                                    level + 1, str(movimento))

                if self.verboso:
                    print(f"{identar}  [MIN nível={level} movimento={movimento}] val={valor:.3f}  Alfa={alfa:.3f}  Beta={beta:.3f}")

                valor_minimo = min(valor_minimo, valor)
                beta    = min(beta, valor_minimo)

                # Poda Alfa
                if alfa >= beta:
                    self.contagem_podas += 1
                    if self.verboso:
                        print(f"{identar} PODA Alfa em {movimento}: Alfa({alfa:.3f}) ≥ Beta({beta:.3f})")
                    break

            return valor_minimo

    # Função de avaliação heurística

    def avaliacao_posicao(self, jogo: Othello) -> float:
        # Avaliar(s) = w1·f1(s) + w2·f2(s) + w3·f3(s)

        # Retorna valor positivo quando o estado é bom para self.jogador.

        tabuleiro = jogo.tabuleiro
        tamanho  = jogo.tamanho
        oponente   = -self.jogador

        # --- f1: Diferença de peças normalizada ---
        minhas_pecas  = int((tabuleiro == self.jogador).sum())
        oponente_pecas = int((tabuleiro == oponente).sum())
        total = minhas_pecas + oponente_pecas
        if total == 0:
            f1 = 0.0
        else:
            f1 = (minhas_pecas - oponente_pecas) / total   # Intervalo [-1, 1]

        # --- f2: Mobilidade relativa ---
        minha_mobilidade  = len(jogo.movimentos_validos(self.jogador))
        mobilidade_oponente = len(jogo.movimentos_validos(oponente))
        soma_mobilidade = minha_mobilidade + mobilidade_oponente + 1e-9 # 1e-9 evita divisão por zero
        f2 = (minha_mobilidade - mobilidade_oponente) / soma_mobilidade # Intervalo [-1, 1]

        # --- f3: Controle de cantos e bordas ---

        # Cantos
        cantos = [(0, 0), (0, tamanho - 1),
                   (tamanho - 1, 0), (tamanho - 1, tamanho - 1)]

        # Cantos são os mais valiosos, pois uma vez capturados, não podem ser revertidos.
        score_cantos = 0.0
        for l, c in cantos:
            if tabuleiro[l][c] == self.jogador:
                score_cantos += 1.0
            elif tabuleiro[l][c] == oponente:
                score_cantos -= 1.0

        # Bordas (excluindo cantos já contados)
        score_bordas = 0.0
        for c in range(1, tamanho - 1): # Borda superior
            if tabuleiro[0][c] == self.jogador:   
                score_bordas += 0.3
            elif tabuleiro[0][c] == oponente:         
                score_bordas -= 0.3

        for c in range(1, tamanho - 1): # Borda inferior
            if tabuleiro[tamanho-1][c] == self.jogador:  
                score_bordas += 0.3
            elif tabuleiro[tamanho-1][c] == oponente:        
                score_bordas -= 0.3
                
        for l in range(1, tamanho - 1):  # Borda esquerda
            if tabuleiro[l][0] == self.jogador:   
                score_bordas += 0.3
            elif tabuleiro[l][0] == oponente:     
                score_bordas -= 0.3

        for l in range(1, tamanho - 1): # Borda direita
            if tabuleiro[l][tamanho-1] == self.jogador:  
                score_bordas += 0.3
            elif tabuleiro[l][tamanho-1] == oponente:        
                score_bordas -= 0.3

        # Normaliza f3 (cantos) e f4 (bordas) para intervalo [-1, 1]
        f3_maximo = 4.0
        f3 = score_cantos / f3_maximo

        f4_maximo = 4 * (tamanho - 2) * 0.3
        if f4_maximo > 0.0:
            f4 = score_bordas / f4_maximo
        else:
            f4 = 0.0

        # --- Avaliação final ---
        if jogo.verifica_fim():
            vencedor = jogo.get_vencedor()
            if vencedor == self.jogador:
                return 1e6 # Eu ganhei
            elif vencedor == oponente:
                return -1e6 # Oponente ganhou
            else:
                return 0.0 # Empate

        return W_DIFF_PECAS * f1 + W_MOBILIDADE * f2 + W_CANTOS * f3 + W_BORDAS * f4
