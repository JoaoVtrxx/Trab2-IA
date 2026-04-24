# Agente Min-Max com Poda Alfa-Beta para o jogo Othello.

# Função de avaliação heurística:
#     Avaliar(s) = w1·f1(s) + w2·f2(s) + w3·f3(s)

# Onde:
#     f1(s) = Diferença de peças normalizada
#             f1 = (minhas - adversárias) / (minhas + adversárias)
#             Peso w1 = 1.0  → métrica base, sempre relevante

#     f2(s) = Mobilidade relativa (número de jogadas disponíveis)
#             f2 = (mob_minha - mob_adv) / (mob_minha + mob_adv + ε)
#             Peso w2 = 3.0  → mobilidade é crítica no meio-jogo;
#                              ter mais opções evita posições forçadas

#     f3(s) = Controle de cantos | Controle de bordas
#             cantos: peso altíssimo (capturam definitivamente)
#             bordas: peso médio (estáveis, difíceis de reverter)
#             Peso w3 = 5.0  → cantos são irreversíveis e dominantes;
#                              literatura de Othello confirma superioridade

# Justificativa dos pesos:
#     A hierarquia w3 > w2 > w1 reflete a teoria clássica do Othello:
#     (1) Cantos e bordas determinam o resultado na maioria dos casos.
#     (2) Mobilidade controla o ritmo do jogo.
#     (3) Diferença bruta de peças é enganosa no início (sacrifícios são táticos).

import math
import time
from typing import Tuple, Optional

from othello_game import Othello

# Pesos da função de avaliação (ver início do módulo)
W_DIFF_PECAS    = 1.0   # w1
W_MOBILIDADE    = 3.0   # w2
W_CANTOS_BORDAS = 5.0   # w3


class MinMaxAgent:
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

    def escolhe_movimento(self, game: Othello) -> Optional[Tuple[int, int]]:
        # Escolhe a melhor jogada para o estado atual do jogo, com base na árvore.
        # Retorna None se não houver jogadas disponíveis (passa a vez).
        # Trata o nó inicial (raiz), sendo ele um de max.

        start = time.time()
        self.nos_avaliados = 0
        self.contagem_podas     = 0

        movimentos_validos = game.movimentos_validos(self.jogador)
        if not movimentos_validos:
            return None

        melhor_movimento  = None
        melhor_valor = -math.inf
        alpha = -math.inf
        beta  =  math.inf

        if self.verboso:
            print(f"\n{'='*60}")
            print(f"  MinMax (profundidade={self.profundidade}) - Jogador {self.jogador}")
            print(f"  Jogadas válidas: {movimentos_validos}")
            print(f"{'='*60}")

        for movimento in movimentos_validos:
            child = game.clonar()
            child.executa_jogada(movimento[0], movimento[1], self.jogador)
            child.troca_jogador()

            valor = self.minimax(child, self.profundidade - 1, # O nível do raiz é nó de MAX, 
                                  alpha, beta,                 # o próximo é nó de MIN, com profundidade já ajustada para profundidade - 1.
                                  maximizando=False,
                                  level=1, movimento_label=str(movimento))

            if self.verboso:
                print(f"  [Raiz → {movimento}] valor propagado = {valor:.3f}")

            if valor > melhor_valor:
                melhor_valor = valor
                melhor_movimento  = movimento

            alpha = max(alpha, melhor_valor)

        self.tempo_do_ultimo_movimento = time.time() - start

        if self.verboso:
            print(f"\n   Melhor jogada: {melhor_movimento}  (valor={melhor_valor:.3f})")
            print(f"  Nós avaliados : {self.nos_avaliados}")
            print(f"  Podas α-β     : {self.contagem_podas}")
            print(f"  Tempo         : {self.tempo_do_ultimo_movimento:.4f}s")

        return melhor_movimento

    def minimax(self, game: Othello, profundidade: int, # PAROU AQUI !!!!!!!!!!!!!!!!!!!!!!
                 alpha: float, beta: float,
                 maximizando: bool, level: int = 0,
                 movimento_label: str = "") -> float:
        """
        Busca recursiva Min-Max com Poda Alfa-Beta.

        Parâmetros
        ----------
        game       : estado atual do jogo
        profundidade      : profundidade restante
        alpha      : melhor valor garantido para MAX
        beta       : melhor valor garantido para MIN
        maximizando : True se é o turno do agente (MAX)
        level      : nível atual (para verboso)
        movimento_label : rótulo para exibição na árvore verboso
        """
        jogador_atual = self.jogador if maximizando else -self.jogador

        # ----- Caso base: profundidade zero ou estado terminal -----
        if profundidade == 0 or game.verifica_fim():
            self.nos_avaliados += 1
            return self._evaluate(game)

        movimentos_validos = game.movimentos_validos(jogador_atual)

        # ----- Sem jogadas → passa a vez (não troca de nível) ------
        if not movimentos_validos:
            child = game.clonar()
            child.troca_jogador()
            return self.minimax(child, profundidade - 1,
                                 alpha, beta, not maximizando,
                                 level + 1, "PASS")

        indent = "  " * level

        # ----- Nó MAX -----------------------------------------------
        if maximizando:
            max_val = -math.inf
            for movimento in movimentos_validos:
                child = game.clonar()
                child.executa_jogada(movimento[0], movimento[1], jogador_atual)
                child.troca_jogador()

                val = self.minimax(child, profundidade - 1,
                                    alpha, beta, False,
                                    level + 1, str(movimento))

                if self.verboso:
                    print(f"{indent}  [MAX nível={level} movimento={movimento}] val={val:.3f}  α={alpha:.3f}  β={beta:.3f}")

                max_val = max(max_val, val)
                alpha   = max(alpha, max_val)

                # Poda β
                if alpha >= beta:
                    self.contagem_podas += 1
                    if self.verboso:
                        print(f"{indent}  ✂ PODA β em {movimento}: α({alpha:.3f}) ≥ β({beta:.3f})")
                    break

            return max_val

        # ----- Nó MIN -----------------------------------------------
        else:
            min_val = math.inf
            for movimento in movimentos_validos:
                child = game.clonar()
                child.executa_jogada(movimento[0], movimento[1], jogador_atual)
                child.troca_jogador()

                val = self.minimax(child, profundidade - 1,
                                    alpha, beta, True,
                                    level + 1, str(movimento))

                if self.verboso:
                    print(f"{indent}  [MIN nível={level} movimento={movimento}] val={val:.3f}  α={alpha:.3f}  β={beta:.3f}")

                min_val = min(min_val, val)
                beta    = min(beta, min_val)

                # Poda α
                if alpha >= beta:
                    self.contagem_podas += 1
                    if self.verboso:
                        print(f"{indent}  ✂ PODA α em {movimento}: α({alpha:.3f}) ≥ β({beta:.3f})")
                    break

            return min_val

    # ------------------------------------------------------------------
    # Função de avaliação heurística: Avaliar(s) = Σ wi·fi(s)
    # ------------------------------------------------------------------

    def _evaluate(self, game: Othello) -> float:
        """
        Avaliar(s) = w1·f1(s) + w2·f2(s) + w3·f3(s)

        Retorna valor positivo quando o estado é bom para self.jogador.
        """
        tabuleiro = game.tabuleiro
        tamanho  = game.tamanho
        opp   = -self.jogador

        # --- f1: Diferença de peças normalizada ---
        my_pieces  = int((tabuleiro == self.jogador).sum())
        opp_pieces = int((tabuleiro == opp).sum())
        total = my_pieces + opp_pieces
        if total == 0:
            f1 = 0.0
        else:
            f1 = (my_pieces - opp_pieces) / total   # ∈ [-1, 1]

        # --- f2: Mobilidade relativa ---
        my_mob  = len(game.movimentos_validos(self.jogador))
        opp_mob = len(game.movimentos_validos(opp))
        mob_sum = my_mob + opp_mob + 1e-9            # ε evita divisão por zero
        f2 = (my_mob - opp_mob) / mob_sum            # ∈ [-1, 1]

        # --- f3: Controle de cantos e bordas ---
        corners = [(0, 0), (0, tamanho - 1),
                   (tamanho - 1, 0), (tamanho - 1, tamanho - 1)]

        corner_score = 0.0
        for r, c in corners:
            if tabuleiro[r][c] == self.jogador:
                corner_score += 1.0
            elif tabuleiro[r][c] == opp:
                corner_score -= 1.0

        # Bordas (excluindo cantos já contados)
        edge_score = 0.0
        for c in range(1, tamanho - 1):                 # Borda superior
            if tabuleiro[0][c] == self.jogador:   edge_score += 0.3
            elif tabuleiro[0][c] == opp:         edge_score -= 0.3
        for c in range(1, tamanho - 1):                 # Borda inferior
            if tabuleiro[tamanho-1][c] == self.jogador:  edge_score += 0.3
            elif tabuleiro[tamanho-1][c] == opp:        edge_score -= 0.3
        for r in range(1, tamanho - 1):                 # Borda esquerda
            if tabuleiro[r][0] == self.jogador:   edge_score += 0.3
            elif tabuleiro[r][0] == opp:         edge_score -= 0.3
        for r in range(1, tamanho - 1):                 # Borda direita
            if tabuleiro[r][tamanho-1] == self.jogador:  edge_score += 0.3
            elif tabuleiro[r][tamanho-1] == opp:        edge_score -= 0.3

        # Normaliza f3 aproximadamente para [-1, 1]
        max_f3 = 4.0 + 4 * (tamanho - 2) * 0.3
        f3 = (corner_score + edge_score) / max_f3

        # --- Avaliação terminal: punição/prêmio máximo ---
        if game.verifica_fim():
            winner = game.get_vencedor()
            if winner == self.jogador:
                return 1e6
            elif winner == opp:
                return -1e6
            else:
                return 0.0

        return W_DIFF_PECAS * f1 + W_MOBILIDADE * f2 + W_CANTOS_BORDAS * f3
