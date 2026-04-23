"""
minmax_agent.py
===============
Agente Min-Max com Poda Alfa-Beta para o jogo Othello.

Função de avaliação heurística:
    Avaliar(s) = w1·f1(s) + w2·f2(s) + w3·f3(s)

Onde:
    f1(s) = Diferença de peças normalizada
            f1 = (minhas - adversárias) / (minhas + adversárias)
            Peso w1 = 1.0  → métrica base, sempre relevante

    f2(s) = Mobilidade relativa (número de jogadas disponíveis)
            f2 = (mob_minha - mob_adv) / (mob_minha + mob_adv + ε)
            Peso w2 = 3.0  → mobilidade é crítica no meio-jogo;
                             ter mais opções evita posições forçadas

    f3(s) = Controle de cantos | Controle de bordas
            cantos: peso altíssimo (capturam definitivamente)
            bordas: peso médio (estáveis, difíceis de reverter)
            Peso w3 = 5.0  → cantos são irreversíveis e dominantes;
                             literatura de Othello confirma superioridade

Justificativa dos pesos:
    A hierarquia w3 > w2 > w1 reflete a teoria clássica do Othello:
    (1) Cantos e bordas determinam o resultado em ~80% dos casos.
    (2) Mobilidade controla o ritmo do jogo.
    (3) Diferença bruta de peças é enganosa no início (sacrifícios são táticos).
"""

import math
import time
from typing import Tuple, Optional, List

from othello_game import Othello


# ---------------------------------------------------------------------------
# Pesos da função de avaliação (ver docstring do módulo)
# ---------------------------------------------------------------------------
W_PIECE_DIFF  = 1.0   # w1
W_MOBILITY    = 3.0   # w2
W_CORNER_EDGE = 5.0   # w3


class MinMaxAgent:
    """
    Agente baseado em Min-Max com Poda Alfa-Beta.

    Parâmetros
    ----------
    jogador : int
        Identidade do agente (1 ou -1).
    depth : int
        Profundidade máxima de busca (3 a 5 recomendado).
    verbose : bool
        Se True, imprime a árvore de decisão dos primeiros 2 níveis.
    """

    def __init__(self, jogador: int, depth: int = 4, verbose: bool = False):
        assert 3 <= depth <= 5, "Profundidade deve estar entre 3 e 5"
        self.jogador  = jogador
        self.depth   = depth
        self.verbose = verbose

        # Estatísticas internas
        self.nodes_evaluated = 0
        self.prune_count     = 0
        self.last_move_time  = 0.0

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def choose_move(self, game: Othello) -> Optional[Tuple[int, int]]:
        """
        Escolhe a melhor jogada para o estado atual do jogo.
        Retorna None se não houver jogadas disponíveis (passa a vez).
        """
        start = time.time()
        self.nodes_evaluated = 0
        self.prune_count     = 0

        valido_moves = game.movimentos_validoos(self.jogador)
        if not valido_moves:
            return None

        best_move  = None
        best_value = -math.inf
        alpha = -math.inf
        beta  =  math.inf

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  MinMax (profundidade={self.depth}) — Jogador {self.jogador}")
            print(f"  Jogadas válidas: {valido_moves}")
            print(f"{'='*60}")

        for move in valido_moves:
            child = game.clone()
            child.executa_jogada(move[0], move[1], self.jogador)
            child.switch_jogador()

            value = self._minimax(child, self.depth - 1,
                                  alpha, beta,
                                  maximizing=False,
                                  level=1, move_label=str(move))

            if self.verbose:
                print(f"  [Raiz → {move}] valor propagado = {value:.3f}")

            if value > best_value:
                best_value = value
                best_move  = move

            alpha = max(alpha, best_value)

        self.last_move_time = time.time() - start

        if self.verbose:
            print(f"\n  ✓ Melhor jogada: {best_move}  (valor={best_value:.3f})")
            print(f"  Nós avaliados : {self.nodes_evaluated}")
            print(f"  Podas α-β     : {self.prune_count}")
            print(f"  Tempo         : {self.last_move_time:.4f}s")

        return best_move

    # ------------------------------------------------------------------
    # Algoritmo Min-Max com Poda Alfa-Beta
    # ------------------------------------------------------------------

    def _minimax(self, game: Othello, depth: int,
                 alpha: float, beta: float,
                 maximizing: bool, level: int = 0,
                 move_label: str = "") -> float:
        """
        Busca recursiva Min-Max com Poda Alfa-Beta.

        Parâmetros
        ----------
        game       : estado atual do jogo
        depth      : profundidade restante
        alpha      : melhor valor garantido para MAX
        beta       : melhor valor garantido para MIN
        maximizing : True se é o turno do agente (MAX)
        level      : nível atual (para verbose)
        move_label : rótulo para exibição na árvore verbose
        """
        jogador_atual = self.jogador if maximizing else -self.jogador

        # ----- Caso base: profundidade zero ou estado terminal -----
        if depth == 0 or game.is_terminal():
            self.nodes_evaluated += 1
            return self._evaluate(game)

        valido_moves = game.movimentos_validoos(jogador_atual)

        # ----- Sem jogadas → passa a vez (não troca de nível) ------
        if not valido_moves:
            child = game.clone()
            child.switch_jogador()
            return self._minimax(child, depth - 1,
                                 alpha, beta, not maximizing,
                                 level + 1, "PASS")

        indent = "  " * level

        # ----- Nó MAX -----------------------------------------------
        if maximizing:
            max_val = -math.inf
            for move in valido_moves:
                child = game.clone()
                child.executa_jogada(move[0], move[1], jogador_atual)
                child.switch_jogador()

                val = self._minimax(child, depth - 1,
                                    alpha, beta, False,
                                    level + 1, str(move))

                if self.verbose:
                    print(f"{indent}  [MAX nível={level} move={move}] val={val:.3f}  α={alpha:.3f}  β={beta:.3f}")

                max_val = max(max_val, val)
                alpha   = max(alpha, max_val)

                # Poda β
                if alpha >= beta:
                    self.prune_count += 1
                    if self.verbose:
                        print(f"{indent}  ✂ PODA β em {move}: α({alpha:.3f}) ≥ β({beta:.3f})")
                    break

            return max_val

        # ----- Nó MIN -----------------------------------------------
        else:
            min_val = math.inf
            for move in valido_moves:
                child = game.clone()
                child.executa_jogada(move[0], move[1], jogador_atual)
                child.switch_jogador()

                val = self._minimax(child, depth - 1,
                                    alpha, beta, True,
                                    level + 1, str(move))

                if self.verbose:
                    print(f"{indent}  [MIN nível={level} move={move}] val={val:.3f}  α={alpha:.3f}  β={beta:.3f}")

                min_val = min(min_val, val)
                beta    = min(beta, min_val)

                # Poda α
                if alpha >= beta:
                    self.prune_count += 1
                    if self.verbose:
                        print(f"{indent}  ✂ PODA α em {move}: α({alpha:.3f}) ≥ β({beta:.3f})")
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
        my_mob  = len(game.movimentos_validoos(self.jogador))
        opp_mob = len(game.movimentos_validoos(opp))
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
        if game.is_terminal():
            winner = game.get_winner()
            if winner == self.jogador:
                return 1e6
            elif winner == opp:
                return -1e6
            else:
                return 0.0

        return W_PIECE_DIFF * f1 + W_MOBILITY * f2 + W_CORNER_EDGE * f3
