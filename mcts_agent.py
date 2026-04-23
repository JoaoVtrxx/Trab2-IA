"""
mcts_agent.py
=============
Agente Monte Carlo Tree Search (MCTS) para o jogo Othello.

Os quatro pilares:
    1. SELEÇÃO   — desce na árvore usando UCT até um nó não totalmente expandido
    2. EXPANSÃO  — adiciona um filho aleatório não visitado
    3. SIMULAÇÃO — jogo aleatório (rollout) até o fim
    4. RETROPROP — propaga vitória/derrota de volta à raiz

Fórmula UCT (Upper Confidence Bound for Trees):
    UCT(v) = Q(v)/N(v)  +  C · √( ln(N(pai)) / N(v) )

    Onde:
        Q(v) = total de vitórias acumuladas no nó v
        N(v) = número de visitas ao nó v
        C    = constante de exploração (padrão √2 ≈ 1.414)
                C grande → mais exploração; C pequeno → mais exploração

Critério de parada: número máximo de simulações (default 500) ou
                    tempo máximo em segundos (default 2.0s),
                    o que vier primeiro.
"""

import math
import time
import random
from typing import Optional, List, Tuple

from othello_game import Othello


C_UCT = math.sqrt(2)    # Constante de exploração UCT


# ---------------------------------------------------------------------------
# Nó da árvore MCTS
# ---------------------------------------------------------------------------

class MCTSNode:
    """
    Representa um nó na árvore de busca Monte Carlo.

    Atributos
    ----------
    game    : estado do jogo associado a este nó
    jogador  : jogador que VAI jogar a partir deste nó
    parent  : nó pai (None para a raiz)
    move    : jogada que levou do pai a este nó
    children: filhos expandidos
    untried : jogadas ainda não expandidas
    wins    : total de vitórias (do ponto de vista de quem CRIOU o nó)
    visits  : número de vezes que o nó foi visitado
    """

    def __init__(self, game: Othello, jogador: int,
                 parent: Optional["MCTSNode"] = None,
                 move: Optional[Tuple[int, int]] = None):
        self.game    = game
        self.jogador  = jogador         # Jogador cujo turno é neste nó
        self.parent  = parent
        self.move    = move
        self.children: List["MCTSNode"] = []
        self.untried: List[Tuple[int, int]] = game.movimentos_validoos(jogador)
        random.shuffle(self.untried)  # Aleatoriedade na expansão
        self.wins   = 0.0
        self.visits = 0

    # ------------------------------------------------------------------
    # UCT: Upper Confidence Bound for Trees
    # ------------------------------------------------------------------

    def uct_value(self) -> float:
        """
        UCT(v) = Q(v)/N(v) + C·√(ln(N_pai)/N(v))

        Nós não visitados recebem +∞ (prioridade total para exploração).
        """
        if self.visits == 0:
            return math.inf
        exploitation = self.wins / self.visits
        exploration  = C_UCT * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def best_child(self) -> "MCTSNode":
        """Retorna o filho com maior valor UCT."""
        return max(self.children, key=lambda n: n.uct_value())

    def most_visited_child(self) -> "MCTSNode":
        """Retorna o filho mais visitado (política final de escolha)."""
        return max(self.children, key=lambda n: n.visits)


# ---------------------------------------------------------------------------
# Agente MCTS
# ---------------------------------------------------------------------------

class MCTSAgent:
    """
    Agente baseado em Monte Carlo Tree Search.

    Parâmetros
    ----------
    jogador       : identidade do agente (1 ou -1)
    max_sims     : número máximo de simulações por jogada
    max_time     : tempo máximo em segundos por jogada
    rollout_type : 'random' (aleatório puro) ou 'heuristic' (prefere cantos)
    """

    def __init__(self, jogador: int, max_sims: int = 500,
                 max_time: float = 2.0, rollout_type: str = "heuristic"):
        self.jogador      = jogador
        self.max_sims    = max_sims
        self.max_time    = max_time
        self.rollout_type = rollout_type

        # Estatísticas
        self.sims_done       = 0
        self.last_move_time  = 0.0

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def choose_move(self, game: Othello) -> Optional[Tuple[int, int]]:
        """
        Executa as simulações MCTS e retorna a melhor jogada encontrada.
        Retorna None se não houver jogadas (passa a vez).
        """
        valido_moves = game.movimentos_validoos(self.jogador)
        if not valido_moves:
            return None

        start = time.time()
        self.sims_done = 0

        # Cria raiz da árvore com o estado atual
        root = MCTSNode(game.clone(), self.jogador)

        # --- Loop principal de simulação ---
        while (self.sims_done < self.max_sims and
               time.time() - start < self.max_time):

            # 1. SELEÇÃO
            node = self._select(root)

            # 2. EXPANSÃO
            if not node.game.is_terminal() and not node.is_fully_expanded():
                node = self._expand(node)

            # 3. SIMULAÇÃO (Rollout)
            result = self._simulate(node)

            # 4. RETROPROPAGAÇÃO
            self._backpropagate(node, result)

            self.sims_done += 1

        self.last_move_time = time.time() - start

        # Escolhe o filho mais visitado (mais robusto que maior UCT)
        best = root.most_visited_child()
        return best.move

    # ------------------------------------------------------------------
    # Pilar 1: SELEÇÃO (desce pela árvore via UCT)
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Desce na árvore seguindo UCT até encontrar:
        - Um nó terminal, OU
        - Um nó não totalmente expandido.
        """
        while not node.game.is_terminal():
            if not node.is_fully_expanded():
                return node          # Encontrou nó para expandir
            node = node.best_child() # Desce pelo filho com maior UCT
        return node

    # ------------------------------------------------------------------
    # Pilar 2: EXPANSÃO (adiciona novo nó filho)
    # ------------------------------------------------------------------

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expande um nó não totalmente expandido:
        pega uma jogada não tentada e cria o filho correspondente.
        """
        move = node.untried.pop()     # Remove uma jogada não tentada
        child_game = node.game.clone()
        child_game.executa_jogada(move[0], move[1], node.jogador)
        next_jogador = -node.jogador

        # Se o próximo jogador não tem movimentos, o turno volta
        if not child_game.movimentos_validoos(next_jogador):
            next_jogador = node.jogador

        child_game.jogador_atual = next_jogador
        child = MCTSNode(child_game, next_jogador, parent=node, move=move)
        node.children.append(child)
        return child

    # ------------------------------------------------------------------
    # Pilar 3: SIMULAÇÃO / ROLLOUT
    # ------------------------------------------------------------------

    def _simulate(self, node: MCTSNode) -> int:
        """
        Joga uma partida completa aleatória (ou heurística) a partir
        do estado do nó. Retorna:
            +1 se self.jogador venceu
            -1 se o adversário venceu
             0 se empate
        """
        sim_game   = node.game.clone()
        sim_jogador = node.jogador

        while not sim_game.is_terminal():
            moves = sim_game.movimentos_validoos(sim_jogador)

            if not moves:
                # Passa a vez
                sim_jogador = -sim_jogador
                moves = sim_game.movimentos_validoos(sim_jogador)
                if not moves:
                    break           # Ambos sem jogadas → fim

            move = self._rollout_policy(sim_game, moves, sim_jogador)
            sim_game.executa_jogada(move[0], move[1], sim_jogador)
            sim_jogador = -sim_jogador

        winner = sim_game.get_winner()
        if winner == self.jogador:
            return 1
        elif winner == -self.jogador:
            return -1
        else:
            return 0

    def _rollout_policy(self, game: Othello,
                        moves: List[Tuple[int, int]],
                        jogador: int) -> Tuple[int, int]:
        """
        Política de rollout:
            'random'    → escolha uniforme aleatória
            'heuristic' → prefere cantos > bordas > outros
        """
        if self.rollout_type == "random":
            return random.choice(moves)

        # --- Heurística simples: prioriza cantos e bordas ---
        tamanho = game.tamanho
        corners = {(0, 0), (0, tamanho-1), (tamanho-1, 0), (tamanho-1, tamanho-1)}
        edges   = {(r, c) for r in range(tamanho) for c in range(tamanho)
                   if r == 0 or r == tamanho-1 or c == 0 or c == tamanho-1}

        corner_moves = [m for m in moves if m in corners]
        if corner_moves:
            return random.choice(corner_moves)

        edge_moves = [m for m in moves if m in edges]
        if edge_moves:
            return random.choice(edge_moves)

        return random.choice(moves)

    # ------------------------------------------------------------------
    # Pilar 4: RETROPROPAGAÇÃO
    # ------------------------------------------------------------------

    def _backpropagate(self, node: MCTSNode, result: int):
        """
        Propaga o resultado da simulação de volta à raiz.

        result = +1 (vitória de self.jogador), -1 (derrota), 0 (empate).
        Cada nó acumula vitórias do ponto de vista do jogador que
        FEZ A JOGADA para chegar nele (= -node.jogador).
        """
        while node is not None:
            node.visits += 1
            # O jogador que jogou para chegar aqui é o PAI do nó
            # Portanto, avaliamos do ponto de vista de -node.jogador
            if result == 0:
                node.wins += 0.5    # Empate vale meio ponto
            elif result == 1 and self.jogador != node.jogador:
                node.wins += 1.0    # Vitória do agente
            elif result == -1 and self.jogador == node.jogador:
                node.wins += 1.0    # Também conta perda do oponente
            node = node.parent

    # ------------------------------------------------------------------
    # Estatísticas
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "sims_done"     : self.sims_done,
            "last_move_time": self.last_move_time,
        }
