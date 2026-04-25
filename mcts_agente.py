# Agente Monte Carlo Tree Search (MCTS) para o jogo Othello.

# Os quatro pilares:
#     1. SELEÇÃO   - desce na árvore usando UCT até um nó não totalmente expandido
#     2. EXPANSÃO  - adiciona um filho aleatório não visitado
#     3. SIMULAÇÃO - jogo aleatório (rollout) até o fim
#     4. RETROPROP - propaga vitória/derrota de volta à raiz

# Fórmula UCT (Upper Confidence Bound for Trees):
#     UCT(v) = Q(v)/N(v)  +  C · √( ln(N(pai)) / N(v) )

#     Onde:
#         Q(v) = total de vitórias acumuladas no nó v
#         N(v) = número de visitas ao nó v
#         C    = constante de exploração (padrão √2 =~ 1.414)
#                 C maior -> mais exploração; C menor -> mais aproveitamento (exploitation)

# Critério de parada: número máximo de simulações (default 500) ou
#                     tempo máximo em segundos (default 2.0s),
#                     o que vier primeiro.

import math
import time
import random
from typing import Optional, List, Tuple

from othello_jogo import Othello

C_UCT = math.sqrt(2)    # Constante de exploração UCT

# Nó da árvore MCTS

class MCTSNode:
    # Representa um nó na árvore de busca Monte Carlo.

    # Atributos
    # ----------
    # jogo    : estado do jogo associado a este nó
    # jogador  : jogador que VAI jogar a partir deste nó
    # pai  : nó pai (None para a raiz)
    # movimento    : jogada que levou do pai a este nó
    # filhos: filhos expandidos
    # nao_expandidos : jogadas ainda não expandidas
    # vitorias    : total de vitórias (do ponto de vista de quem CRIOU o nó)
    # visitas  : número de vezes que o nó foi visitado

    def __init__(self, jogo: Othello, jogador: int,
                 pai: Optional["MCTSNode"] = None,
                 movimento: Optional[Tuple[int, int]] = None):
        self.jogo    = jogo
        self.jogador  = jogador         # Jogador cujo turno é neste nó
        self.pai  = pai
        self.movimento    = movimento
        self.filhos: List["MCTSNode"] = []
        self.nao_expandidos: List[Tuple[int, int]] = jogo.movimentos_validos(jogador)
        random.shuffle(self.nao_expandidos)  # Aleatoriedade na expansão
        self.vitorias   = 0.0
        self.visitas = 0

    # UCT: Upper Confidence Bound for Trees

    def uct_valor(self) -> float:
        # UCT(v) = Q(v)/N(v) + C·√(ln(N_pai)/N(v))

        # Nós não visitados recebem +inf (prioridade total para exploração).

        if self.visitas == 0:
            return math.inf
        exploitation = self.vitorias / self.visitas
        exploration  = C_UCT * math.sqrt(math.log(self.pai.visitas) / self.visitas)
        return exploitation + exploration

    def verifica_totalmente_expandido(self) -> bool:
        return len(self.nao_expandidos) == 0

    def verifica_folha(self) -> bool:
        return len(self.filhos) == 0

    def melhor_filho(self) -> "MCTSNode":
        # Retorna o filho com maior valor UCT.
        return max(self.filhos, key=lambda n: n.uct_valor())

    def filho_mais_visitado(self) -> "MCTSNode":
        # Retorna o filho mais visitado (política final de escolha).
        return max(self.filhos, key=lambda n: n.visitas)

# Agente MCTS

class MCTSAgente:
    # Agente baseado em Monte Carlo Tree Search.

    # Parâmetros
    # ----------
    # jogador       : identidade do agente (1 ou -1)
    # maximo_simulacoes     : número máximo de simulações por jogada
    # maximo_tempo_segundos     : tempo máximo em segundos por jogada
    # tipo_rollout : 'random' ou 'heuristica_cantos' (prefere cantos)

    def __init__(self, jogador: int, maximo_simulacoes: int = 500,
                 maximo_tempo_segundos: float = 2.0, tipo_rollout: str = "heuristica_cantos"):
        self.jogador      = jogador
        self.maximo_simulacoes    = maximo_simulacoes
        self.maximo_tempo_segundos    = maximo_tempo_segundos
        self.tipo_rollout = tipo_rollout

        # Estatísticas
        self.quantidade_simulacoes       = 0
        self.tempo_do_ultimo_movimento  = 0.0

    def escolhe_movimento(self, jogo: Othello) -> Optional[Tuple[int, int]]:
        # Executa as simulações MCTS e retorna a melhor jogada encontrada.
        # Retorna None se não houver jogadas (passa a vez).

        movimentos_validos = jogo.movimentos_validos(self.jogador)
        if not movimentos_validos:
            return None

        start = time.time()
        self.quantidade_simulacoes = 0

        # Cria raiz da árvore com o estado atual
        raiz = MCTSNode(jogo.clonar(), self.jogador)

        # Loop principal de simulação
        while (self.quantidade_simulacoes < self.maximo_simulacoes and
               time.time() - start < self.maximo_tempo_segundos):

            # 1. SELEÇÃO
            node = self.seleciona(raiz)

            # 2. EXPANSÃO
            if not node.jogo.verifica_fim() and not node.verifica_totalmente_expandido():
                node = self.expande(node)

            # 3. SIMULAÇÃO (Rollout)
            result = self.simula(node)

            # 4. RETROPROPAGAÇÃO
            self.retropropaga(node, result)

            self.quantidade_simulacoes += 1

        self.tempo_do_ultimo_movimento = time.time() - start

        # Escolhe o filho mais visitado (mais robusto que maior UCT)
        best = raiz.filho_mais_visitado()
        return best.movimento

    # Pilar 1: SELEÇÃO (desce pela árvore via UCT)

    def seleciona(self, node: MCTSNode) -> MCTSNode:
        # Desce na árvore seguindo UCT até encontrar:
        # - Um nó terminal, OU
        # - Um nó não totalmente expandido.

        while not node.jogo.verifica_fim():
            if not node.verifica_totalmente_expandido():
                return node          # Encontrou nó para expandir
            node = node.melhor_filho() # Desce pelo filho com maior UCT
        return node

    # Pilar 2: EXPANSÃO (adiciona novo nó filho)

    def expande(self, node: MCTSNode) -> MCTSNode:
        # Expande um nó não totalmente expandido:
        # pega uma jogada não tentada e cria o filho correspondente.

        movimento = node.nao_expandidos.pop()     # Remove uma jogada não tentada
        filho_jogo = node.jogo.clonar()
        filho_jogo.executa_jogada(movimento[0], movimento[1], node.jogador)
        proximo_jogador = -node.jogador

        # Se o próximo jogador não tem movimentos, o turno volta
        if not filho_jogo.movimentos_validos(proximo_jogador):
            proximo_jogador = node.jogador

        filho_jogo.jogador_atual = proximo_jogador
        filho = MCTSNode(filho_jogo, proximo_jogador, pai=node, movimento=movimento)
        node.filhos.append(filho)
        return filho

    # Pilar 3: SIMULAÇÃO / ROLLOUT

    def simula(self, node: MCTSNode) -> int:
        # Joga uma partida completa aleatória (ou heurística) a partir
        # do estado do nó. Retorna:
        #     +1 se self.jogador venceu
        #     -1 se o adversário venceu
        #      0 se empate

        sim_jogo   = node.jogo.clonar()
        sim_jogador = node.jogador

        while not sim_jogo.verifica_fim():
            movimentos = sim_jogo.movimentos_validos(sim_jogador)

            if not movimentos:
                # Passa a vez
                sim_jogador = -sim_jogador
                movimentos = sim_jogo.movimentos_validos(sim_jogador)
                if not movimentos:
                    break           # Ambos sem jogadas -> fim

            movimento = self.politica_rollout(sim_jogo, movimentos, sim_jogador)
            sim_jogo.executa_jogada(movimento[0], movimento[1], sim_jogador)
            sim_jogador = -sim_jogador

        vencedor = sim_jogo.get_vencedor()
        if vencedor == self.jogador:
            return 1
        elif vencedor == -self.jogador:
            return -1
        else:
            return 0

    def politica_rollout(self, jogo: Othello,
                        movimentos: List[Tuple[int, int]],
                        jogador: int) -> Tuple[int, int]:
        # Política de rollout:
        #     'random'    -> escolha uniforme aleatória
        #     'heuristica_cantos' -> prefere cantos > bordas > outros

        if self.tipo_rollout == "random":
            return random.choice(movimentos)

        # Heurística simples: prioriza cantos e bordas
        tamanho = jogo.tamanho
        cantos = {(0, 0), (0, tamanho-1), (tamanho-1, 0), (tamanho-1, tamanho-1)}
        bordas   = {(r, c) for r in range(tamanho) for c in range(tamanho)
                   if r == 0 or r == tamanho-1 or c == 0 or c == tamanho-1}

        movimentos_cantos = [m for m in movimentos if m in cantos]
        if movimentos_cantos:
            return random.choice(movimentos_cantos)

        movimentos_bordas = [m for m in movimentos if m in bordas]
        if movimentos_bordas:
            return random.choice(movimentos_bordas)

        return random.choice(movimentos)

    # Pilar 4: RETROPROPAGAÇÃO

    def retropropaga(self, node: MCTSNode, result: int):
        # Propaga o resultado da simulação de volta à raiz.

        # result = +1 (vitória de self.jogador), -1 (derrota), 0 (empate).
        # Cada nó acumula vitórias do ponto de vista do jogador que
        # FEZ A JOGADA para chegar nele (= -node.jogador).

        while node is not None:
            node.visitas += 1
            # O jogador que jogou para chegar aqui é o PAI do nó
            # Portanto, avaliamos do ponto de vista de -node.jogador
            if result == 0:
                node.vitorias += 0.5    # Empate vale meio ponto
            elif result == 1 and self.jogador != node.jogador:
                node.vitorias += 1.0    # Vitória do agente
            elif result == -1 and self.jogador == node.jogador:
                node.vitorias += 1.0    # Também conta perda do oponente
            node = node.pai

    # Estatísticas

    def stats(self) -> dict:
        return {
            "quantidade_simulacoes"     : self.quantidade_simulacoes,
            "tempo_do_ultimo_movimento": self.tempo_do_ultimo_movimento,
        }
