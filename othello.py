import copy
import math
import random
import time
from collections import defaultdict

# Constantes para representar os jogadores
P = 1  # Preto
B = 2  # Branco
VAZIO = 0

class OthelloGame:
    def __init__(self, n=6, regras_avancadas=False):
        """
        Inicializa o jogo Othello.
        :param n: Tamanho do tabuleiro (NxN).
        :param regras_avancadas: Se True, usa pesos diferentes para avaliar ganhadores (cantos valem mais).
        """
        self.n = n
        self.regras_avancadas = regras_avancadas
        self.tabuleiro = self.criar_tabuleiro(n)
        self.jogador_atual = P

    def criar_tabuleiro(self, n):
        tab = [[VAZIO for _ in range(n)] for _ in range(n)]
        meio = n // 2
        tab[meio - 1][meio - 1] = P
        tab[meio - 1][meio] = B
        tab[meio][meio - 1] = B
        tab[meio][meio] = P
        return tab

    def imprimir_tabuleiro(self, tab=None):
        if tab is None:
            tab = self.tabuleiro
        print("  " + " ".join(str(i) for i in range(self.n)))
        for i in range(self.n):
            linha_str = f"{i} "
            for j in range(self.n):
                if tab[i][j] == VAZIO:
                    linha_str += ". "
                elif tab[i][j] == P:
                    linha_str += "P "
                else:
                    linha_str += "B "
            print(linha_str)
        print()

    def jogada_valida(self, tab, linha, col, jogador):
        if tab[linha][col] != VAZIO:
            return False
        
        oponente = B if jogador == P else P
        direcoes = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),          (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]
        
        for dr, dc in direcoes:
            r, c = linha + dr, col + dc
            encontrou_oponente = False
            
            while 0 <= r < self.n and 0 <= c < self.n:
                if tab[r][c] == VAZIO:
                    break
                elif tab[r][c] == oponente:
                    encontrou_oponente = True
                elif tab[r][c] == jogador:
                    if encontrou_oponente:
                        return True
                    else:
                        break
                r += dr
                c += dc
        return False

    def jogadas_validas(self, tab, jogador):
        return [(i, j) for i in range(self.n) for j in range(self.n) if self.jogada_valida(tab, i, j, jogador)]

    def aplicar_jogada(self, tab, linha, col, jogador):
        novo_tab = copy.deepcopy(tab)
        novo_tab[linha][col] = jogador
        oponente = B if jogador == P else P
        direcoes = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),          (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]
        
        for dr, dc in direcoes:
            r, c = linha + dr, col + dc
            celulas_virar = []
            
            while 0 <= r < self.n and 0 <= c < self.n:
                if novo_tab[r][c] == VAZIO:
                    break
                elif novo_tab[r][c] == oponente:
                    celulas_virar.append((r, c))
                elif novo_tab[r][c] == jogador:
                    for pr, pc in celulas_virar:
                        novo_tab[pr][pc] = jogador
                    break
                r += dr
                c += dc
        return novo_tab

    def verificar_fim(self, tab):
        return not self.jogadas_validas(tab, P) and not self.jogadas_validas(tab, B)

    def obter_pontuacao(self, tab):
        if not self.regras_avancadas:
            pontos = {P: 0, B: 0}
            for i in range(self.n):
                for j in range(self.n):
                    if tab[i][j] != VAZIO:
                        pontos[tab[i][j]] += 1
            return pontos
        else:
            # Regra customizada: Cantos valem 5, bordas valem 2, outras 1
            pontos = {P: 0, B: 0}
            for i in range(self.n):
                for j in range(self.n):
                    peca = tab[i][j]
                    if peca != VAZIO:
                        peso = 1
                        if (i == 0 or i == self.n - 1) and (j == 0 or j == self.n - 1):
                            peso = 5
                        elif i == 0 or i == self.n - 1 or j == 0 or j == self.n - 1:
                            peso = 2
                        pontos[peca] += peso
            return pontos
            
    def jogar(self, jogada):
        """Aplica a jogada e passa o turno."""
        self.tabuleiro = self.aplicar_jogada(self.tabuleiro, jogada[0], jogada[1], self.jogador_atual)
        oponente = B if self.jogador_atual == P else P
        # Troca de turno se o oponente tiver jogadas, senão mantém
        if self.jogadas_validas(self.tabuleiro, oponente):
            self.jogador_atual = oponente


class MinMaxAgent:
    def __init__(self, jogador, profundidade=3):
        self.jogador = jogador
        self.oponente = B if jogador == P else P
        self.profundidade = profundidade

    def avaliar(self, tab, game):
        """
        Função de Avaliação Heurística.
        f(s) = w1 * Peças + w2 * Mobilidade + w3 * Cantos
        Pesos matemáticos:
        - Os cantos fornecem maior estabilidade (w3 = 50)
        - A mobilidade (opções de jogada) controla o fluxo do jogo (w2 = 10)
        - A diferença crua de peças é a menos importante até o final do jogo (w1 = 1)
        """
        w_pecas = 1
        w_mob = 10
        w_cantos = 50

        # Diferença de peças
        minhas_pecas = sum(1 for i in range(game.n) for j in range(game.n) if tab[i][j] == self.jogador)
        oponente_pecas = sum(1 for i in range(game.n) for j in range(game.n) if tab[i][j] == self.oponente)
        dif_pecas = minhas_pecas - oponente_pecas

        # Mobilidade
        minha_mob = len(game.jogadas_validas(tab, self.jogador))
        oponente_mob = len(game.jogadas_validas(tab, self.oponente))
        dif_mob = minha_mob - oponente_mob

        # Cantos
        cantos_coords = [(0, 0), (0, game.n - 1), (game.n - 1, 0), (game.n - 1, game.n - 1)]
        minha_canto = sum(1 for r, c in cantos_coords if tab[r][c] == self.jogador)
        oponente_canto = sum(1 for r, c in cantos_coords if tab[r][c] == self.oponente)
        dif_cantos = minha_canto - oponente_canto

        return (w_pecas * dif_pecas) + (w_mob * dif_mob) + (w_cantos * dif_cantos)

    def minmax(self, tab, profundidade, alpha, beta, maximizando, game):
        if profundidade == 0 or game.verificar_fim(tab):
            return self.avaliar(tab, game), None

        jogador_turno = self.jogador if maximizando else self.oponente
        jogadas = game.jogadas_validas(tab, jogador_turno)

        if not jogadas:
            return self.minmax(tab, profundidade - 1, alpha, beta, not maximizando, game)[0], None

        melhor_jogada = None
        if maximizando:
            max_eval = -float('inf')
            for jogada in jogadas:
                novo_tab = game.aplicar_jogada(tab, jogada[0], jogada[1], jogador_turno)
                eval_val, _ = self.minmax(novo_tab, profundidade - 1, alpha, beta, False, game)
                if eval_val > max_eval:
                    max_eval = eval_val
                    melhor_jogada = jogada
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            return max_eval, melhor_jogada
        else:
            min_eval = float('inf')
            for jogada in jogadas:
                novo_tab = game.aplicar_jogada(tab, jogada[0], jogada[1], jogador_turno)
                eval_val, _ = self.minmax(novo_tab, profundidade - 1, alpha, beta, True, game)
                if eval_val < min_eval:
                    min_eval = eval_val
                    melhor_jogada = jogada
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            return min_eval, melhor_jogada

    def escolher_jogada(self, game):
        _, jogada = self.minmax(game.tabuleiro, self.profundidade, -float('inf'), float('inf'), True, game)
        return jogada


class Node:
    def __init__(self, state, jogador_turno, parent=None, jogada=None):
        self.state = state
        self.jogador_turno = jogador_turno
        self.parent = parent
        self.jogada = jogada
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = None

    def uct(self, exploration_weight=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)


class MCTSAgent:
    def __init__(self, jogador, num_simulacoes=500, tempo_limite=None):
        self.jogador = jogador
        self.oponente = B if jogador == P else P
        self.num_simulacoes = num_simulacoes
        self.tempo_limite = tempo_limite

    def obter_untried_moves(self, node, game):
        if node.untried_moves is None:
            node.untried_moves = game.jogadas_validas(node.state, node.jogador_turno)
        return node.untried_moves

    def escolher_jogada(self, game):
        root = Node(game.tabuleiro, self.jogador)
        start_time = time.time()
        sims = 0

        while True:
            if self.tempo_limite and (time.time() - start_time) > self.tempo_limite:
                break
            if not self.tempo_limite and sims >= self.num_simulacoes:
                break

            # 1. Selection
            node = root
            while not self.obter_untried_moves(node, game) and node.children:
                node = max(node.children, key=lambda c: c.uct())

            # 2. Expansion
            untried_moves = self.obter_untried_moves(node, game)
            if untried_moves:
                jogada = random.choice(untried_moves)
                novo_estado = game.aplicar_jogada(node.state, jogada[0], jogada[1], node.jogador_turno)
                prox_jogador = self.oponente if node.jogador_turno == self.jogador else self.jogador
                if not game.jogadas_validas(novo_estado, prox_jogador):
                    prox_jogador = node.jogador_turno  # Passa a vez
                
                child = Node(novo_estado, prox_jogador, parent=node, jogada=jogada)
                node.untried_moves.remove(jogada)
                node.children.append(child)
                node = child

            # 3. Simulation (Rollout)
            estado_atual = node.state
            jogador_atual_sim = node.jogador_turno
            while not game.verificar_fim(estado_atual):
                moves = game.jogadas_validas(estado_atual, jogador_atual_sim)
                if moves:
                    move = random.choice(moves)
                    estado_atual = game.aplicar_jogada(estado_atual, move[0], move[1], jogador_atual_sim)
                
                prox = B if jogador_atual_sim == P else P
                if game.jogadas_validas(estado_atual, prox):
                    jogador_atual_sim = prox

            # 4. Backpropagation
            pontos = game.obter_pontuacao(estado_atual)
            vencedor = P if pontos[P] > pontos[B] else B if pontos[B] > pontos[P] else None
            
            temp_node = node
            while temp_node is not None:
                temp_node.visits += 1
                # O nó armazena o jogador que VAI jogar, a jogada que levou a ele foi do parent.
                # Então, o jogador que fez a jogada foi o contrário de temp_node.jogador_turno.
                if temp_node.parent is not None:
                    jogador_da_jogada = B if temp_node.jogador_turno == P else P
                    if vencedor == jogador_da_jogada:
                        temp_node.wins += 1
                    elif vencedor is None:
                        temp_node.wins += 0.5
                temp_node = temp_node.parent

            sims += 1

        if not root.children:
            return None
        return max(root.children, key=lambda c: c.visits).jogada


def torneio(partidas=10, regras_avancadas=False, n=6):
    agente1_vitorias = 0
    agente2_vitorias = 0
    empates = 0
    tempos_agente1 = []
    tempos_agente2 = []

    print(f"Iniciando torneio com {partidas} partidas no tabuleiro {n}x{n}. Regras Avançadas: {regras_avancadas}")
    
    for i in range(partidas):
        game = OthelloGame(n=n, regras_avancadas=regras_avancadas)
        # O agente que joga com Preto (1) ou Branco (2) será alternado
        p_agent_is_minmax = (i % 2 == 0)
        
        jogador_p = MinMaxAgent(P, profundidade=4) if p_agent_is_minmax else MCTSAgent(P, num_simulacoes=200)
        jogador_b = MCTSAgent(B, num_simulacoes=200) if p_agent_is_minmax else MinMaxAgent(B, profundidade=4)

        while not game.verificar_fim(game.tabuleiro):
            start = time.time()
            if game.jogador_atual == P:
                jogada = jogador_p.escolher_jogada(game)
                tempos_agente1.append(time.time() - start) if p_agent_is_minmax else tempos_agente2.append(time.time() - start)
            else:
                jogada = jogador_b.escolher_jogada(game)
                tempos_agente1.append(time.time() - start) if not p_agent_is_minmax else tempos_agente2.append(time.time() - start)
            
            if jogada:
                game.jogar(jogada)
            else:
                # Passa a vez
                game.jogador_atual = B if game.jogador_atual == P else P

        pontos = game.obter_pontuacao(game.tabuleiro)
        vencedor = P if pontos[P] > pontos[B] else B if pontos[B] > pontos[P] else None
        
        if vencedor == P:
            if p_agent_is_minmax: agente1_vitorias += 1
            else: agente2_vitorias += 1
        elif vencedor == B:
            if p_agent_is_minmax: agente2_vitorias += 1
            else: agente1_vitorias += 1
        else:
            empates += 1
            
        print(f"Partida {i+1}/{partidas} finalizada. Placar: P={pontos[P]} B={pontos[B]}")

    print("\n--- Resultados do Torneio ---")
    print(f"Vitórias Min-Max: {agente1_vitorias} ({(agente1_vitorias/partidas)*100:.1f}%)")
    print(f"Vitórias MCTS: {agente2_vitorias} ({(agente2_vitorias/partidas)*100:.1f}%)")
    print(f"Empates: {empates}")
    print(f"Tempo médio por jogada Min-Max: {sum(tempos_agente1)/len(tempos_agente1) if tempos_agente1 else 0:.4f}s")
    print(f"Tempo médio por jogada MCTS: {sum(tempos_agente2)/len(tempos_agente2) if tempos_agente2 else 0:.4f}s")

if __name__ == "__main__":
    # Roda o script de torneio para análise experimental
    torneio(partidas=4, regras_avancadas=False, n=6)
