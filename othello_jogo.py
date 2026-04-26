# Motor do jogo Othello (Reversi) para tabuleiro NxN.

# Representação do tabuleiro:
#    0  = célula vazia
#   +1  = peça do jogador 1 (preto)
#   -1  = peça do jogador 2 (branco)

import numpy as np
from typing import List, Tuple, Optional

# Direções para verificar capturas (8 direções)
DIRECOES = [(-1, -1), (-1, 0), (-1, 1),
              (0,  -1),           (0,  1),
              (1,  -1),  (1,  0), (1,  1)]


class Othello:
#
#    tamanho : int
#        Tamanho do tabuleiro (tamanho x tamanho). Default = 6.
#    bonus_canto : bool
#        Se True, cantos valem 3x na contagem final.
#        Extensões obrigatórias: (1) Tabuleiro dinâmico (2) Pontuação diferenciada.

    def __init__(self, tamanho: int = 6, bonus_canto: bool = False):
        self.tamanho = tamanho
        self.bonus_canto = bonus_canto
        self.tabuleiro = np.zeros((tamanho, tamanho), dtype=int)
        self._inicializa_tabuleiro()
        self.jogador_atual = 1          # Jogador 1 (preto) começa
        self.contagem_passes = 0              # Contagem de passes consecutivos
        self.historico: List[np.ndarray] = []

    def _inicializa_tabuleiro(self):
        # Inicialização do tabuleiro com peças iniciais (no centro)
        c = self.tamanho // 2
        self.tabuleiro[c - 1][c - 1] =  1
        self.tabuleiro[c - 1][c]     = -1
        self.tabuleiro[c][c - 1]     = -1
        self.tabuleiro[c][c]         =  1

    def movimentos_validos(self, jogador: int) -> List[Tuple[int, int]]:
        # Geração de jogadas válidas
        
        # Retorna lista de (linha, col) onde 'jogador' pode jogar.
        
        # Uma posição é válida se estiver vazia E resultar na captura
        # de ao menos uma peça adversária em qualquer das 8 direções.

        validos = []
        for l in range(self.tamanho):
            for c in range(self.tamanho):
                if self.tabuleiro[l][c] == 0 and self.eh_movimento_valido(l, c, jogador):
                    validos.append((l, c))
        return validos

    def eh_movimento_valido(self, linha: int, col: int, jogador: int) -> bool:
        # Verifica se (linha, col) é um movimento válido para 'jogador'.
        for dl, dc in DIRECOES:
            if self.verifica_captura_na_direcao(linha, col, jogador, dl, dc):
                return True
        return False

    def verifica_captura_na_direcao(self, linha: int, col: int,
                                jogador: int, dl: int, dc: int) -> bool:
        # Verifica se, na direção (dl, dc), jogar em (linha, col) captura
        # ao menos uma peça adversária.

        oponente = -jogador
        l, c = linha + dl, col + dc
        achou_oponente = False

        while 0 <= l < self.tamanho and 0 <= c < self.tamanho:
            if self.tabuleiro[l][c] == oponente: # Precisa achar oponente entre esse lugar que tu quer jogar e outra peça nossa
                achou_oponente = True
            elif self.tabuleiro[l][c] == jogador:
                return achou_oponente   # Há peças adversárias entre as nossas
            else:
                return False            # Célula vazia, então interrompe a cadeia
            l += dl
            c += dc

        return False

    def executa_jogada(self, linha: int, col: int, jogador: int) -> bool:
        # Aplica o movimento (linha, col) para 'jogador', revertendo as peças
        # capturadas. Retorna True se o movimento foi executado com sucesso.

        if self.tabuleiro[linha][col] != 0:
            return False
        if not self.eh_movimento_valido(linha, col, jogador):
            return False

        self.historico.append(self.tabuleiro.copy())
        self.tabuleiro[linha][col] = jogador
        self.inverte_pecas(linha, col, jogador)
        self.contagem_passes = 0
        return True

    def inverte_pecas(self, linha: int, col: int, jogador: int):
        # Inverte todas as peças capturadas pelo movimento em (linha, col).

        for dl, dc in DIRECOES: # Para cada uma das direções, verifica quais peças devem ser invertidas
            lista_de_inversoes = self.get_inversoes(linha, col, jogador, dl, dc) # Lista com todas as (linha, col) que devem ser invertidadas dada uma direcao

            for l, c in lista_de_inversoes: # Itera invertendo as pecas
                self.tabuleiro[l][c] = jogador

    def get_inversoes(self, linha: int, col: int,
                   jogador: int, dl: int, dc: int) -> List[Tuple[int, int]]:
        # Retorna lista de posições a flipar na direção (dl, dc).

        oponente = -jogador
        l, c = linha + dl, col + dc
        candidatos = []

        while 0 <= l < self.tamanho and 0 <= c < self.tamanho:
            if self.tabuleiro[l][c] == oponente:
                candidatos.append((l, c))
            elif self.tabuleiro[l][c] == jogador:
                return candidatos        # Confirma captura
            else:
                return []               # Cadeia interrompida por vazio
            l += dl
            c += dc

        return []

    def verifica_fim(self) -> bool:
        # O jogo termina quando:
        # - Nenhum dos dois jogadores tem jogadas disponíveis, OU
        # - O tabuleiro está cheio.

        if np.count_nonzero(self.tabuleiro == 0) == 0: # Tabuleiro cheio
            return True
        if not self.movimentos_validos(1) and not self.movimentos_validos(-1): # Sem jogadas para ambos os jogadores
            return True
        return False

    def get_vencedor(self) -> Optional[int]:
        # Conta as peças e retorna o vencedor (P = 1 ou B = -1).
        # Empate retorna 0. Só deve ser chamado quando verifica_fim().

        # Extensão: se bonus_canto=True, cantos valem 3x na soma.

        score_preto, score_branco = self.contagem_scores()

        if score_preto > score_branco:
            return 1
        elif score_branco > score_preto:
            return -1
        else:
            return 0

    def contagem_scores(self) -> Tuple[int, int]:
        # Calcula pontuação de cada jogador.
        score_preto = int(np.sum(self.tabuleiro == 1))
        score_branco = int(np.sum(self.tabuleiro == -1))

        if self.bonus_canto:
            cantos = [(0, 0), (0, self.tamanho - 1),
                       (self.tamanho - 1, 0), (self.tamanho - 1, self.tamanho - 1)]
            for l, c in cantos:
                v = self.tabuleiro[l][c]
                if v == 1:
                    score_preto += 2        # +2 extra = total 3× por canto
                elif v == -1:
                    score_branco += 2

        return score_preto, score_branco

    def get_dicionario_score(self) -> dict:
        # Retorna dicionário com pontuação de cada jogador.
        score_preto, score_branco = self.contagem_scores()
        return {1: score_preto, -1: score_branco}

    def clonar(self) -> "Othello":
        # Cria uma cópia profunda do estado atual (usada pelos agentes).
        novo_jogo = Othello(self.tamanho, self.bonus_canto)
        novo_jogo.tabuleiro = self.tabuleiro.copy()
        novo_jogo.jogador_atual = self.jogador_atual
        novo_jogo.contagem_passes = self.contagem_passes
        return novo_jogo

    def troca_jogador(self):
        # Troca o jogador corrente.
        self.jogador_atual *= -1 # Troca entre 1 e -1

    def __str__(self) -> str: 
        # '__str__' faz com que sempre que o objeto seja convertido para string (ex: print(jogo)) ele retorne a representação do tabuleiro
        simbolos = {0: ".", 1: "○", -1: "●"}
        linhas = []
        header = "  " + " ".join(str(i) for i in range(self.tamanho))
        linhas.append(header)
        for l in range(self.tamanho):
            linha_str = str(l) + " " + " ".join(simbolos[self.tabuleiro[l][c]]
                                               for c in range(self.tamanho))
            linhas.append(linha_str)
        return "\n".join(linhas)
