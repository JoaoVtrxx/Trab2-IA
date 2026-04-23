# Motor do jogo Othello (Reversi) para tabuleiro NxN.

# Representação do tabuleiro:
#    0  = célula vazia
#    1  = peça do jogador 1 (preto)
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
#        Se True, cantos valem 3× na contagem final.
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

    def movimentos_validoos(self, jogador: int) -> List[Tuple[int, int]]:
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
        
        # PARAMOS AQUI!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if self.tabuleiro[linha][col] != 0:
            return False
        if not self.eh_movimento_valido(linha, col, jogador):
            return False

        self.historico.append(self.tabuleiro.copy())
        self.tabuleiro[linha][col] = jogador
        self._flip_pieces(linha, col, jogador)
        self.contagem_passes = 0
        return True

    def _flip_pieces(self, linha: int, col: int, jogador: int):
        """Reverte todas as peças capturadas pelo movimento em (linha, col)."""
        for dr, dc in DIRECOES:
            to_flip = self._get_flips(linha, col, jogador, dr, dc)
            for r, c in to_flip:
                self.tabuleiro[r][c] = jogador

    def _get_flips(self, linha: int, col: int,
                   jogador: int, dr: int, dc: int) -> List[Tuple[int, int]]:
        """Retorna lista de posições a flipar na direção (dr, dc)."""
        oponente = -jogador
        r, c = linha + dr, col + dc
        candidate = []

        while 0 <= r < self.tamanho and 0 <= c < self.tamanho:
            if self.tabuleiro[r][c] == oponente:
                candidate.append((r, c))
            elif self.tabuleiro[r][c] == jogador:
                return candidate        # Confirma captura
            else:
                return []               # Cadeia interrompida por vazio
            r += dr
            c += dc

        return []

    # ------------------------------------------------------------------
    # Condição de término e vencedor
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        """
        O jogo termina quando:
        - Nenhum dos dois jogadores tem jogadas disponíveis, OU
        - O tabuleiro está cheio.
        """
        if np.count_nonzero(self.tabuleiro == 0) == 0:
            return True
        if not self.movimentos_validoos(1) and not self.movimentos_validoos(-1):
            return True
        return False

    def get_winner(self) -> Optional[int]:
        """
        Conta as peças e retorna o vencedor (1 ou -1).
        Empate retorna 0. Só deve ser chamado quando is_terminal().

        Extensão: se bonus_canto=True, cantos valem 3× na soma.
        """
        score1, score_neg1 = self._count_scores()

        if score1 > score_neg1:
            return 1
        elif score_neg1 > score1:
            return -1
        else:
            return 0

    def _count_scores(self) -> Tuple[int, int]:
        """Calcula pontuação de cada jogador."""
        score1 = int(np.sum(self.tabuleiro == 1))
        score_neg1 = int(np.sum(self.tabuleiro == -1))

        if self.bonus_canto:
            corners = [(0, 0), (0, self.tamanho - 1),
                       (self.tamanho - 1, 0), (self.tamanho - 1, self.tamanho - 1)]
            for r, c in corners:
                v = self.tabuleiro[r][c]
                if v == 1:
                    score1 += 2        # +2 extra = total 3× por canto
                elif v == -1:
                    score_neg1 += 2

        return score1, score_neg1

    def get_score_dict(self) -> dict:
        """Retorna dicionário com pontuação de cada jogador."""
        s1, sn1 = self._count_scores()
        return {1: s1, -1: sn1}

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def clone(self) -> "Othello":
        """Cria uma cópia profunda do estado atual (usada pelos agentes)."""
        new_game = Othello(self.tamanho, self.bonus_canto)
        new_game.tabuleiro = self.tabuleiro.copy()
        new_game.jogador_atual = self.jogador_atual
        new_game.contagem_passes = self.contagem_passes
        return new_game

    def switch_jogador(self):
        """Troca o jogador corrente."""
        self.jogador_atual *= -1

    def __str__(self) -> str:
        symbols = {0: ".", 1: "●", -1: "○"}
        linhas = []
        header = "  " + " ".join(str(i) for i in range(self.tamanho))
        linhas.append(header)
        for r in range(self.tamanho):
            linha_str = str(r) + " " + " ".join(symbols[self.tabuleiro[r][c]]
                                               for c in range(self.tamanho))
            linhas.append(linha_str)
        return "\n".join(linhas)
