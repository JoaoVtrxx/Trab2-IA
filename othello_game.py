"""
othello_game.py
===============
Motor do jogo Othello (Reversi) para tabuleiro NxN.

Representação do tabuleiro:
    0  = célula vazia
    1  = peça do jogador 1 (preto)
   -1  = peça do jogador 2 (branco)
"""

import copy
import numpy as np
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Direções para verificar capturas (8 direções)
# ---------------------------------------------------------------------------
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0,  -1),           (0,  1),
              (1,  -1),  (1,  0), (1,  1)]


class OthelloGame:
    """
    Encapsula o estado e a lógica completa do jogo Othello/Reversi.

    Parâmetros
    ----------
    size : int
        Tamanho do tabuleiro (size x size). Default = 6.
    corner_bonus : bool
        Se True, cantos valem 3× na contagem final.
        Extensão obrigatória de pontuação diferenciada.
    """

    def __init__(self, size: int = 6, corner_bonus: bool = False):
        self.size = size
        self.corner_bonus = corner_bonus
        self.board = np.zeros((size, size), dtype=int)
        self._init_board()
        self.current_player = 1          # Jogador 1 (preto) começa
        self.pass_count = 0              # Contagem de passes consecutivos
        self.history: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Inicialização
    # ------------------------------------------------------------------

    def _init_board(self):
        """Posiciona as quatro peças iniciais no centro do tabuleiro."""
        c = self.size // 2
        self.board[c - 1][c - 1] =  1
        self.board[c - 1][c]     = -1
        self.board[c][c - 1]     = -1
        self.board[c][c]         =  1

    # ------------------------------------------------------------------
    # Geração de jogadas válidas
    # ------------------------------------------------------------------

    def get_valid_moves(self, player: int) -> List[Tuple[int, int]]:
        """
        Retorna lista de (row, col) onde 'player' pode jogar.

        Uma posição é válida se estiver vazia E resultar na captura
        de ao menos uma peça adversária em qualquer das 8 direções.
        """
        valid = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0 and self._is_valid_move(r, c, player):
                    valid.append((r, c))
        return valid

    def _is_valid_move(self, row: int, col: int, player: int) -> bool:
        """Verifica se (row, col) é um movimento válido para 'player'."""
        for dr, dc in DIRECTIONS:
            if self._captures_in_direction(row, col, player, dr, dc):
                return True
        return False

    def _captures_in_direction(self, row: int, col: int,
                                player: int, dr: int, dc: int) -> bool:
        """
        Verifica se, na direção (dr, dc), jogar em (row, col) captura
        ao menos uma peça adversária.
        """
        opponent = -player
        r, c = row + dr, col + dc
        found_opponent = False

        while 0 <= r < self.size and 0 <= c < self.size:
            if self.board[r][c] == opponent:
                found_opponent = True
            elif self.board[r][c] == player:
                return found_opponent   # Há peças adversárias entre as nossas
            else:
                return False            # Célula vazia — interrompe a cadeia
            r += dr
            c += dc

        return False

    # ------------------------------------------------------------------
    # Execução de jogada
    # ------------------------------------------------------------------

    def make_move(self, row: int, col: int, player: int) -> bool:
        """
        Aplica o movimento (row, col) para 'player', revertendo as peças
        capturadas. Retorna True se o movimento foi executado com sucesso.
        """
        if self.board[row][col] != 0:
            return False
        if not self._is_valid_move(row, col, player):
            return False

        self.history.append(self.board.copy())
        self.board[row][col] = player
        self._flip_pieces(row, col, player)
        self.pass_count = 0
        return True

    def _flip_pieces(self, row: int, col: int, player: int):
        """Reverte todas as peças capturadas pelo movimento em (row, col)."""
        for dr, dc in DIRECTIONS:
            to_flip = self._get_flips(row, col, player, dr, dc)
            for r, c in to_flip:
                self.board[r][c] = player

    def _get_flips(self, row: int, col: int,
                   player: int, dr: int, dc: int) -> List[Tuple[int, int]]:
        """Retorna lista de posições a flipar na direção (dr, dc)."""
        opponent = -player
        r, c = row + dr, col + dc
        candidate = []

        while 0 <= r < self.size and 0 <= c < self.size:
            if self.board[r][c] == opponent:
                candidate.append((r, c))
            elif self.board[r][c] == player:
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
        if np.count_nonzero(self.board == 0) == 0:
            return True
        if not self.get_valid_moves(1) and not self.get_valid_moves(-1):
            return True
        return False

    def get_winner(self) -> Optional[int]:
        """
        Conta as peças e retorna o vencedor (1 ou -1).
        Empate retorna 0. Só deve ser chamado quando is_terminal().

        Extensão: se corner_bonus=True, cantos valem 3× na soma.
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
        score1 = int(np.sum(self.board == 1))
        score_neg1 = int(np.sum(self.board == -1))

        if self.corner_bonus:
            corners = [(0, 0), (0, self.size - 1),
                       (self.size - 1, 0), (self.size - 1, self.size - 1)]
            for r, c in corners:
                v = self.board[r][c]
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

    def clone(self) -> "OthelloGame":
        """Cria uma cópia profunda do estado atual (usada pelos agentes)."""
        new_game = OthelloGame(self.size, self.corner_bonus)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.pass_count = self.pass_count
        return new_game

    def switch_player(self):
        """Troca o jogador corrente."""
        self.current_player *= -1

    def __str__(self) -> str:
        symbols = {0: ".", 1: "●", -1: "○"}
        rows = []
        header = "  " + " ".join(str(i) for i in range(self.size))
        rows.append(header)
        for r in range(self.size):
            row_str = str(r) + " " + " ".join(symbols[self.board[r][c]]
                                               for c in range(self.size))
            rows.append(row_str)
        return "\n".join(rows)
