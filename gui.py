"""
gui.py  —  Interface Gráfica para o Jogo Othello
=================================================
Visualize partidas entre MinMax e MCTS ao estilo "Universo Programado".

Funcionalidades:
    - Tabuleiro animado com peças preto/branco
    - Destaque de jogadas válidas e último movimento
    - Painel de placar em tempo real
    - Controles: Play / Pause / Passo a passo / Reset
    - Slider de velocidade
    - Configuração dos agentes (profundidade MinMax, simulações MCTS)
    - Modo Torneio: N partidas com gráfico de resultados
    - Log de jogadas com tempo por movimento

Execute:
    python gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import queue
import statistics
from typing import Optional, List, Tuple, Dict

from othello_jogo import Othello
from minmax_agente import MinMaxAgente
from mcts_agente import MCTSAgente


# ── Paleta de cores ──────────────────────────────────────────────────────────
VERDE_TABULEIRO  = "#1a6b2f"
VERDE_ESCURO     = "#14511f"
VERDE_CLARO_HINT = "#2ecc71"   # jogadas válidas
AMARELO_ULTIMO   = "#f1c40f"   # último movimento
PRETO_PECA       = "#1a1a1a"
BRANCO_PECA      = "#f0f0f0"
BG_PRINCIPAL     = "#1e1e2e"   # fundo dark
BG_PAINEL        = "#2a2a3e"
BG_PAINEL2       = "#313149"
TEXTO            = "#cdd6f4"
TEXTO_SUBTITULO  = "#a6adc8"
ACENTO_AZUL      = "#89b4fa"
ACENTO_VERDE     = "#a6e3a1"
ACENTO_VERMELHO  = "#f38ba8"
ACENTO_AMARELO   = "#f9e2af"
BORDA_CELULA     = "#0d3a1a"

CELL = 72        # pixels por célula
MARGEM = 30      # margem do canvas


# ── Helpers visuais ──────────────────────────────────────────────────────────

def interpola_cor(c1: str, c2: str, t: float) -> str:
    """Interpola duas cores hex para animação flip de peças."""
    r1, g1, b1 = int(c1[1:3],16), int(c1[3:5],16), int(c1[5:7],16)
    r2, g2, b2 = int(c2[1:3],16), int(c2[3:5],16), int(c2[5:7],16)
    r = int(r1 + (r2-r1)*t)
    g = int(g1 + (g2-g1)*t)
    b = int(b1 + (b2-b1)*t)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Janela principal ──────────────────────────────────────────────────────────

class OthelloGUI:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Othello — Visualizador de Partidas IA")
        self.root.configure(bg=BG_PRINCIPAL)
        self.root.resizable(True, True)

        # Estado do jogo
        self.game: Optional[Othello] = None
        self.agent1 = None
        self.agent2 = None
        self.tamanho = 6
        self.ultimo_movimento: Optional[Tuple[int,int]] = None
        self.movimentos_validos_cache: List[Tuple[int,int]] = []

        # Controle de execução
        self._jogando = False
        self._pausado = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fila: queue.Queue = queue.Queue()
        self._delay_ms = 700   # velocidade padrão

        # Histórico de estados para step-back/forward
        self._historico_estados: List[dict] = []
        self._idx_historico = -1

        # Torneio
        self._resultados_torneio: List[dict] = []

        self._build_ui()
        self._nova_partida_silenciosa()
        self._poll_fila()

    # ── Construção da UI ──────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Barra de título ──
        header = tk.Frame(self.root, bg=BG_PRINCIPAL)
        header.pack(fill="x", padx=12, pady=(10, 0))
        tk.Label(header, text="♟  OTHELLO", font=("Segoe UI", 18, "bold"),
                 bg=BG_PRINCIPAL, fg=ACENTO_AZUL).pack(side="left")
        tk.Label(header, text="Visualizador de IAs",
                 font=("Segoe UI", 11), bg=BG_PRINCIPAL,
                 fg=TEXTO_SUBTITULO).pack(side="left", padx=12, pady=3)

        # ── Corpo central ──
        corpo = tk.Frame(self.root, bg=BG_PRINCIPAL)
        corpo.pack(fill="both", expand=True, padx=8, pady=6)

        # Coluna esquerda (configurações)
        col_esq = tk.Frame(corpo, bg=BG_PRINCIPAL, width=220)
        col_esq.pack(side="left", fill="y", padx=(0,6))
        col_esq.pack_propagate(False)
        self._build_config(col_esq)

        # Tabuleiro (centro)
        col_centro = tk.Frame(corpo, bg=BG_PRINCIPAL)
        col_centro.pack(side="left", fill="both", expand=True)
        self._build_board(col_centro)

        # Coluna direita (placar + log)
        col_dir = tk.Frame(corpo, bg=BG_PRINCIPAL, width=260)
        col_dir.pack(side="left", fill="y", padx=(6,0))
        col_dir.pack_propagate(False)
        self._build_info(col_dir)

    # ── Painel de configurações ───────────────────────────────────────────────

    def _build_config(self, parent):
        def secao(texto):
            f = tk.Frame(parent, bg=BG_PAINEL, bd=0)
            f.pack(fill="x", pady=(0,6))
            tk.Label(f, text=texto, font=("Segoe UI", 9, "bold"),
                     bg=BG_PAINEL, fg=ACENTO_AZUL,
                     padx=8, pady=4).pack(anchor="w")
            inner = tk.Frame(f, bg=BG_PAINEL)
            inner.pack(fill="x", padx=8, pady=(0,6))
            return inner

        # ── Agente 1 ──
        f1 = secao("● PRETO  (Jogador 1)")
        self.var_tipo1 = tk.StringVar(value="MinMax")
        self._radio_agente(f1, self.var_tipo1)
        tk.Label(f1, text="Profundidade:", bg=BG_PAINEL,
                 fg=TEXTO, font=("Segoe UI", 9)).pack(anchor="w")
        self.spin_prof1 = tk.Spinbox(f1, from_=3, to=7, width=6,
                                     font=("Segoe UI",9),
                                     bg=BG_PAINEL2, fg=TEXTO,
                                     buttonbackground=BG_PAINEL2,
                                     insertbackground=TEXTO)
        self.spin_prof1.delete(0, "end"); self.spin_prof1.insert(0, "4")
        self.spin_prof1.pack(anchor="w", pady=2)
        tk.Label(f1, text="Simulações:", bg=BG_PAINEL,
                 fg=TEXTO, font=("Segoe UI", 9)).pack(anchor="w")
        self.spin_sims1 = tk.Spinbox(f1, from_=50, to=2000, increment=50,
                                      width=6, font=("Segoe UI",9),
                                      bg=BG_PAINEL2, fg=TEXTO,
                                      buttonbackground=BG_PAINEL2,
                                      insertbackground=TEXTO)
        self.spin_sims1.delete(0, "end"); self.spin_sims1.insert(0, "300")
        self.spin_sims1.pack(anchor="w", pady=2)

        # ── Agente 2 ──
        f2 = secao("○ BRANCO  (Jogador 2)")
        self.var_tipo2 = tk.StringVar(value="MCTS")
        self._radio_agente(f2, self.var_tipo2)
        tk.Label(f2, text="Profundidade:", bg=BG_PAINEL,
                 fg=TEXTO, font=("Segoe UI", 9)).pack(anchor="w")
        self.spin_prof2 = tk.Spinbox(f2, from_=3, to=7, width=6,
                                     font=("Segoe UI",9),
                                     bg=BG_PAINEL2, fg=TEXTO,
                                     buttonbackground=BG_PAINEL2,
                                     insertbackground=TEXTO)
        self.spin_prof2.delete(0, "end"); self.spin_prof2.insert(0, "4")
        self.spin_prof2.pack(anchor="w", pady=2)
        tk.Label(f2, text="Simulações:", bg=BG_PAINEL,
                 fg=TEXTO, font=("Segoe UI", 9)).pack(anchor="w")
        self.spin_sims2 = tk.Spinbox(f2, from_=50, to=2000, increment=50,
                                      width=6, font=("Segoe UI",9),
                                      bg=BG_PAINEL2, fg=TEXTO,
                                      buttonbackground=BG_PAINEL2,
                                      insertbackground=TEXTO)
        self.spin_sims2.delete(0, "end"); self.spin_sims2.insert(0, "300")
        self.spin_sims2.pack(anchor="w", pady=2)

        # ── Tabuleiro ──
        fb = secao("TABULEIRO")
        tk.Label(fb, text="Tamanho (NxN):", bg=BG_PAINEL,
                 fg=TEXTO, font=("Segoe UI", 9)).pack(anchor="w")
        self.var_tamanho = tk.IntVar(value=6)
        frame_tam = tk.Frame(fb, bg=BG_PAINEL)
        frame_tam.pack(anchor="w")
        for t in (4, 6, 8):
            tk.Radiobutton(frame_tam, text=str(t), variable=self.var_tamanho,
                           value=t, bg=BG_PAINEL, fg=TEXTO,
                           selectcolor=BG_PAINEL2,
                           activebackground=BG_PAINEL,
                           font=("Segoe UI", 9)).pack(side="left")
        self.var_bonus = tk.BooleanVar(value=False)
        tk.Checkbutton(fb, text="Bônus canto 3×",
                       variable=self.var_bonus,
                       bg=BG_PAINEL, fg=TEXTO,
                       selectcolor=BG_PAINEL2,
                       activebackground=BG_PAINEL,
                       font=("Segoe UI", 9)).pack(anchor="w")

        # ── Velocidade ──
        fv = secao("VELOCIDADE")
        tk.Label(fv, text="Rápido ←→ Devagar",
                 bg=BG_PAINEL, fg=TEXTO_SUBTITULO,
                 font=("Segoe UI", 8)).pack(anchor="w")
        self.slider_vel = tk.Scale(fv, from_=100, to=3000,
                                   orient="horizontal",
                                   bg=BG_PAINEL, fg=TEXTO,
                                   troughcolor=BG_PAINEL2,
                                   highlightthickness=0,
                                   font=("Segoe UI", 8),
                                   length=190, showvalue=True,
                                   resolution=100, label="ms/jogada")
        self.slider_vel.set(700)
        self.slider_vel.pack()

        # ── Torneio ──
        ft = secao("TORNEIO")
        tk.Label(ft, text="Nº de partidas:", bg=BG_PAINEL,
                 fg=TEXTO, font=("Segoe UI", 9)).pack(anchor="w")
        self.spin_ngames = tk.Spinbox(ft, from_=2, to=50, width=6,
                                      font=("Segoe UI",9),
                                      bg=BG_PAINEL2, fg=TEXTO,
                                      buttonbackground=BG_PAINEL2,
                                      insertbackground=TEXTO)
        self.spin_ngames.delete(0, "end"); self.spin_ngames.insert(0, "6")
        self.spin_ngames.pack(anchor="w", pady=2)
        tk.Button(ft, text="▶  Iniciar Torneio",
                  command=self._iniciar_torneio,
                  bg=ACENTO_AZUL, fg="#1e1e2e",
                  font=("Segoe UI", 9, "bold"),
                  relief="flat", padx=6, pady=3,
                  cursor="hand2").pack(fill="x", pady=(4,0))

    def _radio_agente(self, parent, var):
        f = tk.Frame(parent, bg=BG_PAINEL)
        f.pack(anchor="w")
        for opt in ("MinMax", "MCTS"):
            tk.Radiobutton(f, text=opt, variable=var, value=opt,
                           bg=BG_PAINEL, fg=TEXTO,
                           selectcolor=BG_PAINEL2,
                           activebackground=BG_PAINEL,
                           font=("Segoe UI", 9)).pack(side="left")

    # ── Tabuleiro ────────────────────────────────────────────────────────────

    def _build_board(self, parent):
        # Frame do tabuleiro
        board_frame = tk.Frame(parent, bg=BG_PRINCIPAL)
        board_frame.pack(expand=True, fill="both")

        # Canvas
        canvas_size = CELL * 8 + MARGEM * 2 + 2
        self.canvas = tk.Canvas(board_frame, width=canvas_size,
                                height=canvas_size,
                                bg=BG_PRINCIPAL, highlightthickness=0)
        self.canvas.pack(expand=True)

        # ── Controles ──
        ctrl = tk.Frame(parent, bg=BG_PRINCIPAL)
        ctrl.pack(fill="x", padx=6, pady=4)

        btn_style = dict(font=("Segoe UI", 11), relief="flat",
                         padx=8, pady=4, cursor="hand2")

        self.btn_nova = tk.Button(ctrl, text="⟳  Nova", bg=BG_PAINEL2,
                                  fg=TEXTO, command=self._nova_partida,
                                  **btn_style)
        self.btn_nova.pack(side="left", padx=2)

        self.btn_play = tk.Button(ctrl, text="▶  Play", bg=ACENTO_VERDE,
                                  fg="#1e1e2e", command=self._toggle_play,
                                  **btn_style)
        self.btn_play.pack(side="left", padx=2)

        self.btn_passo = tk.Button(ctrl, text="⏭  Passo", bg=BG_PAINEL2,
                                   fg=TEXTO, command=self._passo,
                                   **btn_style)
        self.btn_passo.pack(side="left", padx=2)

        self.btn_prev = tk.Button(ctrl, text="◀  Voltar", bg=BG_PAINEL2,
                                  fg=TEXTO, command=self._voltar,
                                  **btn_style)
        self.btn_prev.pack(side="left", padx=2)

        # Status
        self.var_status = tk.StringVar(value="Pronto")
        tk.Label(ctrl, textvariable=self.var_status,
                 bg=BG_PRINCIPAL, fg=ACENTO_AMARELO,
                 font=("Segoe UI", 9, "bold")).pack(side="right", padx=6)

    # ── Info (placar + log) ───────────────────────────────────────────────────

    def _build_info(self, parent):
        # ── Placar ──
        placar_frame = tk.Frame(parent, bg=BG_PAINEL)
        placar_frame.pack(fill="x", pady=(0,6))
        tk.Label(placar_frame, text="PLACAR", font=("Segoe UI",9,"bold"),
                 bg=BG_PAINEL, fg=ACENTO_AZUL,
                 padx=8, pady=4).pack(anchor="w")

        inner_placar = tk.Frame(placar_frame, bg=BG_PAINEL)
        inner_placar.pack(fill="x", padx=8, pady=(0,6))

        # Jogador 1 (preto)
        f_p1 = tk.Frame(inner_placar, bg=BG_PAINEL2)
        f_p1.pack(fill="x", pady=2)
        tk.Label(f_p1, text="●", font=("Segoe UI",18),
                 bg=BG_PAINEL2, fg=PRETO_PECA,
                 padx=6).pack(side="left")
        tk.Label(f_p1, text="PRETO", font=("Segoe UI",9,"bold"),
                 bg=BG_PAINEL2, fg=TEXTO).pack(side="left")
        self.lbl_p1_tipo = tk.Label(f_p1, text="(MinMax)",
                                    font=("Segoe UI",8),
                                    bg=BG_PAINEL2, fg=TEXTO_SUBTITULO)
        self.lbl_p1_tipo.pack(side="left", padx=2)
        self.lbl_score1 = tk.Label(f_p1, text="2",
                                   font=("Segoe UI",20,"bold"),
                                   bg=BG_PAINEL2, fg=ACENTO_VERDE)
        self.lbl_score1.pack(side="right", padx=10)

        # Jogador 2 (branco)
        f_p2 = tk.Frame(inner_placar, bg=BG_PAINEL2)
        f_p2.pack(fill="x", pady=2)
        tk.Label(f_p2, text="○", font=("Segoe UI",18),
                 bg=BG_PAINEL2, fg="#888",
                 padx=6).pack(side="left")
        tk.Label(f_p2, text="BRANCO", font=("Segoe UI",9,"bold"),
                 bg=BG_PAINEL2, fg=TEXTO).pack(side="left")
        self.lbl_p2_tipo = tk.Label(f_p2, text="(MCTS)",
                                    font=("Segoe UI",8),
                                    bg=BG_PAINEL2, fg=TEXTO_SUBTITULO)
        self.lbl_p2_tipo.pack(side="left", padx=2)
        self.lbl_score2 = tk.Label(f_p2, text="2",
                                   font=("Segoe UI",20,"bold"),
                                   bg=BG_PAINEL2, fg=ACENTO_VERDE)
        self.lbl_score2.pack(side="right", padx=10)

        # Barra de progresso visual
        self.canvas_barra = tk.Canvas(inner_placar, height=14,
                                      bg=BG_PAINEL2, highlightthickness=0)
        self.canvas_barra.pack(fill="x", pady=(4,2))

        # Jogada atual
        self.lbl_jogada_num = tk.Label(inner_placar,
                                       text="Jogada: 0",
                                       font=("Segoe UI",9),
                                       bg=BG_PAINEL, fg=TEXTO_SUBTITULO)
        self.lbl_jogada_num.pack(anchor="w")

        # Quem está pensando
        self.lbl_turno = tk.Label(inner_placar,
                                  text="",
                                  font=("Segoe UI",9,"bold"),
                                  bg=BG_PAINEL, fg=ACENTO_AMARELO)
        self.lbl_turno.pack(anchor="w")

        # ── Tempos ──
        tempos_frame = tk.Frame(parent, bg=BG_PAINEL)
        tempos_frame.pack(fill="x", pady=(0,6))
        tk.Label(tempos_frame, text="TEMPO POR JOGADA",
                 font=("Segoe UI",9,"bold"),
                 bg=BG_PAINEL, fg=ACENTO_AZUL,
                 padx=8, pady=4).pack(anchor="w")
        inner_t = tk.Frame(tempos_frame, bg=BG_PAINEL)
        inner_t.pack(fill="x", padx=8, pady=(0,6))
        self.lbl_t1 = tk.Label(inner_t, text="● Último: —",
                                font=("Segoe UI",9), bg=BG_PAINEL, fg=TEXTO)
        self.lbl_t1.pack(anchor="w")
        self.lbl_t2 = tk.Label(inner_t, text="○ Último: —",
                                font=("Segoe UI",9), bg=BG_PAINEL, fg=TEXTO)
        self.lbl_t2.pack(anchor="w")
        self.lbl_tmedio1 = tk.Label(inner_t, text="● Médio:  —",
                                     font=("Segoe UI",9), bg=BG_PAINEL,
                                     fg=TEXTO_SUBTITULO)
        self.lbl_tmedio1.pack(anchor="w")
        self.lbl_tmedio2 = tk.Label(inner_t, text="○ Médio:  —",
                                     font=("Segoe UI",9), bg=BG_PAINEL,
                                     fg=TEXTO_SUBTITULO)
        self.lbl_tmedio2.pack(anchor="w")

        # ── Log de jogadas ──
        log_frame = tk.Frame(parent, bg=BG_PAINEL)
        log_frame.pack(fill="both", expand=True)
        tk.Label(log_frame, text="LOG DE JOGADAS",
                 font=("Segoe UI",9,"bold"),
                 bg=BG_PAINEL, fg=ACENTO_AZUL,
                 padx=8, pady=4).pack(anchor="w")
        self.log = scrolledtext.ScrolledText(
            log_frame, width=28, height=14,
            bg=BG_PAINEL2, fg=TEXTO,
            font=("Courier New", 8),
            insertbackground=TEXTO,
            relief="flat", padx=4, pady=4,
            state="disabled"
        )
        self.log.pack(fill="both", expand=True, padx=6, pady=(0,6))

    # ── Desenhando o tabuleiro ────────────────────────────────────────────────

    def _cell_xy(self, l: int, c: int):
        """Retorna (x0, y0, x1, y1) de uma célula."""
        x0 = MARGEM + c * CELL
        y0 = MARGEM + l * CELL
        return x0, y0, x0 + CELL, y0 + CELL

    def _desenha_tabuleiro(self):
        self.canvas.delete("all")
        if self.game is None:
            return

        n = self.game.tamanho
        validos = self.movimentos_validos_cache

        # Fundo geral do tabuleiro
        total = n * CELL
        self.canvas.create_rectangle(
            MARGEM - 4, MARGEM - 4,
            MARGEM + total + 4, MARGEM + total + 4,
            fill=VERDE_ESCURO, outline=VERDE_ESCURO, width=3
        )

        for l in range(n):
            for c in range(n):
                x0, y0, x1, y1 = self._cell_xy(l, c)
                # Fundo da célula
                cor_cel = VERDE_TABULEIRO
                # Destaque do último movimento
                if self.ultimo_movimento == (l, c):
                    cor_cel = AMARELO_ULTIMO
                self.canvas.create_rectangle(x0, y0, x1, y1,
                                             fill=cor_cel,
                                             outline=BORDA_CELULA, width=1)
                # Hint de jogada válida
                if (l, c) in validos:
                    cx = (x0 + x1) // 2
                    cy = (y0 + y1) // 2
                    r = CELL // 6
                    self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                            fill=VERDE_CLARO_HINT,
                                            outline="", stipple="")

        # Peças
        for l in range(n):
            for c in range(n):
                v = self.game.tabuleiro[l][c]
                if v != 0:
                    x0, y0, x1, y1 = self._cell_xy(l, c)
                    pad = CELL * 0.10
                    cor = PRETO_PECA if v == 1 else BRANCO_PECA
                    sombra_off = 3
                    # Sombra
                    self.canvas.create_oval(
                        x0+pad+sombra_off, y0+pad+sombra_off,
                        x1-pad+sombra_off, y1-pad+sombra_off,
                        fill="#000000", outline="", tags="peca_sombra"
                    )
                    # Peça
                    self.canvas.create_oval(
                        x0+pad, y0+pad, x1-pad, y1-pad,
                        fill=cor, outline="#333333", width=1, tags="peca"
                    )
                    # Reflexo (brilho)
                    pad2 = pad * 2
                    rx0 = x0 + pad2
                    ry0 = y0 + pad2
                    rx1 = x0 + CELL * 0.45
                    ry1 = y0 + CELL * 0.40
                    brilho = "#4a4a4a" if v == 1 else "#ffffff"
                    self.canvas.create_oval(rx0, ry0, rx1, ry1,
                                            fill=brilho, outline="", tags="peca")

        # Coordenadas
        for i in range(n):
            cx = MARGEM + i * CELL + CELL // 2
            self.canvas.create_text(cx, MARGEM - 14, text=str(i),
                                    fill=TEXTO_SUBTITULO,
                                    font=("Segoe UI", 9))
            cy = MARGEM + i * CELL + CELL // 2
            self.canvas.create_text(MARGEM - 14, cy, text=str(i),
                                    fill=TEXTO_SUBTITULO,
                                    font=("Segoe UI", 9))

    def _atualiza_placar(self):
        if self.game is None:
            return
        sc = self.game.get_dicionario_score()
        p1, p2 = sc[1], sc[-1]
        self.lbl_score1.config(text=str(p1))
        self.lbl_score2.config(text=str(p2))

        # Barra proporcional
        total_pcs = p1 + p2
        w = self.canvas_barra.winfo_width()
        if w < 4:
            w = 220
        self.canvas_barra.delete("all")
        if total_pcs > 0:
            frac = p1 / total_pcs
            x_div = int(w * frac)
            self.canvas_barra.create_rectangle(0, 0, x_div, 14,
                                               fill=PRETO_PECA, outline="")
            self.canvas_barra.create_rectangle(x_div, 0, w, 14,
                                               fill=BRANCO_PECA, outline="")
        n_jogadas = len(self.game.historico)
        self.lbl_jogada_num.config(text=f"Jogada: {n_jogadas}")

    def _log(self, texto: str):
        self.log.config(state="normal")
        self.log.insert("end", texto + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _log_clear(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    # ── Nova partida ──────────────────────────────────────────────────────────

    def _criar_agente(self, tipo: str, jogador: int,
                      prof: int, sims: int):
        if tipo == "MinMax":
            return MinMaxAgente(jogador=jogador, profundidade=prof)
        else:
            return MCTSAgente(jogador=jogador, maximo_simulacoes=sims,
                              tipo_rollout="heuristica_cantos")

    def _nova_partida_silenciosa(self):
        self.tamanho = 6
        self.game = Othello(tamanho=self.tamanho, bonus_canto=False)
        self.ultimo_movimento = None
        self.movimentos_validos_cache = self.game.movimentos_validos(
            self.game.jogador_atual)
        self._historico_estados = []
        self._idx_historico = -1
        self._desenha_tabuleiro()
        self._atualiza_placar()

    def _nova_partida(self):
        self._parar_thread()
        self.tamanho = self.var_tamanho.get()
        bonus = self.var_bonus.get()
        self.game = Othello(tamanho=self.tamanho, bonus_canto=bonus)
        self.ultimo_movimento = None

        prof1 = int(self.spin_prof1.get())
        sims1 = int(self.spin_sims1.get())
        prof2 = int(self.spin_prof2.get())
        sims2 = int(self.spin_sims2.get())
        tipo1 = self.var_tipo1.get()
        tipo2 = self.var_tipo2.get()

        self.agent1 = self._criar_agente(tipo1, 1,  prof1, sims1)
        self.agent2 = self._criar_agente(tipo2, -1, prof2, sims2)

        self.lbl_p1_tipo.config(text=f"({tipo1})")
        self.lbl_p2_tipo.config(text=f"({tipo2})")

        self.movimentos_validos_cache = self.game.movimentos_validos(
            self.game.jogador_atual)
        self._historico_estados = []
        self._idx_historico = -1
        self._times1: List[float] = []
        self._times2: List[float] = []

        self._log_clear()
        self._log(f"═══ NOVA PARTIDA ═══")
        self._log(f"● {tipo1}  vs  ○ {tipo2}")
        self._log(f"Tabuleiro {self.tamanho}×{self.tamanho}")
        self._log("")

        self._desenha_tabuleiro()
        self._atualiza_placar()
        self.var_status.set("Pronto — pressione Play")
        self.btn_play.config(text="▶  Play", bg=ACENTO_VERDE)
        self._jogando = False

    # ── Controles de execução ─────────────────────────────────────────────────

    def _toggle_play(self):
        if not self._jogando:
            # Inicia ou retoma
            if self.game is None or self.game.verifica_fim():
                self._nova_partida()
            if self.agent1 is None:
                self._nova_partida()
            self._jogando = True
            self._pausado = False
            self.btn_play.config(text="⏸  Pause", bg=ACENTO_VERMELHO)
            self.var_status.set("Jogando…")
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop_partida, daemon=True)
            self._thread.start()
        else:
            self._parar_thread()
            self.btn_play.config(text="▶  Play", bg=ACENTO_VERDE)
            self.var_status.set("Pausado")
            self._jogando = False

    def _parar_thread(self):
        self._stop_event.set()
        self._jogando = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def _passo(self):
        """Executa um único movimento."""
        if self.game is None or self.game.verifica_fim():
            return
        if self._jogando:
            return  # não mistura com auto-play
        if self.agent1 is None:
            self._nova_partida()
        self._executar_movimento()

    def _voltar(self):
        """Volta um estado no histórico."""
        if len(self._historico_estados) == 0:
            return
        estado = self._historico_estados.pop()
        self.game.tabuleiro = estado["tabuleiro"]
        self.game.jogador_atual = estado["jogador_atual"]
        self.game.contagem_passes = estado["contagem_passes"]
        self.ultimo_movimento = estado.get("ultimo_movimento")
        self.movimentos_validos_cache = self.game.movimentos_validos(
            self.game.jogador_atual)
        self._desenha_tabuleiro()
        self._atualiza_placar()
        n_mov = len(self.game.historico)
        self.lbl_turno.config(text="")
        self._log(f"  ↩ Voltando para jogada {n_mov}")

    # ── Loop de partida (thread) ──────────────────────────────────────────────

    def _loop_partida(self):
        """Roda num thread separado; comunica via fila."""
        while not self._stop_event.is_set():
            if self.game.verifica_fim():
                self._fila.put(("fim", None))
                break
            self._executar_movimento_thread()
            delay = self.slider_vel.get() / 1000.0
            time.sleep(delay)
        self._fila.put(("thread_done", None))

    def _executar_movimento_thread(self):
        """Executado dentro da thread de jogo."""
        jogador = self.game.jogador_atual
        agent = self.agent1 if jogador == 1 else self.agent2
        nome = "●" if jogador == 1 else "○"

        self._fila.put(("pensando", jogador))

        t0 = time.time()
        movimento = agent.escolhe_movimento(self.game)
        elapsed = time.time() - t0

        if self._stop_event.is_set():
            return

        self._fila.put(("movimento", (jogador, movimento, elapsed)))

    def _executar_movimento(self):
        """Versão síncrona para o botão Passo."""
        if self.game.verifica_fim():
            self._mostrar_fim()
            return

        jogador = self.game.jogador_atual
        agent = self.agent1 if jogador == 1 else self.agent2
        nome = "●" if jogador == 1 else "○"

        self.var_status.set(f"{nome} pensando…")
        self.lbl_turno.config(
            text=f"{'● Preto' if jogador == 1 else '○ Branco'} pensando…")
        self.root.update_idletasks()

        t0 = time.time()
        movimento = agent.escolhe_movimento(self.game)
        elapsed = time.time() - t0

        self._aplicar_movimento(jogador, movimento, elapsed)

        if self.game.verifica_fim():
            self._mostrar_fim()

    def _aplicar_movimento(self, jogador: int,
                           movimento: Optional[Tuple[int,int]],
                           elapsed: float):
        """Aplica um movimento ao estado do jogo e atualiza a UI."""
        import copy
        # Salva estado para o voltar
        self._historico_estados.append({
            "tabuleiro": self.game.tabuleiro.copy(),
            "jogador_atual": self.game.jogador_atual,
            "contagem_passes": self.game.contagem_passes,
            "ultimo_movimento": self.ultimo_movimento,
        })

        nome_j = "● Preto" if jogador == 1 else "○ Branco"
        tipo_j = self.var_tipo1.get() if jogador == 1 else self.var_tipo2.get()

        if movimento is None:
            self._log(f"  {nome_j} ({tipo_j}): PASSA")
            self.game.contagem_passes += 1
            self.game.troca_jogador()
            # verifica duplo passe
            if self.game.contagem_passes >= 2:
                return
        else:
            self.ultimo_movimento = movimento
            self.game.executa_jogada(movimento[0], movimento[1], jogador)
            n_mov = len(self.game.historico)
            self._log(f"  {n_mov:2d}. {nome_j} ({tipo_j}): "
                      f"({movimento[0]},{movimento[1]})  {elapsed*1000:.0f}ms")
            # Tempos
            if jogador == 1:
                self._times1.append(elapsed)
                self.lbl_t1.config(
                    text=f"● Último: {elapsed*1000:.0f} ms")
                if len(self._times1) > 1:
                    self.lbl_tmedio1.config(
                        text=f"● Médio:  "
                             f"{statistics.mean(self._times1)*1000:.0f} ms")
            else:
                self._times2.append(elapsed)
                self.lbl_t2.config(
                    text=f"○ Último: {elapsed*1000:.0f} ms")
                if len(self._times2) > 1:
                    self.lbl_tmedio2.config(
                        text=f"○ Médio:  "
                             f"{statistics.mean(self._times2)*1000:.0f} ms")
            self.game.troca_jogador()

        self.movimentos_validos_cache = self.game.movimentos_validos(
            self.game.jogador_atual)
        self._desenha_tabuleiro()
        self._atualiza_placar()
        self.lbl_turno.config(
            text=f"Vez de: {'● Preto' if self.game.jogador_atual==1 else '○ Branco'}")

    def _mostrar_fim(self):
        venc = self.game.get_vencedor()
        sc = self.game.get_dicionario_score()
        self.movimentos_validos_cache = []
        self._desenha_tabuleiro()
        self._atualiza_placar()
        if venc == 1:
            msg = f"● PRETO vence!  {sc[1]}×{sc[-1]}"
            cor = ACENTO_VERDE
        elif venc == -1:
            msg = f"○ BRANCO vence!  {sc[1]}×{sc[-1]}"
            cor = "#f0f0f0"
        else:
            msg = f"EMPATE!  {sc[1]}×{sc[-1]}"
            cor = ACENTO_AMARELO
        self.var_status.set(msg)
        self.lbl_turno.config(text=msg)
        self._log("")
        self._log(f"═══ FIM DE JOGO ═══")
        self._log(f"  {msg}")
        self._log(f"  Jogadas: {len(self.game.historico)}")
        if self._times1:
            self._log(f"  ● tempo médio: "
                      f"{statistics.mean(self._times1)*1000:.0f} ms")
        if self._times2:
            self._log(f"  ○ tempo médio: "
                      f"{statistics.mean(self._times2)*1000:.0f} ms")
        self.btn_play.config(text="▶  Play", bg=ACENTO_VERDE)
        self._jogando = False

    # ── Polling da fila (main thread) ─────────────────────────────────────────

    def _poll_fila(self):
        try:
            while True:
                tipo, dado = self._fila.get_nowait()
                if tipo == "pensando":
                    jogador = dado
                    nome = "● Preto" if jogador == 1 else "○ Branco"
                    self.var_status.set(f"{nome} pensando…")
                    self.lbl_turno.config(text=f"{nome} pensando…")
                elif tipo == "movimento":
                    jogador, mov, elapsed = dado
                    self._aplicar_movimento(jogador, mov, elapsed)
                    if self.game.verifica_fim():
                        self._mostrar_fim()
                        self._stop_event.set()
                elif tipo == "fim":
                    self._mostrar_fim()
                elif tipo == "thread_done":
                    pass
                elif tipo == "torneio_update":
                    self._atualiza_janela_torneio(dado)
                elif tipo == "torneio_fim":
                    self._torneio_fim(dado)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_fila)

    # ── Torneio ───────────────────────────────────────────────────────────────

    def _iniciar_torneio(self):
        self._parar_thread()

        n = int(self.spin_ngames.get())
        tipo1 = self.var_tipo1.get()
        tipo2 = self.var_tipo2.get()
        prof1 = int(self.spin_prof1.get())
        sims1 = int(self.spin_sims1.get())
        prof2 = int(self.spin_prof2.get())
        sims2 = int(self.spin_sims2.get())
        tamanho = self.var_tamanho.get()
        bonus = self.var_bonus.get()

        # Cria janela do torneio
        self._win_torneio = JanelaTorneio(self.root,
                                          tipo1, tipo2, n)
        self._resultados_torneio = []

        # Thread do torneio
        self._stop_event.clear()
        t = threading.Thread(
            target=self._loop_torneio,
            args=(n, tipo1, tipo2, prof1, sims1, prof2, sims2,
                  tamanho, bonus),
            daemon=True
        )
        t.start()

    def _loop_torneio(self, n, tipo1, tipo2,
                      prof1, sims1, prof2, sims2,
                      tamanho, bonus):
        resultados = []
        for i in range(n):
            if self._stop_event.is_set():
                break
            # Alterna cores
            if i % 2 == 0:
                c1, c2 = 1, -1
            else:
                c1, c2 = -1, 1

            ag1 = self._criar_agente(tipo1, c1, prof1, sims1)
            ag2 = self._criar_agente(tipo2, c2, prof2, sims2)

            game = Othello(tamanho=tamanho, bonus_canto=bonus)
            times_ag1: List[float] = []
            times_ag2: List[float] = []
            consec = 0

            while not game.verifica_fim():
                jogador = game.jogador_atual
                ag = ag1 if jogador == c1 else ag2
                t_list = times_ag1 if jogador == c1 else times_ag2

                t0 = time.time()
                mov = ag.escolhe_movimento(game)
                elapsed = time.time() - t0
                t_list.append(elapsed)

                if mov is None:
                    consec += 1
                    if consec >= 2:
                        break
                    game.troca_jogador()
                else:
                    consec = 0
                    game.executa_jogada(mov[0], mov[1], jogador)
                    game.troca_jogador()

                if self._stop_event.is_set():
                    break

            sc = game.get_dicionario_score()
            venc = game.get_vencedor()
            r = {
                "partida": i + 1,
                "venc": venc,
                "cor_ag1": c1,
                "score": sc,
                "movs": len(game.historico),
                "t_ag1": statistics.mean(times_ag1) if times_ag1 else 0,
                "t_ag2": statistics.mean(times_ag2) if times_ag2 else 0,
            }
            resultados.append(r)
            self._fila.put(("torneio_update", list(resultados)))

        self._fila.put(("torneio_fim", resultados))

    def _atualiza_janela_torneio(self, resultados):
        if hasattr(self, "_win_torneio") and self._win_torneio.winfo_exists():
            self._win_torneio.atualiza(resultados)

    def _torneio_fim(self, resultados):
        if hasattr(self, "_win_torneio") and self._win_torneio.winfo_exists():
            self._win_torneio.atualiza(resultados, fim=True)


# ── Janela de Torneio ─────────────────────────────────────────────────────────

class JanelaTorneio(tk.Toplevel):

    def __init__(self, parent, tipo1: str, tipo2: str, n: int):
        super().__init__(parent)
        self.title(f"Torneio  ●{tipo1} vs ○{tipo2}  ({n} partidas)")
        self.configure(bg=BG_PRINCIPAL)
        self.resizable(True, True)
        self.tipo1 = tipo1
        self.tipo2 = tipo2
        self.n = n
        self._build()

    def _build(self):
        # Cabeçalho
        tk.Label(self, text="TORNEIO", font=("Segoe UI",14,"bold"),
                 bg=BG_PRINCIPAL, fg=ACENTO_AZUL,
                 padx=12, pady=8).pack(anchor="w")
        tk.Label(self,
                 text=f"● {self.tipo1}  vs  ○ {self.tipo2}",
                 font=("Segoe UI",10), bg=BG_PRINCIPAL,
                 fg=TEXTO).pack(anchor="w", padx=12)

        # Placar grande
        f_placar = tk.Frame(self, bg=BG_PAINEL)
        f_placar.pack(fill="x", padx=12, pady=8)

        self.lbl_v1 = tk.Label(f_placar, text="0",
                                font=("Segoe UI",36,"bold"),
                                bg=BG_PAINEL, fg=ACENTO_VERDE,
                                padx=20)
        self.lbl_v1.pack(side="left")
        tk.Label(f_placar, text="×", font=("Segoe UI",24),
                 bg=BG_PAINEL, fg=TEXTO).pack(side="left")
        self.lbl_v2 = tk.Label(f_placar, text="0",
                                font=("Segoe UI",36,"bold"),
                                bg=BG_PAINEL, fg=ACENTO_VERMELHO,
                                padx=20)
        self.lbl_v2.pack(side="left")
        self.lbl_empates = tk.Label(f_placar, text="Empates: 0",
                                    font=("Segoe UI",10),
                                    bg=BG_PAINEL, fg=TEXTO_SUBTITULO,
                                    padx=10)
        self.lbl_empates.pack(side="left")
        self.lbl_prog = tk.Label(f_placar, text="0/0",
                                 font=("Segoe UI",10),
                                 bg=BG_PAINEL, fg=TEXTO_SUBTITULO,
                                 padx=10)
        self.lbl_prog.pack(side="right", padx=10)

        # Barra de progresso das vitórias
        self.canvas_barra = tk.Canvas(self, height=20,
                                      bg=BG_PAINEL2,
                                      highlightthickness=0)
        self.canvas_barra.pack(fill="x", padx=12, pady=(0,8))

        # Canvas do gráfico de vitórias acumuladas
        tk.Label(self, text="Vitórias Acumuladas",
                 font=("Segoe UI",9), bg=BG_PRINCIPAL,
                 fg=TEXTO_SUBTITULO).pack(anchor="w", padx=12)
        self.canvas_grafico = tk.Canvas(self, width=520, height=180,
                                        bg=BG_PAINEL2,
                                        highlightthickness=0)
        self.canvas_grafico.pack(padx=12, pady=(0,8))

        # Tabela de partidas
        tk.Label(self, text="Partidas",
                 font=("Segoe UI",9), bg=BG_PRINCIPAL,
                 fg=TEXTO_SUBTITULO).pack(anchor="w", padx=12)
        cols = ("Partida","Cores","Vencedor","Placar","Jogadas",
                f"T.médio {self.tipo1}",f"T.médio {self.tipo2}")
        self.tree = ttk.Treeview(self, columns=cols,
                                 show="headings", height=8)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                        background=BG_PAINEL2,
                        foreground=TEXTO,
                        fieldbackground=BG_PAINEL2,
                        rowheight=22,
                        font=("Segoe UI",8))
        style.configure("Treeview.Heading",
                        background=BG_PAINEL,
                        foreground=ACENTO_AZUL,
                        font=("Segoe UI",8,"bold"))
        widths = [60, 80, 110, 80, 70, 110, 110]
        for col, w in zip(cols, widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=12, pady=(0,8))

        # Estatísticas finais
        self.lbl_stats = tk.Label(self, text="",
                                  font=("Segoe UI",9),
                                  bg=BG_PRINCIPAL, fg=ACENTO_AMARELO,
                                  justify="left", padx=12, pady=6)
        self.lbl_stats.pack(anchor="w")

    def atualiza(self, resultados: List[dict], fim: bool = False):
        """Atualiza a janela com os resultados mais recentes."""
        v1 = sum(1 for r in resultados if r["venc"] == r["cor_ag1"])
        v2 = sum(1 for r in resultados if r["venc"] != r["cor_ag1"]
                 and r["venc"] != 0)
        emp = sum(1 for r in resultados if r["venc"] == 0)
        total = len(resultados)

        self.lbl_v1.config(text=str(v1))
        self.lbl_v2.config(text=str(v2))
        self.lbl_empates.config(text=f"Empates: {emp}")
        self.lbl_prog.config(text=f"{total}/{self.n}")

        # Barra proporcional
        w = self.canvas_barra.winfo_width()
        if w < 4:
            w = 500
        self.canvas_barra.delete("all")
        if total > 0:
            f1 = v1 / total
            f2 = v2 / total
            x1 = int(w * f1)
            x2 = int(w * (f1 + f2))
            self.canvas_barra.create_rectangle(0, 0, x1, 20,
                                               fill=ACENTO_VERDE,
                                               outline="")
            self.canvas_barra.create_rectangle(x1, 0, x2, 20,
                                               fill=ACENTO_VERMELHO,
                                               outline="")
            self.canvas_barra.create_rectangle(x2, 0, w, 20,
                                               fill=BG_PAINEL,
                                               outline="")
            if v1 > 0:
                self.canvas_barra.create_text(x1//2, 10,
                    text=f"{self.tipo1} {v1}", fill="#1e1e2e",
                    font=("Segoe UI",8,"bold"))
            if v2 > 0:
                mx = (x1 + x2) // 2
                self.canvas_barra.create_text(mx, 10,
                    text=f"{self.tipo2} {v2}", fill="#1e1e2e",
                    font=("Segoe UI",8,"bold"))

        # Gráfico de vitórias acumuladas
        self._desenha_grafico(resultados)

        # Tabela
        # Limpa e recria
        for row in self.tree.get_children():
            self.tree.delete(row)
        for r in resultados:
            c1_str = "●" if r["cor_ag1"] == 1 else "○"
            venc_str = (f"● {self.tipo1}" if r["venc"] == r["cor_ag1"]
                        else (f"○ {self.tipo2}" if r["venc"] != 0
                              else "Empate"))
            sc = r["score"]
            self.tree.insert("", "end", values=(
                r["partida"],
                f"{c1_str}{self.tipo1} vs {self.tipo2}",
                venc_str,
                f"{sc[1]}×{sc[-1]}",
                r["movs"],
                f"{r['t_ag1']*1000:.0f} ms",
                f"{r['t_ag2']*1000:.0f} ms",
            ))

        # Estatísticas finais
        if fim and total > 0:
            t1_all = [r["t_ag1"] for r in resultados if r["t_ag1"] > 0]
            t2_all = [r["t_ag2"] for r in resultados if r["t_ag2"] > 0]
            txt = (f"RESULTADO FINAL:  "
                   f"● {self.tipo1}: {v1} vitórias ({100*v1/total:.0f}%)  |  "
                   f"○ {self.tipo2}: {v2} vitórias ({100*v2/total:.0f}%)  |  "
                   f"Empates: {emp}\n"
                   f"Tempo médio/jog  ● {statistics.mean(t1_all)*1000:.0f} ms"
                   f"   ○ {statistics.mean(t2_all)*1000:.0f} ms"
                   if t1_all and t2_all else "")
            self.lbl_stats.config(text=txt)

    def _desenha_grafico(self, resultados: List[dict]):
        """Gráfico de linhas: vitórias acumuladas ao longo das partidas."""
        c = self.canvas_grafico
        c.delete("all")
        w, h = 520, 180
        pad = 36

        if not resultados:
            return

        n = len(resultados)
        acc1 = []
        acc2 = []
        v1_acc = 0
        v2_acc = 0
        for r in resultados:
            if r["venc"] == r["cor_ag1"]:
                v1_acc += 1
            elif r["venc"] != 0:
                v2_acc += 1
            acc1.append(v1_acc)
            acc2.append(v2_acc)

        maximo = max(max(acc1), max(acc2), 1)

        # Grid
        for tick in range(0, maximo + 2):
            y = h - pad - int((tick / (maximo + 1)) * (h - 2*pad))
            c.create_line(pad, y, w-pad, y,
                          fill=BG_PAINEL, width=1)
            c.create_text(pad - 4, y, text=str(tick),
                          fill=TEXTO_SUBTITULO,
                          font=("Segoe UI", 7), anchor="e")

        # Eixo X
        c.create_line(pad, h-pad, w-pad, h-pad,
                      fill=TEXTO_SUBTITULO, width=1)
        # Eixo Y
        c.create_line(pad, pad, pad, h-pad,
                      fill=TEXTO_SUBTITULO, width=1)

        def ponto(i, val):
            x = pad + int(i / max(n-1, 1) * (w - 2*pad))
            y = h - pad - int(val / (maximo + 1) * (h - 2*pad))
            return x, y

        # Linhas de série
        for serie, cor, nome in [
            (acc1, ACENTO_VERDE, self.tipo1),
            (acc2, ACENTO_VERMELHO, self.tipo2),
        ]:
            pts = [ponto(i, v) for i, v in enumerate(serie)]
            for i in range(len(pts)-1):
                c.create_line(*pts[i], *pts[i+1],
                              fill=cor, width=2, smooth=True)
            for x, y in pts:
                c.create_oval(x-3, y-3, x+3, y+3,
                              fill=cor, outline="")

        # Legenda
        c.create_rectangle(w-120, 12, w-80, 22,
                           fill=ACENTO_VERDE, outline="")
        c.create_text(w-75, 17, text=f"● {self.tipo1}",
                      fill=TEXTO, font=("Segoe UI",8), anchor="w")
        c.create_rectangle(w-120, 28, w-80, 38,
                           fill=ACENTO_VERMELHO, outline="")
        c.create_text(w-75, 33, text=f"○ {self.tipo2}",
                      fill=TEXTO, font=("Segoe UI",8), anchor="w")


# ── Ponto de entrada ──────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = OthelloGUI(root)

    # Centraliza na tela
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    rw = root.winfo_width()
    rh = root.winfo_height()
    x = (sw - rw) // 2
    y = (sh - rh) // 2
    root.geometry(f"+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()
