"""
tournament.py
=============
Script de torneio e análise experimental entre MinMaxAgent e MCTSAgent.

Coleta:
    - Taxa de vitória de cada agente
    - Impacto da profundidade no MinMax
    - Impacto do número de simulações no MCTS
    - Tempo médio de processamento por jogada
"""

import time
import statistics
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field

from othello_game import Othello
from minmax_agent import MinMaxAgent
from mcts_agent import MCTSAgent


# ---------------------------------------------------------------------------
# Estrutura de resultado de uma partida
# ---------------------------------------------------------------------------

@dataclass
class GameResult:
    winner: int            # 1, -1 ou 0
    movimentos_count: int
    times_p1: List[float]  # tempo (s) por jogada do jogador 1
    times_p2: List[float]  # tempo (s) por jogada do jogador 2
    score: Dict[int, int]  # pontuação final


# ---------------------------------------------------------------------------
# Jogador de uma partida completa
# ---------------------------------------------------------------------------

def play_game(agent1, agent2, tamanho: int = 6,
              bonus_canto: bool = False,
              verboso: bool = False) -> GameResult:
    """
    Executa uma partida entre agent1 (jogador 1) e agent2 (jogador -1).
    Retorna GameResult com estatísticas detalhadas.
    """
    game = Othello(tamanho=tamanho, bonus_canto=bonus_canto)
    times1: List[float] = []
    times2: List[float] = []
    movimentos_count = 0
    consecutive_passes = 0

    while not game.verifica_fim():
        jogador = game.jogador_atual
        agent  = agent1 if jogador == 1 else agent2
        times  = times1 if jogador == 1 else times2

        t0   = time.time()
        movimento = agent.escolhe_movimento(game)
        elapsed = time.time() - t0
        times.append(elapsed)

        if movimento is None:
            # Sem jogadas → passa a vez
            consecutive_passes += 1
            if consecutive_passes >= 2:
                break
            game.troca_jogador()
            continue

        consecutive_passes = 0
        game.executa_jogada(movimento[0], movimento[1], jogador)
        movimentos_count += 1
        game.troca_jogador()

        if verboso:
            print(f"  Jogador {jogador} → {movimento}  ({elapsed:.3f}s)")
            print(game)
            print()

    return GameResult(
        winner=game.get_vencedor(),
        movimentos_count=movimentos_count,
        times_p1=times1,
        times_p2=times2,
        score=game.get_dicionario_score(),
    )


# ---------------------------------------------------------------------------
# Torneio principal
# ---------------------------------------------------------------------------

def run_tournament(n_games: int = 10, tamanho: int = 6,
                   minmax_profundidade: int = 4, mcts_sims: int = 300,
                   bonus_canto: bool = False) -> None:
    """
    Realiza n_games partidas alternando quem é o jogador 1 (preto).
    Exibe estatísticas completas ao final.
    """
    print("\n" + "="*65)
    print("  TORNEIO: MinMax(α-β) vs MCTS")
    print(f"  Partidas={n_games}  Tabuleiro={tamanho}x{tamanho}  "
          f"Profundidade_MM={minmax_profundidade}  Sims_MCTS={mcts_sims}")
    print("="*65 + "\n")

    wins_mm   = 0
    wins_mcts = 0
    draws     = 0

    all_times_mm  : List[float] = []
    all_times_mcts: List[float] = []

    for i in range(n_games):
        # Alterna quem joga com peças pretas (jogador 1)
        if i % 2 == 0:
            mm_color   = 1
            mcts_color = -1
        else:
            mm_color   = -1
            mcts_color = 1

        mm_agent   = MinMaxAgent(jogador=mm_color,   profundidade=minmax_profundidade)
        mcts_agent = MCTSAgent(jogador=mcts_color, max_sims=mcts_sims,
                               rollout_type="heuristic")

        if mm_color == 1:
            result = play_game(mm_agent, mcts_agent, tamanho=tamanho,
                               bonus_canto=bonus_canto)
            all_times_mm.extend(result.times_p1)
            all_times_mcts.extend(result.times_p2)
        else:
            result = play_game(mcts_agent, mm_agent, tamanho=tamanho,
                               bonus_canto=bonus_canto)
            all_times_mm.extend(result.times_p2)
            all_times_mcts.extend(result.times_p1)

        if result.winner == mm_color:
            wins_mm += 1
            outcome = "MinMax vence"
        elif result.winner == mcts_color:
            wins_mcts += 1
            outcome = "MCTS vence"
        else:
            draws += 1
            outcome = "Empate"

        score = result.score
        print(f"  Partida {i+1:2d}: {outcome:15s}  "
              f"(preto={mm_color:+d}→MM, score={score[1]}x{score[-1]}  "
              f"jogadas={result.movimentos_count})")

    # --- Resumo ---
    total_with_winner = wins_mm + wins_mcts
    print("\n" + "-"*65)
    print("  RESULTADO FINAL")
    print(f"    MinMax  vitórias : {wins_mm:3d}  ({100*wins_mm/n_games:.1f}%)")
    print(f"    MCTS    vitórias : {wins_mcts:3d}  ({100*wins_mcts/n_games:.1f}%)")
    print(f"    Empates          : {draws:3d}  ({100*draws/n_games:.1f}%)")

    if all_times_mm:
        print(f"\n  Tempo médio / jogada MinMax : "
              f"{statistics.mean(all_times_mm)*1000:.1f} ms  "
              f"(±{statistics.stdev(all_times_mm)*1000:.1f} ms)")
    if all_times_mcts:
        print(f"  Tempo médio / jogada MCTS   : "
              f"{statistics.mean(all_times_mcts)*1000:.1f} ms  "
              f"(±{statistics.stdev(all_times_mcts)*1000:.1f} ms)")
    print("="*65 + "\n")


# ---------------------------------------------------------------------------
# Análise de impacto de profundidade (MinMax)
# ---------------------------------------------------------------------------

def profundidade_impact_analysis(profundidades: List[int] = [3, 4, 5],
                           n_games: int = 6, tamanho: int = 6) -> None:
    """
    Avalia o impacto da profundidade do MinMax contra MCTS fixo.
    """
    print("\n" + "="*65)
    print("  ANÁLISE: Impacto da Profundidade no MinMax")
    print("="*65)

    base_mcts = MCTSAgent(jogador=-1, max_sims=300, rollout_type="heuristic")

    for profundidade in profundidades:
        wins_mm = 0
        total_time = []

        for i in range(n_games):
            mm = MinMaxAgent(jogador=1 if i%2==0 else -1, profundidade=profundidade)
            mcts = MCTSAgent(jogador=-1 if i%2==0 else 1,
                             max_sims=300, rollout_type="heuristic")

            if mm.jogador == 1:
                r = play_game(mm, mcts, tamanho=tamanho)
                total_time.extend(r.times_p1)
            else:
                r = play_game(mcts, mm, tamanho=tamanho)
                total_time.extend(r.times_p2)

            if r.winner == mm.jogador:
                wins_mm += 1

        avg_ms = statistics.mean(total_time)*1000 if total_time else 0
        print(f"  Profundidade {profundidade}: MinMax vence {wins_mm}/{n_games} "
              f"({100*wins_mm/n_games:.0f}%)  "
              f"avg_tempo={avg_ms:.1f}ms/jogada")


# ---------------------------------------------------------------------------
# Análise de impacto de simulações (MCTS)
# ---------------------------------------------------------------------------

def sims_impact_analysis(sims_list: List[int] = [100, 300, 600],
                         n_games: int = 6, tamanho: int = 6) -> None:
    """
    Avalia o impacto do número de simulações do MCTS contra MinMax fixo.
    """
    print("\n" + "="*65)
    print("  ANÁLISE: Impacto do Nº de Simulações no MCTS")
    print("="*65)

    for sims in sims_list:
        wins_mcts = 0
        total_time = []

        for i in range(n_games):
            mcts = MCTSAgent(jogador=1 if i%2==0 else -1,
                             max_sims=sims, rollout_type="heuristic")
            mm = MinMaxAgent(jogador=-1 if i%2==0 else 1, profundidade=4)

            if mcts.jogador == 1:
                r = play_game(mcts, mm, tamanho=tamanho)
                total_time.extend(r.times_p1)
            else:
                r = play_game(mm, mcts, tamanho=tamanho)
                total_time.extend(r.times_p2)

            if r.winner == mcts.jogador:
                wins_mcts += 1

        avg_ms = statistics.mean(total_time)*1000 if total_time else 0
        print(f"  Simulações {sims:4d}: MCTS vence {wins_mcts}/{n_games} "
              f"({100*wins_mcts/n_games:.0f}%)  "
              f"avg_tempo={avg_ms:.1f}ms/jogada")


# ---------------------------------------------------------------------------
# Demo de uma única partida com verboso
# ---------------------------------------------------------------------------

def demo_single_game(verboso_mm: bool = True) -> None:
    """
    Executa uma partida de demonstração exibindo a árvore MinMax (2 níveis).
    """
    print("\n" + "="*65)
    print("  DEMO: Partida com árvore Min-Max visível (prof=3)")
    print("="*65 + "\n")

    mm   = MinMaxAgent(jogador=1,  profundidade=3, verboso=verboso_mm)
    mcts = MCTSAgent(jogador=-1, max_sims=200, rollout_type="heuristic")

    game = Othello(tamanho=6)
    print("Estado inicial:")
    print(game)
    print()

    movimento_number = 0
    consecutive_passes = 0

    while not game.verifica_fim():
        jogador = game.jogador_atual
        agent  = mm if jogador == 1 else mcts

        movimento = agent.escolhe_movimento(game)

        if movimento is None:
            consecutive_passes += 1
            print(f"  Jogador {'MM' if jogador==1 else 'MCTS'} passa a vez.")
            if consecutive_passes >= 2:
                break
            game.troca_jogador()
            continue

        consecutive_passes = 0
        game.executa_jogada(movimento[0], movimento[1], jogador)
        movimento_number += 1
        game.troca_jogador()

        label = "MinMax" if jogador == 1 else "MCTS  "
        print(f"\n  [{movimento_number:02d}] {label} (jogador {jogador:+d}) → {movimento}")
        print(game)

    scores = game.get_dicionario_score()
    winner = game.get_vencedor()
    print(f"\n  RESULTADO: ●(MM)={scores[1]}  ○(MCTS)={scores[-1]}  "
          f"Vencedor={'MinMax' if winner==1 else ('MCTS' if winner==-1 else 'Empate')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Demo de uma partida com árvore visível
    demo_single_game(verboso_mm=True)

    # 2. Torneio principal (10 partidas)
    run_tournament(n_games=10, tamanho=6, minmax_profundidade=4,
                   mcts_sims=300, bonus_canto=False)

    # 3. Análise de profundidade do MinMax
    profundidade_impact_analysis(profundidades=[3, 4, 5], n_games=6)

    # 4. Análise de simulações do MCTS
    sims_impact_analysis(sims_list=[100, 300, 600], n_games=6)
