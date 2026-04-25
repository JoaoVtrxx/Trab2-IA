# Script de torneio e análise experimental entre MinMaxAgente e MCTSAgente.

# Coleta:
#     - Taxa de vitória de cada agente
#     - Impacto da profundidade no MinMax
#     - Impacto do número de simulações no MCTS
#     - Tempo médio de processamento por jogada

import time
import statistics
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field

from othello_jogo import Othello
from minmax_agente import MinMaxAgente
from mcts_agente import MCTSAgente

# Estrutura de resultado de uma partida

@dataclass
class jogoResult:
    vencedor: int            # 1, -1 ou 0
    quantidade_movimentos: int
    tempo_p1: List[float]  # tempo (s) por jogada do jogador 1
    tempo_p2: List[float]  # tempo (s) por jogada do jogador 2
    score: Dict[int, int]  # pontuação final

# Jogador de uma partida completa

def executa_jogo(agente1, agente2, tamanho: int = 6,
              bonus_canto: bool = False,
              verboso: bool = False) -> jogoResult:
    # Executa uma partida entre agent1 (jogador 1) e agent2 (jogador -1).
    # Retorna jogoResult com estatísticas detalhadas.

    jogo = Othello(tamanho=tamanho, bonus_canto=bonus_canto)
    tempo1: List[float] = []
    tempo2: List[float] = []
    quantidade_movimentos = 0
    passes_consecutivos = 0

    while not jogo.verifica_fim():
        jogador = jogo.jogador_atual
        agente  = agente1 if jogador == 1 else agente2
        tempo  = tempo1 if jogador == 1 else tempo2

        t0   = time.time()
        movimento = agente.escolhe_movimento(jogo)
        tempo_decorrido = time.time() - t0
        tempo.append(tempo_decorrido)

        if movimento is None:
            # Sem jogadas -> passa a vez
            passes_consecutivos += 1
            if passes_consecutivos >= 2:
                break
            jogo.troca_jogador()
            continue

        passes_consecutivos = 0
        jogo.executa_jogada(movimento[0], movimento[1], jogador)
        quantidade_movimentos += 1
        jogo.troca_jogador()

        if verboso:
            print(f"  Jogador {jogador} -> {movimento}  ({tempo_decorrido:.3f}s)")
            print(jogo)
            print()

    return jogoResult(
        vencedor=jogo.get_vencedor(),
        quantidade_movimentos=quantidade_movimentos,
        tempo_p1=tempo1,
        tempo_p2=tempo2,
        score=jogo.get_dicionario_score(),
    )

# Torneio principal

def run_tournament(n_jogos: int = 10, tamanho: int = 6,
                   minmax_profundidade: int = 4, mcts_sims: int = 300,
                   bonus_canto: bool = False) -> None:
    # Realiza n_jogos partidas alternando quem é o jogador 1 (preto).
    # Exibe estatísticas completas ao final.

    print("\n" + "="*65)
    print("  TORNEIO: MinMax(Alfa-Beta) vs MCTS")
    print(f"  Partidas={n_jogos}  Tabuleiro={tamanho}x{tamanho}  "
          f"Profundidade_MM={minmax_profundidade}  Sims_MCTS={mcts_sims}")
    print("="*65 + "\n")

    vitorias_mm   = 0
    vitorias_mcts = 0
    empates     = 0

    tempo_total_mm  : List[float] = []
    tempo_total_mcts: List[float] = []

    for i in range(n_jogos):
        # Alterna quem joga com peças pretas (jogador 1)
        if i % 2 == 0:
            cor_mm   = 1
            cor_mcts = -1
        else:
            cor_mm   = -1
            cor_mcts = 1

        agente_mm   = MinMaxAgente(jogador=cor_mm,   profundidade=minmax_profundidade)
        agente_mcts = MCTSAgente(jogador=cor_mcts, maximo_simulacoes=mcts_sims,
                               tipo_rollout="heuristico_cantos")

        if cor_mm == 1:
            resultado = executa_jogo(agente_mm, agente_mcts, tamanho=tamanho,
                               bonus_canto=bonus_canto)
            tempo_total_mm.extend(resultado.tempo_p1)
            tempo_total_mcts.extend(resultado.tempo_p2)
        else:
            resultado = executa_jogo(agente_mcts, agente_mm, tamanho=tamanho,
                               bonus_canto=bonus_canto)
            tempo_total_mm.extend(resultado.tempo_p2)
            tempo_total_mcts.extend(resultado.tempo_p1)

        if resultado.vencedor == cor_mm:
            vitorias_mm += 1
            resultado_str = "MinMax vence"
        elif resultado.vencedor == cor_mcts:
            vitorias_mcts += 1
            resultado_str = "MCTS vence"
        else:
            empates += 1
            resultado_str = "Empate"

        score = resultado.score
        print(f"  Partida {i+1:2d}: {resultado_str:15s}  "
              f"(preto={cor_mm:+d}->MM, score={score[1]}x{score[-1]}  "
              f"jogadas={resultado.quantidade_movimentos})")

    # --- Resumo ---
    print("\n" + "-"*65)
    print("  RESULTADO FINAL")
    print(f"    MinMax  vitórias : {vitorias_mm:3d}  ({100*vitorias_mm/n_jogos:.1f}%)")
    print(f"    MCTS    vitórias : {vitorias_mcts:3d}  ({100*vitorias_mcts/n_jogos:.1f}%)")
    print(f"    Empates          : {empates:3d}  ({100*empates/n_jogos:.1f}%)")

    if tempo_total_mm:
        print(f"\n  Tempo médio / jogada MinMax : "
              f"{statistics.mean(tempo_total_mm)*1000:.1f} ms  "
              f"(+-{statistics.stdev(tempo_total_mm)*1000:.1f} ms)")
    if tempo_total_mcts:
        print(f"  Tempo médio / jogada MCTS   : "
              f"{statistics.mean(tempo_total_mcts)*1000:.1f} ms  "
              f"(+-{statistics.stdev(tempo_total_mcts)*1000:.1f} ms)")
    print("="*65 + "\n")

# Análise de impacto de profundidade (MinMax)

def analise_impacto_profundidade(profundidades: List[int] = [3, 4, 5],
                           n_jogos: int = 6, tamanho: int = 6) -> None:
    # Avalia o impacto da profundidade do MinMax contra MCTS fixo.

    print("\n" + "="*65)
    print("  ANÁLISE: Impacto da Profundidade no MinMax")
    print("="*65)

    for profundidade in profundidades:
        vitorias_mm = 0
        tempo_total = []

        for i in range(n_jogos):
            mm = MinMaxAgente(jogador=1 if i%2==0 else -1, profundidade=profundidade)
            mcts = MCTSAgente(jogador=-1 if i%2==0 else 1,
                             maximo_simulacoes=300, tipo_rollout="heuristico_cantos")

            if mm.jogador == 1:
                r = executa_jogo(mm, mcts, tamanho=tamanho)
                tempo_total.extend(r.tempo_p1)
            else:
                r = executa_jogo(mcts, mm, tamanho=tamanho)
                tempo_total.extend(r.tempo_p2)

            if r.vencedor == mm.jogador:
                vitorias_mm += 1

        media_ms = statistics.mean(tempo_total)*1000 if tempo_total else 0
        print(f"  Profundidade {profundidade}: MinMax vence {vitorias_mm}/{n_jogos} "
              f"({100*vitorias_mm/n_jogos:.0f}%)  "
              f"avg_tempo={media_ms:.1f}ms/jogada")


# Análise de impacto de simulações (MCTS)

def analise_impacto_simulacoes(lista_simulacoes: List[int] = [100, 300, 600],
                         n_jogos: int = 6, tamanho: int = 6) -> None:
    # Avalia o impacto do número de simulações do MCTS contra MinMax fixo.

    print("\n" + "="*65)
    print("  ANÁLISE: Impacto do Nº de Simulações no MCTS")
    print("="*65)

    for sims in lista_simulacoes:
        vitorias_mcts = 0
        tempo_total = []

        for i in range(n_jogos):
            mcts = MCTSAgente(jogador=1 if i%2==0 else -1,
                             maximo_simulacoes=sims, tipo_rollout="heuristico_cantos")
            mm = MinMaxAgente(jogador=-1 if i%2==0 else 1, profundidade=4)

            if mcts.jogador == 1:
                r = executa_jogo(mcts, mm, tamanho=tamanho)
                tempo_total.extend(r.tempo_p1)
            else:
                r = executa_jogo(mm, mcts, tamanho=tamanho)
                tempo_total.extend(r.tempo_p2)

            if r.vencedor == mcts.jogador:
                vitorias_mcts += 1

        media_ms = statistics.mean(tempo_total)*1000 if tempo_total else 0
        print(f"  Simulações {sims:4d}: MCTS vence {vitorias_mcts}/{n_jogos} "
              f"({100*vitorias_mcts/n_jogos:.0f}%)  "
              f"avg_tempo={media_ms:.1f}ms/jogada")



# Demo de uma única partida com verboso

def demo_unica_partida(verboso_mm: bool = True) -> None:
    # Executa uma partida de demonstração exibindo a árvore MinMax.

    print("\n" + "="*65)
    print("  DEMO: Partida com árvore Min-Max visível")
    print("="*65 + "\n")

    mm   = MinMaxAgente(jogador=1,  profundidade=3, verboso=verboso_mm)
    mcts = MCTSAgente(jogador=-1, maximo_simulacoes=200, tipo_rollout="heuristico_cantos")

    jogo = Othello(tamanho=6)
    print("Estado inicial:")
    print(jogo)
    print()

    numero_movimentos = 0
    passes_consecutivos = 0

    while not jogo.verifica_fim():
        jogador = jogo.jogador_atual
        agente  = mm if jogador == 1 else mcts

        movimento = agente.escolhe_movimento(jogo)

        if movimento is None:
            passes_consecutivos += 1
            print(f"  Jogador {'MM' if jogador==1 else 'MCTS'} passa a vez.")
            if passes_consecutivos >= 2:
                break
            jogo.troca_jogador()
            continue

        passes_consecutivos = 0
        jogo.executa_jogada(movimento[0], movimento[1], jogador)
        numero_movimentos += 1
        jogo.troca_jogador()

        label = "MinMax" if jogador == 1 else "MCTS  "
        print(f"\n  [{numero_movimentos:02d}] {label} (jogador {jogador:+d}) -> {movimento}")
        print(jogo)

    scores = jogo.get_dicionario_score()
    vencedor = jogo.get_vencedor()
    print(f"\n  RESULTADO: ●(MM)={scores[1]}  ○(MCTS)={scores[-1]}  "
          f"Vencedor={'MinMax' if vencedor==1 else ('MCTS' if vencedor==-1 else 'Empate')}")


# Entry point

if __name__ == "__main__":
    # 1. Demo de uma partida com árvore visível
    demo_unica_partida(verboso_mm=True)

    # 2. Torneio principal (10 partidas)
    run_tournament(n_jogos=10, tamanho=6, minmax_profundidade=4,
                   mcts_sims=300, bonus_canto=False)

    # 3. Análise de profundidade do MinMax
    analise_impacto_profundidade(profundidades=[3, 4, 5], n_jogos=6)

    # 4. Análise de simulações do MCTS
    analise_impacto_simulacoes(lista_simulacoes=[100, 300, 600], n_jogos=6)
