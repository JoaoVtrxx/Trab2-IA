from othello_jogo import Othello
from minmax_agente import MinMaxAgente
from mcts_agente import MCTSAgente

def demo_unica_partida(verboso_mm: bool = True) -> None:
    # Executa uma partida de demonstração exibindo a árvore MinMax.

    print("\n" + "="*65)
    print("  DEMO: Partida com árvore Min-Max visível")
    print("="*65 + "\n")

    mm   = MinMaxAgente(jogador=1,  profundidade=3, verboso=verboso_mm)
    mcts = MCTSAgente(jogador=-1, maximo_simulacoes=200, tipo_rollout="heuristica_cantos")

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
    print(f"\n  RESULTADO: ○(MM)={scores[1]}  ●(MCTS)={scores[-1]}  "
          f"Vencedor={'MinMax' if vencedor==1 else ('MCTS' if vencedor==-1 else 'Empate')}")

if __name__ == "__main__":
    # Demo de uma partida com árvore visível
    demo_unica_partida(verboso_mm=True)
