# Othello (Reversi) - Trabalho 2

Implementação do jogo **Othello** (também conhecido como Reversi) em Python, desenvolvida como trabalho da disciplina de Inteligência Artificial.

## Sobre o jogo

Othello é um jogo de tabuleiro para dois jogadores em um grid 8×8. O objetivo é ter mais peças da sua cor no tabuleiro ao final do jogo. Um jogador captura as peças do oponente ao flanqueá-las em linha reta (horizontal, vertical ou diagonal).

- `P` = Peça Preta (Jogador 1)
- `B` = Peça Branca (Jogador 2)
- `.` = Célula vazia

## Funcionalidades implementadas

- `criar_tabuleiro()` — cria o tabuleiro 8×8 com as 4 peças iniciais
- `imprimir_tabuleiro(tab)` — exibe o tabuleiro no terminal
- `jogada_validoa(tab, linha, col, jogador)` — verifica se uma jogada é válida
- `jogadas_validoas(tab, jogador)` — retorna todas as jogadas válidas para um jogador
- `aplicar_jogada(tab, linha, col, jogador)` — aplica uma jogada e vira as peças capturadas
- `verificar_fim(tab)` — verifica se o jogo acabou
- `contar_pecas(tab)` — conta as peças de cada jogador no tabuleiro

## Como executar

```bash
python othello.py
```

## Requisitos

- Python 3.x
