def imprimir_tabuleiro(tab):
    print("  0 1 2 3 4 5 6 7")   

    for i in range(8):            
        print(i, end=" ")         
        for j in range(8):       
            if tab[i][j] == 0:
                print(".", end=" ")   # vazio
            elif tab[i][j] == 1:
                print("P", end=" ")   # preto
            else:
                print("B", end=" ")   # branco
        print()                   

def criar_tabuleiro():
    tab = []
    for i in range(8):
        linha = [0] * 8
        tab.append(linha)

    # posições centrais com as 4 peças iniciais
    tab[3][3] = 2    # branco
    tab[3][4] = 1    # preto
    tab[4][3] = 1    # preto
    tab[4][4] = 2    # branco

    return tab

def jogada_valida(tab, linha, col, jogador):
    if tab[linha][col] != 0:          # célula ocupada
        return False
    
    oponente = 2 if jogador == 1 else 1
    direcoes = [(-1,-1), (-1,0), (-1,1),
                ( 0,-1),         ( 0,1),
                ( 1,-1), ( 1,0), ( 1,1)]
    
    for dr, dc in direcoes:
        r, c = linha + dr, col + dc   # primeiro passo na direção
        encontrou_oponente = False
        
        while 0 <= r < 8 and 0 <= c < 8:   # enquanto dentro do tabuleiro
            if tab[r][c] == 0:              # vazio → para (caso 1)
                break
            elif tab[r][c] == oponente:     # oponente → continua (caso 2)
                encontrou_oponente = True
            elif tab[r][c] == jogador:      # própria peça
                if encontrou_oponente:      # tinha oponente antes → válido!
                    return True
                else:                       # sem oponente antes → caso 3
                    break
            r += dr
            c += dc
    
    return False

def jogadas_validas(tab, jogador):

    resultados = []

    for i in range(8):            
        for j in range(8):       
            if jogada_valida(tab, i, j, jogador):
                resultados.append((i, j))

    return resultados

def aplicar_jogada(tab, linha, col, jogador):
    tab[linha][col] = jogador         # coloca a peça
    oponente = 2 if jogador == 1 else 1
    direcoes = [(-1,-1), (-1,0), (-1,1),
                ( 0,-1),         ( 0,1),
                ( 1,-1), ( 1,0), ( 1,1)]

    for dr, dc in direcoes:
        r, c = linha + dr, col + dc
        celulas_virar = []

        while 0 <= r < 8 and 0 <= c < 8:
            if tab[r][c] == 0:                    # vazio → descarta
                break
            elif tab[r][c] == oponente:           # oponente → coleta
                celulas_virar.append((r, c))
            elif tab[r][c] == jogador:            # própria peça
                for pr, pc in celulas_virar:      # vira todas coletadas
                    tab[pr][pc] = jogador
                break
            r += dr
            c += dc

def verificar_fim(tab):
    if jogadas_validas(tab, 1) or jogadas_validas(tab, 2):
        return False    # ainda tem jogadas possiveis
    return True        

def contar_pecas(tab):
    contagem = {1: 0, 2: 0}
    for i in range(8):
        for j in range(8):
            if tab[i][j] == 1:
                contagem[1] += 1
            elif tab[i][j] == 2:
                contagem[2] += 1
    return contagem

tab = criar_tabuleiro()
print(contar_pecas(tab))  # {1: 2, 2: 2}
print(verificar_fim(tab))  # False — jogo recém começou