# Define 'P' como 1
P = 1
# Define 'B' como 2
B = 2

def criar_tabuleiro(N = 8):
    tab = []
    for i in range(N):
        linha = [0] * N
        tab.append(linha)

    # posições centrais com as 4 peças iniciais
    tab[N//2-1][N//2-1] = P    # preto
    tab[N//2-1][N//2] = B       # branco
    tab[N//2][N//2-1] = B       # branco
    tab[N//2][N//2] = P         # preto

    return tab

def imprimir_tabuleiro(tab):
    N = len(tab)
    print("  " + " ".join(str(i) for i in range(N)))

    for i in range(N):
        print(i, end=" ")
        for j in range(N):
            if tab[i][j] == 0:
                print(".", end=" ")   # vazio
            elif tab[i][j] == P:
                print("P", end=" ")   # preto
            else:
                print("B", end=" ")   # branco
        print()                   

def jogada_valida(tab, linha, col, jogador):
    if tab[linha][col] != 0:          # célula ocupada
        return False
    N = len(tab)
    if not (0 <= linha < N and 0 <= col < N):  # fora do tabuleiro
        return False
    oponente = B if jogador == P else P
    direcoes = [(-1,-1), (-1,0), (-1,1),
                ( 0,-1),         ( 0,1),
                ( 1,-1), ( 1,0), ( 1,1)]
    
    for dr, dc in direcoes:
        r, c = linha + dr, col + dc   # primeiro passo na direção
        encontrou_oponente = False

        while 0 <= r < N and 0 <= c < N:   # enquanto dentro do tabuleiro
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
    N = len(tab)
    for i in range(N):
        for j in range(N):
            if jogada_valida(tab, i, j, jogador):
                resultados.append((i, j))

    return resultados

def aplicar_jogada(tab, linha, col, jogador):
    tab[linha][col] = jogador         # coloca a peça
    oponente = B if jogador == P else P
    direcoes = [(-1,-1), (-1,0), (-1,1),
                ( 0,-1),         ( 0,1),
                ( 1,-1), ( 1,0), ( 1,1)]
    N = len(tab)
    for dr, dc in direcoes:
        r, c = linha + dr, col + dc
        celulas_virar = []

        while 0 <= r < N and 0 <= c < N:
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
    if jogadas_validas(tab, P) or jogadas_validas(tab, B):
        return False    # ainda tem jogadas possiveis
    return True        

def contar_pecas(tab):
    contagem = {P: 0, B: 0}
    N = len(tab)
    for i in range(N):
        for j in range(N):
            if tab[i][j] == P:
                contagem[P] += 1
            elif tab[i][j] == B:
                contagem[B] += 1
    return contagem

def imprimir_contagem(contagem):
    print(f"Preto (P): {contagem[P]} peças")
    print(f"Branco (B): {contagem[B]} peças")

tab = criar_tabuleiro(8)
imprimir_tabuleiro(tab)
print(jogadas_validas(tab, P))  # [(2,3), (3,2), (4,5), (5,4)]
print(jogadas_validas(tab, B))  # [(2,4), (3,5), (4,2), (5,3)]

imprimir_contagem(contar_pecas(tab))  # Preto (P): 2 peças, Branco (B): 2 peças
print(verificar_fim(tab))  # False — jogo recém começou