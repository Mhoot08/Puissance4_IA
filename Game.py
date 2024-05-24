import copy
import random
import numpy as np
import time
import networkx as nx
import matplotlib.pyplot as plt

ROWS = 6
COLS = 7



def create_board():
    return np.zeros((ROWS, COLS), dtype=int)


def drop_piece(board, row, col, piece):
    board[row][col] = piece


game_over = False


def is_valid_location(board, col):
    return board[ROWS - 1][col] == 0


def get_next_open_row(board, col):
    for r in range(ROWS):
        if board[r][col] == 0:
            return r
    return None


def winning_move(board, row, col):
    piece = board[row][col]
    if piece == 0:
        return False

    # Check horizontal locations
    if col <= COLS - 4:
        if all([board[row][col + i] == piece for i in range(4)]):
            return True
    if 1 <= col <= COLS - 3:
        if all([board[row][col - 1 + i] == piece for i in range(4)]):
            return True
    if 2 <= col <= COLS - 2:
        if all([board[row][col - 2 + i] == piece for i in range(4)]):
            return True
    if 3 <= col <= COLS - 1:
        if all([board[row][col - 3 + i] == piece for i in range(4)]):
            return True

    # Check vertical locations
    if row <= ROWS - 4:
        if all([board[row + i][col] == piece for i in range(4)]):
            return True
    if 1 <= row <= ROWS - 3:
        if all([board[row - 1 + i][col] == piece for i in range(4)]):
            return True
    if 2 <= row <= ROWS - 2:
        if all([board[row - 2 + i][col] == piece for i in range(4)]):
            return True
    if 3 <= row <= ROWS - 1:
        if all([board[row - 3 + i][col] == piece for i in range(4)]):
            return True

    # Check positively sloped diagonals
    if row <= ROWS - 4 and col <= COLS - 4:
        if all([board[row + i][col + i] == piece for i in range(4)]):
            return True
    if 1 <= row <= ROWS - 3 and 1 <= col <= COLS - 3:
        if all([board[row - 1 + i][col - 1 + i] == piece for i in range(4)]):
            return True
    if 2 <= row <= ROWS - 2 and 2 <= col <= COLS - 2:
        if all([board[row - 2 + i][col - 2 + i] == piece for i in range(4)]):
            return True
    if 3 <= row <= ROWS - 1 and 3 <= col <= COLS - 1:
        if all([board[row - 3 + i][col - 3 + i] == piece for i in range(4)]):
            return True

    # Check negatively sloped diagonals
    if 3 <= row <= ROWS - 1 and col <= COLS - 4:
        if all([board[row - i][col + i] == piece for i in range(4)]):
            return True
    if 2 <= row <= ROWS - 2 and 1 <= col <= COLS - 3:
        if all([board[row + 1 - i][col - 1 + i] == piece for i in range(4)]):
            return True
    if 1 <= row <= ROWS - 3 and 2 <= col <= COLS - 2:
        if all([board[row + 2 - i][col - 2 + i] == piece for i in range(4)]):
            return True
    if row <= ROWS - 4 and 3 <= col <= COLS - 1:
        if all([board[row + 3 - i][col - 3 + i] == piece for i in range(4)]):
            return True
    return False


def print_board(board):
    # Convertir les éléments 1 en 'X', les éléments 2 en 'O' et les autres en ' '
    board = np.where(board == 1, '\033[91mX\033[0m', np.where(board == 2, '\033[93mO\033[0m', ' '))

    # Créer une chaîne de caractères pour la grille avec des lignes de séparation
    grid = '\n' + '+---' * len(board[0]) + '+\n'
    for row in reversed(board):
        grid += '| ' + ' | '.join(row) + ' |\n' + '+---' * len(board[0]) + '+\n'

    # Afficher la grille
    print(grid)


def eval_fonction(board):
    score_player1 = eval_player(board, 1, 2)
    score_player0 = eval_player(board, 2, 1)
    return score_player1 - score_player0


def get_score(nb):
    match nb:
        case 0:
            return 0
        case 1:
            return 1
        case 2:
            return 5
        case 3:
            return 100
        case 4:
            return 1000


def eval_player(board, piece, enemy_piece):
    score_player = 0

    # Check horizontal locations
    for c in range(COLS - 3):
        for r in range(ROWS):
            element = board[r][c], board[r][c + 1], board[r][c + 2], board[r][c + 3]
            if element.count(enemy_piece) == 0:
                score_player += get_score(element.count(piece))

    # Check vertical locations
    for c in range(COLS):
        for r in range(ROWS - 3):
            element = board[r][c], board[r + 1][c], board[r + 2][c], board[r + 3][c]
            if element.count(enemy_piece) == 0:
                score_player += get_score(element.count(piece))

    # Check positively sloped diagonals
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            element = board[r][c], board[r + 1][c + 1], board[r + 2][c + 2], board[r + 3][c + 3] == piece
            if element.count(enemy_piece) == 0:
                score_player += get_score(element.count(piece))

    # Check negatively sloped diagonals
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            element = board[r][c], board[r - 1][c + 1], board[r - 2][c + 2], board[r - 3][
                                                                                 c + 3] == piece
            if element.count(enemy_piece) == 0:
                score_player += get_score(element.count(piece))
    return score_player


##Algo MiniMax
list_time_minimax = {"type" : ["minimax"], "profondeur" : [0,0],"1": [], "2": []}
def minimax(board, maxProfondeur, player):
    list_time_minimax["profondeur"][player-1] = maxProfondeur
    start_time = time.time()
    if player == 1:
        eval, action = joueurMax(board, maxProfondeur, 0, 0, player)
        end_time = time.time()
        list_time_minimax["1"].append(end_time - start_time)
    else:
        eval, action = joueurMin(board, maxProfondeur, 0, 0, player)
        end_time = time.time()
        list_time_minimax["2"].append(end_time - start_time)
    print(f"temps de calcul minimax:{end_time-start_time}")
    return action


def joueurMax(n, p, row, col, player):
    if p == 0 or winning_move(n, row, col):
        return eval_fonction(n), None
    u = float('-inf')
    action = None
    for c in range(COLS):
        n_deepcopy = copy.deepcopy(n)
        r = get_next_open_row(n_deepcopy, c)
        if r is not None:
            n_deepcopy[r][c] = 1
            eval = joueurMin(n_deepcopy, p - 1, r, c, player)[0]
            if eval > u:
                u = eval
                action = c
    return u, action


board = create_board()


def joueurMin(n, p, row, col, player):
    if p == 0 or winning_move(n, row, col):
        return eval_fonction(n), None
    u = float('inf')
    action = None
    for c in range(COLS):
        n_deepcopy = copy.deepcopy(n)
        r = get_next_open_row(n_deepcopy, c)
        if r is not None:
            n_deepcopy[r][c] = 2
            eval = joueurMax(n_deepcopy, p - 1, r, c, player)[0]
            if eval < u:
                u = eval
                action = c
    return u, action


##Algo Alpha-Beta
list_time_alphabeta = {"type" : ["alphabeta"], "profondeur" : [0,0], "1": [], "2": []}
def alphabeta(board, maxProfondeur, player):
    list_time_alphabeta["profondeur"][player-1] = maxProfondeur
    start_time = time.time()
    if player == 1:
        eval, action = joueurMaxAlphaBeta(board, maxProfondeur, float("-inf"), float("inf"), 0, 0, player)
        end_time = time.time()
        list_time_alphabeta["1"].append(end_time - start_time)
    else:
        eval, action = joueurMinAlphaBeta(board, maxProfondeur, float("-inf"), float("inf"), 0, 0, player)
        end_time = time.time()
        list_time_alphabeta["2"].append(end_time - start_time)
    print(f"temps de calcul alphabeta:{end_time - start_time}")
    return action


def joueurMaxAlphaBeta(n, p, alpha, beta, row, col, player):
    if p == 0 or winning_move(n, row, col):
        return eval_fonction(n), None
    u = float('-inf')
    action = None
    for c in range(COLS):
        n_deepcopy = copy.deepcopy(n)
        r = get_next_open_row(n_deepcopy, c)
        if r is not None:
            n_deepcopy[r][c] = 1
            eval = joueurMinAlphaBeta(n_deepcopy, p - 1, alpha, beta, r, c, player)[0]
            if eval > u:
                u = eval
                action = c
            alpha = max(alpha, u)
            if alpha >= beta:
                return u, action
    return u, action


def joueurMinAlphaBeta(n, p, alpha, beta, row, col, player):
    if p == 0 or winning_move(n, row, col):
        return eval_fonction(n), None
    u = float('inf')
    action = None
    for c in range(COLS):
        n_deepcopy = copy.deepcopy(n)
        r = get_next_open_row(n_deepcopy, c)
        if r is not None:
            n_deepcopy[r][c] = 2
            eval = joueurMaxAlphaBeta(n_deepcopy, p - 1, alpha, beta, r, c, player)[0]
            if eval < u:
                u = eval
                action = c
            beta = min(beta, u)
            if beta <= alpha:
                return u, action
    return u, action


## Algo MCTS
turn = 0


class Node:
    def __init__(self, board, piece, parent, row, col):
        self.piece = piece
        self.board = copy.deepcopy(board)
        self.row = row
        self.col = col
        self.colNotFull = self.computeColNotfullAndAddchield(board, parent)
        self.parent = parent
        self.children = []
        self.terminated = False
        self.isFullyExpanded = False
        self.visits = 1
        self.score = 0
        self.computeTerminated()
        # self.toString()

    def computeColNotfullAndAddchield(self, board, parent):
        if parent is not None:
            parent.add_child(self)
            colNotFull = copy.deepcopy(parent.colNotFull)
            if self.row == ROWS - 1:
                colNotFull.remove(self.col)
        else:
            colNotFull = []
            for c in range(COLS):
                if board[ROWS - 1][c] == 0:
                    colNotFull.append(c)
        return colNotFull

    def add_child(self, child):
        self.children.append(child)
        self.isFullyExpanded = len(self.children) == len(self.colNotFull)

    def computeTerminated(self):
        if self.colNotFull == [] or winning_move(self.board, self.row, self.col):
            self.terminated = True
            self.isFullyExpanded = True

    def toString(self):
        print(f"Piece: {self.piece}")
        print_board(self.board)


def defaultPolicy(v):
    while v.terminated is False:
        # Gerer quandd on retombe sur un même fils
        col = random.choice(v.colNotFull)
        n = copy.deepcopy(v.board)
        piece_inverse = 2 if v.piece == 1 else 1
        row = get_next_open_row(n, col)
        n[row][col] = piece_inverse
        v = Node(n, piece_inverse, v, row, col)
    return eval_fonction(v.board)


def backup(v, delta):
    while v is not None:
        v.visits += 1
        v.score += delta
        delta = -delta
        v = v.parent


def bestChild(v, c):
    max = float('-inf')
    max_child = None
    for child in v.children:
        temp = child.score / child.visits + c * np.sqrt(2 * np.log(v.visits) / child.visits)
        if temp > max:
            max = temp
            max_child = child
    return max_child


def uctsearch(board, piece, v0=None):
    if v0 == None:
        colNotFull = []
        for c in range(COLS):
            if board[ROWS - 1][c] == 0:
                colNotFull.append(c)
        v0 = Node(board, piece, None, 0, 0)
    for i in range(1000):
        v1 = treePolicy(v0)
        delta = defaultPolicy(v1)
        backup(v1, delta)
    return bestChild(v0, 0)


def treePolicy(v0):
    v = v0
    while not v.terminated:
        if not v.isFullyExpanded:
            return expand(v)
        else:
            # Deuxième argument = c dans BestChild
            v = bestChild(v, 1 / np.sqrt(2))
    return v


def expand(v):
    for c in v.colNotFull:
        if c not in [child.col for child in v.children]:
            n = copy.deepcopy(v.board)
            piece_inverse = 2 if v.piece == 1 else 1
            row = get_next_open_row(n, c)
            n[row][c] = piece_inverse
            child = Node(n, piece_inverse, v, row, c)
            return child


import matplotlib.pyplot as plt


def plot_from_dicts(*dicts):
    """
    Affiche plusieurs courbes à partir de dictionnaires contenant des listes de nombres.

    :param dicts: Une ou plusieurs dictionnaires contenant des listes de nombres à afficher.
    """
    plt.figure(figsize=(10, 5))

    # Liste des labels et des courbes correspondantes qui ont des données
    labels = []
    plots = []

    for d in dicts:
        label = d.get("type", ["Unknown"])[0]
        depths = d.get("profondeur", [])
        for i, key in enumerate(d.keys()):
            if key != "type" and key != "profondeur" and d[key]:
                depth_label = depths[0] if key == "1" else depths[1]
                labels.append(f'Joueur {key} - {label} - Profondeur : {depth_label}')
                plots.append(d[key])

    # Si au moins une courbe a des données, afficher le graphique et la légende
    if plots:
        for label, values in zip(labels, plots):
            plt.plot(values, marker='o', linestyle='-', label=label)

        plt.title("temps d'éxécution")
        plt.xlabel('Itération')
        plt.ylabel('Temps')
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print("Aucune donnée à afficher.")

def selectionner_joueur(numero):
    print(f"Sélectionnez le type pour le joueur {numero}:")
    print("1. Humain")
    print("2. MCTS")
    print("3. Minimax")
    print("4. AlphaBeta")
    choix = int(input("Votre choix: "))
    profondeur = None
    if choix == 3 or choix == 4:
        print(f"Sélectionnez la profondeur souhaité pour l'algorithme ")
        profondeur = int(input("Votre choix: "))
    return choix, profondeur


joueur1, profondeur1 = selectionner_joueur(1)
joueur2, profondeur2 = selectionner_joueur(2)

n = None
col_lastplayer = None
while not game_over:
    if turn == 0:
        choix = joueur1
        profondeur = profondeur1
    else:
        choix = joueur2
        profondeur = profondeur2

    if choix == 1:
        # joueur_humain
        col = int(input(f"Player {turn + 1}, choose a column (0-6):"))
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)

    elif choix == 2:
        # joueur_ucts
        if n == None:
            n = uctsearch(board, turn + 1)
        else:
            for child in n.children:
                if child.col == col_lastplayer:
                    n = uctsearch(board, turn + 1, child)

        col = n.col
        print(f"Player {turn + 1}, choose a column (0-6): {col}")
        row = get_next_open_row(board, col)


    elif choix == 3:
        # joueur_minimax
        col = minimax(board, profondeur, turn + 1)
        print(f"Player {turn + 1}, choose a column (0-6): {col}")
        row = get_next_open_row(board, col)

    elif choix == 4:
        # joueur_alphabeta
        col = alphabeta(board, profondeur, turn + 1)
        print(f"Player {turn + 1}, choose a column (0-6): {col}")
        row = get_next_open_row(board, col)

    else:
        print("erreur lors de la saisi des joueurs")
        exit()

    drop_piece(board, row, col, turn + 1)
    col_lastplayer = col
    if winning_move(board, row, col):
        print(f"Player {turn + 1} wins!")
        game_over = True

    print_board(board)
    turn += 1
    turn %= 2

plot_from_dicts(list_time_minimax, list_time_alphabeta)
