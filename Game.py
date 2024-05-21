import copy

import numpy as np

ROWS = 6
COLS = 7

def create_board():
    return np.zeros((ROWS, COLS), dtype=int)

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[ROWS-1][col] == 0

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
    if col <= COLS-4:
        if all([board[row][col+i] == piece for i in range(4)]):
            return True
    if 1 <= col <= COLS-3:
        if all([board[row][col-1+i] == piece for i in range(4)]):
            return True
    if 2 <= col <= COLS-2:
        if all([board[row][col-2+i] == piece for i in range(4)]):
            return True
    if 3 <= col <= COLS-1:
        if all([board[row][col-3+i] == piece for i in range(4)]):
            return True



    # Check vertical locations
    if row <= ROWS-4:
        if all([board[row+i][col] == piece for i in range(4)]):
            return True
    if 1 <= row <= ROWS-3:
        if all([board[row-1+i][col] == piece for i in range(4)]):
            return True
    if 2 <= row <= ROWS-2:
        if all([board[row-2+i][col] == piece for i in range(4)]):
            return True
    if 3 <= row <= ROWS-1:
        if all([board[row-3+i][col] == piece for i in range(4)]):
            return True


    # Check positively sloped diagonals
    if row <= ROWS-4 and col <= COLS-4:
        if all([board[row+i][col+i] == piece for i in range(4)]):
            return True
    if 1 <= row <= ROWS-3 and 1 <= col <= COLS-3:
        if all([board[row-1+i][col-1+i] == piece for i in range(4)]):
            return True
    if 2 <= row <= ROWS-2 and 2 <= col <= COLS-2:
        if all([board[row-2+i][col-2+i] == piece for i in range(4)]):
            return True
    if 3 <= row <= ROWS-1 and 3 <= col <= COLS-1:
        if all([board[row-3+i][col-3+i] == piece for i in range(4)]):
            return True

    # Check negatively sloped diagonals
    if 3 <= row <= ROWS-1 and col <= COLS-4:
        if all([board[row-i][col+i] == piece for i in range(4)]):
            return True
    if 2 <= row <= ROWS-2 and 1 <= col <= COLS-3:
        if all([board[row+1-i][col-1+i] == piece for i in range(4)]):
            return True
    if 1 <= row <= ROWS-3 and 2 <= col <= COLS-2:
        if all([board[row+2-i][col-2+i] == piece for i in range(4)]):
            return True
    if row <= ROWS-4 and 3 <= col <= COLS-1:
        if all([board[row+3-i][col-3+i] == piece for i in range(4)]):
            return True
    return False

def print_board(board):
    print(np.flip(board, 0))

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
        for r in range(ROWS-3):
            element = board[r][c], board[r+1][c], board[r + 2][c], board[r + 3][c]
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

def minimax(board, maxProfondeur, player):
    if player == 1:
        eval, action = joueurMax(board, maxProfondeur, 0, 0)
    else:
        eval, action = joueurMin(board, maxProfondeur, 0, 0)
    return action

def joueurMax(n,p, row, col):
    if p == 0 or winning_move(n, row, col):
        return eval_fonction(n), None
    u = float('-inf')
    action = None
    for c in range(COLS):
        n_deepcopy = copy.deepcopy(n)
        r = get_next_open_row(n_deepcopy, c)
        if r is not None:
            n_deepcopy[r][c] = 1
            eval = joueurMin(n_deepcopy, p - 1, r, c)[0]
            if eval > u:
                u = eval
                action = c
    return u, action

def joueurMin(n,p, row, col):
    if p == 0 or winning_move(n, row, col):
        return eval_fonction(n), None
    u = float('inf')
    action = None
    for c in range(COLS):
        n_deepcopy = copy.deepcopy(n)
        r = get_next_open_row(n_deepcopy, c)
        if r is not None:
            n_deepcopy[r][c] = 2
            eval = joueurMax(n_deepcopy, p-1, r, c)[0]
            if eval < u:
                u = eval
                action = c
    return u, action

##Algo Alpha-Beta

def alphabeta(board, maxProfondeur, player):
    if player == 1:
        eval, action = joueurMaxAlphaBeta(board, maxProfondeur, float("-inf"), float("inf"), 0, 0)
    else:
        eval, action = joueurMinAlphaBeta(board, maxProfondeur, float("inf"), float("-inf"), 0, 0)
    return action

def joueurMaxAlphaBeta(n,p, alpha, beta, row, col):
    if p == 0 or winning_move(n, row, col):
        return eval_fonction(n), None
    u = float('-inf')
    action = None
    for c in range(COLS):
        n_deepcopy = copy.deepcopy(n)
        r = get_next_open_row(n_deepcopy, c)
        if r is not None:
            n_deepcopy[r][c] = 1
            eval = joueurMinAlphaBeta(n_deepcopy, p - 1,alpha, beta, r, c)[0]
            if eval > u:
                u = eval
                action = c
            alpha = max(alpha, u)
            if alpha >= beta:
                return u, action
    return u, action

def joueurMinAlphaBeta(n,p,alpha, beta, row, col):
    if p == 0 or winning_move(n, row, col):
        return eval_fonction(n), None
    u = float('inf')
    action = None
    for c in range(COLS):
        n_deepcopy = copy.deepcopy(n)
        r = get_next_open_row(n_deepcopy, c)
        if r is not None:
            n_deepcopy[r][c] = 2
            eval = joueurMaxAlphaBeta(n, p-1,alpha, beta, r, c)[0]
            if eval < u:
                u = eval
                action = c
            beta = min(beta, u)
            if beta <= alpha:
                return u, action
    return u, action

board = create_board()
game_over = False
turn = 0

while not game_over:
    # Ask for Player 1 input
    if turn == 0:
        #col = int(input("Player 1, choose a column (0-6):"))
        col = alphabeta(board, 4, 1)
        print(f"Player 1, choose a column (0-6): {col}")
        row = get_next_open_row(board, col)
        drop_piece(board, row, col, 1)
        if winning_move(board,  row, col):
            print("Player 1 wins!")
            game_over = True

    # Ask for Player 2 input
    else:
        col = alphabeta(board, 3, 2)
        print(f"Player 2, choose a column (0-6): {col}")
        row = get_next_open_row(board, col)
        drop_piece(board, row, col, 2)
        if winning_move(board,  row, col):
            print("Player 2 wins!")
            game_over = True


        # col = int(input("Player 2, choose a column (0-6):"))
        #
        # if is_valid_location(board, col):
        #     row = get_next_open_row(board, col)
        #     drop_piece(board, row, col, 2)
        #
        #     if winning_move(board,  row, col):
        #         print("Player 2 wins!")
        #         game_over = True

    print_board(board)
    turn += 1
    turn %= 2
