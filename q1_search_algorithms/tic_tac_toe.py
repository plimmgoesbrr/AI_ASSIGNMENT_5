"""
Tic-Tac-Toe game state — shared by Minimax, Alpha-Beta, Heuristic Alpha-Beta,
and MCTS implementations in Q1.
"""

import copy


class TicTacToe:
    """A simple immutable-style 3x3 Tic-Tac-Toe board.

    `make_move()` returns a new board (does NOT mutate the original) — this
    keeps recursive search clean.
    """

    def __init__(self, board=None, to_move='X'):
        if board is None:
            board = [[' '] * 3 for _ in range(3)]
        self.board = board
        self.to_move = to_move  # whose turn it is

    def legal_moves(self):
        """Return list of (row, col) tuples for empty squares."""
        return [(r, c) for r in range(3) for c in range(3)
                if self.board[r][c] == ' ']

    def make_move(self, move, player):
        """Return a new TicTacToe with `player`'s mark placed at `move`."""
        new_board = copy.deepcopy(self.board)
        new_board[move[0]][move[1]] = player
        next_player = 'O' if player == 'X' else 'X'
        return TicTacToe(new_board, next_player)

    def winner(self):
        """Return 'X', 'O', 'draw', or None (game not over)."""
        b = self.board
        lines = []
        # Rows and columns
        for i in range(3):
            lines.append([b[i][0], b[i][1], b[i][2]])
            lines.append([b[0][i], b[1][i], b[2][i]])
        # Diagonals
        lines.append([b[0][0], b[1][1], b[2][2]])
        lines.append([b[0][2], b[1][1], b[2][0]])

        for line in lines:
            if line[0] != ' ' and line.count(line[0]) == 3:
                return line[0]

        # Check for draw (no empty squares left)
        if not self.legal_moves():
            return 'draw'
        return None

    def is_terminal(self):
        return self.winner() is not None

    def display(self):
        print()
        for r in range(3):
            print(" " + " | ".join(self.board[r]))
            if r < 2:
                print("---+---+---")
        print()
