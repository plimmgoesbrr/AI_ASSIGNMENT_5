"""
Problem 1c: Heuristic Alpha-Beta Search
=======================================
Plain minimax / alpha-beta explore down to terminal nodes. For larger games
this is infeasible. Heuristic alpha-beta cuts off the search at a fixed depth
and applies a heuristic evaluation function to the non-terminal leaves.

Heuristic used here for Tic-Tac-Toe (Russell & Norvig style):
  eval(s) = (number of rows/cols/diagonals open to X)
          - (number of rows/cols/diagonals open to O)

A line is "open to X" if it contains no O's; similarly for O. The score thus
estimates how many ways each player still has to win.
"""

import math
from tic_tac_toe import TicTacToe

nodes_explored = 0


def evaluate(state: TicTacToe) -> float:
    """Heuristic evaluation: difference in open lines."""
    winner = state.winner()
    if winner == 'X':
        return 1000
    if winner == 'O':
        return -1000
    if winner == 'draw':
        return 0

    b = state.board
    lines = []
    for i in range(3):
        lines.append([b[i][0], b[i][1], b[i][2]])
        lines.append([b[0][i], b[1][i], b[2][i]])
    lines.append([b[0][0], b[1][1], b[2][2]])
    lines.append([b[0][2], b[1][1], b[2][0]])

    x_open = sum(1 for line in lines if 'O' not in line)
    o_open = sum(1 for line in lines if 'X' not in line)
    return x_open - o_open


def heuristic_alpha_beta(state: TicTacToe, depth: int,
                         alpha: float, beta: float,
                         is_maximizing: bool) -> float:
    global nodes_explored
    nodes_explored += 1

    # Cutoff test: terminal OR depth limit reached
    if state.is_terminal() or depth == 0:
        return evaluate(state)

    if is_maximizing:
        value = -math.inf
        for move in state.legal_moves():
            child = state.make_move(move, 'X')
            value = max(value, heuristic_alpha_beta(
                child, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = math.inf
        for move in state.legal_moves():
            child = state.make_move(move, 'O')
            value = min(value, heuristic_alpha_beta(
                child, depth - 1, alpha, beta, True))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def best_move_heuristic(state: TicTacToe, player: str, depth: int = 2):
    """Return the best move using heuristic alpha-beta limited to `depth`."""
    global nodes_explored
    nodes_explored = 0

    is_max = (player == 'X')
    best_value = -math.inf if is_max else math.inf
    best_action = None
    alpha, beta = -math.inf, math.inf

    for move in state.legal_moves():
        child = state.make_move(move, player)
        value = heuristic_alpha_beta(child, depth - 1, alpha, beta, not is_max)
        if is_max:
            if value > best_value:
                best_value, best_action = value, move
            alpha = max(alpha, value)
        else:
            if value < best_value:
                best_value, best_action = value, move
            beta = min(beta, value)

    return best_action, best_value, nodes_explored


# -------------------- TEST CASES --------------------
def run_tests():
    print("=" * 60)
    print("HEURISTIC ALPHA-BETA — TEST CASES")
    print("=" * 60)

    # Test 1: shallow depth still finds immediate win
    print("\nTest 1: depth=1, immediate win available")
    board = [['X', 'X', ' '],
             ['O', 'O', ' '],
             [' ', ' ', ' ']]
    state = TicTacToe(board, 'X')
    state.display()
    move, value, nodes = best_move_heuristic(state, 'X', depth=1)
    print(f"Best move for X: {move}  | Value: {value}  | Nodes: {nodes}")
    assert move == (0, 2)
    print("PASSED")

    # Test 2: depth=2 should still block O's threat
    print("\nTest 2: depth=2, must block O")
    board = [['O', 'O', ' '],
             ['X', ' ', ' '],
             [' ', ' ', 'X']]
    state = TicTacToe(board, 'X')
    state.display()
    move, value, nodes = best_move_heuristic(state, 'X', depth=2)
    print(f"Best move for X: {move}  | Value: {value}  | Nodes: {nodes}")
    assert move == (0, 2)
    print("PASSED")

    # Test 3: heuristic eval works on an empty board
    print("\nTest 3: empty board, depth=3 — should expand far fewer nodes")
    state = TicTacToe([[' '] * 3 for _ in range(3)], 'X')
    move, value, nodes = best_move_heuristic(state, 'X', depth=3)
    print(f"Best move for X: {move}  | Heuristic value: {value}  | "
          f"Nodes: {nodes}")
    print("(Center or corner is typical — both are optimal openings.)")
    assert move in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
    print("PASSED")

    print("\nAll Heuristic Alpha-Beta tests PASSED.")


if __name__ == "__main__":
    run_tests()
