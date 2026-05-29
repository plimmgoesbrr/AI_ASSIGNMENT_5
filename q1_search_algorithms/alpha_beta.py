"""
Problem 1b: Alpha-Beta Search Algorithm
=======================================
Alpha-Beta pruning is an optimization of minimax that maintains two values:
  alpha = best value MAX can guarantee so far on this path
  beta  = best value MIN can guarantee so far on this path

If at any point alpha >= beta, the remaining children can be pruned because
the opponent already has a better option on a previously visited branch.
The final answer is identical to minimax — pruning only saves work.
"""

import math
from tic_tac_toe import TicTacToe

nodes_explored = 0


def alpha_beta(state: TicTacToe, alpha: float, beta: float,
               is_maximizing: bool) -> int:
    """Alpha-Beta search; returns the minimax value of `state`."""
    global nodes_explored
    nodes_explored += 1

    winner = state.winner()
    if winner is not None:
        if winner == 'X':
            return 1
        elif winner == 'O':
            return -1
        else:
            return 0

    if is_maximizing:
        value = -math.inf
        for move in state.legal_moves():
            child = state.make_move(move, 'X')
            value = max(value, alpha_beta(child, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # beta cutoff — MIN won't allow MAX to reach here
        return value
    else:
        value = math.inf
        for move in state.legal_moves():
            child = state.make_move(move, 'O')
            value = min(value, alpha_beta(child, alpha, beta, True))
            beta = min(beta, value)
            if alpha >= beta:
                break  # alpha cutoff
        return value


def best_move_alpha_beta(state: TicTacToe, player: str):
    """Return the best (row, col) move for `player` using Alpha-Beta."""
    global nodes_explored
    nodes_explored = 0

    is_max = (player == 'X')
    best_value = -math.inf if is_max else math.inf
    best_action = None
    alpha, beta = -math.inf, math.inf

    for move in state.legal_moves():
        child = state.make_move(move, player)
        value = alpha_beta(child, alpha, beta, not is_max)
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
    print("ALPHA-BETA SEARCH — TEST CASES")
    print("=" * 60)

    # Test 1: identical result to minimax — immediate win
    print("\nTest 1: X to move, immediate winning move")
    board = [['X', 'X', ' '],
             ['O', 'O', ' '],
             [' ', ' ', ' ']]
    state = TicTacToe(board, 'X')
    state.display()
    move, value, nodes = best_move_alpha_beta(state, 'X')
    print(f"Best move for X: {move}  | Value: {value}  | Nodes: {nodes}")
    assert move == (0, 2) and value == 1
    print("PASSED")

    # Test 2: blocking move
    print("\nTest 2: X to move, must block O")
    board = [['O', 'O', ' '],
             ['X', ' ', ' '],
             [' ', ' ', 'X']]
    state = TicTacToe(board, 'X')
    state.display()
    move, value, nodes = best_move_alpha_beta(state, 'X')
    print(f"Best move for X: {move}  | Value: {value}  | Nodes: {nodes}")
    assert move == (0, 2)
    print("PASSED")

    # Test 3: empty board — alpha-beta should explore far fewer nodes
    # than plain minimax.
    print("\nTest 3: Empty board — checking pruning savings")
    state = TicTacToe([[' '] * 3 for _ in range(3)], 'X')
    move, value, nodes_ab = best_move_alpha_beta(state, 'X')
    print(f"Best move for X: {move}  | Value: {value}  | "
          f"Alpha-Beta nodes: {nodes_ab}")
    assert value == 0  # optimal play on empty board is a draw

    # Comparison with plain minimax
    from minimax import best_move_minimax
    _, _, nodes_mm = best_move_minimax(state, 'X')
    print(f"Minimax explored {nodes_mm} nodes; Alpha-Beta explored "
          f"{nodes_ab} nodes  (saved {nodes_mm - nodes_ab})")
    assert nodes_ab < nodes_mm, "Alpha-Beta should explore fewer nodes"
    print("PASSED — pruning works")

    print("\nAll Alpha-Beta tests PASSED.")


if __name__ == "__main__":
    run_tests()
