"""
Problem 1a: Minimax Search Algorithm
=====================================
Implement the Minimax search algorithm.

Minimax is a recursive algorithm used in two-player zero-sum games. It assumes
both players play optimally:
  - MAX tries to maximize the score
  - MIN tries to minimize the score
The algorithm explores the entire game tree to the terminal nodes, then
propagates values back up.

Demonstration: Tic-Tac-Toe (X = MAX, O = MIN)
"""

import math
from tic_tac_toe import TicTacToe

# Counter to track number of nodes explored (for comparison with alpha-beta)
nodes_explored = 0


def minimax(state: TicTacToe, is_maximizing: bool) -> int:
    """
    Pure minimax with no pruning.
    Returns the utility value (+1 = X wins, -1 = O wins, 0 = draw).
    """
    global nodes_explored
    nodes_explored += 1

    # Terminal test
    winner = state.winner()
    if winner is not None:
        if winner == 'X':
            return 1
        elif winner == 'O':
            return -1
        else:  # draw
            return 0

    if is_maximizing:
        best = -math.inf
        for move in state.legal_moves():
            child = state.make_move(move, 'X')
            value = minimax(child, False)
            best = max(best, value)
        return best
    else:
        best = math.inf
        for move in state.legal_moves():
            child = state.make_move(move, 'O')
            value = minimax(child, True)
            best = min(best, value)
        return best


def best_move_minimax(state: TicTacToe, player: str) -> tuple:
    """Return the (row, col) move that minimax recommends for `player`."""
    global nodes_explored
    nodes_explored = 0

    is_max = (player == 'X')
    best_value = -math.inf if is_max else math.inf
    best_action = None

    for move in state.legal_moves():
        child = state.make_move(move, player)
        value = minimax(child, not is_max)
        if is_max and value > best_value:
            best_value = value
            best_action = move
        elif not is_max and value < best_value:
            best_value = value
            best_action = move

    return best_action, best_value, nodes_explored


# -------------------- TEST CASES --------------------
def run_tests():
    print("=" * 60)
    print("MINIMAX ALGORITHM — TEST CASES")
    print("=" * 60)

    # Test 1: X can win immediately
    print("\nTest 1: X to move, immediate winning move available")
    board = [['X', 'X', ' '],
             ['O', 'O', ' '],
             [' ', ' ', ' ']]
    state = TicTacToe(board, 'X')
    state.display()
    move, value, nodes = best_move_minimax(state, 'X')
    print(f"Best move for X: {move}  | Value: {value}  | Nodes: {nodes}")
    assert move == (0, 2), f"Expected (0,2) but got {move}"
    assert value == 1
    print("PASSED")

    # Test 2: X must block O's winning threat
    print("\nTest 2: X to move, must block O")
    board = [['O', 'O', ' '],
             ['X', ' ', ' '],
             [' ', ' ', 'X']]
    state = TicTacToe(board, 'X')
    state.display()
    move, value, nodes = best_move_minimax(state, 'X')
    print(f"Best move for X: {move}  | Value: {value}  | Nodes: {nodes}")
    assert move == (0, 2), f"Expected (0,2) but got {move}"
    print("PASSED")

    # Test 3: Empty board — optimal play leads to a draw
    print("\nTest 3: Empty board — optimal play should give value 0 (draw)")
    state = TicTacToe([[' '] * 3 for _ in range(3)], 'X')
    move, value, nodes = best_move_minimax(state, 'X')
    print(f"Best move for X: {move}  | Value: {value}  | Nodes: {nodes}")
    assert value == 0, f"Expected 0 (draw) but got {value}"
    print("PASSED")

    print("\nAll Minimax tests PASSED.")


if __name__ == "__main__":
    run_tests()
