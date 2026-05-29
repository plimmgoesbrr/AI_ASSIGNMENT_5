"""
Q1 master test runner — exercises all four algorithms and compares the
number of nodes each one explores.
"""

import time
import random
from tic_tac_toe import TicTacToe
import minimax
import alpha_beta
import heuristic_alpha_beta
import mcts


def main():
    print("\n" + "#" * 60)
    print("# Q1 — RUNNING ALL FOUR SEARCH ALGORITHMS")
    print("#" * 60)

    minimax.run_tests()
    print()
    alpha_beta.run_tests()
    print()
    heuristic_alpha_beta.run_tests()
    print()
    mcts.run_tests()

    # Performance comparison on an empty board
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON — Empty board, X to move")
    print("=" * 60)
    state = TicTacToe([[' '] * 3 for _ in range(3)], 'X')

    t0 = time.time()
    mv, val, n_mm = minimax.best_move_minimax(state, 'X')
    t_mm = time.time() - t0
    print(f"Minimax            : move={mv}  value={val}  "
          f"nodes={n_mm:<6d} time={t_mm*1000:.1f} ms")

    t0 = time.time()
    mv, val, n_ab = alpha_beta.best_move_alpha_beta(state, 'X')
    t_ab = time.time() - t0
    print(f"Alpha-Beta         : move={mv}  value={val}  "
          f"nodes={n_ab:<6d} time={t_ab*1000:.1f} ms")

    t0 = time.time()
    mv, val, n_h = heuristic_alpha_beta.best_move_heuristic(
        state, 'X', depth=3)
    t_h = time.time() - t0
    print(f"Heuristic AB (d=3) : move={mv}  value={val}  "
          f"nodes={n_h:<6d} time={t_h*1000:.1f} ms")

    random.seed(1)
    t0 = time.time()
    mv, visits, wins = mcts.mcts(state, iterations=1000)
    t_mc = time.time() - t0
    print(f"MCTS (1000 iters)  : move={mv}  visits={visits}  "
          f"wins={wins:<5.1f}  time={t_mc*1000:.1f} ms")

    print("\nAll Q1 algorithms ran successfully.\n")


if __name__ == "__main__":
    main()
