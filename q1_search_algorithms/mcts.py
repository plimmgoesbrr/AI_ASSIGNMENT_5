"""
Problem 1d: Monte-Carlo Tree Search (MCTS)
==========================================
MCTS estimates the value of moves through random simulations (rollouts)
rather than full game-tree expansion. Each iteration consists of 4 phases:

  1. SELECTION   — descend the tree using the UCB1 formula until a node
                   with untried moves (or a terminal node) is reached.
  2. EXPANSION   — add one new child for an untried move.
  3. SIMULATION  — play out the game from the new child with random moves
                   until it terminates.
  4. BACKPROPAGATION — propagate the result back up, updating visit and
                       win counts on every node along the path.

After N iterations, the move whose child has the highest visit count
(the "robust" choice) is returned.

UCB1: argmax_a [ Q(a)/N(a) + c * sqrt( ln(N(parent)) / N(a) ) ]
"""

import math
import random
from tic_tac_toe import TicTacToe


class MCTSNode:
    """A node in the Monte-Carlo search tree."""

    def __init__(self, state: TicTacToe, parent=None, move=None,
                 player_just_moved=None):
        self.state = state
        self.parent = parent
        self.move = move  # move that led to this state
        self.player_just_moved = player_just_moved  # who made that move
        self.children = []
        self.wins = 0.0  # cumulative reward (from the perspective of
                         # `player_just_moved`)
        self.visits = 0
        self.untried_moves = state.legal_moves()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c=1.4):
        """Select the child with the highest UCB1 value."""
        return max(
            self.children,
            key=lambda ch: (ch.wins / ch.visits) +
                           c * math.sqrt(math.log(self.visits) / ch.visits)
        )

    def expand(self):
        """Pop one untried move, create a child node, and return it."""
        move = self.untried_moves.pop()
        # Player about to move is self.state.to_move
        player = self.state.to_move
        child_state = self.state.make_move(move, player)
        child = MCTSNode(child_state, parent=self, move=move,
                         player_just_moved=player)
        self.children.append(child)
        return child


def rollout(state: TicTacToe) -> str:
    """Random playout from `state` until the game ends. Returns winner."""
    current = state
    while not current.is_terminal():
        move = random.choice(current.legal_moves())
        current = current.make_move(move, current.to_move)
    return current.winner()


def mcts(root_state: TicTacToe, iterations: int = 1000, c: float = 1.4):
    """Run MCTS for `iterations` rollouts and return the best move."""
    root = MCTSNode(root_state)

    for _ in range(iterations):
        node = root

        # 1. Selection — descend using UCB1 while the node is fully expanded
        #    and non-terminal
        while node.is_fully_expanded() and not node.state.is_terminal():
            node = node.best_child(c)

        # 2. Expansion — add a child if we can
        if not node.state.is_terminal():
            node = node.expand()

        # 3. Simulation — random rollout from the new node
        winner = rollout(node.state)

        # 4. Backpropagation — update statistics along the path
        while node is not None:
            node.visits += 1
            if winner == 'draw':
                node.wins += 0.5
            elif winner == node.player_just_moved:
                node.wins += 1.0
            # else: 0 (loss for the player who made `node.move`)
            node = node.parent

    # Return the move whose child has the most visits (robust choice)
    best = max(root.children, key=lambda ch: ch.visits)
    return best.move, best.visits, best.wins


# -------------------- TEST CASES --------------------
def run_tests():
    random.seed(42)  # reproducible test runs
    print("=" * 60)
    print("MONTE-CARLO TREE SEARCH — TEST CASES")
    print("=" * 60)

    # Test 1: MCTS should find an immediate winning move
    print("\nTest 1: X to move, immediate winning move")
    board = [['X', 'X', ' '],
             ['O', 'O', ' '],
             [' ', ' ', ' ']]
    state = TicTacToe(board, 'X')
    state.display()
    move, visits, wins = mcts(state, iterations=500)
    print(f"MCTS chose: {move}  | visits: {visits}  | wins: {wins}")
    assert move == (0, 2), f"Expected (0,2) but got {move}"
    print("PASSED")

    # Test 2: blocking move
    print("\nTest 2: X to move, must block O")
    board = [['O', 'O', ' '],
             ['X', ' ', ' '],
             [' ', ' ', 'X']]
    state = TicTacToe(board, 'X')
    state.display()
    move, visits, wins = mcts(state, iterations=1000)
    print(f"MCTS chose: {move}  | visits: {visits}  | wins: {wins}")
    assert move == (0, 2), f"Expected (0,2) but got {move}"
    print("PASSED")

    # Test 3: MCTS vs Minimax on empty board — MCTS shouldn't lose
    print("\nTest 3: Empty board — MCTS plays a reasonable opening move")
    state = TicTacToe([[' '] * 3 for _ in range(3)], 'X')
    move, visits, wins = mcts(state, iterations=2000)
    print(f"MCTS chose: {move}  | visits: {visits}  | wins: {wins}")
    # Center or any corner are optimal openings
    assert move in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
    print("PASSED")

    print("\nAll MCTS tests PASSED.")


if __name__ == "__main__":
    run_tests()
