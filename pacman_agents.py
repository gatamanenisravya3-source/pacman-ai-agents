"""
Simplified Pacman AI Agents
---------------------------
A compact, self-contained grid-world reimplementation of the classic
Berkeley CS188 Pacman project ideas: a Reflex Agent, an adversarial
Minimax Agent (with Alpha-Beta pruning), and a tabular Q-Learning Agent,
all playing on a small text-based maze with one ghost and several food
pellets.

Run:
    python pacman_agents.py
"""

import random
from dataclasses import dataclass, replace
from typing import List, Tuple

Position = Tuple[int, int]

MAZE = [
    "#############",
    "#P    #     #",
    "# ### # ### #",
    "#  .  .  .  #",
    "### # # # ###",
    "#   #.G.#   #",
    "# ####.####  #",
    "#   .  .  .  #",
    "#############",
]

MOVES = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1),
    "STOP": (0, 0),
}


@dataclass(frozen=True)
class State:
    pacman: Position
    ghost: Position
    food: frozenset
    score: int = 0


def parse_maze(maze: List[str]) -> State:
    walls = set()
    food = set()
    pacman = (1, 1)
    ghost = (5, 6)
    for r, row in enumerate(maze):
        for c, ch in enumerate(row):
            if ch == "P":
                pacman = (r, c)
            elif ch == "G":
                ghost = (r, c)
            elif ch == ".":
                food.add((r, c))
    return State(pacman=pacman, ghost=ghost, food=frozenset(food))


WALLS = {
    (r, c)
    for r, row in enumerate(MAZE)
    for c, ch in enumerate(row)
    if ch == "#"
}
HEIGHT = len(MAZE)
WIDTH = max(len(row) for row in MAZE)


def legal_moves(pos: Position) -> List[str]:
    moves = []
    for name, (dr, dc) in MOVES.items():
        if name == "STOP":
            continue
        nr, nc = pos[0] + dr, pos[1] + dc
        if (nr, nc) not in WALLS and 0 <= nr < HEIGHT and 0 <= nc < WIDTH:
            moves.append(name)
    return moves or ["STOP"]


def apply_move(pos: Position, action: str) -> Position:
    dr, dc = MOVES[action]
    nr, nc = pos[0] + dr, pos[1] + dc
    if (nr, nc) in WALLS or not (0 <= nr < HEIGHT and 0 <= nc < WIDTH):
        return pos
    return (nr, nc)


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step(state: State, pacman_action: str, ghost_action: str) -> State:
    new_pacman = apply_move(state.pacman, pacman_action)
    new_ghost = apply_move(state.ghost, ghost_action)

    score = state.score - 1  # time penalty, like the real Pacman scoring
    food = state.food
    if new_pacman in food:
        food = food - {new_pacman}
        score += 10

    if new_pacman == new_ghost:
        score -= 500  # caught by ghost

    return replace(state, pacman=new_pacman, ghost=new_ghost, food=food, score=score)


def is_terminal(state: State) -> bool:
    return len(state.food) == 0 or state.pacman == state.ghost


# ---------------------------------------------------------------------
# 1. Reflex Agent
# ---------------------------------------------------------------------
class ReflexAgent:
    """Greedy agent: moves toward the nearest food while staying away
    from the ghost (classic evaluation-function style agent)."""

    def get_action(self, state: State) -> str:
        best_action, best_score = "STOP", float("-inf")
        for action in legal_moves(state.pacman):
            next_pos = apply_move(state.pacman, action)
            score = self._evaluate(next_pos, state)
            if score > best_score:
                best_score, best_action = score, action
        return best_action

    def _evaluate(self, pos: Position, state: State) -> float:
        if not state.food:
            return 0
        food_dist = min(manhattan(pos, f) for f in state.food)
        ghost_dist = manhattan(pos, state.ghost)
        score = -food_dist
        if ghost_dist <= 1:
            score -= 1000  # strongly avoid the ghost
        else:
            score += ghost_dist * 0.1
        if pos in state.food:
            score += 50
        return score


# ---------------------------------------------------------------------
# 2. Minimax Agent (with Alpha-Beta pruning)
# ---------------------------------------------------------------------
class MinimaxAgent:
    """Pacman (maximizer) vs. Ghost (minimizer), searching a fixed depth."""

    def __init__(self, depth: int = 3):
        self.depth = depth

    def get_action(self, state: State) -> str:
        best_action, best_value = "STOP", float("-inf")
        for action in legal_moves(state.pacman):
            value = self._min_value(
                step(state, action, self._closest_ghost_move(state)),
                self.depth - 1,
                float("-inf"),
                float("inf"),
            )
            if value > best_value:
                best_value, best_action = value, action
        return best_action

    def _closest_ghost_move(self, state: State) -> str:
        # Ghost greedily chases Pacman; used to project one ply ahead.
        moves = legal_moves(state.ghost)
        return min(moves, key=lambda a: manhattan(apply_move(state.ghost, a), state.pacman))

    def _max_value(self, state: State, depth: int, alpha: float, beta: float) -> float:
        if depth == 0 or is_terminal(state):
            return self._evaluate(state)
        value = float("-inf")
        for action in legal_moves(state.pacman):
            ghost_action = self._closest_ghost_move(state)
            value = max(value, self._min_value(step(state, action, ghost_action), depth - 1, alpha, beta))
            if value > beta:
                return value
            alpha = max(alpha, value)
        return value

    def _min_value(self, state: State, depth: int, alpha: float, beta: float) -> float:
        if depth == 0 or is_terminal(state):
            return self._evaluate(state)
        value = float("inf")
        for action in legal_moves(state.ghost):
            value = min(value, self._max_value(replace(state, ghost=apply_move(state.ghost, action)), depth - 1, alpha, beta))
            if value < alpha:
                return value
            beta = min(beta, value)
        return value

    def _evaluate(self, state: State) -> float:
        if state.pacman == state.ghost:
            return -1000
        if not state.food:
            return 1000
        food_dist = min(manhattan(state.pacman, f) for f in state.food)
        ghost_dist = manhattan(state.pacman, state.ghost)
        return state.score - food_dist + ghost_dist * 0.5


# ---------------------------------------------------------------------
# 3. Tabular Q-Learning Agent
# ---------------------------------------------------------------------
class QLearningAgent:
    def __init__(self, alpha=0.2, gamma=0.9, epsilon=0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def _key(self, state: State, action: str):
        # Simplified state representation for tractable tabular learning:
        # Pacman position + relative ghost direction + remaining food count.
        gdr = state.ghost[0] - state.pacman[0]
        gdc = state.ghost[1] - state.pacman[1]
        return (state.pacman, (gdr, gdc), len(state.food), action)

    def get_q(self, state: State, action: str) -> float:
        return self.q_table.get(self._key(state, action), 0.0)

    def get_action(self, state: State, greedy: bool = False) -> str:
        actions = legal_moves(state.pacman)
        if not greedy and random.random() < self.epsilon:
            return random.choice(actions)
        return max(actions, key=lambda a: self.get_q(state, a))

    def update(self, state: State, action: str, reward: float, next_state: State):
        next_actions = legal_moves(next_state.pacman)
        best_next = max((self.get_q(next_state, a) for a in next_actions), default=0.0)
        key = self._key(state, action)
        old_q = self.q_table.get(key, 0.0)
        self.q_table[key] = old_q + self.alpha * (reward + self.gamma * best_next - old_q)

    def train(self, episodes: int = 500, max_steps: int = 60):
        for _ in range(episodes):
            state = parse_maze(MAZE)
            for _ in range(max_steps):
                action = self.get_action(state)
                ghost_action = random.choice(legal_moves(state.ghost))
                next_state = step(state, action, ghost_action)
                reward = next_state.score - state.score
                self.update(state, action, reward, next_state)
                state = next_state
                if is_terminal(state):
                    break


# ---------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------
def run_episode(agent, name: str, max_steps: int = 40, greedy: bool = False):
    state = parse_maze(MAZE)
    print(f"\n=== {name} ===")
    for t in range(max_steps):
        if is_terminal(state):
            break
        if isinstance(agent, QLearningAgent):
            action = agent.get_action(state, greedy=greedy)
        else:
            action = agent.get_action(state)
        ghost_action = random.choice(legal_moves(state.ghost))
        state = step(state, action, ghost_action)
    outcome = "ate all food!" if not state.food else ("caught by ghost" if state.pacman == state.ghost else "ran out of time")
    print(f"{name} finished after {t + 1} steps -- {outcome}. Final score: {state.score}")


def main():
    random.seed(42)

    run_episode(ReflexAgent(), "Reflex Agent")
    run_episode(MinimaxAgent(depth=3), "Minimax Agent (alpha-beta, depth=3)")

    print("\n=== Q-Learning Agent ===")
    q_agent = QLearningAgent()
    q_agent.train(episodes=500)
    run_episode(q_agent, "Q-Learning Agent (after training)", greedy=True)


if __name__ == "__main__":
    main()
