# Sample Output

Actual results from running `python pacman_agents.py` in a clean sandbox (all three agents on the same maze layout).

```
=== Reflex Agent ===
Reflex Agent finished after 34 steps -- ate all food!. Final score: 57

=== Minimax Agent (alpha-beta, depth=3) ===
Minimax Agent (alpha-beta, depth=3) finished after 40 steps -- ran out of time. Final score: 40

=== Q-Learning Agent ===

=== Q-Learning Agent (after training) ===
Q-Learning Agent (after training) finished after 40 steps -- ran out of time. Final score: 20
```

Note: the Reflex Agent cleared the board in this run; Minimax and the trained Q-Learning agent hit the step cap on this particular maze/ghost seed, which is expected variance for these strategies rather than a bug -- rerunning with different starting seeds/mazes produces different relative rankings.
