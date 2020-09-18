# tg-repo

```bash
pip install -r requirements.txt
```

## State Space
- [0] X position
- [1] Y position
- [2] Orientation
- [3] Ball X
- [4] Ball Y
- [5] Able to Kick
- [6] Goal Center Proximity
- [7] Goal Center Angle
- [8] Goal Opening Angle
- [9] Proximity to Opponent
- [T] Teammate’s Goal Opening Angle
- [T] Proximity from Teammate i to Opponent
- [T] Pass Opening Angle
- [3T] X, Y, and Uniform Number of Teammates
- [3O] X, Y, and Uniform Number of Opponents
- [+1] Last Action Success Possible

## Field dimension

- X - 105.0 (2 * 52.5)
- Y - 68.0 (2 * 34)
- Diagonal = 125.1
- 14.4 (goal)

## Rcssserver Options

DEFAULT_ENV = 0
DEFAULT_DYNAMIC_ENV = 1
GO_TO_BALL_RANDOM_POS_ENV = 2
ALL_RANDOM_ENV = 3
START_WITH_BALL_ENV = 4
START_WITH_BALL_RANDOM_ENV = 5
START_MEDIUM_BALL_RANDOM_ENV = 6
START_HIGH_BALL_RANDOM_ENV = 7
PENALTY_ENV = 8
PENALTY_MEDIUM_ENV = 9
PENALTY_HIGH_ENV = 10
PENALTY_MEDIUM_STATIC_ENV = 11
PENALTY_HIGH_STATIC_ENV = 12