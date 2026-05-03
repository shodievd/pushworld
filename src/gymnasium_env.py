"""
Gymnasium environment for PushWorld puzzles.

Observation modes
-----------------
Both modes produce float32 arrays with values in [0, 1], ready for direct
neural-network input without any casting or rescaling.

"rgb_array" : float32 RGB image, shape (H_px, W_px, 3).
"grid"      : float32 2-D grid, shape (H_cells, W_cells).
                Raw cell codes (0–6) divided by CELL_AGENT (6):
                0/6 EMPTY  1/6 WALL  2/6 AGENT_WALL  3/6 GOAL_TARGET
                4/6 GOAL_OBJECT  5/6 EXTRA_MOVABLE  6/6 AGENT
"rgb_grid"  : Dict{"rgb": <rgb_array obs>, "grid": <grid obs>}.

Reward scheme (Appendix D of https://arxiv.org/pdf/1707.06203.pdf):
  +10.0 on solving the puzzle, +1.0 per newly achieved sub-goal, -0.01 per step.
"""

import random
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from pushworld.config import PUZZLE_EXTENSION
from pushworld.puzzle import (
    DEFAULT_BORDER_WIDTH,
    DEFAULT_PIXELS_PER_CELL,
    NUM_ACTIONS,
    PushWorldPuzzle,
    State,
)
from pushworld.utils.env_utils import get_max_puzzle_dimensions, render_observation_padded
from pushworld.utils.filesystem import iter_files_with_extension


# Integer values used in the "grid" observation
CELL_EMPTY = 0
CELL_WALL = 1
CELL_AGENT_WALL = 2
CELL_GOAL_TARGET = 3   # static target marker (where a goal object must go)
CELL_GOAL_OBJECT = 4   # movable that has an associated goal
CELL_EXTRA_MOVABLE = 5 # movable without a goal (tool)
CELL_AGENT = 6

OBSERVATION_MODES = ("rgb_array", "grid", "rgb_grid")


def _encode_grid(
    puzzle: PushWorldPuzzle,
    state: State,
    max_cell_height: int,
    max_cell_width: int,
) -> np.ndarray:
    """Return a padded float32 grid encoding of *state*, values in [0, 1].

    Mirrors the drawing order used by PushWorldPuzzle.render():
      walls → agent walls → goal targets → movable objects (agent last / highest).

    Each cell gets an integer code (0–CELL_AGENT) then the whole array is
    divided by CELL_AGENT so the range matches the RGB observation.
    """
    width, height = puzzle.dimensions
    grid = np.zeros((height, width), dtype=np.float32)

    # Walls: wall_positions stores absolute (x, y) cell coords (position is (0,0))
    for x, y in puzzle.wall_positions:
        grid[y, x] = CELL_WALL

    # Agent-only walls: same layout as walls
    for x, y in puzzle.agent_wall_positions:
        grid[y, x] = CELL_AGENT_WALL

    # Goal target markers (static). Each goal object can be multi-cell, so iterate
    # its relative cell offsets exactly as render() does: pos + cell.
    for g in puzzle._goals:
        gx, gy = g.position
        for cx, cy in g.cells:
            grid[gy + cy, gx + cx] = CELL_GOAL_TARGET

    # Movable objects paired with current positions from state — mirrors:
    #   zip(self._movable_objects, state)
    # movable_objects[0] = agent, [1..n_goals] = goal objects, rest = extra movables
    n_goals = len(puzzle.goal_state)
    for i, (obj, pos) in enumerate(zip(puzzle.movable_objects, state)):
        px, py = pos
        if i == 0:
            cell_val = CELL_AGENT
        elif i <= n_goals:
            cell_val = CELL_GOAL_OBJECT
        else:
            cell_val = CELL_EXTRA_MOVABLE
        for cx, cy in obj.cells:
            grid[py + cy, px + cx] = cell_val

    # Normalize to [0, 1] so dtype and range match the RGB observation
    grid /= CELL_AGENT

    # Pad to max dimensions; extra cells stay 0.0 (EMPTY)
    padded = np.zeros((max_cell_height, max_cell_width), dtype=np.float32)
    padded[:height, :width] = grid
    return padded


class PushWorldGymnasiumEnv(gym.Env):
    """Gymnasium environment wrapping PushWorld puzzles.

    Args:
        puzzle_path: Path to a single .pwp file or a directory tree of .pwp files.
            All discovered puzzles are loaded; `reset` samples one at random.
        observation_mode: One of ``"rgb_array"``, ``"grid"``, or ``"rgb_grid"``.
        max_steps: Optional episode step limit; causes truncation when reached.
        border_width: Pixel border width used when rendering (must be >= 1).
        pixels_per_cell: Pixel size of one grid cell when rendering (must be >= 3).
        standard_padding: If True, pad observations to the maximum dimensions of
            the official benchmark puzzles instead of the loaded puzzle set.
    """

    metadata = {"render_modes": ["rgb_array"]}
    render_mode = "rgb_array"

    def __init__(
        self,
        puzzle_path: str,
        observation_mode: str = "rgb_array",
        max_steps: Optional[int] = None,
        border_width: int = DEFAULT_BORDER_WIDTH,
        pixels_per_cell: int = DEFAULT_PIXELS_PER_CELL,
        standard_padding: bool = False,
    ) -> None:
        super().__init__()

        if observation_mode not in OBSERVATION_MODES:
            raise ValueError(
                f"observation_mode must be one of {OBSERVATION_MODES}, "
                f"got {observation_mode!r}."
            )
        if border_width < 1:
            raise ValueError("border_width must be >= 1.")
        if pixels_per_cell < 3:
            raise ValueError("pixels_per_cell must be >= 3.")

        self._puzzles = [
            PushWorldPuzzle(path)
            for path in iter_files_with_extension(puzzle_path, PUZZLE_EXTENSION)
        ]
        if not self._puzzles:
            raise ValueError(f"No PushWorld puzzles found in: {puzzle_path}")

        self._observation_mode = observation_mode
        self._max_steps = max_steps
        self._pixels_per_cell = pixels_per_cell
        self._border_width = border_width

        widths, heights = zip(*[p.dimensions for p in self._puzzles])
        self._max_cell_width = max(widths)
        self._max_cell_height = max(heights)

        if standard_padding:
            standard_cell_height, standard_cell_width = get_max_puzzle_dimensions()

            if standard_cell_height < self._max_cell_height:
                raise ValueError(
                    "`standard_padding` is True, but the maximum puzzle height in "
                    "BENCHMARK_PUZZLES_PATH is less than the height of the puzzle(s) "
                    "in the given `puzzle_path`."
                )
            else:
                self._max_cell_height = standard_cell_height

            if standard_cell_width < self._max_cell_width:
                raise ValueError(
                    "`standard_padding` is True, but the maximum puzzle width in "
                    "BENCHMARK_PUZZLES_PATH is less than the width of the puzzle(s) "
                    "in the given `puzzle_path`."
                )
            else:
                self._max_cell_width = standard_cell_width

        # Fixed seed for reproducibility; overridable via reset(seed=...)
        self._random_generator = random.Random(123)

        self._current_puzzle: Optional[PushWorldPuzzle] = None
        self._current_state: Optional[State] = None
        self._steps: int = 0

        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self.observation_space = self._build_observation_space()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation_space(self) -> gym.spaces.Space:
        dummy = self._puzzles[0]
        rgb_shape = render_observation_padded(
            dummy,
            dummy.initial_state,
            self._max_cell_height,
            self._max_cell_width,
            self._pixels_per_cell,
            self._border_width,
        ).shape
        rgb_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=rgb_shape, dtype=np.float32
        )
        grid_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._max_cell_height, self._max_cell_width),
            dtype=np.float32,
        )

        if self._observation_mode == "rgb_array":
            return rgb_space
        if self._observation_mode == "grid":
            return grid_space
        return gym.spaces.Dict({"rgb": rgb_space, "grid": grid_space})

    def _make_obs(self, puzzle: PushWorldPuzzle, state: State) -> Any:
        if self._observation_mode == "rgb_array":
            return render_observation_padded(
                puzzle, state,
                self._max_cell_height, self._max_cell_width,
                self._pixels_per_cell, self._border_width,
            )
        if self._observation_mode == "grid":
            return _encode_grid(
                puzzle, state, self._max_cell_height, self._max_cell_width
            )
        # "rgb_grid"
        return {
            "rgb": render_observation_padded(
                puzzle, state,
                self._max_cell_height, self._max_cell_width,
                self._pixels_per_cell, self._border_width,
            ),
            "grid": _encode_grid(
                puzzle, state, self._max_cell_height, self._max_cell_width
            ),
        }

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Any, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._random_generator = random.Random(seed)

        self._current_puzzle = self._random_generator.choice(self._puzzles)
        self._current_state = self._current_puzzle.initial_state
        self._current_achieved_goals = self._current_puzzle.count_achieved_goals(
            self._current_state
        )
        self._steps = 0

        return self._make_obs(self._current_puzzle, self._current_state), {
            "puzzle_state": self._current_state,
        }

    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action!r}; must be in {self.action_space}."
            )
        if self._current_state is None:
            raise RuntimeError("reset() must be called before step().")

        self._steps += 1
        previous_state = self._current_state
        self._current_state = self._current_puzzle.get_next_state(
            self._current_state, action
        )

        terminated = self._current_puzzle.is_goal_state(self._current_state)

        if terminated:
            reward = 10.0
        else:
            previous_achieved_goals = self._current_puzzle.count_achieved_goals(
                previous_state
            )
            current_achieved_goals = self._current_puzzle.count_achieved_goals(
                self._current_state
            )
            reward = current_achieved_goals - previous_achieved_goals - 0.01

        truncated = False if self._max_steps is None else self._steps >= self._max_steps
        obs = self._make_obs(self._current_puzzle, self._current_state)
        return obs, reward, terminated, truncated, {"puzzle_state": self._current_state}

    def render(self) -> Optional[np.ndarray]:
        """Return a uint8 RGB image of the current state, or None before reset."""
        if self._current_state is None:
            return None
        return self._current_puzzle.render(
            self._current_state,
            border_width=self._border_width,
            pixels_per_cell=self._pixels_per_cell,
        )

    # ------------------------------------------------------------------
    # Extra properties
    # ------------------------------------------------------------------

    @property
    def current_puzzle(self) -> Optional[PushWorldPuzzle]:
        """The active puzzle, or None if reset has not been called yet."""
        return self._current_puzzle

    @property
    def observation_mode(self) -> str:
        return self._observation_mode
