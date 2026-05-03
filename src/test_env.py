import os
import numpy as np
import mediapy

from pushworld.puzzle import NUM_ACTIONS
from pushworld.gym_env import PushWorldEnv


from absl import app
from absl import flags

BENCH_DIR = "/workspace/HeuristicMethods/pushworld/pushworld/benchmark/puzzles/level0/level0"
tasks_types = ['size', 'shapes', 'goals', 'obstacles', 'all', 'base', 'walls']

tasks_all = os.listdir(f"{BENCH_DIR}/{'all'}/train/")

_PATH = flags.DEFINE_string('path', f"{BENCH_DIR}/{'all'}/train/{tasks_all[0]}", 'Puzzle file path.')
_OUT = flags.DEFINE_string('out', 'frames', 'Output directory for frames.')


def main(argv):
    frames = []
    env = PushWorldEnv(_PATH.value)

    os.makedirs(_OUT.value, exist_ok=True)
    image, info = env.reset()
    
    frames.append(image)
    for i in range(10):
        image = env.step(np.random.randint(NUM_ACTIONS))[0]
        frames.append(image)

    mediapy.write_video('frames/all_train.mp4', frames, fps=1)
    print(f"Frames saved to {_OUT.value}/")


if __name__ == '__main__':
    app.run(main)