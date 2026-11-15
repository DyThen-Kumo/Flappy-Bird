"""Môi trường FlappyBird. Đây là module duy nhất import gymnasium (giống Code/01_environment.ipynb)."""
import numpy as np
import gymnasium
import flappy_bird_gymnasium  # noqa: F401  (đăng ký "FlappyBird-v0" với gymnasium)

ENV_ID = "FlappyBird-v0"
OBS_DIM = 12
N_ACTIONS = 2
IDLE, FLAP = 0, 1


class FlappyEnv:
    """Bọc FlappyBird-v0: observation float32, chặn step() sau khi ván đã kết thúc."""

    def __init__(self, score_limit=None, render_mode=None):
        self._env = gymnasium.make(ENV_ID, use_lidar=False,
                                   score_limit=score_limit, render_mode=render_mode)
        self._episode_over = True

    def reset(self, seed=None):
        obs, _ = self._env.reset(seed=seed)
        self._episode_over = False
        return obs.astype(np.float32)

    def step(self, action):
        """Trả về (obs, reward, terminated, truncated, score)."""
        # Thư viện vẫn cho step() với con chim đã chết, nên phải tự chặn.
        if self._episode_over:
            raise RuntimeError("Ván đã kết thúc: phải gọi reset() trước khi gọi step().")
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        self._episode_over = terminated or truncated
        return obs.astype(np.float32), float(reward), bool(terminated), bool(truncated), info["score"]

    def close(self):
        self._env.close()
