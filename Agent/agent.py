"""Agent DQN đã huấn luyện, chỉ phần chơi (ε = 0). Phần học nằm ở Code/04_agent.ipynb."""
from pathlib import Path

import numpy as np
import tensorflow as tf

from q_network import build_q_network

# Checkpoint tốt nhất theo README mục 6.5: trung bình 910,7 ống trên 100 ván.
DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "Results" / "dqn_run3" / "best.weights.h5"


class DQNAgent:
    def __init__(self, obs_dim=12, n_actions=2, hidden_units=256):
        self.online = build_q_network(obs_dim, n_actions, hidden_units)

    def greedy_action(self, state):
        """Hành động có Q lớn nhất cho một observation."""
        state = np.asarray(state, dtype=np.float32)[None, :]
        return int(self._greedy(state).numpy()[0])

    # tf.function thay cho model.predict: nhanh hơn khoảng 150 lần khi gọi từng khung hình.
    @tf.function
    def _greedy(self, states):
        return tf.argmax(self.online(states, training=False), axis=1)

    def q_values(self, states):
        """Q(s, IDLE) và Q(s, FLAP) cho một batch observation."""
        return self.online(np.asarray(states, dtype=np.float32), training=False).numpy()

    def load(self, path):
        self.online.load_weights(path)


def load_agent(weights_path=DEFAULT_WEIGHTS):
    """Dựng mạng rồi nạp trọng số từ file .weights.h5."""
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Không thấy file trọng số: {weights_path}")
    agent = DQNAgent()
    agent.load(weights_path)
    return agent
