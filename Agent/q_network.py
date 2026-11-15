"""Q-Network với đầu ra Dueling (giống Code/02_q_network.ipynb). Không biết gì về trò chơi."""
import keras


def build_q_network(obs_dim=12, n_actions=2, hidden_units=256):
    """MLP 2 lớp ẩn; Q(s,a) = V(s) + A(s,a) - trung bình của A."""
    states = keras.Input(shape=(obs_dim,), name="state")
    h = keras.layers.Dense(hidden_units, activation="relu", name="hidden_1")(states)
    h = keras.layers.Dense(hidden_units, activation="relu", name="hidden_2")(h)
    value = keras.layers.Dense(1, name="value")(h)
    advantage = keras.layers.Dense(n_actions, name="advantage")(h)
    q_values = value + advantage - keras.ops.mean(advantage, axis=1, keepdims=True)
    return keras.Model(states, q_values, name="dueling_q_network")
