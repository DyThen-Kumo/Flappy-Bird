"""Chạy: pytest Agent"""
import numpy as np
import pytest

from agent import DEFAULT_WEIGHTS, DQNAgent, load_agent
from environment import FLAP, IDLE
from evaluate import play_episodes, summarize_scores
from hand_rule import hand_rule


def test_agent_on_fake_states():
    agent = DQNAgent()
    states = np.random.default_rng(0).uniform(-1, 1, size=(5, 12)).astype(np.float32)
    q = agent.q_values(states)
    assert q.shape == (5, 2)
    assert agent.greedy_action(states[0]) == int(q[0].argmax())


def test_missing_weights_raises():
    with pytest.raises(FileNotFoundError):
        load_agent("khong_ton_tai.weights.h5")


def test_summarize_scores():
    assert summarize_scores([1, 2, 6]) == {"mean": 3.0, "median": 2.0, "min": 1, "max": 6}


def test_hand_rule_flaps_only_when_falling_below_line():
    obs = np.zeros(12, dtype=np.float32)
    obs[0] = 100 / 288   # ống cần vượt là bộ thứ nhất
    obs[2] = 250 / 512   # mép dưới khe ở y = 250, vạch ở y = 238
    obs[9] = 150 / 512   # đáy chim ở y = 174, còn xa vạch
    assert hand_rule(obs) == IDLE
    obs[9] = 220 / 512   # đáy chim khung hình sau ở 220 + 1 + 24 = 245 > 238
    assert hand_rule(obs) == FLAP
    obs[0] = -40 / 288   # chim đã qua bộ thứ nhất: xét bộ thứ hai
    obs[5] = 512 / 512   # bộ thứ hai bị che, mép dưới khe coi như 512
    assert hand_rule(obs) == IDLE


@pytest.mark.skipif(not DEFAULT_WEIGHTS.is_file(), reason="trọng số không nằm trong git")
def test_trained_agent_passes_pipes():
    agent = load_agent()
    scores = play_episodes(agent.greedy_action, n_episodes=2, seed=12345, score_limit=5)
    assert list(scores) == [5, 5]
