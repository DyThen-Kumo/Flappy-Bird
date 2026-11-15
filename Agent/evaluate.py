"""Chấm điểm agent trên nhiều ván, không mở cửa sổ.

Chạy: python Agent/evaluate.py [--episodes 20] [--score-limit 200] [--seed 12345] [--weights FILE]
"""
import argparse

import numpy as np

from agent import DEFAULT_WEIGHTS, load_agent
from environment import FlappyEnv


def play_episodes(policy, n_episodes, seed, score_limit, render_mode=None, on_episode_end=None):
    """Chơi n_episodes ván với policy(obs) -> action, trả về mảng số ống vượt được mỗi ván.

    Seed chỉ đặt ở ván đầu, nên cùng seed cho cùng chuỗi ván.
    """
    env = FlappyEnv(score_limit=score_limit, render_mode=render_mode)
    scores = []
    try:
        for episode in range(n_episodes):
            obs = env.reset(seed=seed if episode == 0 else None)
            score, episode_over = 0, False
            while not episode_over:
                obs, _, terminated, truncated, score = env.step(policy(obs))
                episode_over = terminated or truncated
            scores.append(score)
            if on_episode_end:
                on_episode_end(episode, score)
    finally:
        env.close()
    return np.array(scores)


def summarize_scores(scores):
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "min": int(np.min(scores)),
        "max": int(np.max(scores)),
    }


def main():
    parser = argparse.ArgumentParser(description="Chấm điểm agent DQN (ε = 0).")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--score-limit", type=int, default=200, help="dừng ván khi đạt số ống này")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    agent = load_agent(args.weights)
    scores = play_episodes(agent.greedy_action, args.episodes, args.seed, args.score_limit)
    stats = summarize_scores(scores)
    print(f"{args.episodes} ván, giới hạn {args.score_limit} ống")
    print(f"Trung bình {stats['mean']:.2f} | trung vị {stats['median']:.0f} | "
          f"thấp nhất {stats['min']} | cao nhất {stats['max']}")
    print(f"Số ván chạm giới hạn: {int(np.sum(scores >= args.score_limit))}/{len(scores)}")
    print("Mốc so sánh: ngẫu nhiên 0, luật viết tay 34,2")


if __name__ == "__main__":
    main()
