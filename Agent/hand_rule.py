"""Luật viết tay: mốc so sánh cho agent DQN (Problem_Definition.md mục 1.6).

Vỗ cánh khi và chỉ khi: nếu không vỗ, đáy chim ở khung hình sau sẽ thấp hơn
một vạch đặt cao hơn mép dưới khe của ống cần vượt 12 pixel.
Luật chỉ đọc observation, giống agent.

Chạy: python Agent/hand_rule.py [--episodes 200] [--seed 0]
"""
import argparse

import numpy as np

from environment import FLAP, IDLE

SCREEN_W, SCREEN_H = 288, 512
BIRD_X, BIRD_H = 57, 24
PIPE_W = 52
GROUND_Y = 404.48
MAX_FALL_SPEED = 10
MARGIN = 12  # khoảng cách từ vạch tới mép dưới khe, tính bằng pixel


def target_gap_bottom(obs):
    """y (pixel) của mép dưới khe của ống cần vượt: ống đầu tiên chưa bị chim bay qua hết."""
    first_pipe_right = obs[0] * SCREEN_W + PIPE_W
    gap_bottom = obs[2] if first_pipe_right > BIRD_X else obs[5]
    return gap_bottom * SCREEN_H


def hand_rule(obs):
    y = obs[9] * SCREEN_H
    vy_next = min(obs[10] * MAX_FALL_SPEED + 1, MAX_FALL_SPEED)  # vận tốc nếu không vỗ
    y_next = y + min(vy_next, GROUND_Y - y - BIRD_H)
    bottom_next = y_next + BIRD_H
    return FLAP if bottom_next > target_gap_bottom(obs) - MARGIN else IDLE


def main():
    from evaluate import play_episodes, summarize_scores

    parser = argparse.ArgumentParser(description="Chấm điểm luật viết tay.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score-limit", type=int, default=1_000)
    args = parser.parse_args()

    scores = play_episodes(hand_rule, args.episodes, args.seed, args.score_limit)
    stats = summarize_scores(scores)
    print(f"Luật viết tay, {args.episodes} ván: trung bình {stats['mean']:.1f} | "
          f"trung vị {stats['median']:.0f} | thấp nhất {stats['min']} | cao nhất {stats['max']}")


if __name__ == "__main__":
    main()
