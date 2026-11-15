"""Demo: mở cửa sổ game và xem agent tự chơi (30 khung hình/giây).

Chạy: python Agent/play.py [--episodes 1] [--score-limit 50] [--seed N] [--weights FILE]
Dừng sớm: Ctrl+C.
"""
import argparse

from agent import DEFAULT_WEIGHTS, load_agent
from evaluate import play_episodes


def main():
    parser = argparse.ArgumentParser(description="Xem agent DQN chơi Flappy Bird.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--score-limit", type=int, default=50,
                        help="dừng ván khi đạt số ống này (0 = chơi tới khi chim chết)")
    parser.add_argument("--seed", type=int, default=None, help="mặc định: mỗi lần một chuỗi ống khác")
    args = parser.parse_args()

    agent = load_agent(args.weights)
    try:
        play_episodes(agent.greedy_action, args.episodes, args.seed, args.score_limit or None,
                      render_mode="human",
                      on_episode_end=lambda i, score: print(f"Ván {i + 1}: vượt {score} ống"))
    except KeyboardInterrupt:
        print("Đã dừng.")


if __name__ == "__main__":
    main()
