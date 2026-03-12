"""
LeRobot 데이터셋 비주얼라이저 (act_custom 레코딩 재생)

기능
  - 레코딩 데이터셋을 읽어 이미지를 화면에 표시
  - observation.state, action, timestamp 등을 텍스트 오버레이
  - 자동 플레이 (데이터셋 fps 기준)
  - 에피소드 단위 순차 재생

조작
  SPACE    : 일시정지 / 재개
  <- / ->  : 이전 / 다음 프레임 (일시정지 중)
  N        : 다음 에피소드로 건너뛰기
  ESC / Q  : 종료

사용법
  python examples/tutorial/act/act_replay.py
  python examples/tutorial/act/act_replay.py --dataset outputs/act_custom_dataset
"""

import argparse
import io
import math
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def decode_image(img_data) -> np.ndarray:
    """데이터셋의 이미지 데이터를 BGR numpy 배열로 디코딩."""
    if isinstance(img_data, dict) and "bytes" in img_data:
        img_bytes = img_data["bytes"]
        pil_img = Image.open(io.BytesIO(img_bytes))
        rgb = np.array(pil_img)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(img_data, np.ndarray):
        if img_data.shape[-1] == 3:
            return cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        return img_data
    raise ValueError(f"Unsupported image format: {type(img_data)}")


def overlay_text(img: np.ndarray, lines: list[str], x: int = 10, y_start: int = 30,
                 scale: float = 0.6, color=(0, 255, 255), thickness: int = 1) -> np.ndarray:
    """이미지 위에 여러 줄 텍스트 오버레이."""
    y = y_start
    line_gap = int(28 * scale)
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)  # 그림자
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)
        y += line_gap
    return img


def main():
    parser = argparse.ArgumentParser(description="LeRobot Dataset Visualizer")
    parser.add_argument("--dataset", type=str, default="outputs/act_custom_dataset",
                        help="데이터셋 루트 경로")
    args = parser.parse_args()

    dataset_root = args.dataset

    # ── 메타데이터 로드 ──────────────────────────────────────
    import json
    with open(f"{dataset_root}/meta/info.json", "r") as f:
        info = json.load(f)

    fps = info["fps"]
    total_episodes = info["total_episodes"]
    total_frames = info["total_frames"]

    print(f"Dataset: {dataset_root}")
    print(f"  Episodes: {total_episodes}  |  Frames: {total_frames}  |  FPS: {fps}")
    print(f"  Features: {list(info['features'].keys())}")

    # ── 데이터 로드 ──────────────────────────────────────────
    import glob
    parquet_files = sorted(glob.glob(f"{dataset_root}/data/**/*.parquet", recursive=True))
    if not parquet_files:
        print("Error: 데이터 파일이 없습니다.")
        return

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    print(f"  Loaded {len(df)} rows from {len(parquet_files)} file(s)")

    # 에피소드별 그룹
    episodes = df.groupby("episode_index")
    episode_indices = sorted(episodes.groups.keys())

    print(f"\n[조작]  SPACE: 일시정지  |  <-/->: 프레임 이동  |  N: 다음 에피소드  |  ESC/Q: 종료\n")

    # ── 재생 루프 ────────────────────────────────────────────
    frame_interval = 1.0 / fps
    paused = False
    current_ep_pos = 0  # 현재 에피소드 위치 (episode_indices 인덱스)

    while current_ep_pos < len(episode_indices):
        ep_idx = episode_indices[current_ep_pos]
        ep_data = episodes.get_group(ep_idx).reset_index(drop=True)
        ep_len = len(ep_data)
        frame_idx = 0
        skip_episode = False

        print(f"[EP {ep_idx}] {ep_len} frames")

        while 0 <= frame_idx < ep_len:
            t_start = time.monotonic()

            row = ep_data.iloc[frame_idx]

            # 이미지 디코딩
            img_bgr = decode_image(row["observation.images.cam"])

            # 텍스트 정보 구성
            state_val = row["observation.state"]
            action_val = row["action"]
            timestamp_val = row["timestamp"]

            state_deg = math.degrees(float(state_val))
            action_deg = math.degrees(float(action_val))

            text_lines = [
                f"Episode {ep_idx}/{total_episodes-1}  |  Frame {frame_idx}/{ep_len-1}  |  {fps}fps",
                f"Timestamp: {float(timestamp_val):.3f}s",
                f"State (pan): {float(state_val):.4f} rad  ({state_deg:+.1f} deg)",
                f"Action (target): {float(action_val):.4f} rad  ({action_deg:+.1f} deg)",
            ]
            if paused:
                text_lines.append("[PAUSED]  <-/-> Frame  |  SPACE Resume  |  N Next EP")

            overlay_text(img_bgr, text_lines)

            # 하단 프로그레스 바
            bar_y = img_bgr.shape[0] - 8
            bar_w = img_bgr.shape[1]
            progress = (frame_idx + 1) / ep_len
            cv2.rectangle(img_bgr, (0, bar_y), (bar_w, bar_y + 6), (40, 40, 40), -1)
            cv2.rectangle(img_bgr, (0, bar_y), (int(bar_w * progress), bar_y + 6), (0, 200, 255), -1)

            cv2.imshow("Dataset Replay", img_bgr)

            # ── 키 입력 ──────────────────────────────────────
            if paused:
                key = cv2.waitKeyEx(0)  # blocking
            else:
                elapsed = time.monotonic() - t_start
                wait_ms = max(1, int((frame_interval - elapsed) * 1000))
                key = cv2.waitKeyEx(wait_ms)

            if key in (27, ord('q'), ord('Q')):
                cv2.destroyAllWindows()
                print("종료.")
                return
            elif key == ord(' '):
                paused = not paused
            elif key in (ord('n'), ord('N')):
                skip_episode = True
                break
            elif key in (2424832, 65361, 81) and paused:   # <- 이전 프레임
                frame_idx = max(0, frame_idx - 1)
                continue
            elif key in (2555904, 65363, 83) and paused:   # -> 다음 프레임
                frame_idx = min(ep_len - 1, frame_idx + 1)
                continue

            if not paused:
                frame_idx += 1

        current_ep_pos += 1

    print("모든 에피소드 재생 완료.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
