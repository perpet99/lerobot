"""
ACT Policy: MuJoCo Aloha 시뮬레이션에서 학습 + 추론 테스트
(act_training_example.py + act_using_example.py를 MuJoCo 환경에 맞게 통합)
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def train(device: torch.device, output_directory: Path, training_steps: int = 2):
    """Step 1: Aloha 시뮬레이션 데이터셋으로 ACT 모델 학습"""
    print("=" * 60)
    print("Step 1: ACT 모델 학습")
    print("=" * 60)

    dataset_id = "lerobot/aloha_sim_insertion_human_image"

    print(f"  데이터셋 메타데이터 다운로드 중: {dataset_id}")
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    print(f"  입력 features: {list(input_features.keys())}")
    print(f"  출력 features: {list(output_features.keys())}")

    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    policy = ACTPolicy(cfg)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(device)

    # ACT uses action chunking: multiple future actions as targets
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    # n_obs_steps=1 이므로 observation에는 delta_timestamps 불필요 (temporal 차원 방지)

    print(f"  데이터셋 로딩 중: {dataset_id}")
    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, pin_memory=device.type != "cpu", drop_last=True
    )

    print(f"  학습 시작 ({training_steps} steps)...")
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"    step {step}: loss={loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

    # 저장
    output_directory.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print(f"  모델 저장 완료: {output_directory}")

    return dataset_metadata


def evaluate(device: torch.device, output_directory: Path, dataset_metadata, max_episodes: int = 1, max_steps: int = 50):
    """Step 2: MuJoCo Aloha 시뮬레이션에서 ACT 모델 추론"""
    print("\n" + "=" * 60)
    print("Step 2: MuJoCo Aloha 시뮬레이션에서 추론")
    print("=" * 60)

    # 학습된 모델 로드
    print(f"  모델 로드 중: {output_directory}")
    model = ACTPolicy.from_pretrained(str(output_directory))
    model.to(device)
    model.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        model.config,
        pretrained_path=output_directory,
        dataset_stats=dataset_metadata.stats,
    )

    # MuJoCo 환경 생성 (human 모드로 UI 표시)
    print("  MuJoCo Aloha 환경 생성 중...")
    import cv2
    import gym_aloha  # noqa: F401

    env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    print(f"  환경: {env.spec.id}")
    print(f"  행동 공간: {env.action_space.shape}")
    print("  [ESC] 키로 종료 가능")

    for ep in range(max_episodes):
        print(f"\n  --- Episode {ep + 1}/{max_episodes} ---")
        obs, info = env.reset(seed=42 + ep)
        total_reward = 0.0

        for step in range(max_steps):
            # 시뮬레이션 화면 표시
            frame = env.render()  # (H, W, 3) RGB
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("MuJoCo Aloha - ACT Inference", frame_bgr)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                print("    사용자 종료 (ESC)")
                env.close()
                cv2.destroyAllWindows()
                return

            # 관측을 모델 입력 형태로 변환
            obs_dict = {}
            # 상태 (agent_pos)
            obs_dict["observation.state"] = torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device)
            # 이미지 (top camera)
            img = obs["pixels"]["top"]  # (H, W, C) uint8
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C, H, W)
            obs_dict["observation.images.top"] = img_tensor

            obs_batch = preprocessor(obs_dict)

            with torch.no_grad():
                action = model.select_action(obs_batch)

            action = postprocessor(action)

            # 환경에 행동 적용 (postprocessor returns a tensor, not a dict)
            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward

            if step % 10 == 0:
                print(f"    step {step}: reward={reward:.4f}, total={total_reward:.4f}")

            if terminated or truncated:
                print(f"    종료! step={step}, terminated={terminated}, truncated={truncated}")
                break

        print(f"  Episode {ep + 1} 완료: total_reward={total_reward:.4f}")

    env.close()
    cv2.destroyAllWindows()
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


def main():
    device = torch.device("cpu")  # or "cuda" if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
    else:
        print(f"Using device: {device} (CUDA not available)")

    output_directory = Path("outputs/act_mujoco_test")

    # Step 1: 학습 (저장된 모델이 없을 때만)
    model_file = output_directory / "model.safetensors"
    if model_file.exists():
        print(f"  학습된 모델 발견: {output_directory} → 추론으로 바로 진행")
        dataset_metadata = LeRobotDatasetMetadata("lerobot/aloha_sim_insertion_human_image")
    else:
        print(f"  학습된 모델 없음 → 학습 시작")
        dataset_metadata = train(device, output_directory, training_steps=30)

    # Step 2: MuJoCo에서 추론
    evaluate(device, output_directory, dataset_metadata, max_episodes=100, max_steps=20)


if __name__ == "__main__":
    main()
