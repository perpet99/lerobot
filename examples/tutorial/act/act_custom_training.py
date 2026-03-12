"""
act_custom 데이터셋으로 ACT Policy 학습

입력 : observation.images.cam (이미지) + observation.state (상태)
출력 : action (관절 목표 각도)

학습 후 loss 그래프를 화면에 표시하고 PNG로 저장.

사용법
  python examples/tutorial/act/act_custom_training.py
  python examples/tutorial/act/act_custom_training.py --steps 500 --batch 8
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


# ─── 데이터셋 설정 ──────────────────────────────────────────────
DATASET_REPO_ID = "act_custom/cam_turret"
DATASET_ROOT = "outputs/act_custom_dataset"


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ACT Custom Training")
    parser.add_argument("--steps", type=int, default=100, help="학습 스텝 수")
    parser.add_argument("--batch", type=int, default=8, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-5, help="학습률")
    parser.add_argument("--output", type=str, default="outputs/act_custom_training",
                        help="체크포인트 저장 경로")
    args = parser.parse_args()

    output_directory = Path(args.output)
    output_directory.mkdir(parents=True, exist_ok=True)

    # ── 디바이스 ─────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── 데이터셋 메타데이터 로드 ─────────────────────────────
    dataset_metadata = LeRobotDatasetMetadata(DATASET_REPO_ID, root=DATASET_ROOT)
    features = dataset_to_policy_features(dataset_metadata.features)

    print(f"Dataset: {DATASET_ROOT}")
    print(f"  Episodes: {dataset_metadata.total_episodes}  |  Frames: {dataset_metadata.total_frames}  |  FPS: {dataset_metadata.fps}")
    print(f"  Policy features: {list(features.keys())}")

    # ── 입력/출력 분리 ───────────────────────────────────────
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    print(f"  Input:  {list(input_features.keys())}")
    print(f"  Output: {list(output_features.keys())}")

    # ── ACT 설정 & 모델 생성 ─────────────────────────────────
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=50,
        n_action_steps=50,
        optimizer_lr=args.lr,
    )
    policy = ACTPolicy(cfg)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(device)

    param_count = sum(p.numel() for p in policy.parameters())
    print(f"  Model params: {param_count:,}")

    # ── delta_timestamps 설정 ────────────────────────────────
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    # 이미지가 비디오로 저장된 경우에만 delta_timestamps 추가 (parquet 이미지는 제외)
    if cfg.observation_delta_indices is not None:
        delta_timestamps |= {
            k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
            for k in cfg.image_features
        }

    # ── 데이터셋 & 데이터로더 ────────────────────────────────
    dataset = LeRobotDataset(DATASET_REPO_ID, root=DATASET_ROOT, delta_timestamps=delta_timestamps)

    # 스칼라 피처(shape=(1,)) → 마지막 차원 추가 보정하는 collate
    default_collate = torch.utils.data.dataloader.default_collate

    def collate_fn(batch):
        out = default_collate(batch)
        for key, ft in features.items():
            if key in out and isinstance(out[key], torch.Tensor) and len(ft.shape) == 1 and ft.shape[0] == 1:
                # (batch,) → (batch, 1) 또는 (batch, chunk) → (batch, chunk, 1)
                out[key] = out[key].unsqueeze(-1)
        return out

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        collate_fn=collate_fn,
    )

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    # ── 학습 루프 ────────────────────────────────────────────
    training_steps = args.steps
    log_freq = max(1, training_steps // 50)
    loss_history = []

    print(f"\n{'='*50}")
    print(f"Training: {training_steps} steps, batch={args.batch}, lr={args.lr}")
    print(f"{'='*50}\n")

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if step % log_freq == 0:
                print(f"  step {step:>5d}/{training_steps}  |  loss: {loss_val:.4f}")

            step += 1
            if step >= training_steps:
                done = True
                break

    print(f"\nTraining complete! Final loss: {loss_history[-1]:.4f}")

    # ── 체크포인트 저장 ──────────────────────────────────────
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print(f"Checkpoint saved: {output_directory}")

    # ── Loss 그래프 ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, linewidth=1.2, color="#2196F3")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("ACT Training Loss (act_custom/cam_turret)", fontsize=14)
    ax.grid(True, alpha=0.3)

    # 이동평균 추가 (10스텝)
    if len(loss_history) > 10:
        window = min(10, len(loss_history) // 5)
        smoothed = []
        for i in range(len(loss_history)):
            start = max(0, i - window + 1)
            smoothed.append(sum(loss_history[start:i+1]) / (i - start + 1))
        ax.plot(range(1, len(smoothed) + 1), smoothed, linewidth=2, color="#FF5722",
                label=f"Moving avg (w={window})")
        ax.legend()

    plt.tight_layout()

    graph_path = output_directory / "training_loss.png"
    fig.savefig(graph_path, dpi=150)
    print(f"Loss graph saved: {graph_path}")

    plt.show()


if __name__ == "__main__":
    main()
