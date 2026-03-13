"""act_custom 데이터셋으로 SmolVLA Policy 학습 (시계열 이미지 + 각도 변화량)

입력 : observation.images.cam 3프레임 (N-2, N-1, N) + observation.state (현재 각도)
출력 : 각도 변화량 (N 시점 각도 - N-2 시점 각도)

3장 이미지로 이동방향/속도를 인식하고, 각도 변화량(delta)을 예측.
SmolVLA는 VLM 백본 + 액션 전문가 모델로 구성된 VLA(Vision-Language-Action) 모델.

사전 설치: pip install -e ".[smolvla]"

사용법
  python examples/tutorial/act/act_custom_training2.py
  python examples/tutorial/act/act_custom_training2.py --steps 500 --batch 4
"""

import msvcrt
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


# ─── 데이터셋 설정 ──────────────────────────────────────────────
DATASET_REPO_ID = "act_custom/cam_turret"
DATASET_ROOT = "outputs/act_custom_dataset"
SINGLE_TASK = "Rotate camera turret to track target"

# ─── 시계열 프레임 수 ────────────────────────────────────────────
N_TEMPORAL_FRAMES = 3  # N-2, N-1, N


def transform_batch(batch: dict, device: torch.device, tokenizer) -> dict:
    """시계열 데이터를 3개 카메라(=3프레임) 입력 + 각도 변화량 타겟 + 언어 토큰으로 변환.

    변환 내용:
      observation.images.cam (B, 3, C, H, W)
        → observation.images.cam_t0 (B, C, H, W)  프레임 N-2
        → observation.images.cam_t1 (B, C, H, W)  프레임 N-1
        → observation.images.cam_t2 (B, C, H, W)  프레임 N
      observation.state (B, 2, state_dim)
        → observation.state (B, state_dim)  현재(N) 각도만
      action
        → (B, 1, state_dim)  각도 변화량 = state[N] - state[N-2]
      task (list[str])
        → observation.language.tokens + observation.language.attention_mask
    """
    # 모든 텐서를 device로 이동
    batch = {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    B = batch["observation.images.cam"].shape[0]

    # ── 시계열 이미지 → 3개 독립 카메라로 분리 ───────────────
    imgs = batch.pop("observation.images.cam")  # (B, 3, C, H, W)
    batch["observation.images.cam_t0"] = imgs[:, 0]  # N-2
    batch["observation.images.cam_t1"] = imgs[:, 1]  # N-1
    batch["observation.images.cam_t2"] = imgs[:, 2]  # N (현재)

    # ── 각도 변화량 계산 ─────────────────────────────────────
    states = batch["observation.state"]  # (B, 2, state_dim) or (B, 2)
    if states.dim() == 3:
        state_n2 = states[:, 0, :]  # (B, state_dim) — N-2 시점
        state_n = states[:, 1, :]   # (B, state_dim) — N 시점 (현재)
    else:
        state_n2 = states[:, 0:1]   # (B, 1)
        state_n = states[:, 1:2]    # (B, 1)

    delta_angle = state_n - state_n2  # (B, state_dim)

    # 현재 각도만 입력으로 사용
    batch["observation.state"] = state_n

    # 학습 타깃은 단일 스텝 delta angle 형태로 맞춘다.
    print(f"delta_angle shape: {delta_angle.shape}, values: {delta_angle[:2]}")
    print(f"action shape: {delta_angle.unsqueeze(1).shape}, values: {delta_angle.unsqueeze(1)[:2]}")
    
    batch["action"] = delta_angle.unsqueeze(1)
    batch["action_is_pad"] = torch.zeros(B, 1, dtype=torch.bool, device=device)

    # ── 태스크 문자열 → 언어 토큰 (SmolVLA 필수 입력) ────────
    task_texts = batch.pop("task", None)
    if task_texts is None:
        task_texts = [SINGLE_TASK + "\n"] * B
    elif isinstance(task_texts, str):
        task_texts = [task_texts + "\n"] * B
    else:
        task_texts = [t + "\n" if not t.endswith("\n") else t for t in task_texts]
    tokens = tokenizer(task_texts, padding="longest", padding_side="right",
                       max_length=48, truncation=True, return_tensors="pt")
    batch["observation.language.tokens"] = tokens.input_ids.to(device)
    batch["observation.language.attention_mask"] = tokens.attention_mask.to(device)

    # 시계열 관련 _is_pad 키 제거 (policy에서 사용하지 않음)
    for key in list(batch.keys()):
        if key.endswith("_is_pad") and key != "action_is_pad":
            del batch[key]

    return batch


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SmolVLA Custom Training (Temporal + Delta)")
    parser.add_argument("--steps", type=int, default=100, help="학습 스텝 수")
    parser.add_argument("--batch", type=int, default=4, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument("--output", type=str, default="outputs/smolvla_custom_training",
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
    fps = dataset_metadata.fps

    # 원본 피처 정보 확인
    img_shape_hwc = dataset_metadata.features["observation.images.cam"]["shape"]  # [H, W, C]
    img_c, img_h, img_w = img_shape_hwc[2], img_shape_hwc[0], img_shape_hwc[1]
    state_dim = dataset_metadata.features["observation.state"]["shape"][0]

    print(f"Dataset: {DATASET_ROOT}")
    print(f"  Episodes: {dataset_metadata.total_episodes}  |  Frames: {dataset_metadata.total_frames}  |  FPS: {fps}")
    print(f"  Image: ({img_c}, {img_h}, {img_w})  |  State dim: {state_dim}")
    print(f"  Temporal frames: {N_TEMPORAL_FRAMES} (N-2, N-1, N)")

    # ── 커스텀 피처 정의 ─────────────────────────────────────
    # 3개 시계열 이미지를 독립 카메라처럼 취급 → SmolVLA가 자연스럽게 처리
    input_features = {
        "observation.images.cam_t0": PolicyFeature(type=FeatureType.VISUAL, shape=(img_c, img_h, img_w)),
        "observation.images.cam_t1": PolicyFeature(type=FeatureType.VISUAL, shape=(img_c, img_h, img_w)),
        "observation.images.cam_t2": PolicyFeature(type=FeatureType.VISUAL, shape=(img_c, img_h, img_w)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(state_dim,)),
    }

    print(f"  Input:  {list(input_features.keys())}")
    print(f"  Output: {list(output_features.keys())} (delta angle)")

    # ── SmolVLA 설정 & 모델 생성 ──────────────────────────────
    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=1,            # 단일 스텝 변화량 예측
        n_action_steps=1,
        optimizer_lr=args.lr,
        load_vlm_weights=True,   # 사전학습된 VLM 백본 사용
    )
    policy = SmolVLAPolicy(cfg)

    # 토크나이저 초기화 (SmolVLA 태스크 텍스트 → 언어 토큰)
    tokenizer = AutoTokenizer.from_pretrained(cfg.vlm_model_name)

    # 같은 출력 경로에 저장된 체크포인트가 있으면 이어서 학습한다.
    pretrained_path = output_directory / "model.safetensors"
    if pretrained_path.exists():
        print(f"  기존 모델 발견 → 가중치 로드: {pretrained_path}")
        pretrained_policy = SmolVLAPolicy.from_pretrained(output_directory)
        policy.load_state_dict(pretrained_policy.state_dict())
        del pretrained_policy
    else:
        print("  기존 모델 없음 → 새로 학습 시작")

    policy.train()
    policy.to(device)

    trainable_count = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in policy.parameters())
    print(f"  Model params: {total_count:,} (trainable: {trainable_count:,})")

    # ── delta_timestamps 설정 (시계열 데이터 로드) ───────────
    delta_timestamps = {
        "observation.images.cam": [-2 / fps, -1 / fps, 0.0],  # N-2, N-1, N
        "observation.state": [-2 / fps, 0.0],                  # N-2, N
        "action": [0.0],                                        # placeholder (변환 시 교체됨)
    }

    # ── 데이터셋 & 데이터로더 ────────────────────────────────
    dataset = LeRobotDataset(DATASET_REPO_ID, root=DATASET_ROOT, delta_timestamps=delta_timestamps)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    optimizer = cfg.get_optimizer_preset().build(
        [p for p in policy.parameters() if p.requires_grad]
    )

    # ── 학습 루프 ────────────────────────────────────────────
    training_steps = args.steps
    log_freq = max(1, training_steps // 50)
    loss_history = []

    print(f"\n{'='*60}")
    print(f"Training: {training_steps} steps, batch={args.batch}, lr={args.lr}")
    print(f"  Input:  3 temporal images (N-2, N-1, N) + current angle")
    print(f"  Output: delta angle (N angle - N-2 angle)")
    print(f"  Press 'x' to stop training early")
    print(f"{'='*60}\n")

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            # x 키 입력 시 학습 종료
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key in (b'x', b'X'):
                    print(f"\n  [x] 키 입력 → 학습 조기 종료 (step {step})")
                    done = True
                    break

            batch = transform_batch(batch, device, tokenizer)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if step % 10 == 0:
                target_vals = batch["action"][:, 0, :]   # (B, action_dim)
                print(f"  step {step:>5d}/{training_steps}  |  loss: {loss_val:.4f}"
                      f"  |  target: {target_vals[0].tolist()}")

            step += 1
            if step >= training_steps:
                done = True
                break
            # 손실이 충분히 작아지면 불필요한 추가 학습을 멈춘다.
            if loss_val < 0.0001:
                print(f"\n  loss {loss_val:.6f} < 0.0001 → 학습 조기 종료 (step {step})")
                done = True
                break

    print(f"\nTraining complete! Final loss: {loss_history[-1]:.4f}")

    # ── 체크포인트 저장 ──────────────────────────────────────
    policy.save_pretrained(output_directory)
    print(f"Checkpoint saved: {output_directory}")

    # ── Loss 그래프 ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, linewidth=1.2, color="#2196F3")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("SmolVLA Training Loss — Temporal 3-Frame + Delta Angle", fontsize=14)
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
