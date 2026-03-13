"""act_custom 데이터셋으로 SmolVLA Policy 학습 & MuJoCo 환경 추론

입력 : observation.images.cam (현재 프레임) + observation.state (현재 각도)
       + 언어 지시: "camera follow players"
출력 : 각도 변화량 (delta angle)

SmolVLA는 VLM 백본 + 액션 전문가 모델로 구성된 VLA(Vision-Language-Action) 모델.

사전 설치: pip install -e ".[smolvla]"

사용법 (학습)
  python examples/tutorial/act/act_custom_training2.py
  python examples/tutorial/act/act_custom_training2.py --steps 500 --batch 4

사용법 (추론 — MuJoCo 환경)
  python examples/tutorial/act/act_custom_training2.py --infer
  python examples/tutorial/act/act_custom_training2.py --infer --model outputs/smolvla_custom_training
"""

import math
import msvcrt
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from transformers import AutoTokenizer

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


# ─── 데이터셋 설정 ──────────────────────────────────────────────
DATASET_REPO_ID = "act_custom/cam_turret"
DATASET_ROOT = "outputs/act_custom_dataset"
SINGLE_TASK = "camera follow players"

# ─── MuJoCo 추론 환경 설정 ──────────────────────────────────────
VIDEO_PATH = "examples/tutorial/act/play3.mp4"
VIEW_W, VIEW_H = 800, 450
ANGLE_STEP = math.radians(3)
MAX_ANGLE = math.radians(30)

MUJOCO_XML = """
<mujoco model="cam_turret">
  <option gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="800" offheight="450"/>
  </visual>
  <worldbody>
    <light name="main" pos="0 -1.0 3.5" dir="0 0.4 -1"
           diffuse="0.9 0.9 0.9" specular="0.3 0.3 0.3" castshadow="true"/>
    <light name="fill_left" pos="-2.5 0.5 2.5" dir="1 -0.2 -0.8"
           diffuse="0.35 0.35 0.40" specular="0 0 0" castshadow="false"/>
    <light name="fill_right" pos="2.5 0.5 2.5" dir="-1 -0.2 -0.8"
           diffuse="0.35 0.35 0.40" specular="0 0 0" castshadow="false"/>
    <geom type="plane" size="6 6 0.1" rgba="0.50 0.50 0.50 1"/>
    <body name="base" pos="0 0 0.5">
      <geom type="cylinder" size="0.14 0.07" pos="0 0 1.77" rgba="0.40 0.40 0.40 1"/>
      <body name="turret" pos="0 0 0.14">
        <joint name="pan" type="hinge" axis="0 0 1" range="-0.5236 0.5236" damping="4.0"/>
        <geom type="cylinder" size="0.10 0.10" pos="0 0 0.10" rgba="0.20 0.40 0.80 1"/>
        <body name="cam_body" pos="0 0 0.22">
          <geom type="box" size="0.04 0.04 0.08" rgba="0.15 0.15 0.15 1"/>
          <camera name="cam" pos="0 0.05 0" xyaxes="1 0 0  0 0 1" fovy="35"/>
        </body>
      </body>
    </body>
    <body name="screen_body" pos="0 1.4 0.9">
      <geom type="box" size="2.06 0.025 0.612" rgba="0.06 0.06 0.06 1"/>
      <geom name="screen_surface" type="box" size="2.0 0.010 0.5625"
            pos="0 -0.012 0" rgba="0.05 0.05 0.05 1"/>
    </body>
    <camera name="world_cam" pos="2.0 -1.5 2.2"
            xyaxes="0.832 0.555 0  -0.188 0.282 0.941"/>
  </worldbody>
</mujoco>
"""


# ── MuJoCo 렌더링 유틸리티 (act_custom.py 참조) ────────────────

def get_screen_corners_world(model, data):
    """스크린 표면 geom 4 모서리 월드 좌표."""
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "screen_surface")
    body_id = model.geom_bodyid[geom_id]
    body_pos = data.xpos[body_id].copy()
    body_rot = data.xmat[body_id].reshape(3, 3)
    lpos = model.geom_pos[geom_id]
    lsize = model.geom_size[geom_id]
    geom_center = body_pos + body_rot @ lpos
    hw_x, hw_z, d_y = lsize[0], lsize[2], -lsize[1]
    corners_local = np.array([
        [-hw_x, d_y, -hw_z], [hw_x, d_y, -hw_z],
        [hw_x, d_y, hw_z], [-hw_x, d_y, hw_z],
    ])
    return np.array([geom_center + body_rot @ c for c in corners_local])


def project_to_image(corners_world, gl_cam, view_w, view_h):
    """월드 3D 모서리 → 카메라 2D 픽셀."""
    cam_pos = np.array(gl_cam.pos)
    fwd = np.array(gl_cam.forward)
    up = np.array(gl_cam.up)
    right = np.cross(fwd, up)
    near = gl_cam.frustum_near
    fy = (view_h / 2.0) * near / gl_cam.frustum_top
    fw = gl_cam.frustum_width
    fx = (view_w / 2.0) * near / fw if fw > 1e-9 else fy
    cx, cy = view_w / 2.0, view_h / 2.0
    pts = []
    for p in corners_world:
        dp = p - cam_pos
        depth = max(float(np.dot(dp, fwd)), 1e-6)
        x_c = float(np.dot(dp, right))
        y_c = float(np.dot(dp, up))
        pts.append([cx + fx * x_c / depth, cy - fy * y_c / depth])
    return np.array(pts, dtype=np.float32)


def composite_video(img_bgr, pts_2d, vid_bgr):
    """비디오 프레임을 스크린에 원근 변환하여 합성."""
    h_v, w_v = vid_bgr.shape[:2]
    h_i, w_i = img_bgr.shape[:2]
    src = np.float32([[0, h_v], [w_v, h_v], [w_v, 0], [0, 0]])
    try:
        M = cv2.getPerspectiveTransform(src, pts_2d)
    except cv2.error:
        return img_bgr
    warped = cv2.warpPerspective(vid_bgr, M, (w_i, h_i))
    mask = cv2.warpPerspective(
        np.ones((h_v, w_v), dtype=np.uint8) * 255, M, (w_i, h_i))
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    result = img_bgr.copy()
    result[mask > 0] = warped[mask > 0]
    return result


def label_img(img_bgr, text):
    """이미지 좌상단 텍스트 레이블."""
    cv2.putText(img_bgr, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)


def transform_batch(batch: dict, device: torch.device, tokenizer) -> dict:
    """현재 프레임 이미지 + 언어 지시 + 각도 변화량 타겟으로 변환.

    변환 내용:
      observation.images.cam (B, C, H, W)
        → observation.images.cam (B, C, H, W)  현재 프레임
      observation.state (B, state_dim)
        → observation.state (B, state_dim)  현재 각도
      action
        → (B, 1, state_dim)  각도 변화량
      task (list[str])
        → observation.language.tokens + observation.language.attention_mask
    """
    # 모든 텐서를 device로 이동
    batch = {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    B = batch["observation.images.cam"].shape[0]

    # action을 (B, 1, action_dim) 형태로 맞춘다.
    action = batch["action"]
    if action.dim() == 1:
        action = action.unsqueeze(-1)
    batch["action"] = action.unsqueeze(1) if action.dim() == 2 else action
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
    batch["observation.language.attention_mask"] = tokens.attention_mask.to(device).bool()

    # 불필요한 _is_pad 키 제거
    for key in list(batch.keys()):
        if key.endswith("_is_pad") and key != "action_is_pad":
            del batch[key]

    return batch


def run_inference(model_path: str, device: torch.device) -> None:
    """MuJoCo 환경에서 학습된 SmolVLA 모델로 추론 실행."""
    INFER_FPS = 10

    # ── 비디오 열기 ──────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"play.mp4 열기 실패: {VIDEO_PATH}")
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video: {VIDEO_PATH}  "
          f"{int(cap.get(3))}x{int(cap.get(4))}  {vid_fps:.1f}fps")

    # ── MuJoCo 초기화 ───────────────────────────────────────
    model = mujoco.MjModel.from_xml_string(MUJOCO_XML)
    data = mujoco.MjData(model)
    pan_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pan")
    renderer = mujoco.Renderer(model, height=VIEW_H, width=VIEW_W)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="world_cam")
    renderer.render()

    # ── SmolVLA 모델 로드 ────────────────────────────────────
    print(f"[INF] 모델 로딩: {model_path}")
    policy = SmolVLAPolicy.from_pretrained(model_path)
    policy.eval()
    policy.to(device)
    print(f"[INF] 모델 로드 완료 (device={device})")

    tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)

    # 언어 토큰 사전 생성 (매 프레임 동일)
    tokens = tokenizer([SINGLE_TASK + "\n"], padding="longest", padding_side="right",
                       max_length=48, truncation=True, return_tensors="pt")
    lang_tokens = tokens.input_ids.to(device)
    lang_mask = tokens.attention_mask.to(device).bool()

    # ── 루프 변수 ────────────────────────────────────────────
    target_angle = 0.0
    vid_interval = 1.0 / vid_fps
    next_vid_time = time.monotonic()
    current_bgr = None
    infer_interval = 1.0 / INFER_FPS
    next_infer_time = time.monotonic()
    infer_step = 0

    print(f"\n[조작]  <- -> : 수동 회전  |  ESC/Q : 종료")
    print(f"[INF] 자동 추론 중 ({INFER_FPS}fps)  |  Task: '{SINGLE_TASK}'\n")

    while True:
        key = cv2.waitKeyEx(1)
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (2424832, 65361, 81):   # ←
            target_angle = max(-MAX_ANGLE, target_angle - ANGLE_STEP)
        elif key in (2555904, 65363, 83):   # →
            target_angle = min(MAX_ANGLE, target_angle + ANGLE_STEP)

        # 관절 부드러운 추종
        cur = data.qpos[pan_id]
        data.qpos[pan_id] += (target_angle - cur) * 0.25
        data.qvel[pan_id] = 0.0
        mujoco.mj_forward(model, data)

        # 비디오 프레임 갱신
        now = time.monotonic()
        if now >= next_vid_time:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if ret:
                current_bgr = frame
            next_vid_time += vid_interval

        # ── World view ───────────────────────────────────────
        renderer.update_scene(data, camera="world_cam")
        world_bgr = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)
        if current_bgr is not None:
            corners_w = get_screen_corners_world(model, data)
            pts = project_to_image(corners_w, renderer.scene.camera[0], VIEW_W, VIEW_H)
            if pts is not None:
                world_bgr = composite_video(world_bgr, pts, current_bgr)

        # ── Camera view ──────────────────────────────────────
        renderer.update_scene(data, camera="cam")
        cam_bgr = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)
        if current_bgr is not None:
            pts = project_to_image(corners_w, renderer.scene.camera[0], VIEW_W, VIEW_H)
            if pts is not None:
                cam_bgr = composite_video(cam_bgr, pts, current_bgr)

        # ── 추론: 현재 프레임 + 언어 → delta angle ───────────
        now = time.monotonic()
        if now >= next_infer_time:
            cam_rgb = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB)
            cam_tensor = torch.from_numpy(cam_rgb).float() / 255.0
            cam_tensor = cam_tensor.permute(2, 0, 1)  # (C, H, W)
            cur_angle = float(data.qpos[pan_id])

            with torch.no_grad():
                batch = {
                    "observation.images.cam": cam_tensor.unsqueeze(0).to(device),
                    "observation.state": torch.tensor([[cur_angle]], dtype=torch.float32, device=device),
                    "observation.language.tokens": lang_tokens,
                    "observation.language.attention_mask": lang_mask,
                }
                action = policy.select_action(batch)
                delta = float(action[0, 0].cpu())
                target_angle = -max(-MAX_ANGLE, min(MAX_ANGLE, cur_angle + delta))
                infer_step += 1
                print(
                    f"[INF] step={infer_step:04d}  cur={math.degrees(cur_angle):+6.2f} deg  "
                    f"delta={math.degrees(delta):+7.3f} deg  target={math.degrees(target_angle):+6.2f} deg"
                )

            next_infer_time = now + infer_interval

        # 레이블
        deg = math.degrees(data.qpos[pan_id])
        label_img(world_bgr, f"World View  |  Pan: {deg:+.1f} deg  [INF step={infer_step}]")
        label_img(cam_bgr, f"Camera View  |  [<- ->]  [Q] Quit  |  Task: '{SINGLE_TASK}'")

        cv2.imshow("SmolVLA Inference - MuJoCo", np.vstack([world_bgr, cam_bgr]))

    cap.release()
    renderer.close()
    cv2.destroyAllWindows()
    print(f"추론 종료 (총 {infer_step} steps)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SmolVLA Custom Training (Temporal + Delta)")
    parser.add_argument("--steps", type=int, default=100, help="학습 스텝 수")
    parser.add_argument("--batch", type=int, default=4, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument("--output", type=str, default="outputs/smolvla_custom_training",
                        help="체크포인트 저장 경로")
    parser.add_argument("--infer", action="store_true",
                        help="학습 대신 MuJoCo 환경에서 추론 실행")
    parser.add_argument("--model", type=str, default=None,
                        help="추론에 사용할 모델 경로 (기본: --output 경로)")
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

    # ── 추론 모드 ────────────────────────────────────────────
    if args.infer:
        model_dir = args.model or args.output
        if not Path(model_dir).exists():
            print(f"Error: 모델 경로 없음: {model_dir}")
            return
        run_inference(model_dir, device)
        return

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
    print(f"  Language task: \"{SINGLE_TASK}\"")

    # ── 커스텀 피처 정의 ─────────────────────────────────────
    # 현재 프레임 1장 + 상태 + 언어 지시
    input_features = {
        "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(img_c, img_h, img_w)),
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

    # ── delta_timestamps 설정 ────────────────────────────────
    delta_timestamps = {
        "observation.images.cam": [0.0],   # 현재 프레임만
        "observation.state": [0.0],        # 현재 각도만
        "action": [0.0],
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
    print(f"  Input:  current image + language ('{SINGLE_TASK}') + current angle")
    print(f"  Output: delta angle")
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
    ax.set_title("SmolVLA Training Loss — Single Frame + Language + Delta Angle", fontsize=14)
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
