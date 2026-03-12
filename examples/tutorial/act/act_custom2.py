"""
커스텀 MuJoCo 환경: 회전 관절 + 카메라 + 비디오 스크린

구성
  - 회전 가능한 관절(pan) 위에 카메라 장착
  - 방향키 ← / → : -90° ~ +90° 회전
  - 카메라 정면 스크린에 play.mp4 재생

화면 레이아웃 (2×1)
  ┌──────────────────────┐
  │   World View         │  <- 씬 전체 조망 (고정 카메라)
  ├──────────────────────┤
  │   Camera View        │  <- 관절 위 카메라가 보는 시점
  └──────────────────────┘

조작
  <- / ->  : 좌/우 회전 (3도씩, 최대 +-30도)
  R        : 리플레이 시작 (녹화된 데이터 재생)
  T        : 리플레이 중단
  I        : 추론 시작 (학습된 모델로 자동 제어)
  J        : 추론 중단
  ESC / Q  : 종료

비디오 합성 방식
  MuJoCo UV 텍스처 대신, 카메라 투영(pinhole)으로
  스크린 4 모서리를 2D 픽셀로 변환 -> cv2 원근 변환으로 합성.
  -> UV 왜곡 없이 정확한 영상 출력.
"""

import logging
import math
import os
import time

from collections import deque

import cv2
import numpy as np
import mujoco
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

# ─── 경로 / 크기 설정 ───────────────────────────────────────────
VIDEO_PATH = "examples/tutorial/act/play.mp4"

VIEW_W, VIEW_H = 800, 450      # 각 뷰(world/cam) 렌더 크기

ANGLE_STEP = math.radians(3)   # 키 1회 입력당 회전량
MAX_ANGLE  = math.radians(30)  # 최대 회전 +-30도

# ─── 데이터셋 / 추론 설정 ────────────────────────────────────────
DATASET_REPO_ID = "act_custom/cam_turret"
DATASET_ROOT    = "outputs/act_custom_dataset"
DATASET_FPS     = 10
MODEL_PATH      = "outputs/act_custom_training"
INFERENCE_FPS   = DATASET_FPS
# ────────────────────────────────────────────────────────────────

XML = """
<mujoco model="cam_turret">
  <option gravity="0 0 -9.81"/>

  <visual>
    <global offwidth="800" offheight="450"/>
  </visual>

  <worldbody>
    <!-- 라이트 -->
    <light name="main"
           pos="0 -1.0 3.5" dir="0 0.4 -1"
           diffuse="0.9 0.9 0.9" specular="0.3 0.3 0.3"
           castshadow="true"/>
    <light name="fill_left"
           pos="-2.5 0.5 2.5" dir="1 -0.2 -0.8"
           diffuse="0.35 0.35 0.40" specular="0 0 0"
           castshadow="false"/>
    <light name="fill_right"
           pos="2.5 0.5 2.5" dir="-1 -0.2 -0.8"
           diffuse="0.35 0.35 0.40" specular="0 0 0"
           castshadow="false"/>

    <!-- 바닥 -->
    <geom type="plane" size="6 6 0.1" rgba="0.50 0.50 0.50 1"/>

    <!-- 터렛: 베이스 + 회전 관절 + 카메라 -->
    <body name="base" pos="0 0 0.5">
      <geom type="cylinder" size="0.14 0.07"
            pos="0 0 1.77" rgba="0.40 0.40 0.40 1"/>

      <body name="turret" pos="0 0 0.14">
        <joint name="pan" type="hinge"
               axis="0 0 1" range="-0.5236 0.5236" damping="4.0"/>
        <geom type="cylinder" size="0.10 0.10"
              pos="0 0 0.10" rgba="0.20 0.40 0.80 1"/>

        <body name="cam_body" pos="0 0 0.22">
          <geom type="box" size="0.04 0.04 0.08"
                rgba="0.15 0.15 0.15 1"/>
          <!-- 카메라: Y+ 방향을 바라봄 -->
          <camera name="cam"
                  pos="0 0.05 0"
                  xyaxes="1 0 0  0 0 1"
                  fovy="35"/>
        </body>
      </body>
    </body>

    <!-- 비디오 스크린 (Y=1.4m 앞, 높이 0.9m) -->
    <body name="screen_body" pos="0 1.4 0.9">
      <!-- 검은 테두리 -->
      <geom type="box" size="2.06 0.025 0.612"
            rgba="0.06 0.06 0.06 1"/>
      <!-- 스크린 표면: 어두운 회색 (영상은 2D 합성으로 올림) -->
      <geom name="screen_surface" type="box" size="2.0 0.010 0.5625"
            pos="0 -0.012 0" rgba="0.05 0.05 0.05 1"/>
    </body>

    <!-- 조망 카메라 -->
    <camera name="world_cam"
            pos="2.0 -1.5 2.2"
            xyaxes="0.832 0.555 0  -0.188 0.282 0.941"/>
  </worldbody>
</mujoco>
"""


# ────────────────────────────────────────────────────────────────
#  카메라 투영 유틸리티
# ────────────────────────────────────────────────────────────────

def get_screen_corners_world(model: mujoco.MjModel,
                             data:  mujoco.MjData) -> np.ndarray:
    """스크린 표면 geom의 카메라 쪽 면 4 모서리 (월드 좌표, shape=(4,3)).

    순서: [bottom-left, bottom-right, top-right, top-left]
    """
    geom_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "screen_surface")
    body_id  = model.geom_bodyid[geom_id]

    body_pos = data.xpos[body_id].copy()           # (3,)  world 위치
    body_rot = data.xmat[body_id].reshape(3, 3)    # (3,3) world 회전

    # geom local 위치 / 반-크기
    lpos  = model.geom_pos[geom_id]    # (0, -0.012, 0)
    lsize = model.geom_size[geom_id]   # (1.0, 0.010, 0.5625)

    geom_center = body_pos + body_rot @ lpos

    # 카메라 방향(-Y 로컬) 면의 4 모서리
    hw_x = lsize[0]   # 1.0
    hw_z = lsize[2]   # 0.5625
    d_y  = -lsize[1]  # -0.010  (카메라 쪽 면)

    corners_local = np.array([
        [-hw_x, d_y, -hw_z],   # bottom-left
        [ hw_x, d_y, -hw_z],   # bottom-right
        [ hw_x, d_y,  hw_z],   # top-right
        [-hw_x, d_y,  hw_z],   # top-left
    ])
    return np.array([geom_center + body_rot @ c for c in corners_local])


def project_to_image(corners_world: np.ndarray,
                     gl_cam,
                     view_w: int, view_h: int) -> np.ndarray | None:
    """월드 3D 모서리 -> 카메라 2D 픽셀 (shape=(4,2), float32).

    MjvGLCamera의 frustum 파라미터로 핀홀 투영.
    모서리 중 하나라도 카메라 뒤에 있으면 None 반환.
    """
    cam_pos = np.array(gl_cam.pos)
    fwd     = np.array(gl_cam.forward)   # 단위 벡터, 시선 방향
    up      = np.array(gl_cam.up)        # 단위 벡터, 위쪽
    right   = np.cross(fwd, up)          # 단위 벡터, 오른쪽

    near = gl_cam.frustum_near
    # fy = (H/2) * near / frustum_top
    fy = (view_h / 2.0) * near / gl_cam.frustum_top
    # 대칭 frustum(frustum_width==0)이면 정방 픽셀 가정: fx == fy
    fw = gl_cam.frustum_width
    fx = (view_w / 2.0) * near / fw if fw > 1e-9 else fy
    cx, cy = view_w / 2.0, view_h / 2.0

    pts = []
    for p in corners_world:
        dp    = p - cam_pos
        depth = float(np.dot(dp, fwd))
        if depth <= 1e-6:
            depth = 1e-6                        # 카메라 뒤쪽도 클램핑하여 계속 투영
        x_c = float(np.dot(dp, right))
        y_c = float(np.dot(dp, up))
        u = cx + fx * x_c / depth
        v = cy - fy * y_c / depth              # 이미지 Y 는 아래 방향
        pts.append([u, v])

    return np.array(pts, dtype=np.float32)


def composite_video(img_bgr:      np.ndarray,
                    pts_2d:       np.ndarray,
                    vid_bgr:      np.ndarray) -> np.ndarray:
    """비디오 프레임을 스크린 4 모서리에 원근 변환하여 합성."""
    h_v, w_v = vid_bgr.shape[:2]
    h_i, w_i = img_bgr.shape[:2]

    # 소스: 비디오 프레임 4 모서리 [BL, BR, TR, TL]
    src = np.float32([
        [0,   h_v],
        [w_v, h_v],
        [w_v, 0  ],
        [0,   0  ],
    ])
    dst = pts_2d  # (4,2) 카메라 이미지 픽셀

    try:
        M = cv2.getPerspectiveTransform(src, dst)
    except cv2.error:
        return img_bgr

    warped = cv2.warpPerspective(vid_bgr, M, (w_i, h_i))

    # 마스크: warpPerspective 한 영역
    mask_src = np.ones((h_v, w_v), dtype=np.uint8) * 255
    mask     = cv2.warpPerspective(mask_src, M, (w_i, h_i))
    _, mask  = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    result = img_bgr.copy()
    result[mask > 0] = warped[mask > 0]
    return result


def label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img_bgr, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 255, 255), 2, cv2.LINE_AA)
    return img_bgr


# ────────────────────────────────────────────────────────────────
#  메인
# ────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # 비디오 열기
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"play.mp4 열기 실패: {VIDEO_PATH}")
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video: {VIDEO_PATH}  "
          f"{int(cap.get(3))}x{int(cap.get(4))}  {vid_fps:.1f}fps  "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames")

    # MuJoCo 초기화
    model = mujoco.MjModel.from_xml_string(XML)
    data  = mujoco.MjData(model)
    pan_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pan")

    renderer = mujoco.Renderer(model, height=VIEW_H, width=VIEW_W)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="world_cam")
    renderer.render()   # GL 컨텍스트 초기화

    # ── 추론 상태 ───────────────────────────────────────────
    is_inferring = False
    infer_policy = None
    infer_device = None
    infer_frame_buffer = deque(maxlen=3)   # 최근 3프레임 (N-2, N-1, N) RGB
    infer_interval = 1.0 / INFERENCE_FPS
    next_infer_time = 0.0
    infer_step_count = 0

    # ── 리플레이 상태 ───────────────────────────────────────
    is_replaying = False
    replay_actions = None       # 리플레이할 액션 리스트
    replay_frame_idx = 0        # 현재 리플레이 프레임 인덱스
    replay_total_frames = 0     # 리플레이 총 프레임 수
    replay_episode_idx = 0      # 현재 리플레이 에피소드 인덱스
    replay_num_episodes = 0     # 리플레이 총 에피소드 수

    # 루프 변수
    target_angle  = 0.0
    vid_interval  = 1.0 / vid_fps
    next_vid_time = time.monotonic()   # 즉시 첫 프레임
    current_bgr   = None               # 최신 비디오 프레임 (BGR)

    print("\n[조작]  <- -> : 회전  |  R : 리플레이  |  T : 리플레이 중단  |  I : 추론  |  J : 추론중단  |  ESC/Q : 종료\n")

    while True:
        # 키 입력 (non-blocking)
        key = cv2.waitKeyEx(1)
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (2424832, 65361, 81):   # <- 왼쪽
            target_angle = max(-MAX_ANGLE, target_angle - ANGLE_STEP)
        elif key in (2555904, 65363, 83):   # -> 오른쪽
            target_angle = min(MAX_ANGLE,  target_angle + ANGLE_STEP)
        elif key in (ord('r'), ord('R')):
            if not is_replaying:
                # 데이터셋 존재 여부 확인 후 로드
                replay_data_dir = os.path.join(DATASET_ROOT, "data")
                if os.path.isdir(replay_data_dir):
                    replay_ds = LeRobotDataset(DATASET_REPO_ID, root=DATASET_ROOT)
                    replay_num_episodes = replay_ds.meta.total_episodes
                    if replay_num_episodes > 0:
                        replay_episode_idx = 0
                        ep_frames = replay_ds.hf_dataset.filter(
                            lambda x: x["episode_index"] == replay_episode_idx
                        )
                        replay_actions = ep_frames.select_columns("action")
                        replay_total_frames = len(replay_actions)
                        replay_frame_idx = 0
                        is_replaying = True
                        print(f"[REPLAY] 시작 — 에피소드 {replay_episode_idx}/{replay_num_episodes} ({replay_total_frames} frames)")
                    else:
                        print("[REPLAY] 데이터셋에 에피소드가 없습니다.")
                else:
                    print(f"[REPLAY] 녹화된 데이터가 없습니다: {DATASET_ROOT}")
        elif key in (ord('t'), ord('T')):
            if is_replaying:
                is_replaying = False
                replay_actions = None
                print("[REPLAY] 중단")
        elif key in (ord('i'), ord('I')):
            if not is_inferring and not is_replaying:
                # 모델 로드
                if not os.path.isdir(MODEL_PATH):
                    print(f"[INFER] 모델 경로 없음: {MODEL_PATH}")
                else:
                    if torch.cuda.is_available():
                        infer_device = torch.device("cuda")
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        infer_device = torch.device("mps")
                    else:
                        infer_device = torch.device("cpu")
                    # 학습 시 사용한 피처 정의 재구성
                    img_c, img_h, img_w = 3, VIEW_H, VIEW_W
                    state_dim = 1
                    inf_input_features = {
                        "observation.images.cam_t0": PolicyFeature(type=FeatureType.VISUAL, shape=(img_c, img_h, img_w)),
                        "observation.images.cam_t1": PolicyFeature(type=FeatureType.VISUAL, shape=(img_c, img_h, img_w)),
                        "observation.images.cam_t2": PolicyFeature(type=FeatureType.VISUAL, shape=(img_c, img_h, img_w)),
                        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
                    }
                    inf_output_features = {
                        "action": PolicyFeature(type=FeatureType.ACTION, shape=(state_dim,)),
                    }
                    inf_cfg = ACTConfig(
                        input_features=inf_input_features,
                        output_features=inf_output_features,
                        chunk_size=1,
                        n_action_steps=1,
                    )
                    infer_policy = ACTPolicy(inf_cfg)
                    infer_policy.load_state_dict(ACTPolicy.from_pretrained(MODEL_PATH).state_dict())
                    infer_policy.eval()
                    infer_policy.to(infer_device)
                    infer_frame_buffer.clear()
                    infer_step_count = 0
                    next_infer_time = time.monotonic()
                    is_inferring = True
                    print(f"[INFER] 추론 시작 (모델: {MODEL_PATH}, device: {infer_device})")
        elif key in (ord('j'), ord('J')):
            if is_inferring:
                is_inferring = False
                infer_policy = None
                infer_frame_buffer.clear()
                print("[INFER] 추론 중단")

        # ── 리플레이: 액션 적용 ──────────────────────────────
        if is_replaying and replay_actions is not None:
            if replay_frame_idx < replay_total_frames:
                action_val = replay_actions[replay_frame_idx]["action"]
                target_angle = float(action_val.item()) if action_val.dim() == 0 else float(action_val[0])
                replay_frame_idx += 1
            else:
                # 현재 에피소드 완료 → 다음 에피소드 시도
                replay_episode_idx += 1
                if replay_episode_idx < replay_num_episodes:
                    ep_frames = replay_ds.hf_dataset.filter(
                        lambda x, eidx=replay_episode_idx: x["episode_index"] == eidx
                    )
                    replay_actions = ep_frames.select_columns("action")
                    replay_total_frames = len(replay_actions)
                    replay_frame_idx = 0
                    print(f"[REPLAY] 다음 에피소드 {replay_episode_idx}/{replay_num_episodes} ({replay_total_frames} frames)")
                else:
                    is_replaying = False
                    replay_actions = None
                    print("[REPLAY] 모든 에피소드 재생 완료")

        # 관절 부드러운 추종
        cur = data.qpos[pan_id]
        data.qpos[pan_id] += (target_angle - cur) * 0.25
        data.qvel[pan_id]  = 0.0
        mujoco.mj_forward(model, data)

        # 비디오 프레임 갱신 (시간 기반)
        now = time.monotonic()
        if now >= next_vid_time:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if ret:
                current_bgr = frame
            next_vid_time += vid_interval

        # ── World view 렌더 ──────────────────────────────────
        renderer.update_scene(data, camera="world_cam")
        world_bgr = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)

        # 스크린 모서리 투영 -> 비디오 합성 (world_cam 시점)
        if current_bgr is not None:
            corners_w = get_screen_corners_world(model, data)
            pts_world = project_to_image(
                corners_w, renderer.scene.camera[0], VIEW_W, VIEW_H)
            if pts_world is not None:
                world_bgr = composite_video(world_bgr, pts_world, current_bgr)

        # ── Camera view 렌더 ─────────────────────────────────
        renderer.update_scene(data, camera="cam")
        cam_bgr = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)

        # 스크린 모서리 투영 -> 비디오 합성 (cam 시점)
        if current_bgr is not None:
            pts_cam = project_to_image(
                corners_w, renderer.scene.camera[0], VIEW_W, VIEW_H)
            if pts_cam is not None:
                cam_bgr = composite_video(cam_bgr, pts_cam, current_bgr)

        # ── 추론: 프레임 버퍼에 저장 & 모델 실행 ─────────────
        if is_inferring and infer_policy is not None and now >= next_infer_time:
            # 카메라 뷰 이미지를 RGB float [0,1]로 변환하여 버퍼에 추가
            cam_rgb_infer = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB)
            cam_tensor = torch.from_numpy(cam_rgb_infer).float() / 255.0  # (H, W, C)
            cam_tensor = cam_tensor.permute(2, 0, 1)  # (C, H, W)
            infer_frame_buffer.append(cam_tensor)

            # 3프레임이 모이면 추론 실행
            if len(infer_frame_buffer) >= 3:
                with torch.no_grad():
                    frames = list(infer_frame_buffer)  # [N-2, N-1, N] 각 (C, H, W)
                    cur_angle = data.qpos[pan_id]
                    batch_infer = {
                        "observation.images.cam_t0": frames[0].unsqueeze(0).to(infer_device),
                        "observation.images.cam_t1": frames[1].unsqueeze(0).to(infer_device),
                        "observation.images.cam_t2": frames[2].unsqueeze(0).to(infer_device),
                        "observation.state": torch.tensor([[cur_angle]], dtype=torch.float32).to(infer_device),
                    }
                    action = infer_policy.select_action(batch_infer)  # (1, action_dim)
                    delta = float(action[0, 0].cpu())
                    target_angle = max(-MAX_ANGLE, min(MAX_ANGLE, cur_angle + delta))
                    infer_step_count += 1

            next_infer_time = now + infer_interval

        # 레이블
        deg = math.degrees(data.qpos[pan_id])
        if is_inferring:
            status = f"  [INFER step={infer_step_count}]"
        elif is_replaying:
            status = f"  [REPLAY {replay_frame_idx}/{replay_total_frames} ep{replay_episode_idx}]"
        else:
            status = ""
        label(world_bgr, f"World View  |  Pan: {deg:+.1f} deg{status}")
        label(cam_bgr,   "Camera View  |  [<- ->] [R]Replay [T]Stop [I]Infer [J]Stop [Q]Quit")

        # 화면 표시
        cv2.imshow("Camera Turret - MuJoCo",
                   np.vstack([world_bgr, cam_bgr]))

    # ── 종료 처리 ────────────────────────────────────────────

    cap.release()
    renderer.close()
    cv2.destroyAllWindows()
    print("종료.")


if __name__ == "__main__":
    main()
