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
  S        : 레코딩 시작
  E        : 레코딩 종료 (에피소드 저장)
  I        : 추론 시작 (학습된 모델 로드)
  J        : 추론 정지
  ESC / Q  : 종료

비디오 합성 방식
  MuJoCo UV 텍스처 대신, 카메라 투영(pinhole)으로
  스크린 4 모서리를 2D 픽셀로 변환 -> cv2 원근 변환으로 합성.
  -> UV 왜곡 없이 정확한 영상 출력.
"""

import logging
import math
import os
import shutil
import time

import cv2
import numpy as np
import mujoco
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# ─── 경로 / 크기 설정 ───────────────────────────────────────────
VIDEO_PATH = "examples/tutorial/act/play.mp4"

VIEW_W, VIEW_H = 800, 450      # 각 뷰(world/cam) 렌더 크기

ANGLE_STEP = math.radians(3)   # 키 1회 입력당 회전량
MAX_ANGLE  = math.radians(30)  # 최대 회전 +-30도

# ─── 레코딩 설정 ─────────────────────────────────────────────────
DATASET_REPO_ID = "act_custom/cam_turret"
DATASET_ROOT    = "outputs/act_custom_dataset"
DATASET_FPS     = 10
SINGLE_TASK     = "Rotate camera turret to track target"

# ─── 추론 설정 ───────────────────────────────────────────────────
MODEL_PATH      = "outputs/act_custom_training"
INFERENCE_FPS   = DATASET_FPS

DATASET_FEATURES = {
    "observation.images.cam": {
        "dtype": "image",
        "shape": (VIEW_H, VIEW_W, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["pan_angle"],
    },
    "action": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["pan_target"],
    },
}
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

    # ── LeRobot 데이터셋 초기화 ─────────────────────────────
    dataset_data_dir = os.path.join(DATASET_ROOT, "data")
    dataset_meta_path = os.path.join(DATASET_ROOT, "meta", "info.json")
    if os.path.exists(dataset_meta_path) and os.path.isdir(dataset_data_dir):
        # 에피소드가 저장된 완전한 데이터셋 → 로드
        dataset = LeRobotDataset(DATASET_REPO_ID, root=DATASET_ROOT)
        logging.info(f"기존 데이터셋 로드: {dataset.meta.total_episodes} 에피소드")
    else:
        # 데이터셋이 없거나 불완전(에피소드 없음) → (재)생성
        if os.path.exists(DATASET_ROOT):
            shutil.rmtree(DATASET_ROOT)
        dataset = LeRobotDataset.create(
            repo_id=DATASET_REPO_ID,
            fps=DATASET_FPS,
            root=DATASET_ROOT,
            robot_type="cam_turret",
            features=DATASET_FEATURES,
            use_videos=False,
        )
    is_recording = False
    episode_count = dataset.meta.total_episodes
    frame_count = 0
    rec_interval = 1.0 / DATASET_FPS
    next_rec_time = 0.0

    logging.info(f"Dataset initialized: {DATASET_ROOT}")

    # ── 추론 상태 ────────────────────────────────────────────
    is_inferring = False
    infer_policy = None
    infer_preprocessor = None
    infer_postprocessor = None
    infer_device = None
    infer_interval = 1.0 / INFERENCE_FPS
    next_infer_time = 0.0

    # 루프 변수
    target_angle  = 0.0
    vid_interval  = 1.0 / vid_fps
    next_vid_time = time.monotonic()   # 즉시 첫 프레임
    current_bgr   = None               # 최신 비디오 프레임 (BGR)

    print("\n[조작]  <- -> : 회전  |  S : 레코딩  |  E : 레코딩 종료  |  I : 추론  |  J : 추론정지  |  ESC/Q : 종료\n")

    while True:
        # 키 입력 (non-blocking)
        key = cv2.waitKeyEx(1)
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (2424832, 65361, 81):   # <- 왼쪽
            target_angle = max(-MAX_ANGLE, target_angle - ANGLE_STEP)
        elif key in (2555904, 65363, 83):   # -> 오른쪽
            target_angle = min(MAX_ANGLE,  target_angle + ANGLE_STEP)
        elif key in (ord('s'), ord('S')):
            if not is_recording:
                is_recording = True
                frame_count = 0
                next_rec_time = time.monotonic()
                print(f"[REC] 레코딩 시작 (에피소드 {episode_count}, {DATASET_FPS}fps)")
        elif key in (ord('e'), ord('E')):
            if is_recording:
                is_recording = False
                if frame_count > 0:
                    dataset.save_episode()
                    print(f"[REC] 레코딩 종료 — 에피소드 {episode_count} 저장 ({frame_count} frames)")
                    episode_count += 1
                else:
                    dataset.clear_episode_buffer()
                    print("[REC] 레코딩 종료 — 프레임 없음, 저장 안 함")
        elif key in (ord('i'), ord('I')):
            if not is_inferring:
                try:
                    if infer_policy is None:
                        print(f"[INF] 모델 로딩 중: {MODEL_PATH}")
                        if torch.cuda.is_available():
                            infer_device = torch.device("cuda")
                        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                            infer_device = torch.device("mps")
                        else:
                            infer_device = torch.device("cpu")
                        infer_policy = ACTPolicy.from_pretrained(MODEL_PATH)
                        infer_policy.to(infer_device)
                        infer_policy.eval()
                        ds_meta = LeRobotDatasetMetadata(DATASET_REPO_ID, root=DATASET_ROOT)
                        infer_preprocessor, infer_postprocessor = make_pre_post_processors(
                            infer_policy.config,
                            pretrained_path=MODEL_PATH,
                            dataset_stats=ds_meta.stats,
                        )
                        print(f"[INF] 모델 로드 완료 (device={infer_device})")
                    else:
                        infer_policy.reset()
                    is_inferring = True
                    next_infer_time = time.monotonic()
                    print("[INF] 추론 시작")
                except Exception as e:
                    print(f"[INF] 모델 로드 실패: {e}")
        elif key in (ord('j'), ord('J')):
            if is_inferring:
                is_inferring = False
                print("[INF] 추론 정지")

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

        # ── 추론: 모델이 예측한 action으로 관절 제어 ────────
        if is_inferring and now >= next_infer_time:
            cam_rgb = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(cam_rgb).float().permute(2, 0, 1).unsqueeze(0).to(infer_device)
            state_tensor = torch.tensor([[data.qpos[pan_id]]], dtype=torch.float32, device=infer_device)
            obs_dict = {
                "observation.images.cam": img_tensor,
                "observation.state": state_tensor,
            }
            obs_batch = infer_preprocessor(obs_dict)
            with torch.no_grad():
                action = infer_policy.select_action(obs_batch)
            action = infer_postprocessor(action)
            pred_angle = action.squeeze().item()
            target_angle = max(-MAX_ANGLE, min(MAX_ANGLE, pred_angle))
            next_infer_time = now + infer_interval

        # ── 레코딩: 프레임 저장 (10fps 간격) ────────────────
        if is_recording and now >= next_rec_time:
            # cam_bgr은 BGR이므로 RGB로 변환하여 저장
            cam_rgb = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB)
            frame_data = {
                "observation.images.cam": cam_rgb,
                "observation.state": np.array([data.qpos[pan_id]], dtype=np.float32),
                "action": np.array([target_angle], dtype=np.float32),
                "task": SINGLE_TASK,
            }
            dataset.add_frame(frame_data)
            frame_count += 1
            next_rec_time = now + rec_interval

        # 레이블
        deg = math.degrees(data.qpos[pan_id])
        status = f"  [REC {frame_count}]" if is_recording else ""
        if is_inferring:
            status += "  [INF]"
        label(world_bgr, f"World View  |  Pan: {deg:+.1f} deg{status}")
        label(cam_bgr,   "Camera View  |  [<- ->] [S]Rec [E]Stop [I]Infer [J]Stop [Q]Quit")

        # 화면 표시
        cv2.imshow("Camera Turret - MuJoCo",
                   np.vstack([world_bgr, cam_bgr]))

    # ── 종료 처리 ────────────────────────────────────────────
    # 레코딩 중 종료 시 현재 에피소드 저장
    if is_recording and frame_count > 0:
        dataset.save_episode()
        print(f"[REC] 종료 시 에피소드 {episode_count} 자동 저장 ({frame_count} frames)")
        episode_count += 1

    dataset.finalize()
    print(f"Dataset 확정: 총 {episode_count} 에피소드 저장됨 -> {DATASET_ROOT}")

    cap.release()
    renderer.close()
    cv2.destroyAllWindows()
    print("종료.")


if __name__ == "__main__":
    main()
