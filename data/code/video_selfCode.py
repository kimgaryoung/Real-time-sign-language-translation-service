"""
직접영상제작 폴더의 동영상을 MediaPipe로 처리하여 키포인트 JSON 파일로 변환

============================================================
[Windows  필독]
============================================================
1. 동영상 확장자가 .mp4인지 확인하세요.
   - .mov 파일은 Windows에서 인식이 안 될 수 있습니다.
   - .mov → .mp4 변환 권장 (온라인 변환기 또는 FFmpeg 사용)

2. .mov 파일을 그대로 사용하려면 K-Lite Codec Pack 설치:
   - 다운로드: https://codecguide.com/download_kl.htm
   - "Basic" 또는 "Standard" 버전 설치 후 재부팅

============================================================
실행 방법 (터미널):
============================================================
[macOS]
1. 가상환경 활성화:
   conda activate py311_env

2. 스크립트 실행:
   cd /Users/garyeong/Desktop/Real-time-sign-language-translation-service/data/code
   python video_selfCode.py

[Windows]
1. 가상환경 활성화:
   conda activate py311_env

2. 스크립트 실행:
   cd C:\Users\사용자\Desktop\Real-time-sign-language-translation-service\data\code
   python video_selfCode.py

============================================================
실행 전 설정 (맨 아래 if __name__ == "__main__": 부분):
============================================================
- video_folder: 입력 동영상 폴더 경로
- output_base_folder: 출력 JSON 폴더 경로
- skip_mode: 빈 프레임 필터 모드 ("all", "hands", "hands_or_face", "none")
============================================================
"""
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import sys
import glob
import platform
from pathlib import Path

# ==================== Windows 호환성 설정 ====================
def setup_windows_compatibility():
    """Windows 환경에서의 호환성 설정"""
    if platform.system() == 'Windows':
        # 콘솔 한글 출력 인코딩 설정
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')

        # Windows 콘솔 코드페이지 설정 (선택적)
        try:
            os.system('chcp 65001 > nul 2>&1')
        except:
            pass

    print(f"운영체제: {platform.system()}")
    print(f"Python 인코딩: {sys.stdout.encoding}")

# 시작 시 호환성 설정 실행
setup_windows_compatibility()


#conda activate py311_env
#


# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


def is_empty_keypoints(keypoint_data, mode="all"):
    """
    키포인트 데이터가 비어있는지 확인

    Args:
        keypoint_data: 키포인트 딕셔너리
        mode: 빈 프레임 판단 기준
            - "all": 모든 키포인트가 0일 때만 빈 프레임 (기본값)
            - "hands": 양손 모두 감지 안 되면 빈 프레임
            - "hands_or_face": 손과 얼굴 모두 감지 안 되면 빈 프레임
            - "none": 필터링 안 함 (모든 프레임 저장)

    Returns:
        bool: 빈 프레임이면 True
    """
    if mode == "none":
        return False

    if not keypoint_data["people"]:
        return True

    person = keypoint_data["people"][0]

    if mode == "all":
        # 모든 키포인트가 0인지 확인
        all_keypoints = (
            person["pose_keypoints_2d"] +
            person["face_keypoints_2d"] +
            person["hand_left_keypoints_2d"] +
            person["hand_right_keypoints_2d"]
        )
        return all(v == 0.0 for v in all_keypoints)

    elif mode == "hands":
        # 손 키포인트만 확인 (왼손 또는 오른손 감지되면 유효)
        left_detected = any(v != 0.0 for v in person["hand_left_keypoints_2d"])
        right_detected = any(v != 0.0 for v in person["hand_right_keypoints_2d"])
        return not (left_detected or right_detected)

    elif mode == "hands_or_face":
        # 손 또는 얼굴이 감지되면 유효
        left_detected = any(v != 0.0 for v in person["hand_left_keypoints_2d"])
        right_detected = any(v != 0.0 for v in person["hand_right_keypoints_2d"])
        face_detected = any(v != 0.0 for v in person["face_keypoints_2d"])
        return not (left_detected or right_detected or face_detected)

    return False


def extract_keypoints_from_frame(frame, pose, hands, face_mesh):
    """
    단일 프레임에서 키포인트 추출 (model.ipynb의 411차원 형식)

    Returns:
        dict: OpenPose 형식의 키포인트 데이터
              {
                  "people": [{
                      "pose_keypoints_2d": [75개],    # 25 landmarks * 3 (x, y, confidence)
                      "face_keypoints_2d": [210개],   # 70 landmarks * 3
                      "hand_left_keypoints_2d": [63개],   # 21 landmarks * 3
                      "hand_right_keypoints_2d": [63개]   # 21 landmarks * 3
                  }]
              }
    """
    # BGR → RGB 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe 처리
    pose_results = pose.process(rgb_frame)
    hands_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    # 1. Pose keypoints (25 landmarks * 3 = 75)
    pose_kps = []
    if pose_results.pose_landmarks:
        for i in range(25):  # 처음 25개 랜드마크만 사용
            if i < len(pose_results.pose_landmarks.landmark):
                lm = pose_results.pose_landmarks.landmark[i]
                pose_kps.extend([lm.x, lm.y, lm.visibility])
            else:
                pose_kps.extend([0.0, 0.0, 0.0])
    else:
        pose_kps = [0.0] * 75

    # 2. Face keypoints (70 landmarks * 3 = 210)
    face_kps = []
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        for i in range(70):  # 처음 70개 랜드마크만 사용
            if i < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[i]
                face_kps.extend([lm.x, lm.y, lm.z])
            else:
                face_kps.extend([0.0, 0.0, 0.0])
    else:
        face_kps = [0.0] * 210

    # 3. Hand keypoints (21 * 3 * 2 = 126)
    left_hand_kps = [0.0] * 63
    right_hand_kps = [0.0] * 63

    if hands_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            handedness = hands_results.multi_handedness[idx].classification[0].label
            hand_kps = []
            for lm in hand_landmarks.landmark:
                hand_kps.extend([lm.x, lm.y, lm.z])

            if handedness == "Left":
                left_hand_kps = hand_kps
            elif handedness == "Right":
                right_hand_kps = hand_kps

    # OpenPose 형식으로 반환 (총 75 + 210 + 63 + 63 = 411)
    keypoint_data = {
        "people": [{
            "pose_keypoints_2d": pose_kps,
            "face_keypoints_2d": face_kps,
            "hand_left_keypoints_2d": left_hand_kps,
            "hand_right_keypoints_2d": right_hand_kps
        }]
    }

    return keypoint_data


def process_video_to_json(video_path, output_folder, label_name, skip_mode="all"):
    """
    동영상을 프레임별로 처리하여 JSON 키포인트 파일들로 저장

    Args:
        video_path: 입력 동영상 경로
        output_folder: 출력 폴더 경로
        label_name: 라벨 이름 (예: 'ㄱ', 'ㄴ', ...)
        skip_mode: 빈 프레임 건너뛰기 기준
            - "all": 모든 키포인트가 0일 때만 건너뛰기 (기본값)
            - "hands": 손이 감지 안 되면 건너뛰기
            - "hands_or_face": 손과 얼굴 모두 없으면 건너뛰기
            - "none": 건너뛰지 않음 (모든 프레임 저장)
    """
    # Path 객체로 변환 (호환성)
    output_folder = Path(output_folder)

    # 출력 폴더 생성
    output_folder.mkdir(parents=True, exist_ok=True)

    # 동영상 열기 (경로를 문자열로 변환하여 호환성 확보)
    video_path_str = str(video_path)

    # Windows에서 한글 경로 처리
    if platform.system() == 'Windows':
        # cv2.VideoCapture가 한글 경로를 못 읽을 경우 대비
        try:
            cap = cv2.VideoCapture(video_path_str)
        except:
            # numpy를 통한 우회 방법
            cap = cv2.VideoCapture(video_path_str, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(video_path_str)

    if not cap.isOpened():
        print(f"동영상을 열 수 없습니다: {video_path}")
        if platform.system() == 'Windows':
            print("  [Windows 해결 방법]")
            print("  1. .mov 파일인 경우: K-Lite Codec Pack 설치 또는 .mp4로 변환")
            print("  2. 한글 경로 문제: 영문 경로로 이동 후 재시도")
            print("  3. 경로에 특수문자가 없는지 확인")
        return False

    # 동영상 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n 처리 중: {label_name}")
    print(f"   경로: {video_path}")
    print(f"   FPS: {fps}, 총 프레임: {total_frames}")

    # MediaPipe 초기화
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands, mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        frame_idx = 0
        saved_frames = 0
        skipped_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 키포인트 추출
            keypoint_data = extract_keypoints_from_frame(frame, pose, hands, face_mesh)

            # 빈 프레임 체크 및 건너뛰기
            if is_empty_keypoints(keypoint_data, mode=skip_mode):
                frame_idx += 1
                skipped_frames += 1
                continue

            # JSON 파일로 저장 (OpenPose 형식: {video_id}_{frame_number}_keypoints.json)
            json_filename = f"{label_name}_{frame_idx:06d}_keypoints.json"
            json_path = output_folder / json_filename

            with open(str(json_path), 'w', encoding='utf-8') as f:
                json.dump(keypoint_data, f, ensure_ascii=False, indent=2)

            frame_idx += 1
            saved_frames += 1

            # 진행 상황 표시
            if frame_idx % 10 == 0:
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                print(f"   진행: {frame_idx}/{total_frames} ({progress:.1f}%)", end='\r')

        cap.release()
        print(f"\n   완료: {saved_frames}개 저장 / {skipped_frames}개 건너뜀 (총 {frame_idx}프레임)")
        return True


def create_morpheme_label_file(output_folder, label_name):
    """
    model.ipynb에서 사용하는 형식의 라벨 JSON 파일 생성

    형식:
    {
        "data": [{
            "attributes": [{
                "name": "ㄱ"
            }]
        }]
    }
    """
    label_data = {
        "data": [{
            "attributes": [{
                "name": label_name
            }]
        }]
    }

    # 라벨 파일 저장
    label_filename = f"{label_name}_morpheme.json"
    label_path = Path(output_folder) / label_filename

    with open(str(label_path), 'w', encoding='utf-8') as f:
        json.dump(label_data, f, ensure_ascii=False, indent=2)

    print(f"라벨 파일 생성: {label_filename}")


def process_all_videos(video_folder, output_base_folder, skip_mode="all"):
    """
    직접영상제작 폴더의 모든 동영상을 처리

    Args:
        video_folder: 입력 동영상 폴더 (직접영상제작)
        output_base_folder: 출력 기본 폴더
        skip_mode: 빈 프레임 건너뛰기 기준
            - "all": 모든 키포인트가 0일 때만 건너뛰기 (기본값)
            - "hands": 손이 감지 안 되면 건너뛰기
            - "hands_or_face": 손과 얼굴 모두 없으면 건너뛰기
            - "none": 건너뛰지 않음 (모든 프레임 저장)
    """
    # Path 객체로 변환 (호환성)
    video_folder = Path(video_folder)
    output_base_folder = Path(output_base_folder)

    # 동영상 파일 찾기 (Path.glob 사용으로 호환성 향상)
    video_files = list(video_folder.glob("*.mov"))
    video_files.extend(list(video_folder.glob("*.mp4")))
    video_files.extend(list(video_folder.glob("*.MOV")))  # 대문자 확장자
    video_files.extend(list(video_folder.glob("*.MP4")))
    video_files.extend(list(video_folder.glob("*.avi")))  # AVI 지원 추가
    video_files.extend(list(video_folder.glob("*.AVI")))
    video_files = sorted(set(video_files))  # 중복 제거 및 정렬

    if not video_files:
        print(" 동영상 파일을 찾을 수 없습니다!")
        return

    skip_mode_desc = {
        "all": "모든 키포인트가 0일 때만 건너뛰기",
        "hands": "손이 감지 안 되면 건너뛰기",
        "hands_or_face": "손과 얼굴 모두 없으면 건너뛰기",
        "none": "건너뛰지 않음 (모든 프레임 저장)"
    }

    print(f"\n{'='*70}")
    print(f"직접영상제작 폴더 동영상 → JSON 키포인트 변환")
    print(f"{'='*70}")
    print(f"입력 폴더: {video_folder}")
    print(f"출력 폴더: {output_base_folder}")
    print(f"총 동영상: {len(video_files)}개")
    print(f"빈 프레임 필터: {skip_mode} ({skip_mode_desc.get(skip_mode, '알 수 없음')})")
    print(f"{'='*70}\n")

    # 각 동영상 처리
    success_count = 0
    fail_count = 0

    for idx, video_path in enumerate(video_files, 1):
        # 파일명에서 라벨 추출 (예: 'ㄱ.mov' → 'ㄱ')
        # Path 객체 사용으로 OS 호환성 확보
        label_name = video_path.stem  # 확장자 제외한 파일명

        # 출력 폴더: output_base_folder/label_name/
        output_folder = output_base_folder / label_name

        print(f"\n[{idx}/{len(video_files)}] {label_name}")

        # 동영상 처리
        success = process_video_to_json(video_path, output_folder, label_name, skip_mode)

        if success:
            # 라벨 파일 생성
            create_morpheme_label_file(output_folder, label_name)
            success_count += 1
        else:
            fail_count += 1

    # 결과 요약
    print(f"\n{'='*70}")
    print(f"변환 완료!")
    print(f"{'='*70}")
    print(f" 성공: {success_count}개")
    print(f" 실패: {fail_count}개")
    print(f"\n출력 구조:")
    print(f"  {output_base_folder}/")
    print(f"  ├── ㄱ/")
    print(f"  │   ├── ㄱ_000000_keypoints.json")
    print(f"  │   ├── ㄱ_000001_keypoints.json")
    print(f"  │   ├── ...")
    print(f"  │   └── ㄱ_morpheme.json")
    print(f"  ├── ㄴ/")
    print(f"  │   └── ...")
    print(f"  └── ...")
    print(f"\n 이제 model.ipynb에서 이 데이터를 사용할 수 있습니다!")
    print(f"{'='*70}")


if __name__ == "__main__":
    # ============================================================
    # 경로 설정 (운영체제에 맞게 수정하세요)
    # ============================================================
    if platform.system() == 'Windows':
        # Windows 경로 예시 (본인 환경에 맞게 수정)
        video_folder = r"C:\Users\사용자\Desktop\Real-time-sign-language-translation-service\data\set\가령_영상데이터"
        output_base_folder = r"C:\Users\사용자\Desktop\Real-time-sign-language-translation-service\data\set\가령_영상데이터"
    else:
        # macOS / Linux 경로
        video_folder = "/Users/garyeong/Desktop/Real-time-sign-language-translation-service/data/set/가령_영상데이터"
        output_base_folder = "/Users/garyeong/Desktop/Real-time-sign-language-translation-service/data/set/가령_영상데이터"

    # Path 객체로 변환 (경로 호환성 향상)
    video_folder = Path(video_folder)
    output_base_folder = Path(output_base_folder)

    # ============================================================
    # 빈 프레임 필터 모드 설정 
    # ============================================================
    # "all"           : 모든 키포인트가 0일 때만 건너뛰기 (기본값)
    # "hands"         : 손이 감지 안 되면 건너뛰기
    # "hands_or_face" : 손과 얼굴 모두 없으면 건너뛰기
    # "none"          : 건너뛰지 않음 (모든 프레임 저장)
    # ============================================================
    skip_mode = "hands"

    # 폴더 존재 확인
    if not video_folder.exists():
        print(f"입력 폴더가 존재하지 않습니다: {video_folder}")
        if platform.system() == 'Windows':
            print("\n[Windows 사용자 안내]")
            print("위의 경로를 본인 환경에 맞게 수정하세요.")
            print("예: r\"C:\\Users\\본인이름\\Desktop\\...\"")
        sys.exit(1)

    # 전체 동영상 처리
    process_all_videos(video_folder, output_base_folder, skip_mode)

    print(f"\n\n다음 단계:")
    print(f"1. 생성된 JSON 파일 확인: {output_base_folder}")
    print(f"2. model.ipynb에서 label_base_dir, keypoint_base_dir 경로 수정")
    print(f"3. model.ipynb 실행하여 학습 데이터 생성")
