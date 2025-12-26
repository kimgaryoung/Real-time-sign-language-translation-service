"""
직접영상제작 폴더의 동영상을 MediaPipe로 처리하여 키포인트 JSON 파일로 변환
model.ipynb의 데이터 로딩 방식과 동일한 구조로 저장
"""
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import glob
from pathlib import Path


#conda activate py311_env
#


# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


if __name__ == "__main__":
    # 경로 설정
    video_folder = "/Users/garyeong/project-1/직접영상제작_단어" # 동영상 단어 경로 
    output_base_folder = "/Users/garyeong/project-1/dataset/자체제작_단어_keypoints" 

    # 폴더 존재 확인
    if not os.path.exists(video_folder):
        print(f" 입력 폴더가 존재하지 않습니다: {video_folder}")
        exit(1)

    # 전체 동영상 처리
    process_all_videos(video_folder, output_base_folder)

    print(f"\n\n다음 단계:")
    print(f"1. 생성된 JSON 파일 확인: {output_base_folder}")
    print(f"2. model.ipynb에서 label_base_dir, keypoint_base_dir 경로 수정")
    print(f"3. model.ipynb 실행하여 학습 데이터 생성")



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


def process_video_to_json(video_path, output_folder, label_name):
    """
    동영상을 프레임별로 처리하여 JSON 키포인트 파일들로 저장

    Args:
        video_path: 입력 동영상 경로
        output_folder: 출력 폴더 경로
        label_name: 라벨 이름 (예: 'ㄱ', 'ㄴ', ...)
    """
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # 동영상 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"동영상을 열 수 없습니다: {video_path}")
        return False

    # 동영상 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n📹 처리 중: {label_name}")
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
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 키포인트 추출
            keypoint_data = extract_keypoints_from_frame(frame, pose, hands, face_mesh)

            # JSON 파일로 저장 (OpenPose 형식: {video_id}_{frame_number}_keypoints.json)
            json_filename = f"{label_name}_{frame_idx:06d}_keypoints.json"
            json_path = os.path.join(output_folder, json_filename)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(keypoint_data, f, ensure_ascii=False, indent=2)

            frame_idx += 1
            processed_frames += 1

            # 진행 상황 표시
            if frame_idx % 10 == 0:
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                print(f"   진행: {frame_idx}/{total_frames} ({progress:.1f}%)", end='\r')

        cap.release()
        print(f"\n 완료: {processed_frames}개 프레임 처리됨")
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
    label_path = os.path.join(output_folder, label_filename)

    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(label_data, f, ensure_ascii=False, indent=2)

    print(f" 라벨 파일 생성: {label_filename}")


def process_all_videos(video_folder, output_base_folder):
    """
    직접영상제작 폴더의 모든 동영상을 처리

    Args:
        video_folder: 입력 동영상 폴더 (직접영상제작)
        output_base_folder: 출력 기본 폴더
    """
    # 동영상 파일 찾기
    video_files = glob.glob(os.path.join(video_folder, "*.mov"))
    video_files.extend(glob.glob(os.path.join(video_folder, "*.mp4")))
    video_files = sorted(video_files)

    if not video_files:
        print(" 동영상 파일을 찾을 수 없습니다!")
        return

    print(f"\n{'='*70}")
    print(f"직접영상제작 폴더 동영상 → JSON 키포인트 변환")
    print(f"{'='*70}")
    print(f"입력 폴더: {video_folder}")
    print(f"출력 폴더: {output_base_folder}")
    print(f"총 동영상: {len(video_files)}개")
    print(f"{'='*70}\n")

    # 각 동영상 처리
    success_count = 0
    fail_count = 0

    for idx, video_path in enumerate(video_files, 1):
        # 파일명에서 라벨 추출 (예: 'ㄱ.mov' → 'ㄱ')
        filename = os.path.basename(video_path)
        label_name = os.path.splitext(filename)[0]

        # 출력 폴더: output_base_folder/label_name/
        output_folder = os.path.join(output_base_folder, label_name)

        print(f"\n[{idx}/{len(video_files)}] {label_name}")

        # 동영상 처리
        success = process_video_to_json(video_path, output_folder, label_name)

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


