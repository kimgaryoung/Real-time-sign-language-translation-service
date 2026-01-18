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
import sys  # exit 사용을 위해 추가

# MediaPipe 초기화 (전역 변수 유지)
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

def extract_keypoints_from_frame(frame, pose, hands, face_mesh):
    """
    단일 프레임에서 키포인트 추출 (model.ipynb의 411차원 형식)
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
            # handedness 안전하게 가져오기
            if hands_results.multi_handedness:
                handedness = hands_results.multi_handedness[idx].classification[0].label
            else:
                handedness = "Unknown"
            
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
    
    # MediaPipe 초기화 (with 구문 사용)
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

            # JSON 파일로 저장
            json_filename = f"{label_name}_{frame_idx:06d}_keypoints.json"
            json_path = os.path.join(output_folder, json_filename)

            # 들여쓰기(indent)를 None으로 하여 용량 줄임 (옵션)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(keypoint_data, f, ensure_ascii=False)

            frame_idx += 1
            processed_frames += 1

            # 진행 상황 표시
            if frame_idx % 10 == 0:
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                print(f"   진행: {frame_idx}/{total_frames} ({progress:.1f}%)", end='\r')

        cap.release()
        print(f"\n   완료: {processed_frames}개 프레임 처리됨")
        return True

def create_morpheme_label_file(output_folder, label_name):
    """
    model.ipynb에서 사용하는 형식의 라벨 JSON 파일 생성
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

    # print(f" 라벨 파일 생성: {label_filename}")

def process_all_videos(video_folder, output_base_folder):
    """
    폴더의 모든 동영상을 처리 (파일명에 따라 라벨 그룹화)
    """
    # 동영상 파일 찾기 (mp4, mov 대소문자 무관)
    video_files = []
    for ext in ["*.mov", "*.MOV", "*.mp4", "*.MP4"]:
        video_files.extend(glob.glob(os.path.join(video_folder, ext)))
    
    video_files = sorted(video_files)

    if not video_files:
        print("❌ 동영상 파일을 찾을 수 없습니다!")
        return

    print(f"\n{'='*70}")
    print(f"동영상 → JSON 키포인트 변환 시작")
    print(f"입력: {video_folder}")
    print(f"출력: {output_base_folder}")
    print(f"총 파일: {len(video_files)}개")
    print(f"{'='*70}")

    success_count = 0
    fail_count = 0

    for idx, video_path in enumerate(video_files, 1):
        filename = os.path.basename(video_path)
        
        # [중요] 파일명에서 라벨 추출 로직 수정
        # 예: "안녕하세요_1.mp4" -> "안녕하세요"
        raw_name = os.path.splitext(filename)[0]
        if '_' in raw_name:
            label_name = raw_name.split('_')[0]
        else:
            label_name = raw_name

        # 출력 폴더: output_base_folder/label_name/
        output_folder = os.path.join(output_base_folder, label_name)

        print(f"\n[{idx}/{len(video_files)}] 파일: {filename} → 라벨: {label_name}")

        # 동영상 처리
        success = process_video_to_json(video_path, output_folder, label_name)

        if success:
            # 라벨 파일 생성 (이미 존재해도 덮어쓰기 되므로 안전)
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
    print(f"\n출력 경로: {output_base_folder}")
    print(f"이제 model.ipynb에서 이 데이터를 학습에 사용하세요.")
    print(f"{'='*70}")

# ==========================================
# 실행부 (Main) 
# ==========================================
if __name__ == "__main__":
    # 경로 설정 (윈도우 경로 raw string 적용)
    # 1. 입력: 촬영한 동영상이 있는 폴더
    video_folder = r"C:/Users/yues7/OneDrive/사진/Camera Roll"
    
    # 2. 출력: 결과가 저장될 경로
    output_base_folder = r"C:/j/dataset자체제작_단어_keypoints" 

    # 폴더 존재 확인
    if not os.path.exists(video_folder):
        print(f"❌ 입력 폴더가 존재하지 않습니다: {video_folder}")
        print("경로를 다시 확인해주세요.")
        sys.exit(1)

    # 전체 동영상 처리 시작
    process_all_videos(video_folder, output_base_folder)