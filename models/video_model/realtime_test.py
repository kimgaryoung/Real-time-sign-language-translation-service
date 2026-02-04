# =============================================================================
# 실시간 웹캠 수어 인식 테스트
# =============================================================================

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import time

# ============================================================
# 설정 (모델 경로 수정)
# ============================================================
MODEL_PATH = r"C:/j/models/sign_language_cnn_gru.pt"
# ============================================================


# =============================================================================
# 1. 모델 정의 (학습할 때와 동일한 구조)
# =============================================================================

class CnnGruModel(nn.Module):
    def __init__(self, input_dim=411, hidden_dim=128, num_layers=2, 
                 num_classes=10, dropout=0.3):
        super(CnnGruModel, self).__init__()
        
        self.spatial = nn.Sequential(
            nn.Conv1d(input_dim, 256, 1),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
        )
        
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
        )
        
        gru_dropout = dropout if num_layers > 1 else 0
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=gru_dropout,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.spatial(x)
        x = self.temporal(x)
        x = x.permute(0, 2, 1)
        gru_output, _ = self.gru(x)
        attn = self.attention(gru_output)
        context = torch.sum(gru_output * attn, dim=1)
        output = self.classifier(context)
        return output


# =============================================================================
# 2. 모델 로드
# =============================================================================

def load_model(model_path):
    """학습된 모델 로드"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = CnnGruModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=checkpoint['num_classes'],
        dropout=0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 정규화 파라미터 (차원 맞추기 - squeeze 적용)
    norm_params = {
        'mean': np.array(checkpoint['norm_params']['mean']).squeeze(),
        'std': np.array(checkpoint['norm_params']['std']).squeeze()
    }
    
    idx_to_label = checkpoint['idx_to_label']
    sequence_length = config['sequence_length']
    
    print("모델 로드 완료!")
    print("클래스 수:", checkpoint['num_classes'])
    print("시퀀스 길이:", sequence_length)
    
    return model, idx_to_label, norm_params, sequence_length, device


# =============================================================================
# 3. MediaPipe로 키포인트 추출 (411차원)
# =============================================================================

def extract_keypoints(results_pose, results_hands, results_face):
    """MediaPipe 결과에서 411차원 키포인트 추출"""
    
    # Pose (25 * 3 = 75)
    pose = np.zeros(75, dtype=np.float32)
    if results_pose.pose_landmarks:
        for i, lm in enumerate(results_pose.pose_landmarks.landmark[:25]):
            pose[i*3] = lm.x
            pose[i*3 + 1] = lm.y
            pose[i*3 + 2] = lm.visibility
    
    # Face (70 * 3 = 210)
    face = np.zeros(210, dtype=np.float32)
    if results_face.multi_face_landmarks:
        face_lm = results_face.multi_face_landmarks[0]
        for i, lm in enumerate(face_lm.landmark[:70]):
            face[i*3] = lm.x
            face[i*3 + 1] = lm.y
            face[i*3 + 2] = lm.z
    
    # Hands (21 * 3 = 63 each)
    hand_left = np.zeros(63, dtype=np.float32)
    hand_right = np.zeros(63, dtype=np.float32)
    
    if results_hands.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            handedness = results_hands.multi_handedness[idx].classification[0].label
            hand_kp = np.zeros(63, dtype=np.float32)
            
            for i, lm in enumerate(hand_landmarks.landmark):
                hand_kp[i*3] = lm.x
                hand_kp[i*3 + 1] = lm.y
                hand_kp[i*3 + 2] = lm.z
            
            # 거울 모드라서 좌우 반전
            if handedness == 'Left':
                hand_right = hand_kp
            else:
                hand_left = hand_kp
    
    return np.concatenate([pose, face, hand_left, hand_right])


# =============================================================================
# 4. 실시간 웹캠 인식
# =============================================================================

def run_realtime_recognition():
    """실시간 웹캠 수어 인식"""
    
    # 모델 로드
    print("=" * 50)
    print("모델 로드 중...")
    print("=" * 50)
    model, idx_to_label, norm_params, sequence_length, device = load_model(MODEL_PATH)
    
    # MediaPipe 초기화
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    
    # 키포인트 버퍼 (시퀀스 저장용)
    keypoint_buffer = []
    
    # 현재 예측 결과
    current_prediction = "Waiting..."
    current_confidence = 0.0
    
    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다!")
        return
    
    print("\n" + "=" * 50)
    print("실시간 인식 시작!")
    print("'q' 키를 누르면 종료")
    print("=" * 50 + "\n")
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose, mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands, mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 좌우 반전 (거울 모드)
            frame = cv2.flip(frame, 1)
            
            # BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe 처리
            results_pose = pose.process(rgb_frame)
            results_hands = hands.process(rgb_frame)
            results_face = face_mesh.process(rgb_frame)
            
            # 키포인트 추출 (411차원)
            keypoints = extract_keypoints(results_pose, results_hands, results_face)
            keypoint_buffer.append(keypoints)
            
            # 버퍼가 시퀀스 길이에 도달하면 예측
            if len(keypoint_buffer) >= sequence_length:
                # 최근 sequence_length개만 사용
                sequence = np.array(keypoint_buffer[-sequence_length:], dtype=np.float32)
                
                # 정규화 (이미 squeeze 적용됨)
                sequence = (sequence - norm_params['mean']) / norm_params['std']
                
                # 텐서 변환 및 예측
                input_tensor = torch.tensor(sequence, dtype=torch.float32)
                input_tensor = input_tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    confidence, predicted_idx = torch.max(probs, dim=1)
                
                current_prediction = idx_to_label[predicted_idx.item()]
                current_confidence = confidence.item()
                
                # 버퍼 절반 유지 (슬라이딩 윈도우)
                keypoint_buffer = keypoint_buffer[-sequence_length//2:]
            
            # =========== 화면에 표시 ===========
            
            # 손 랜드마크 그리기
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 포즈 랜드마크 그리기
            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # 상단 배경 박스
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
            
            # 예측 결과 표시
            cv2.putText(frame, "Prediction: {}".format(current_prediction), 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, "Confidence: {:.1f}%".format(current_confidence * 100), 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            # 버퍼 상태 표시
            buffer_ratio = min(len(keypoint_buffer) / sequence_length, 1.0)
            bar_width = int(200 * buffer_ratio)
            cv2.rectangle(frame, (10, 100), (210, 115), (100, 100, 100), -1)
            cv2.rectangle(frame, (10, 100), (10 + bar_width, 115), (0, 255, 0), -1)
            cv2.putText(frame, "{}/{}".format(min(len(keypoint_buffer), sequence_length), sequence_length), 
                       (220, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 안내 문구
            cv2.putText(frame, "Press 'q' to quit", 
                       (frame.shape[1] - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # 화면 출력
            cv2.imshow('Sign Language Recognition', frame)
            
            # 'q' 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n인식 종료!")


# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    run_realtime_recognition()