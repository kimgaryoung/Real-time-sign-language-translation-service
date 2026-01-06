"""
실시간 지문자 인식 + 단어 조합 
- V2 모델 사용 (기존 + JSON keypoints 데이터 학습)
- 자동 인식 (사용자 조작 불필요)
- Top 3 예측 결과 + 확률 표시
- 한글 폰트 지원
- 자모음 자동 조합하여 단어 표시
- 일정 시간 유지 시 자동 확정 -> 지금은 1초로 설정해둠.
- 키보드: 백스페이스(삭제), c(전체 초기화), q(종료)



"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import warnings
import time
warnings.filterwarnings('ignore')

# ==================== 설정 ====================
MODEL_PATH = "../../models/photo_model/photo_finger_alphabet_411_model_V2.pt"

# 키포인트 차원
POSE_DIM = 75
FACE_DIM = 210
HAND_DIM = 63
TOTAL_DIM = 411

# 가중치
POSE_WEIGHT = 0.005
FACE_WEIGHT = 0.005
HAND_WEIGHT = 0.99

# 자모음 확정 시간 (초)
CONFIRM_TIME = 1.0  # 1초 동안 같은 자모음 유지 시 확정
MIN_CONFIDENCE = 80.0  # 최소 신뢰도 (%)

# 디바이스
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"디바이스: {device}")

# ==================== 한글 조합 클래스 ====================
class HangulComposer:
    """한글 자모음을 조합하여 완성형 글자로 만드는 클래스"""

    # 초성 (19개)
    CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
               'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    # 중성 (21개)
    JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
                'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

    # 종성 (28개, 첫번째는 종성 없음)
    JONGSUNG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
                'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    # 자음 목록 (초성/종성 가능)
    JAEUM = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
             'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    # 모음 목록
    MOEUM = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
             'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

    def __init__(self):
        self.reset()

    def reset(self):
        """현재 조합 상태 초기화"""
        self.cho = None   # 초성
        self.jung = None  # 중성
        self.jong = None  # 종성
        self.completed_text = ""  # 완성된 텍스트

    def is_jaeum(self, char):
        """자음인지 확인"""
        return char in self.JAEUM

    def is_moeum(self, char):
        """모음인지 확인"""
        return char in self.MOEUM

    def compose_syllable(self, cho, jung, jong=None):
        """초성, 중성, 종성을 조합하여 한글 글자 생성"""
        if cho is None or jung is None:
            return None

        try:
            cho_idx = self.CHOSUNG.index(cho)
            jung_idx = self.JUNGSUNG.index(jung)
            jong_idx = 0 if jong is None else self.JONGSUNG.index(jong)

            # 한글 유니코드 조합 공식
            code = 0xAC00 + (cho_idx * 21 * 28) + (jung_idx * 28) + jong_idx
            return chr(code)
        except ValueError:
            return None

    def add_jamo(self, jamo):
        """자모음 추가 및 조합"""
        if not jamo:
            return

        if self.is_jaeum(jamo):
            # 자음 입력
            if self.cho is None:
                # 초성 없음 -> 초성으로 설정
                self.cho = jamo
            elif self.jung is None:
                # 초성만 있고 중성 없음 -> 기존 초성 완료, 새 초성
                self.completed_text += self.cho
                self.cho = jamo
            elif self.jong is None:
                # 초성+중성 있음 -> 종성으로 설정
                if jamo in self.JONGSUNG:
                    self.jong = jamo
                else:
                    # 종성으로 쓸 수 없는 자음 (ㄸ, ㅃ, ㅉ)
                    syllable = self.compose_syllable(self.cho, self.jung)
                    if syllable:
                        self.completed_text += syllable
                    self.cho = jamo
                    self.jung = None
                    self.jong = None
            else:
                # 초성+중성+종성 모두 있음 -> 현재 글자 완성, 새 초성
                syllable = self.compose_syllable(self.cho, self.jung, self.jong)
                if syllable:
                    self.completed_text += syllable
                self.cho = jamo
                self.jung = None
                self.jong = None

        elif self.is_moeum(jamo):
            # 모음 입력
            if self.cho is None:
                # 초성 없음 -> 모음만 추가 (ㅏ, ㅓ 등)
                self.completed_text += jamo
            elif self.jung is None:
                # 초성만 있음 -> 중성으로 설정
                self.jung = jamo
            elif self.jong is None:
                # 초성+중성 있음 -> 현재 글자 완성, 새 모음
                syllable = self.compose_syllable(self.cho, self.jung)
                if syllable:
                    self.completed_text += syllable
                self.completed_text += jamo
                self.cho = None
                self.jung = None
            else:
                # 초성+중성+종성 있음 -> 종성을 초성으로, 새 중성
                syllable = self.compose_syllable(self.cho, self.jung)
                if syllable:
                    self.completed_text += syllable
                self.cho = self.jong
                self.jung = jamo
                self.jong = None

    def get_current_text(self):
        """현재까지 조합된 텍스트 반환 (조합 중인 글자 포함)"""
        result = self.completed_text

        if self.cho is not None:
            if self.jung is not None:
                if self.jong is not None:
                    syllable = self.compose_syllable(self.cho, self.jung, self.jong)
                    result += syllable if syllable else f"{self.cho}{self.jung}{self.jong}"
                else:
                    syllable = self.compose_syllable(self.cho, self.jung)
                    result += syllable if syllable else f"{self.cho}{self.jung}"
            else:
                result += self.cho

        return result

    def get_composing_jamo(self):
        """현재 조합 중인 자모음 표시"""
        jamo_list = []
        if self.cho:
            jamo_list.append(self.cho)
        if self.jung:
            jamo_list.append(self.jung)
        if self.jong:
            jamo_list.append(self.jong)
        return ''.join(jamo_list)

    def delete_last(self):
        """마지막 입력 삭제"""
        if self.jong is not None:
            self.jong = None
        elif self.jung is not None:
            self.jung = None
        elif self.cho is not None:
            self.cho = None
        elif self.completed_text:
            self.completed_text = self.completed_text[:-1]

    def clear_all(self):
        """전체 초기화"""
        self.reset()


# ==================== 한글 폰트 설정 ====================
def get_korean_font(size=30):
    """시스템에서 한글 폰트 찾기"""
    font_paths = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/Library/Fonts/AppleGothic.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except:
            continue
    return ImageFont.load_default()

font_large = get_korean_font(50)
font_medium = get_korean_font(30)
font_small = get_korean_font(20)

def put_korean_text(img, text, pos, font, color=(255, 255, 255)):
    """OpenCV 이미지에 한글 텍스트 그리기"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==================== 모델 정의 ====================
class KeypointClassifier(nn.Module):
    def __init__(self, input_size=411, num_classes=28, dropout=0.3):
        super(KeypointClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# ==================== 모델 로드 ====================
print("모델 로드 중...")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

num_classes = checkpoint['num_classes']
idx_to_label = checkpoint['idx_to_label']
norm_mean = checkpoint['norm_params']['mean'].to(device)
norm_std = checkpoint['norm_params']['std'].to(device)

model = KeypointClassifier(input_size=TOTAL_DIM, num_classes=num_classes, dropout=0.0)
model.load_state_dict(checkpoint['model_kp_state'])
model = model.to(device)
model.eval()

print(f"클래스 수: {num_classes}")
print(f"라벨: {list(idx_to_label.values())}")

# ==================== MediaPipe 초기화 ====================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ==================== 키포인트 추출 함수 ====================
def extract_keypoints(image, pose, hands, face_mesh):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb_image)
    hands_results = hands.process(rgb_image)
    face_results = face_mesh.process(rgb_image)

    detection_info = {'pose': False, 'face': False, 'left_hand': False, 'right_hand': False}

    # Pose (75)
    pose_kps = []
    if pose_results.pose_landmarks:
        detection_info['pose'] = True
        for i in range(25):
            if i < len(pose_results.pose_landmarks.landmark):
                lm = pose_results.pose_landmarks.landmark[i]
                pose_kps.extend([lm.x, lm.y, lm.visibility])
            else:
                pose_kps.extend([0.0, 0.0, 0.0])
    else:
        pose_kps = [0.0] * POSE_DIM

    # Face (210)
    face_kps = []
    if face_results.multi_face_landmarks:
        detection_info['face'] = True
        face_landmarks = face_results.multi_face_landmarks[0]
        for i in range(70):
            if i < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[i]
                face_kps.extend([lm.x, lm.y, lm.z])
            else:
                face_kps.extend([0.0, 0.0, 0.0])
    else:
        face_kps = [0.0] * FACE_DIM

    # Hands (126)
    left_hand_kps = [0.0] * HAND_DIM
    right_hand_kps = [0.0] * HAND_DIM

    if hands_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            handedness = hands_results.multi_handedness[idx].classification[0].label
            hand_kps = []
            for lm in hand_landmarks.landmark:
                hand_kps.extend([lm.x, lm.y, lm.z])

            if handedness == "Left":
                left_hand_kps = hand_kps
                detection_info['left_hand'] = True
            elif handedness == "Right":
                right_hand_kps = hand_kps
                detection_info['right_hand'] = True

    combined = np.concatenate([
        np.array(pose_kps),
        np.array(face_kps),
        np.array(left_hand_kps),
        np.array(right_hand_kps)
    ])

    return combined.astype(np.float32), detection_info, hands_results

def apply_weights(keypoints):
    weighted = keypoints.copy()
    weighted[0:POSE_DIM] *= POSE_WEIGHT
    weighted[POSE_DIM:POSE_DIM+FACE_DIM] *= FACE_WEIGHT
    weighted[POSE_DIM+FACE_DIM:POSE_DIM+FACE_DIM+HAND_DIM] *= HAND_WEIGHT
    weighted[POSE_DIM+FACE_DIM+HAND_DIM:TOTAL_DIM] *= HAND_WEIGHT
    return weighted

# ==================== 예측 함수 ====================
def predict_top3(keypoints):
    weighted = apply_weights(keypoints)
    x = torch.tensor(weighted, dtype=torch.float32).unsqueeze(0).to(device)
    x = (x - norm_mean) / norm_std

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        top3_probs, top3_indices = torch.topk(probs, 3)

    results = []
    for i in range(3):
        idx = top3_indices[0][i].item()
        prob = top3_probs[0][i].item() * 100
        label = idx_to_label[idx]
        results.append((label, prob))

    return results

# ==================== 자모음 확정 트래커 ====================
class JamoTracker:
    """자모음 인식 확정을 추적하는 클래스"""

    def __init__(self, confirm_time=1.5, min_confidence=70.0):
        self.confirm_time = confirm_time
        self.min_confidence = min_confidence
        self.current_jamo = None
        self.start_time = None
        self.last_confirmed = None

    def update(self, jamo, confidence):
        """
        자모음 업데이트 및 확정 여부 반환
        Returns: (confirmed_jamo, progress)
        - confirmed_jamo: 확정된 자모음 (None이면 미확정)
        - progress: 확정 진행률 (0.0 ~ 1.0)
        """
        current_time = time.time()

        # 신뢰도 미달
        if confidence < self.min_confidence:
            self.current_jamo = None
            self.start_time = None
            return None, 0.0

        # 새로운 자모음 또는 변경됨
        if jamo != self.current_jamo:
            self.current_jamo = jamo
            self.start_time = current_time
            return None, 0.0

        # 같은 자모음 유지 중
        elapsed = current_time - self.start_time
        progress = min(elapsed / self.confirm_time, 1.0)

        # 확정 시간 도달
        if elapsed >= self.confirm_time:
            # 같은 자모음 연속 확정 방지
            if jamo != self.last_confirmed:
                self.last_confirmed = jamo
                self.start_time = current_time  # 리셋
                return jamo, 1.0

        return None, progress

    def reset(self):
        """트래커 리셋"""
        self.current_jamo = None
        self.start_time = None
        self.last_confirmed = None


# ==================== 메인 루프 ====================
print("\n" + "="*50)
print("실시간 지문자 인식 + 단어 조합")
print("조작키:")
print("  q: 종료")
print("  백스페이스: 마지막 글자 삭제")
print("  c: 전체 초기화")
print("="*50 + "\n")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 한글 조합기 및 트래커 초기화
composer = HangulComposer()
tracker = JamoTracker(confirm_time=CONFIRM_TIME, min_confidence=MIN_CONFIDENCE)

with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as pose, \
     mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5) as hands, \
     mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 키포인트 추출 및 예측
        keypoints, detection_info, hands_results = extract_keypoints(frame, pose, hands, face_mesh)

        # 손 랜드마크 그리기
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )

        # 손이 감지되면 예측
        hand_detected = detection_info['left_hand'] or detection_info['right_hand']

        # 예측 결과 박스 (왼쪽 상단)
        cv2.rectangle(frame, (10, 10), (350, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 200), (255, 255, 255), 2)

        # 조합된 단어 박스 (오른쪽 상단)
        cv2.rectangle(frame, (frame.shape[1]-400, 10), (frame.shape[1]-10, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (frame.shape[1]-400, 10), (frame.shape[1]-10, 150), (0, 255, 255), 2)

        confirmed_jamo = None
        progress = 0.0

        if hand_detected:
            top3 = predict_top3(keypoints)
            top1_label, top1_conf = top3[0]

            # 자모음 트래커 업데이트
            confirmed_jamo, progress = tracker.update(top1_label, top1_conf)

            # 확정된 자모음이 있으면 조합기에 추가
            if confirmed_jamo:
                composer.add_jamo(confirmed_jamo)

            # 한글 텍스트 표시 (예측 결과)
            frame = put_korean_text(frame, "[ TOP 3 예측 결과 ]", (20, 20), font_medium, (0, 255, 255))

            colors = [(0, 255, 0), (255, 200, 0), (255, 150, 0)]
            for i, (label, prob) in enumerate(top3):
                rank = i + 1
                text = f"{rank}위: {label}  ({prob:.1f}%)"
                y_pos = 70 + i * 40
                frame = put_korean_text(frame, text, (20, y_pos), font_medium, colors[i])

            # 진행률 바 그리기
            bar_x = 20
            bar_y = 180
            bar_width = 310
            bar_height = 10
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
        else:
            frame = put_korean_text(frame, "[ 손을 기다리는 중... ]", (20, 20), font_medium, (0, 255, 255))
            frame = put_korean_text(frame, "손을 카메라에 보여주세요", (20, 70), font_small, (200, 200, 200))
            tracker.reset()

        # 조합된 단어 표시
        current_text = composer.get_current_text()
        composing_jamo = composer.get_composing_jamo()

        frame = put_korean_text(frame, "[ 조합된 단어 ]", (frame.shape[1]-390, 20), font_medium, (0, 255, 255))

        # 현재 조합된 텍스트 (큰 폰트)
        display_text = current_text if current_text else "(입력 대기)"
        text_color = (255, 255, 255) if current_text else (150, 150, 150)
        frame = put_korean_text(frame, display_text, (frame.shape[1]-390, 60), font_large, text_color)

        # 현재 조합 중인 자모음 표시
        if composing_jamo:
            frame = put_korean_text(frame, f"조합 중: {composing_jamo}", (frame.shape[1]-390, 120), font_small, (200, 200, 0))

        # 상태 표시
        status = f"손 감지: {'O' if hand_detected else 'X'}"
        frame = put_korean_text(frame, status, (20, 210), font_small, (150, 150, 150))

        # 도움말 표시
        help_text = "q:종료 | 백스페이스:삭제 | c:초기화"
        frame = put_korean_text(frame, help_text, (20, frame.shape[0]-30), font_small, (150, 150, 150))

        cv2.imshow('Finger Alphabet + Word (Press Q to quit)', frame)

        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 8 or key == 127:  # 백스페이스 (macOS: 127, Windows: 8)
            composer.delete_last()
            tracker.reset()
            print(f"삭제 -> 현재 텍스트: {composer.get_current_text()}")
        elif key == ord('c'):
            composer.clear_all()
            tracker.reset()
            print("전체 초기화")

cap.release()
cv2.destroyAllWindows()
print(f"\n최종 텍스트: {composer.get_current_text()}")
print("종료되었습니다.")