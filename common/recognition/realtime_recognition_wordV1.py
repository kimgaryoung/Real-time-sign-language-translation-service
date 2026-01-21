"""
실시간 지문자 인식 + 단어 조합 + NLP 자연어 처리
- V2 모델 사용 (기존 + JSON keypoints 데이터 학습)
- 자동 인식 (사용자 조작 불필요)
- Top 3 예측 결과 + 확률 표시
- 한글 폰트 지원
- 자모음 자동 조합하여 단어 표시
- NLP 자동 문장 변환 및 맞춤법 교정
- 일정 시간 유지 시 자동 확정 (1초)
- 키보드: 백스페이스(삭제), c(전체 초기화), s(문장 변환), q(종료)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import warnings
import time
import sys
import os
import platform

warnings.filterwarnings('ignore')

# 프로젝트 루트 경로를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# MediaPipe solutions 정의
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    print("✓ MediaPipe 모듈 로드 완료")
except ImportError:
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    print("✓ MediaPipe 레거시 방식 로드")

# NLP 모듈 import
try:
    from common.nlp import HyemiTextManager, KoreanSpellChecker
    NLP_AVAILABLE = True
    print("✓ NLP 모듈 로드 완료")
except ImportError as e:
    NLP_AVAILABLE = False
    print(f"⚠️  NLP 모듈을 찾을 수 없습니다: {e}")
    print("   NLP 없이 기본 모드로 실행합니다.")

# ==================== 설정 ====================
MODEL_PATH = "models/photo_model/photo_finger_alphabet_411_model_V2.pt"

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
CONFIRM_TIME = 1.0
MIN_CONFIDENCE = 80.0

# 디바이스
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"✓ 디바이스: {device}")

# ==================== 한글 조합 클래스 ====================
class HangulComposer:
    """한글 자모음을 조합하여 완성형 글자로 만드는 클래스"""
    # 초성 (19개)       
    CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
               'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    
    # 중성 (21개)
    JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
                'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ','ㅣ']
    
    # 종성 (28개, 첫번째는 종성 없음)
    JONGSUNG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
                'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    
    # 자음 목록 (초성/종성 가능)
    JAEUM = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
             'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    
    # 모음 목록
    MOEUM = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
             'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ','ㅣ']

    def __init__(self):
        self.reset()
        self.word_history = []

    def reset(self):
        """현재 조합 상태 초기화"""
        self.cho = None
        self.jung = None
        self.jong = None
        self.completed_text = ""

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
        """현재까지 조합된 텍스트 반환"""
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

    def get_words_list(self):
        """
        조합된 텍스트를 단어 리스트로 변환
        공백 기준으로 분리하되, 공백이 없으면 전체를 하나의 단어로
        """
        text = self.get_current_text()
        if not text:
            return []
        
        # 공백으로 분리
        words = text.split()
        if words:
            return words
        else:
            # 공백이 없으면 전체를 하나의 단어로
            return [text] if text else []

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
    """시스템에서 한글 폰트 찾기 (OS별 대응)"""
    font_paths = [
        # macOS
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/Library/Fonts/AppleGothic.ttf",
        # Windows
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
        "C:/Windows/Fonts/batang.ttc",
        # Linux
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception as e:
                continue
    
    print(f"⚠️  크기 {size}의 한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
    return ImageFont.load_default()

font_large = get_korean_font(50)
font_medium = get_korean_font(30)
font_small = get_korean_font(20)
font_tiny = get_korean_font(16)

def put_korean_text(img, text, pos, font, color=(255, 255, 255)):
    """OpenCV 이미지에 한글 텍스트 그리기"""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"텍스트 렌더링 오류: {e}")
        return img


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
def load_model():
    """모델 로드 및 초기화"""
    try:
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

        print(f"✓ 클래스 수: {num_classes}")
        print(f"✓ 라벨: {list(idx_to_label.values())}")
        
        return model, idx_to_label, norm_mean, norm_std
    except FileNotFoundError:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("   models/photo_model/ 폴더에 모델 파일이 있는지 확인하세요.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 모델 로드 중 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ==================== 키포인트 추출 함수 ====================
def extract_keypoints(image, pose, hands, face_mesh):
    """이미지에서 Pose, Hands, Face Mesh 키포인트 추출"""
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
    """키포인트에 가중치 적용"""
    weighted = keypoints.copy()
    weighted[0:POSE_DIM] *= POSE_WEIGHT
    weighted[POSE_DIM:POSE_DIM+FACE_DIM] *= FACE_WEIGHT
    weighted[POSE_DIM+FACE_DIM:POSE_DIM+FACE_DIM+HAND_DIM] *= HAND_WEIGHT
    weighted[POSE_DIM+FACE_DIM+HAND_DIM:TOTAL_DIM] *= HAND_WEIGHT
    return weighted


# ==================== 예측 함수 ====================
def predict_top3(keypoints, model, idx_to_label, norm_mean, norm_std):
    """상위 3개 예측 결과 반환"""
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
                self.start_time = current_time
                return jamo, 1.0

        return None, progress

    def reset(self):
        """트래커 리셋"""
        self.current_jamo = None
        self.start_time = None
        self.last_confirmed = None


# ==================== NLP 처리 함수 ====================
def process_with_nlp(words_list, text_manager, spell_checker):
    """
    단어 리스트를 자연스러운 문장으로 변환
    """
    if not words_list:
        return ""
    
    try:
        # 1. 단어를 문장으로 변환
        sentence, _ = text_manager.process_text(words_list)
        
        # 2. 맞춤법 교정
        corrected = spell_checker.check(sentence)
        
        return corrected
    except Exception as e:
        print(f"NLP 처리 오류: {e}")
        return " ".join(words_list)  # 오류 시 단순 연결


# ==================== USB 카메라 안정화 함수 ====================
def find_and_init_camera(preferred_width=640, preferred_height=480):
    """
    OS별로 최적화된 카메라 초기화
    """
    print("="*60)
    print("카메라 검색 중...")
    system = platform.system()
    print(f"운영체제: {system}")
    print("="*60)
    
    # OS별 백엔드 선택
    if system == 'Windows':
        backend = cv2.CAP_DSHOW
        print("백엔드: DirectShow (Windows)")
    elif system == 'Darwin':  # macOS
        backend = cv2.CAP_AVFOUNDATION
        print("백엔드: AVFoundation (macOS)")
    else:  # Linux
        backend = cv2.CAP_V4L2
        print("백엔드: V4L2 (Linux)")
    
    available_cameras = []
    
    # 카메라 검색 (0~4번까지)
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    available_cameras.append({
                        'idx': i,
                        'width': w,
                        'height': h,
                        'cap': cap
                    })
                    print(f"  ✓ 카메라 {i}: {w}x{h}")
                else:
                    cap.release()
            else:
                # 백엔드 실패 시 ANY로 재시도
                cap.release()
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        available_cameras.append({
                            'idx': i,
                            'width': w,
                            'height': h,
                            'cap': cap
                        })
                        print(f"  ✓ 카메라 {i}: {w}x{h} (fallback)")
                    else:
                        cap.release()
                else:
                    cap.release()
        except Exception as e:
            print(f"  ✗ 카메라 {i} 오류: {e}")
            continue
    
    if not available_cameras:
        print("\n❌ 사용 가능한 카메라를 찾을 수 없습니다.")
        print("\n확인 사항:")
        print("  1. 카메라가 제대로 연결되어 있는지")
        print("  2. 다른 프로그램에서 사용 중이지 않은지")
        if system == 'Windows':
            print("  3. Windows 카메라 설정에서 앱 권한 확인")
        elif system == 'Darwin':
            print("  3. 시스템 환경설정 > 보안 및 개인정보보호 > 카메라 권한 확인")
        return None, -1
    
    # 첫 번째 카메라 사용
    camera_info = available_cameras[0]
    cap = camera_info['cap']
    camera_idx = camera_info['idx']
    
    # 나머지 카메라 닫기
    for cam in available_cameras[1:]:
        cam['cap'].release()
    
    print(f"\n✓ 카메라 {camera_idx} 선택")
    
    # 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, preferred_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preferred_height)
    
    # 버퍼 최소화
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # 실제 설정값 확인
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"  해상도: {actual_width}x{actual_height}")
    print(f"  FPS: {fps}")
    
    # 밝기 설정 (Windows/Linux만, macOS는 지원 제한적)
    if system != 'Darwin':
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, -4)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 200)
            cap.set(cv2.CAP_PROP_CONTRAST, 80)
            cap.set(cv2.CAP_PROP_GAIN, 100)
            print("  ✓ 밝기 설정 완료")
        except:
            print("  ⚠️  밝기 설정 스킵 (지원 안 됨)")
    
        # 워밍업
    print("워밍업 중...", end="", flush=True)
    for _ in range(30):
        cap.read()
        time.sleep(0.03)
    print(" 완료!")
    
    print("="*60)
    return cap, camera_idx


# ==================== 메인 함수 ====================
def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("실시간 지문자 인식 + 단어 조합 + NLP 자연어 처리")  # 수정
    print("="*60)
    print("조작키:")
    print("  q: 종료")
    print("  백스페이스: 마지막 글자 삭제")
    print("  c: 전체 초기화")
    print("  s: 현재 단어를 NLP로 문장 변환")
    print("  스페이스: 단어 띄어쓰기")
    if NLP_AVAILABLE:
        print("\n✓ NLP 모드: 활성화")
    else:
        print("\n⚠️  NLP 모드: 비활성화 (기본 모드)")
    print("="*60 + "\n")

# 모델 로드
model, idx_to_label, norm_mean, norm_std = load_model()

# 한글 조합기 및 트래커 초기화
composer = HangulComposer()
tracker = JamoTracker(confirm_time=CONFIRM_TIME, min_confidence=MIN_CONFIDENCE)

# NLP 모듈 초기화
if NLP_AVAILABLE:
    try:
        text_manager = HyemiTextManager()
        spell_checker = KoreanSpellChecker()
        nlp_sentence = ""
        print("✓ NLP 모듈 초기화 완료")
    except Exception as e:
        print(f"⚠️  NLP 모듈 초기화 실패: {e}")
        text_manager = None
        spell_checker = None
        nlp_sentence = ""
else:
    text_manager = None
    spell_checker = None
    nlp_sentence = ""

print("\nMediaPipe 초기화 중...")

try:
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as pose, \
         mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5) as hands, \
         mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

        # 카메라 초기화
        cap, camera_idx = find_and_init_camera(preferred_width=640, preferred_height=480)
        
        if cap is None:
            print("프로그램을 종료합니다.")
            sys.exit()
        print("\n✓ 카메라 연결 성공! 창이 열립니다...\n")
        
        # 연속 실패 카운터
        consecutive_failures = 0
        MAX_FAILURES = 50
        
        # 메인 루프
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret or frame is None:
                consecutive_failures += 1
                print(f"프레임 읽기 실패 ({consecutive_failures}/{MAX_FAILURES})")
                
                if consecutive_failures >= MAX_FAILURES:
                    print("\n카메라 연결이 끊어졌습니다.")
                    break
                
                time.sleep(0.1)
                continue
            
            # 프레임 읽기 성공 시 카운터 리셋
            consecutive_failures = 0
            
            # 좌우 반전 (거울 모드)
            frame = cv2.flip(frame, 1)
            
            # 키포인트 추출 및 예측
            keypoints, detection_info, hands_results = extract_keypoints(frame, pose, hands, face_mesh)

            # 손 랜드마크 그리기
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )

            hand_detected = detection_info['left_hand'] or detection_info['right_hand']

# UI 박스 그리기 (반투명)
            overlay = frame.copy()
            
            # 1. 예측 결과 박스 (왼쪽 상단)
            box1_x1, box1_y1 = 10, 10
            box1_x2, box1_y2 = 280, 140
            cv2.rectangle(overlay, (box1_x1, box1_y1), (box1_x2, box1_y2), (0, 0, 0), -1)
            cv2.rectangle(frame, (box1_x1, box1_y1), (box1_x2, box1_y2), (255, 255, 255), 2)

            # 2. 조합된 단어 박스 (오른쪽 상단)
            box2_x1 = frame.shape[1] - 350
            box2_y1 = 10
            box2_x2 = frame.shape[1] - 10
            box2_y2 = 140
            cv2.rectangle(overlay, (box2_x1, box2_y1), (box2_x2, box2_y2), (0, 0, 0), -1)
            cv2.rectangle(frame, (box2_x1, box2_y1), (box2_x2, box2_y2), (0, 255, 255), 2)

            # 3. NLP 변환 결과 박스 (하단 중앙)
            if NLP_AVAILABLE and nlp_sentence:
                box3_height = 80
                box3_y1 = frame.shape[0] - box3_height - 10
                box3_y2 = frame.shape[0] - 10
                box3_x1 = 200
                box3_x2 = frame.shape[1] - 200
                cv2.rectangle(overlay, (box3_x1, box3_y1), (box3_x2, box3_y2), (0, 0, 0), -1)
                cv2.rectangle(frame, (box3_x1, box3_y1), (box3_x2, box3_y2), (0, 255, 0), 3)

            # 오버레이 적용 (투명도 70%)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            confirmed_jamo = None
            progress = 0.0

            # 예측 및 조합 처리
            if hand_detected:
                top3 = predict_top3(keypoints, model, idx_to_label, norm_mean, norm_std)
                top1_label, top1_conf = top3[0]

                # 자모음 트래커 업데이트
                confirmed_jamo, progress = tracker.update(top1_label, top1_conf)

                # 확정된 자모음이 있으면 조합기에 추가
                if confirmed_jamo:
                    composer.add_jamo(confirmed_jamo)

                # 예측 결과 표시 - 간격 조정
                frame = put_korean_text(frame, "TOP 3 예측", (20, 20), font_small, (0, 255, 255))

                for i, (label, prob) in enumerate(top3):
                    rank = i + 1
                    text = f"{rank}. {label} {prob:.1f}%"
                    y_pos = 50 + i * 35  # 30 → 35로 간격 넓히기
                    frame = put_korean_text(frame, text, (20, y_pos), font_small, colors[i])

                # 진행률 바 그리기
                bar_x, bar_y = 20, 140
                bar_width, bar_height = 260, 8
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (80, 80, 80), -1)
                if progress > 0:
                    cv2.rectangle(frame, (bar_x, bar_y), 
                                (bar_x + int(bar_width * progress), bar_y + bar_height), 
                                (0, 255, 0), -1)
            else:
                frame = put_korean_text(frame, "손을 보여주세요", (20, 20), font_small, (255, 100, 100))
                frame = put_korean_text(frame, "카메라 앞에 손을", (20, 50), font_tiny, (200, 200, 200))
                frame = put_korean_text(frame, "펼쳐주세요", (20, 75), font_tiny, (200, 200, 200))
                tracker.reset()

            # 조합된 단어 표시
            current_text = composer.get_current_text()
            composing_jamo = composer.get_composing_jamo()

            frame = put_korean_text(frame, "조합 단어", (box2_x1 + 10, 20), font_small, (0, 255, 255))

            # 현재 조합된 텍스트
            display_text = current_text if current_text else "(대기)"
            text_color = (255, 255, 255) if current_text else (150, 150, 150)
            frame = put_korean_text(frame, display_text, (box2_x1 + 10, 55), font_medium, text_color)

            # 조합 중인 자모음 표시
            if composing_jamo:
                frame = put_korean_text(frame, f"→ {composing_jamo}", (box2_x1 + 10, 95), font_tiny, (255, 255, 0))

            # 단어 개수 표시
            if current_text and NLP_AVAILABLE:
                words = composer.get_words_list()
                word_count_text = f"{len(words)}개"
                frame = put_korean_text(frame, word_count_text, (box2_x1 + 10, 115), font_tiny, (150, 200, 255))

            # NLP 변환 결과 표시
            if NLP_AVAILABLE and nlp_sentence:
                box_y = frame.shape[0] - 90
                frame = put_korean_text(frame, "NLP 변환", (220, box_y + 5), font_small, (0, 255, 0))
                frame = put_korean_text(frame, nlp_sentence, (220, box_y + 35), font_medium, (255, 255, 255))

            # 상태 표시 (하단)
            status_color = (0, 255, 0) if hand_detected else (100, 100, 100)
            status_text = "●" if hand_detected else "○"
            cv2.circle(frame, (20, frame.shape[0] - 20), 8, status_color, -1)
            frame = put_korean_text(frame, f"손 감지 {status_text}", (40, frame.shape[0] - 30), 
                                font_tiny, status_color)

            # 도움말 (오른쪽 하단)
            help_y = frame.shape[0] - 30
            frame = put_korean_text(frame, "q:종료 | BS:삭제 | c:초기화 | s:문장 | 스페이스:띄어쓰기", 
                                (frame.shape[1] - 550, help_y), font_tiny, (150, 150, 150))

            # 화면 표시
            cv2.imshow('Sign Language Recognition + NLP (Press Q to quit)', frame)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n사용자 종료 요청")
                break
            elif key == 8 or key == 127:  # 백스페이스
                composer.delete_last()
                tracker.reset()
                print(f"삭제 → {composer.get_current_text()}")
            elif key == ord('c'):  # 초기화
                composer.clear_all()
                tracker.reset()
                nlp_sentence = ""
                print("전체 초기화")
            elif key == ord('s'):  # NLP 변환
                if NLP_AVAILABLE and text_manager and spell_checker:
                    words = composer.get_words_list()
                    if words:
                        nlp_sentence = process_with_nlp(words, text_manager, spell_checker)
                        print(f"\n{'='*60}")
                        print(f"단어: {words}")
                        print(f"변환: {nlp_sentence}")
                        print(f"{'='*60}\n")
                    else:
                        print("변환할 단어가 없습니다.")
                else:
                    print("NLP 모듈이 비활성화되어 있습니다.")
            elif key == 32:  # 스페이스
                current = composer.get_current_text()
                if current and not current.endswith(' '):
                    composer.completed_text += ' '
                    print(f"띄어쓰기 → {composer.get_current_text()}")

        # 카메라 해제
        cap.release()
        cv2.destroyAllWindows()

except KeyboardInterrupt:
    print("\n\n사용자에 의해 중단되었습니다.")
except Exception as e:
    print(f"\n\n오류 발생: {e}")
    import traceback
    traceback.print_exc()
finally:
    # 리소스 정리
    if 'cap' in locals() and cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    
    # 종료 시 최종 결과 출력
    print("\n" + "="*60)
    print("종료 - 최종 결과")
    print("="*60)
    if 'composer' in locals():
        print(f"조합된 텍스트: {composer.get_current_text()}")

        if NLP_AVAILABLE and 'nlp_sentence' in locals() and nlp_sentence:
            print(f"NLP 변환 결과: {nlp_sentence}")
        elif NLP_AVAILABLE and 'text_manager' in locals() and text_manager:
            final_words = composer.get_words_list()
            if final_words:
                try:
                    final_sentence = process_with_nlp(final_words, text_manager, spell_checker)
                    print(f"최종 NLP 변환: {final_sentence}")
                except:
                    pass

    print("="*60)
    print("프로그램을 종료합니다.")
    print("="*60 + "\n")


    # 수정: 함수 밖으로 이동
if __name__ == "__main__":
    main()

'''
전체 데이터 흐름

카메라 입력
↓
MediaPipe 키포인트 추출 (411차원)
↓
AI 모델 예측 → Top 3 (ㄱ 95%, ㄴ 3%, ㄷ 2%)
↓
Tracker: 1초 유지 확인 → 확정!
↓
Composer: 자모음 조합 → '가', '각', '갑'...
↓
사용자가 's' 키 누름
↓
TextManager: ['나', '밥', '먹다'] → '저는 밥을 먹습니다'
↓
SpellChecker: 맞춤법 교정
↓
화면에 최종 문장 표시
'''