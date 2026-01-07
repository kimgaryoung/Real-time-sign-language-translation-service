import os
import hashlib
import re
import logging
from typing import List, Tuple, Optional, Dict, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum, auto

# ================================================================================
# 환경 설정 및 로깅
# ================================================================================

# API 키 설정 (OpenAI GPT 사용 시 필요)
OPENAI_API_KEY = ""  # 예: "sk-xxxxx..."
CACHE_DIR = "tts_cache" # 캐시 디렉토리 (TTS 오디오 파일 저장용)
LOG_DIR = "logs" # 로그 디렉토리

if not os.path.exists(LOG_DIR): # 로그 디렉토리가 없으면 생성
    os.makedirs(LOG_DIR)

logging.basicConfig( # 로깅 설정: 파일과 콘솔 모두에 출력
    filename=os.path.join(LOG_DIR, f"nlp_system_{datetime.now().strftime('%Y%m%d')}.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)
logger = logging.getLogger("test_manager")

console_handler = logging.StreamHandler() # 콘솔 핸들러 추가 (터미널에서도 로그 확인)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# OpenAI 라이브러리 import 시도
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("'openai' 라이브러리 미설치. 규칙 기반 모드로 동작합니다.")
    openai = None
    OPENAI_AVAILABLE = False


# ================================================================================
# 언어학적 데이터 타입 정의
# ================================================================================

class WordType(Enum):
    """
    품사 분류
    
    한국어 품사 체계를 수어 번역에 맞게 단순화
    - 체언: 명사, 대명사, 수사
    - 용언: 동사, 형용사
    - 수식언: 부사
    - 기타: 의문사, 시간, 장소, 인사 등 (의미 기반 분류)
    """
    NOUN = auto()          # 명사: 사람, 사물, 개념 (밥, 학교, 사랑)
    VERB = auto()          # 동사: 동작, 작용 (먹다, 가다, 하다)
    ADJECTIVE = auto()     # 형용사: 상태, 성질 (좋다, 크다, 예쁘다)
    ADVERB = auto()        # 부사: 용언 수식 (빨리, 많이, 잘)
    PRONOUN = auto()       # 대명사: 사람/사물 대신 (나, 너, 우리)
    INTERROGATIVE = auto() # 의문사: 질문 (무엇, 어디, 언제, 왜)
    TIME = auto()          # 시간 표현: (오늘, 어제, 내일, 지금)
    PLACE = auto()         # 장소 표현: (집, 학교, 병원)
    GREETING = auto()      # 인사/감정: (안녕, 감사, 미안)
    NUMBER = auto()        # 숫자: (하나, 둘, 1, 2)
    NEGATION = auto()      # 부정: (안, 못, 아니다)
    CONNECTIVE = auto()    # 접속/연결: (그리고, 그래서, 하지만)
    UNKNOWN = auto()       # 미분류


class Tense(Enum):
    """
    시제 분류
    
    수어에서는 시간 부사로 시제를 표현하므로,
    시간 부사를 감지하여 적절한 시제 적용
    """
    PRESENT = auto()  # 현재: 기본 시제
    PAST = auto()     # 과거: 어제, 지난주, ~전 등
    FUTURE = auto()   # 미래: 내일, 다음주, 나중에 등


class SentenceType(Enum):
    """
    문장 유형 분류
    
    종결어미 선택에 영향
    """
    DECLARATIVE = auto()   # 평서문: ~합니다
    INTERROGATIVE = auto() # 의문문: ~합니까?, ~인가요?
    IMPERATIVE = auto()    # 명령문: ~하세요
    PROPOSITIVE = auto()   # 청유문: ~합시다
    EXCLAMATORY = auto()   # 감탄문: ~하네요!


@dataclass
class WordInfo:
    """
    단어 정보를 담는 데이터 클래스
    
    Attributes:
        word: 원본 단어
        word_type: 품사 분류
        base_form: 기본형 (동사/형용사의 경우 사전형)
        role: 문장 내 역할 (주어, 목적어, 서술어 등)
        particle: 부착할 조사
        is_negated: 부정 표현 여부
    """
    word: str                           # 원본 단어
    word_type: WordType                 # 품사
    base_form: str = ""                 # 기본형
    role: str = ""                      # 문장 성분 (subject, object, predicate 등)
    particle: str = ""                  # 부착할 조사
    is_negated: bool = False            # 부정 여부 (안/못 + 동사)
    original_index: int = 0             # 원래 위치 (어순 재배열 추적용)


@dataclass  
class SentenceContext:
    """
    문장 전체의 문맥 정보
    
    개별 단어 처리 시 참조하여 일관된 문장 생성
    """
    tense: Tense = Tense.PRESENT              # 시제
    sentence_type: SentenceType = SentenceType.DECLARATIVE  # 문장 유형
    has_subject: bool = False                 # 주어 존재 여부
    has_object: bool = False                  # 목적어 존재 여부
    has_predicate: bool = False               # 서술어 존재 여부
    is_question: bool = False                 # 의문문 여부
    is_request: bool = False                  # 요청/부탁 여부
    is_negated: bool = False                  # 부정문 여부
    honorific_level: int = 2                  # 높임 단계 (1:해요, 2:합니다)
    time_word: str = ""                       # 감지된 시간 표현


# ================================================================================
# 한국어 언어학 데이터베이스
# ================================================================================

class KoreanLinguisticsDB:
    """
    한국어 언어학 규칙 및 어휘 데이터베이스
    
    이 클래스는 한국어 문장 생성에 필요한 모든 언어학적 데이터를 포함:
    - 품사 사전
    - 동사/형용사 활용 테이블
    - 조사 선택 규칙
    - 복합 표현 패턴
    """
    
    # ============================================================
    # - 품사 분류 사전
    # ============================================================
    # key: 단어, value: 품사
    # 수어에서 자주 사용되는 어휘를 중심으로 구성
    
    WORD_TYPE_MAP: Dict[str, WordType] = {
        # ── 대명사 (사람을 가리킴) ──
        "나": WordType.PRONOUN,
        "저": WordType.PRONOUN,      # '나'의 겸양 표현
        "너": WordType.PRONOUN,
        "당신": WordType.PRONOUN,
        "우리": WordType.PRONOUN,
        "저희": WordType.PRONOUN,    # '우리'의 겸양 표현
        "그": WordType.PRONOUN,
        "그녀": WordType.PRONOUN,
        "이것": WordType.PRONOUN,
        "저것": WordType.PRONOUN,
        "그것": WordType.PRONOUN,
        "여기": WordType.PRONOUN,
        "거기": WordType.PRONOUN,
        "저기": WordType.PRONOUN,
        "누구": WordType.INTERROGATIVE,  # 의문 대명사
        
        # ── 의문사 (질문할 때 사용) ──
        "무엇": WordType.INTERROGATIVE,
        "뭐": WordType.INTERROGATIVE,    # '무엇'의 구어체
        "어디": WordType.INTERROGATIVE,
        "언제": WordType.INTERROGATIVE,
        "왜": WordType.INTERROGATIVE,
        "어떻게": WordType.INTERROGATIVE,
        "몇": WordType.INTERROGATIVE,
        "얼마": WordType.INTERROGATIVE,
        "어느": WordType.INTERROGATIVE,
        "무슨": WordType.INTERROGATIVE,
        
        # ── 시간 표현 ──
        # 현재
        "오늘": WordType.TIME,
        "지금": WordType.TIME,
        "요즘": WordType.TIME,
        # 과거
        "어제": WordType.TIME,
        "그제": WordType.TIME,
        "아까": WordType.TIME,
        "전": WordType.TIME,           # ~전
        "과거": WordType.TIME,
        "옛날": WordType.TIME,
        # 미래
        "내일": WordType.TIME,
        "모레": WordType.TIME,
        "나중": WordType.TIME,
        "나중에": WordType.TIME,
        "다음": WordType.TIME,
        "곧": WordType.TIME,
        "앞으로": WordType.TIME,
        # 시간대
        "아침": WordType.TIME,
        "점심": WordType.TIME,
        "저녁": WordType.TIME,
        "밤": WordType.TIME,
        "낮": WordType.TIME,
        "새벽": WordType.TIME,
        # 요일/주기
        "주말": WordType.TIME,
        "평일": WordType.TIME,
        "매일": WordType.TIME,
        "항상": WordType.TIME,
        
        # ── 장소 표현 ──
        "집": WordType.PLACE,
        "학교": WordType.PLACE,
        "병원": WordType.PLACE,
        "회사": WordType.PLACE,
        "사무실": WordType.PLACE,
        "은행": WordType.PLACE,
        "마트": WordType.PLACE,
        "편의점": WordType.PLACE,
        "역": WordType.PLACE,
        "공항": WordType.PLACE,
        "식당": WordType.PLACE,
        "카페": WordType.PLACE,
        "도서관": WordType.PLACE,
        "공원": WordType.PLACE,
        "화장실": WordType.PLACE,
        "교실": WordType.PLACE,
        "방": WordType.PLACE,
        "밖": WordType.PLACE,
        "안": WordType.PLACE,
        "위": WordType.PLACE,
        "아래": WordType.PLACE,
        "앞": WordType.PLACE,
        "뒤": WordType.PLACE,
        "옆": WordType.PLACE,
        
        # ── 인사/감정 표현 ──
        "안녕": WordType.GREETING,
        "감사": WordType.GREETING,
        "고맙다": WordType.GREETING,
        "고마워": WordType.GREETING,
        "미안": WordType.GREETING,
        "죄송": WordType.GREETING,
        "축하": WordType.GREETING,
        "수고": WordType.GREETING,
        "반갑다": WordType.GREETING,
        "환영": WordType.GREETING,
        "실례": WordType.GREETING,
        
        # ── 부사 (동사/형용사 수식) ──
        "많이": WordType.ADVERB,
        "조금": WordType.ADVERB,
        "아주": WordType.ADVERB,
        "매우": WordType.ADVERB,
        "너무": WordType.ADVERB,
        "정말": WordType.ADVERB,
        "진짜": WordType.ADVERB,
        "빨리": WordType.ADVERB,
        "천천히": WordType.ADVERB,
        "잘": WordType.ADVERB,
        "열심히": WordType.ADVERB,
        "같이": WordType.ADVERB,
        "함께": WordType.ADVERB,
        "혼자": WordType.ADVERB,
        "다시": WordType.ADVERB,
        "또": WordType.ADVERB,
        "이미": WordType.ADVERB,
        "벌써": WordType.ADVERB,
        "아직": WordType.ADVERB,
        "계속": WordType.ADVERB,
        "바로": WordType.ADVERB,
        "먼저": WordType.ADVERB,
        
        # ── 부정 표현 ──
        "안": WordType.NEGATION,
        "못": WordType.NEGATION,
        "아니다": WordType.NEGATION,
        "아니": WordType.NEGATION,
        "없다": WordType.VERB,  # 동사로 처리 (있다/없다)
        
        # ── 접속/연결 표현 ──
        "그리고": WordType.CONNECTIVE,
        "그래서": WordType.CONNECTIVE,
        "하지만": WordType.CONNECTIVE,
        "그런데": WordType.CONNECTIVE,
        "그러나": WordType.CONNECTIVE,
    }
    
    # ============================================================
    # - 시제 판별 사전
    # ============================================================
    # 시간 표현 → 시제 매핑
    
    TENSE_MARKERS: Dict[str, Tense] = {
        # 과거 표현
        "어제": Tense.PAST,
        "그제": Tense.PAST,
        "아까": Tense.PAST,
        "전": Tense.PAST,
        "과거": Tense.PAST,
        "옛날": Tense.PAST,
        "지난": Tense.PAST,
        
        # 미래 표현  
        "내일": Tense.FUTURE,
        "모레": Tense.FUTURE,
        "나중": Tense.FUTURE,
        "나중에": Tense.FUTURE,
        "다음": Tense.FUTURE,
        "곧": Tense.FUTURE,
        "앞으로": Tense.FUTURE,
        
        # 현재 (기본값)
        "오늘": Tense.PRESENT,
        "지금": Tense.PRESENT,
        "요즘": Tense.PRESENT,
    }
    
    # ============================================================
    # - 동사 활용 테이블
    # ============================================================
    # 기본형 → {시제/유형: 활용형}
    # 
    # 한국어 동사 활용 규칙:
    # - 규칙 활용: 어간 + 어미
    # - 불규칙 활용: ㅂ불규칙, ㄷ불규칙, ㅅ불규칙, ㄹ불규칙 등
    
    VERB_CONJUGATION: Dict[str, Dict[str, str]] = {
        # ── 기본 동사 ──
        "가다": {
            "present": "갑니다",
            "past": "갔습니다", 
            "future": "갈 것입니다",
            "present_polite": "가요",
            "past_polite": "갔어요",
            "question": "가시나요",
            "request": "가세요",
            "negative": "가지 않습니다",
            "connective": "가서",  # ~해서
        },
        "오다": {
            "present": "옵니다",
            "past": "왔습니다",
            "future": "올 것입니다",
            "question": "오시나요",
            "request": "오세요",
            "negative": "오지 않습니다",
        },
        "먹다": {
            "present": "먹습니다",
            "past": "먹었습니다",
            "future": "먹을 것입니다",
            "question": "드시나요",      # 높임
            "request": "드세요",
            "negative": "먹지 않습니다",
            "honorific": "드십니다",     # 주체 높임
        },
        "마시다": {
            "present": "마십니다",
            "past": "마셨습니다",
            "future": "마실 것입니다",
            "question": "드시나요",
            "honorific": "드십니다",
        },
        "자다": {
            "present": "잡니다",
            "past": "잤습니다",
            "future": "잘 것입니다",
            "question": "주무시나요",    # 높임
            "honorific": "주무십니다",
        },
        "일어나다": {
            "present": "일어납니다",
            "past": "일어났습니다",
            "future": "일어날 것입니다",
        },
        "하다": {
            "present": "합니다",
            "past": "했습니다",
            "future": "할 것입니다",
            "question": "하시나요",
            "request": "하세요",
            "negative": "하지 않습니다",
        },
        "보다": {
            "present": "봅니다",
            "past": "봤습니다",
            "future": "볼 것입니다",
            "question": "보시나요",
        },
        "주다": {
            "present": "줍니다",
            "past": "줬습니다",
            "future": "줄 것입니다",
            "request": "주세요",
            "honorific": "드립니다",     # 상대 높임
        },
        "받다": {
            "present": "받습니다",
            "past": "받았습니다",
            "future": "받을 것입니다",
        },
        "만나다": {
            "present": "만납니다",
            "past": "만났습니다",
            "future": "만날 것입니다",
            "question": "만나시나요",
            "honorific": "뵙니다",       # 높임
        },
        "기다리다": {
            "present": "기다립니다",
            "past": "기다렸습니다",
            "future": "기다릴 것입니다",
        },
        "알다": {
            "present": "압니다",
            "past": "알았습니다",
            "negative": "모릅니다",
        },
        "모르다": {
            "present": "모릅니다",
            "past": "몰랐습니다",
        },
        "있다": {
            "present": "있습니다",
            "past": "있었습니다",
            "question": "있으신가요",
            "negative": "없습니다",
            "honorific": "계십니다",
        },
        "없다": {
            "present": "없습니다",
            "past": "없었습니다",
        },
        "사다": {
            "present": "삽니다",
            "past": "샀습니다",
            "future": "살 것입니다",
        },
        "팔다": {
            "present": "팝니다",
            "past": "팔았습니다",
        },
        "배우다": {
            "present": "배웁니다",
            "past": "배웠습니다",
        },
        "가르치다": {
            "present": "가르칩니다",
            "past": "가르쳤습니다",
        },
        "공부하다": {
            "present": "공부합니다",
            "past": "공부했습니다",
            "future": "공부할 것입니다",
        },
        "운동하다": {
            "present": "운동합니다",
            "past": "운동했습니다",
        },
        "일하다": {
            "present": "일합니다",
            "past": "일했습니다",
        },
        "쉬다": {
            "present": "쉽니다",
            "past": "쉬었습니다",
        },
        "듣다": {  # ㄷ 불규칙
            "present": "듣습니다",
            "past": "들었습니다",
        },
        "걷다": {  # ㄷ 불규칙
            "present": "걷습니다",
            "past": "걸었습니다",
        },
        "말하다": {
            "present": "말합니다",
            "past": "말했습니다",
            "request": "말씀해 주세요",
            "honorific": "말씀하십니다",
        },
        "읽다": {
            "present": "읽습니다",
            "past": "읽었습니다",
        },
        "쓰다": {
            "present": "씁니다",
            "past": "썼습니다",
        },
        "타다": {
            "present": "탑니다",
            "past": "탔습니다",
        },
        "내리다": {
            "present": "내립니다",
            "past": "내렸습니다",
        },
        "뛰다": {
            "present": "뜁니다",
            "past": "뛰었습니다",
        },
        "앉다": {
            "present": "앉습니다",
            "past": "앉았습니다",
            "request": "앉으세요",
        },
        "서다": {
            "present": "섭니다",
            "past": "섰습니다",
        },
        "열다": {
            "present": "엽니다",
            "past": "열었습니다",
        },
        "닫다": {
            "present": "닫습니다",
            "past": "닫았습니다",
        },
        "찾다": {
            "present": "찾습니다",
            "past": "찾았습니다",
            "question": "찾으시나요",
        },
        "돕다": {  # ㅂ 불규칙
            "present": "돕습니다",
            "past": "도왔습니다",
            "request": "도와주세요",
        },
        "살다": {
            "present": "삽니다",
            "past": "살았습니다",
            "question": "사시나요",
        },
        "필요하다": {
            "present": "필요합니다",
            "past": "필요했습니다",
        },
        "원하다": {
            "present": "원합니다",
            "past": "원했습니다",
        },
        "좋아하다": {
            "present": "좋아합니다",
            "past": "좋아했습니다",
        },
        "싫어하다": {
            "present": "싫어합니다",
            "past": "싫어했습니다",
        },
        "사랑하다": {
            "present": "사랑합니다",
            "past": "사랑했습니다",
        },
        "생각하다": {
            "present": "생각합니다",
            "past": "생각했습니다",
        },
        "이해하다": {
            "present": "이해합니다",
            "past": "이해했습니다",
            "negative": "이해하지 못합니다",
        },
        "기억하다": {
            "present": "기억합니다",
            "past": "기억했습니다",
        },
        "부르다": {  # 르 불규칙
            "present": "부릅니다",
            "past": "불렀습니다",
        },
        "고르다": {  # 르 불규칙
            "present": "고릅니다",
            "past": "골랐습니다",
        },
        "부탁하다": {
            "present": "부탁합니다",
            "past": "부탁했습니다",
            "request": "부탁드립니다",
        },
        "전화하다": {
            "present": "전화합니다",
            "past": "전화했습니다",
        },
        "연락하다": {
            "present": "연락합니다",
            "past": "연락했습니다",
        },
        "도착하다": {
            "present": "도착합니다",
            "past": "도착했습니다",
        },
        "출발하다": {
            "present": "출발합니다",
            "past": "출발했습니다",
        },
        "시작하다": {
            "present": "시작합니다",
            "past": "시작했습니다",
        },
        "끝나다": {
            "present": "끝납니다",
            "past": "끝났습니다",
        },
    }
    
    # ============================================================
    # - 형용사 활용 테이블
    # ============================================================
    
    ADJECTIVE_CONJUGATION: Dict[str, Dict[str, str]] = {
        "좋다": {
            "present": "좋습니다",
            "past": "좋았습니다",
            "connective": "좋아서",
            "question": "좋으신가요",
        },
        "나쁘다": {
            "present": "나쁩니다",
            "past": "나빴습니다",
        },
        "크다": {
            "present": "큽니다",
            "past": "컸습니다",
        },
        "작다": {
            "present": "작습니다",
            "past": "작았습니다",
        },
        "많다": {
            "present": "많습니다",
            "past": "많았습니다",
        },
        "적다": {
            "present": "적습니다",
            "past": "적었습니다",
        },
        "덥다": {  # ㅂ 불규칙
            "present": "덥습니다",
            "past": "더웠습니다",
        },
        "춥다": {  # ㅂ 불규칙
            "present": "춥습니다",
            "past": "추웠습니다",
        },
        "아프다": {
            "present": "아픕니다",
            "past": "아팠습니다",
        },
        "예쁘다": {
            "present": "예쁩니다",
            "past": "예뻤습니다",
        },
        "맛있다": {
            "present": "맛있습니다",
            "past": "맛있었습니다",
        },
        "맛없다": {
            "present": "맛없습니다",
            "past": "맛없었습니다",
        },
        "재미있다": {
            "present": "재미있습니다",
            "past": "재미있었습니다",
        },
        "재미없다": {
            "present": "재미없습니다",
            "past": "재미없었습니다",
        },
        "비싸다": {
            "present": "비쌉니다",
            "past": "비쌌습니다",
        },
        "싸다": {
            "present": "쌉니다",
            "past": "쌌습니다",
        },
        "바쁘다": {
            "present": "바쁩니다",
            "past": "바빴습니다",
        },
        "피곤하다": {
            "present": "피곤합니다",
            "past": "피곤했습니다",
        },
        "배고프다": {
            "present": "배고픕니다",
            "past": "배고팠습니다",
        },
        "고프다": {
            "present": "고픕니다",
            "past": "고팠습니다",
        },
        "배부르다": {
            "present": "배부릅니다",
            "past": "배불렀습니다",
        },
        "목마르다": {
            "present": "목마릅니다",
            "past": "목말랐습니다",
        },
        "기쁘다": {
            "present": "기쁩니다",
            "past": "기뻤습니다",
        },
        "슬프다": {
            "present": "슬픕니다",
            "past": "슬펐습니다",
        },
        "무섭다": {  # ㅂ 불규칙
            "present": "무섭습니다",
            "past": "무서웠습니다",
        },
        "어렵다": {  # ㅂ 불규칙
            "present": "어렵습니다",
            "past": "어려웠습니다",
        },
        "쉽다": {  # ㅂ 불규칙
            "present": "쉽습니다",
            "past": "쉬웠습니다",
        },
        "빠르다": {  # 르 불규칙
            "present": "빠릅니다",
            "past": "빨랐습니다",
        },
        "느리다": {
            "present": "느립니다",
            "past": "느렸습니다",
        },
        "멀다": {
            "present": "멉니다",
            "past": "멀었습니다",
        },
        "가깝다": {  # ㅂ 불규칙
            "present": "가깝습니다",
            "past": "가까웠습니다",
        },
        "괜찮다": {
            "present": "괜찮습니다",
            "past": "괜찮았습니다",
            "question": "괜찮으신가요",
        },
        "행복하다": {
            "present": "행복합니다",
            "past": "행복했습니다",
        },
        "건강하다": {
            "present": "건강합니다",
            "past": "건강했습니다",
        },
        "중요하다": {
            "present": "중요합니다",
            "past": "중요했습니다",
        },
        "필요하다": {
            "present": "필요합니다",
            "past": "필요했습니다",
        },
        "가능하다": {
            "present": "가능합니다",
            "past": "가능했습니다",
        },
        "불가능하다": {
            "present": "불가능합니다",
            "past": "불가능했습니다",
        },
    }
    
    # ============================================================
    # - 인사/감정 표현 매핑
    # ============================================================
    
    GREETING_MAP: Dict[str, str] = {
        "안녕": "안녕하세요",
        "감사": "감사합니다",
        "고맙다": "감사합니다",
        "고마워": "감사합니다",
        "미안": "죄송합니다",
        "죄송": "죄송합니다",
        "축하": "축하드립니다",
        "수고": "수고하셨습니다",
        "반갑다": "반갑습니다",
        "환영": "환영합니다",
        "실례": "실례합니다",
    }
    
    # ============================================================
    # - 복합 표현 패턴
    # ============================================================
    # (단어1, 단어2, ...) → 완성된 문장
    # 수어의 고정 표현을 자연스러운 한국어로 변환
    
    COMPOUND_PATTERNS: Dict[Tuple[str, ...], str] = {
        # ── 인사/소개 ──
        ("처음", "만나다"): "처음 뵙겠습니다",
        ("잘", "부탁"): "잘 부탁드립니다",
        ("오래", "만나다"): "오랜만입니다",
        
        # ── 개인 정보 질문 ──
        ("이름", "무엇"): "성함이 어떻게 되시나요?",
        ("이름", "뭐"): "성함이 어떻게 되시나요?",
        ("나이", "몇"): "연세가 어떻게 되시나요?",
        ("나이", "얼마"): "연세가 어떻게 되시나요?",
        ("직업", "무엇"): "직업이 무엇인가요?",
        ("직업", "뭐"): "직업이 무엇인가요?",
        ("전화번호", "무엇"): "전화번호가 어떻게 되시나요?",
        ("전화번호", "뭐"): "전화번호가 어떻게 되시나요?",
        
        # ── 장소 질문 ──
        ("화장실", "어디"): "화장실이 어디인가요?",
        ("병원", "어디"): "병원이 어디인가요?",
        ("출구", "어디"): "출구가 어디인가요?",
        ("역", "어디"): "역이 어디인가요?",
        ("어디", "가다"): "어디에 가시나요?",
        ("어디", "살다"): "어디에 사시나요?",
        ("어디", "있다"): "어디에 있나요?",
        
        # ── 시간 질문 ──
        ("몇", "시"): "몇 시인가요?",
        ("언제", "오다"): "언제 오시나요?",
        ("언제", "가다"): "언제 가시나요?",
        ("언제", "시작"): "언제 시작하나요?",
        ("언제", "끝나다"): "언제 끝나나요?",
        
        # ── 기타 질문 ──
        ("뭐", "하다"): "무엇을 하시나요?",
        ("무엇", "하다"): "무엇을 하시나요?",
        ("왜", "오다"): "왜 오셨나요?",
        ("얼마", "이다"): "얼마인가요?",
        ("몇", "개"): "몇 개인가요?",
        ("몇", "명"): "몇 명인가요?",
        
        # ── 신체/건강 ──
        ("배", "고프다"): "배가 고픕니다",
        ("배", "아프다"): "배가 아픕니다",
        ("머리", "아프다"): "머리가 아픕니다",
        ("목", "아프다"): "목이 아픕니다",
        ("다리", "아프다"): "다리가 아픕니다",
        ("몸", "아프다"): "몸이 아픕니다",
        ("목", "마르다"): "목이 마릅니다",
        
        # ── 날씨/상태 ── (시제는 별도 처리)
        ("날씨", "좋다"): "날씨가 좋습니다",
        ("날씨", "나쁘다"): "날씨가 나쁩니다",
        ("날씨", "덥다"): "날씨가 덥습니다",
        ("날씨", "춥다"): "날씨가 춥습니다",
        
        # ── 같이/함께 표현 ──
        ("같이", "먹다"): "같이 먹습니다",
        ("함께", "먹다"): "함께 먹습니다",
        ("같이", "가다"): "같이 갑니다",
        ("함께", "가다"): "함께 갑니다",
        
        # ── 가능/불가능 ──
        ("시간", "없다"): "시간이 없습니다",
        ("시간", "있다"): "시간이 있습니다",
        ("돈", "없다"): "돈이 없습니다",
        ("돈", "있다"): "돈이 있습니다",
        
        # ── 요청/부탁 ──
        ("도움", "필요"): "도움이 필요합니다",
        ("도움", "주다"): "도와주세요",
        ("다시", "말하다"): "다시 말씀해 주세요",
        ("천천히", "말하다"): "천천히 말씀해 주세요",
        ("크게", "말하다"): "크게 말씀해 주세요",
        ("잠깐", "기다리다"): "잠깐만 기다려 주세요",
        
        # ── 이해/확인 ──
        ("이해", "못하다"): "이해하지 못했습니다",
        ("알다", "못하다"): "모르겠습니다",
        ("이해", "하다"): "이해했습니다",
        ("알다", "하다"): "알겠습니다",
        
        # ── 감정/상태 ──
        ("기분", "좋다"): "기분이 좋습니다",
        ("기분", "나쁘다"): "기분이 나쁩니다",
        ("걱정", "하다"): "걱정됩니다",
        ("걱정", "안", "하다"): "걱정하지 마세요",
    }
    
    # ============================================================
    # - 장소 + 동사 조사 패턴
    # ============================================================
    # 동사에 따라 장소 뒤에 붙는 조사가 달라짐
    
    PLACE_PARTICLE_BY_VERB: Dict[str, str] = {
        "가다": "에",      # ~에 갑니다
        "오다": "에서",    # ~에서 옵니다
        "있다": "에",      # ~에 있습니다
        "없다": "에",      # ~에 없습니다
        "살다": "에",      # ~에 삽니다
        "일하다": "에서",  # ~에서 일합니다
        "공부하다": "에서", # ~에서 공부합니다
        "먹다": "에서",    # ~에서 먹습니다
    }


# ================================================================================
# 핵심 NLP 엔진
# ================================================================================

class HyemiTextManager:
    """
    수어 번역을 위한 text_manager
    
    [처리 파이프라인]
    1. 전처리: 중복 제거, 정규화
    2. 형태소 분석: 품사 태깅, 문맥 파악
    3. 어순 재배열: 수어 → 한국어 어순
    4. 조사 부착: 문맥에 맞는 조사 선택
    5. 용언 활용: 시제/높임/문장유형에 따른 활용
    6. 후처리: 띄어쓰기 정리
    """
    
    def __init__(self):
        """
        초기화
        - 디렉토리 생성
        - OpenAI 클라이언트 연결 (선택적)
        - 언어학 데이터베이스 로드
        """
        self._init_directories()
        self.client = self._init_openai()
        self.ling = KoreanLinguisticsDB()  # 언어학 데이터베이스
        logger.info("✅ text_manager 초기화 완료")
    
    def _init_directories(self) -> None:
        """캐시 디렉토리 생성"""
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            logger.info(f"캐시 디렉토리 생성: {CACHE_DIR}")
    
    def _init_openai(self):
        """OpenAI 클라이언트 초기화"""
        if OPENAI_AVAILABLE and OPENAI_API_KEY.startswith("sk-"):
            try:
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                logger.info("NLP 모드: GPT-4o (Active)")
                return client
            except Exception as e:
                logger.error(f"OpenAI 연결 실패: {e}")
        
        logger.info("NLP 모드: Advanced Rule Engine")
        return None
    
    # ================================================================
    # - 메인 인터페이스
    # ================================================================
    
    def process_text(self, word_list: List[str]) -> Tuple[str, Optional[str]]:
        """
        메인 처리 함수
        
        Args:
            word_list: 수어 인식 단어 리스트 (예: ['나', '밥', '먹다'])
        
        Returns:
            Tuple[str, Optional[str]]: (변환된 문장, 캐시 경로 또는 None)
        
        Example:
            >>> manager = HyemiTextManager()
            >>> sentence, cache = manager.process_text(['어제', '나', '친구', '만나다'])
            >>> print(sentence)
            "어제 저는 친구를 만났습니다"
        """
        if not word_list: # 빈 입력 처리
            return "", None
        
        # Step 1: 전처리 (중복 제거, 정규화)
        cleaned_words = self._preprocess(word_list) 
        
        if not cleaned_words:
            return "", None
        
        # Step 2: 문장 생성
        if self.client: 
            final_sentence = self._generate_with_gpt(cleaned_words) # GPT 사용 가능하면 GPT로 처리
        else:
            final_sentence = self._generate_with_rules(cleaned_words) # 규칙 기반 엔진으로 처리

        # Step 3: 후처리 (띄어쓰기 정리)
        final_sentence = self._postprocess(final_sentence) 
        
        # Step 4: 캐시 확인
        cached_path = self._check_cache(final_sentence)
        
        logger.info(f"- 변환: {word_list} → '{final_sentence}'")
        return final_sentence, cached_path
    
    # ================================================================
    # - 전처리 단계
    # ================================================================
    
    def _preprocess(self, words: List[str]) -> List[str]:
        """
        입력 단어 리스트 전처리
        
        1. 공백 제거
        2. 특수문자 제거 (?, ! 제외)
        3. 안전한 조사 제거 (화이트리스트 방식)
        4. 불용어 필터링
        5. 연속 중복 제거
        
        Args:
            words: 원본 단어 리스트
        
        Returns:
            전처리된 단어 리스트
        """
        
        stop_words = {"음", "어", "그", "저기", "막", "아", "음...", "...", ""} # 불용어 (의미 없는 단어)
        
        # 안전하게 조사를 제거할 수 있는 단어 목록 (명사/의문사만)
        # key: "단어+조사", value: "원형"
        safe_particle_removal = {
            # 의문사
            "무엇이": "무엇",
            "무엇을": "무엇",
            "무엇은": "무엇",
            "뭐이": "뭐",
            "뭐를": "뭐",
            "뭐는": "뭐",
            "어디가": "어디",
            "어디를": "어디",
            "어디는": "어디",
            "언제가": "언제",
            "언제는": "언제",
            "누구가": "누구",
            "누구를": "누구",
            "누구는": "누구",
            # 자주 쓰이는 명사 (동사와 헷갈리지 않는 것만)
            "이름이": "이름",
            "이름은": "이름",
            "이름을": "이름",
            "나이가": "나이",
            "나이는": "나이",
            "직업이": "직업",
            "직업은": "직업",
            "전화번호가": "전화번호",
            "전화번호는": "전화번호",
            "화장실이": "화장실",
            "화장실은": "화장실",
            "병원이": "병원",
            "병원은": "병원",
            "학교가": "학교",
            "학교는": "학교",
            "집이": "집",
            "집은": "집",
        }
        
        cleaned = []
        prev_word = ""
        
        for word in words:
            word = word.strip()  # 공백 제거
            word = re.sub(r'\s+', '', word) # 내부 공백 제거               
            word = re.sub(r'[^\w가-힣?!]', '', word) # 특수문자 제거 (?, ! 만 유지)

            if not word or word in stop_words: # 빈 문자열이나 불용어 스킵
                continue
            
            # 화이트리스트 기반 안전한 조사 제거
            if word in safe_particle_removal:
                word = safe_particle_removal[word]
            
            if word == prev_word: # 연속 중복 제거 (같은 단어가 연속으로 나오면 하나만)
                continue
            
            cleaned.append(word)
            prev_word = word
        
        return cleaned
    
    # ================================================================
    # - GPT 기반 생성
    # ================================================================
    
    def _generate_with_gpt(self, words: List[str]) -> str:
        """
        GPT를 이용한 고품질 문장 생성
        
        수어 전문 프롬프트를 사용하여 자연스러운 한국어 변환
        """
        text = " ".join(words)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """당신은 한국 수어(KSL)를 자연스러운 한국어로 변환하는 전문 통역사입니다.

[변환 규칙]
1. 존댓말(합니다체) 사용
2. 수어 어순을 한국어 어순(시간-주어-장소-목적어-서술어)으로 재배열
3. 적절한 조사(은/는, 이/가, 을/를, 에/에서) 추가
4. 시간 표현에 따른 시제 적용:
   - 어제, 아까, ~전 → 과거형 (-았/었습니다)
   - 내일, 나중에 → 미래형 (-ㄹ 것입니다)
5. 의문사 있으면 의문문으로, 요청이면 청유형으로
6. 생략된 주어는 문맥에 맞게 추가
7. 결과 문장만 출력 (설명 없이)"""
                    },
                    {"role": "user", "content": f"수어 단어: {text}"}
                ],
                temperature=0.2,  # 낮은 temperature로 일관성 유지
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"GPT 오류: {e}")
            # 실패 시 규칙 기반으로 폴백
            return self._generate_with_rules(words)
    
    # ================================================================
    # - 규칙 기반 문장 생성 엔진
    # ================================================================
    
    def _generate_with_rules(self, words: List[str]) -> str:
        """
        규칙 기반 문장 생성 (GPT 없이 동작)
        
        [처리 순서]
        1. 복합 패턴 체크 (고정 표현)
        2. 문맥 분석 (시제, 문장유형 파악)
        3. 형태소 분석 (품사 태깅)
        4. 어순 재배열
        5. 조사 부착 및 용언 활용
        6. 문장 완성
        """
        if not words:
            return ""
        
        # Step 1: 복합 패턴 체크 (우선 처리)
        compound_result = self._check_compound_patterns(words)
        if compound_result:
            return compound_result
        
        # Step 2: 문맥 분석
        context = self._analyze_context(words)
        
        # Step 3: 형태소 분석 (품사 태깅)
        analyzed_words = self._analyze_words(words, context)
        
        # Step 4: 어순 재배열
        reordered_words = self._reorder_words(analyzed_words, context)
        
        # Step 5: 조사 부착 및 활용
        sentence = self._compose_sentence(reordered_words, context)
        
        # Step 6: 문장 유형에 따른 마무리
        sentence = self._finalize_sentence(sentence, context)
        
        return sentence
    
    # ================================================================
    # - 복합 패턴 체크
    # ================================================================
    
    def _check_compound_patterns(self, words: List[str]) -> Optional[str]:
        """
        복합 표현 패턴 매칭
        
        수어의 고정 표현은 직접 매핑된 자연스러운 문장으로 변환
        
        예: ['화장실', '어디'] → "화장실이 어디인가요?"
        """
        # 3단어 패턴 체크
        if len(words) >= 3:
            for i in range(len(words) - 2):
                key = (words[i], words[i+1], words[i+2])
                if key in self.ling.COMPOUND_PATTERNS:
                    return self._build_compound_result(words, i, 3)
        
        # 2단어 패턴 체크
        if len(words) >= 2:
            for i in range(len(words) - 1):
                key = (words[i], words[i+1])
                if key in self.ling.COMPOUND_PATTERNS:
                    return self._build_compound_result(words, i, 2)
        
        return None
    
    def _build_compound_result(self, words: List[str], start_idx: int, pattern_len: int) -> str:
        """
        복합 패턴 결과 조합
        
        패턴 앞뒤에 다른 단어가 있으면 함께 처리
        시간 표현이 있으면 시제도 적용
        """
        pattern_key = tuple(words[start_idx:start_idx + pattern_len])
        matched_phrase = self.ling.COMPOUND_PATTERNS[pattern_key]
        
        prefix_words = words[:start_idx]
        suffix_words = words[start_idx + pattern_len:]
        
        result_parts = []
        
        # 앞부분 처리 (재귀 호출)
        if prefix_words:
            prefix_result = self._generate_with_rules(prefix_words)
            if prefix_result:
                result_parts.append(prefix_result)
                
                # 시제 적용: 앞부분에 시간 표현이 있으면 복합 결과에 시제 적용
                for word in prefix_words:
                    if word in self.ling.TENSE_MARKERS:
                        tense = self.ling.TENSE_MARKERS[word]
                        matched_phrase = self._apply_tense_to_phrase(matched_phrase, tense)
                        break
        
        # 매칭된 패턴
        result_parts.append(matched_phrase)
        
        # 뒷부분 처리 (재귀 호출)
        if suffix_words:
            suffix_result = self._generate_with_rules(suffix_words)
            if suffix_result:
                result_parts.append(suffix_result)
        
        return " ".join(result_parts)
    
    def _apply_tense_to_phrase(self, phrase: str, tense: Tense) -> str:
        """
        완성된 문구에 시제 적용
        
        현재형 종결어미를 과거형/미래형으로 변환
        """
        if tense == Tense.PAST:
            # 현재형 → 과거형 변환 (특수 케이스 우선)
            special_replacements = [
                # ㅂ 불규칙 형용사
                ("덥습니다", "더웠습니다"),
                ("춥습니다", "추웠습니다"),
                ("아픕니다", "아팠습니다"),
                ("고픕니다", "고팠습니다"),
                ("무섭습니다", "무서웠습니다"),
                ("어렵습니다", "어려웠습니다"),
                ("쉽습니다", "쉬웠습니다"),
                ("가깝습니다", "가까웠습니다"),
                # 일반 형용사
                ("좋습니다", "좋았습니다"),
                ("나쁩니다", "나빴습니다"),
                ("없습니다", "없었습니다"),
                ("있습니다", "있었습니다"),
                ("많습니다", "많았습니다"),
                ("적습니다", "적었습니다"),
                # ~하다 형태
                ("피곤합니다", "피곤했습니다"),
                ("행복합니다", "행복했습니다"),
                ("합니다", "했습니다"),
                # 일반 종결어미
                ("입니다", "이었습니다"),
                ("습니다", "었습니다"),
                ("ㅂ니다", "었습니다"),
            ]
            
            for current, past in special_replacements:
                if phrase.endswith(current):
                    return phrase[:-len(current)] + past
        
        elif tense == Tense.FUTURE: # 현재형 → 미래형 변환
            if phrase.endswith("습니다"):
                return phrase[:-3] + "ㄹ 것입니다"
            elif phrase.endswith("합니다"):
                return phrase[:-3] + "할 것입니다"
        
        return phrase
    
    # ================================================================
    # - 문맥 분석
    # ================================================================
    
    def _analyze_context(self, words: List[str]) -> SentenceContext:
        """
        문장 전체의 문맥 분석
        
        - 시제 파악 (시간 표현 기반)
        - 문장 유형 파악 (의문사, 요청 표현 등)
        - 부정 표현 여부
        
        Returns:
            SentenceContext: 문맥 정보
        """
        context = SentenceContext()
        
        for word in words:
            # 시제 판별
            if word in self.ling.TENSE_MARKERS:
                context.tense = self.ling.TENSE_MARKERS[word]
                context.time_word = word
            
            # 의문문 판별
            if word in {"무엇", "뭐", "어디", "언제", "왜", "어떻게", "몇", "얼마", "누구"}:
                context.is_question = True
                context.sentence_type = SentenceType.INTERROGATIVE
            
            # 부정 판별
            if word in {"안", "못"}:
                context.is_negated = True
            
            # 요청/부탁 판별
            if word in {"주다", "부탁", "도움"}:
                context.is_request = True
        
        return context
    
    # ================================================================
    # = 형태소 분석 (품사 태깅)
    # ================================================================
    
    def _analyze_words(self, words: List[str], context: SentenceContext) -> List[WordInfo]:
        """
        각 단어의 품사 분석 및 WordInfo 생성
        
        Args:
            words: 단어 리스트
            context: 문맥 정보
        
        Returns:
            List[WordInfo]: 분석된 단어 정보 리스트
        """
        analyzed = []
        
        for idx, word in enumerate(words):
            word_type = self._get_word_type(word)
            base_form = self._get_base_form(word)
            
            # 부정 표현 체크 (다음 단어가 동사면 현재 단어가 부정)
            is_negated = False
            if idx < len(words) - 1:
                next_word = words[idx + 1]
                if word in {"안", "못"} and self._get_word_type(next_word) == WordType.VERB:
                    is_negated = True
            
            analyzed.append(WordInfo(
                word=word,
                word_type=word_type,
                base_form=base_form,
                is_negated=is_negated,
                original_index=idx
            ))
        
        return analyzed
    
    def _get_word_type(self, word: str) -> WordType:
        """
        단어의 품사 판별
        
        우선순위:
        1. 사전에서 찾기
        2. 동사/형용사 활용 사전에서 찾기
        3. 패턴 기반 추론
        4. 기본값: 명사
        """
        # 1. 품사 사전에서 찾기
        if word in self.ling.WORD_TYPE_MAP:
            return self.ling.WORD_TYPE_MAP[word]
        
        # 2. 동사 활용 사전에서 찾기
        if word in self.ling.VERB_CONJUGATION:
            return WordType.VERB
        
        # 3. 형용사 활용 사전에서 찾기
        if word in self.ling.ADJECTIVE_CONJUGATION:
            return WordType.ADJECTIVE
        
        # 4. 패턴 기반 추론
        if word.endswith("다"):
            # ~하다 형태는 동사
            if word.endswith("하다"):
                return WordType.VERB
            # ~되다 형태는 동사
            if word.endswith("되다"):
                return WordType.VERB
            # 그 외 ~다로 끝나면 동사 추정
            return WordType.VERB
        
        if word.endswith("게") or word.endswith("히") or word.endswith("이"):
            return WordType.ADVERB
        
        # 5. 기본값: 명사
        return WordType.NOUN
    
    def _get_base_form(self, word: str) -> str:
        """
        기본형(사전형) 추출
        
        활용형 → 기본형 변환
        예: '먹습니다' → '먹다'
        """
        # 이미 사전에 있으면 그대로
        if word in self.ling.VERB_CONJUGATION:
            return word
        if word in self.ling.ADJECTIVE_CONJUGATION:
            return word
        
        # 활용형 어미 제거 시도
        endings_to_remove = [
            "습니다", "ㅂ니다", "었습니다", "았습니다",
            "세요", "어요", "아요", "해요", "니까", "는데"
        ]
        
        for ending in endings_to_remove:
            if word.endswith(ending):
                stem = word[:-len(ending)]
                return stem + "다"
        
        return word
    
    # ================================================================
    # - 어순 재배열
    # ================================================================
    
    def _reorder_words(self, analyzed: List[WordInfo], context: SentenceContext) -> List[WordInfo]:
        """
        수어 어순 → 한국어 어순 재배열
        
        한국어 기본 어순:
        시간 + 주어 + 부사(양태) + 장소 + 목적어(명사) + 부사(동작) + 서술어(동사/형용사)
        
        수어 특성:
        - 시간이 문장 앞에 옴 (한국어와 동일)
        - 의문사가 문장 끝에 오는 경우 있음
        - Topic-Comment 구조
        """
        # 품사별 분류
        time_words = []       # 시간 표현
        greeting_words = []   # 인사말
        subject_words = []    # 주어 (대명사)
        place_words = []      # 장소
        object_words = []     # 목적어 (일반 명사)
        manner_adverbs = []   # 양태 부사 (같이, 함께, 혼자)
        action_adverbs = []   # 동작 부사 (빨리, 천천히, 많이)
        negation_words = []   # 부정 (안, 못)
        interrogative_words = []  # 의문사
        predicate_words = []  # 서술어 (동사, 형용사)
        connective_words = [] # 접속사
        other_words = []      # 기타
        
        # 양태 부사 목록 (함께하는 방식)
        manner_adverb_list = {"같이", "함께", "혼자", "다시", "또"}
        
        for info in analyzed:
            if info.word_type == WordType.TIME:
                time_words.append(info)
            elif info.word_type == WordType.GREETING:
                greeting_words.append(info)
            elif info.word_type == WordType.PRONOUN:
                subject_words.append(info)
            elif info.word_type == WordType.PLACE:
                place_words.append(info)
            elif info.word_type == WordType.ADVERB:
                # 양태 부사는 목적어 앞에, 동작 부사는 서술어 앞에
                if info.word in manner_adverb_list:
                    manner_adverbs.append(info)
                else:
                    action_adverbs.append(info)
            elif info.word_type == WordType.NEGATION:
                negation_words.append(info)
            elif info.word_type == WordType.INTERROGATIVE:
                interrogative_words.append(info)
            elif info.word_type in {WordType.VERB, WordType.ADJECTIVE}:
                predicate_words.append(info)
            elif info.word_type == WordType.CONNECTIVE:
                connective_words.append(info)
            elif info.word_type == WordType.NOUN:
                object_words.append(info)
            else:
                other_words.append(info)
        
        # 한국어 어순으로 재배열
        # 인사 + 시간 + 주어 + 양태부사 + 장소 + 목적어 + 의문사 + 부정 + 동작부사 + 서술어
        reordered = (
            greeting_words +
            time_words +
            subject_words +
            manner_adverbs +    # 같이, 함께 등은 목적어 앞
            place_words +
            object_words +
            other_words +
            interrogative_words +
            negation_words +
            action_adverbs +    # 빨리, 많이 등은 서술어 앞
            predicate_words
        )
        
        return reordered
    
    # ================================================================
    # - 문장 조합 (조사 부착 + 용언 활용)
    # ================================================================
    
    def _compose_sentence(self, words: List[WordInfo], context: SentenceContext) -> str:
        """
        조사 부착 및 용언 활용을 통한 문장 조합
        
        각 단어 유형에 따라 적절한 처리 수행
        """
        result_parts = []
        total_words = len(words)
        
        for idx, info in enumerate(words):
            word = info.word
            is_last = (idx == total_words - 1)
            next_info = words[idx + 1] if idx < total_words - 1 else None
            
            # ────────────────────────────────────
            # Case 1: 인사/감정 표현
            # ────────────────────────────────────
            if info.word_type == WordType.GREETING:
                if word in self.ling.GREETING_MAP:
                    result_parts.append(self.ling.GREETING_MAP[word])
                else:
                    result_parts.append(word)
                continue
            
            # ────────────────────────────────────
            # Case 2: 대명사 (주어)
            # ────────────────────────────────────
            if info.word_type == WordType.PRONOUN:
                converted = self._convert_pronoun(word)
                result_parts.append(converted)
                continue
            
            # ────────────────────────────────────
            # Case 3: 시간 표현
            # ────────────────────────────────────
            if info.word_type == WordType.TIME:
                # 시간 표현은 조사 없이 그대로
                result_parts.append(word)
                continue
            
            # ────────────────────────────────────
            # Case 4: 부사
            # ────────────────────────────────────
            if info.word_type == WordType.ADVERB:
                result_parts.append(word)
                continue
            
            # ────────────────────────────────────
            # Case 5: 부정 표현 (안, 못)
            # ────────────────────────────────────
            if info.word_type == WordType.NEGATION:
                result_parts.append(word)
                continue
            
            # ────────────────────────────────────
            # Case 6: 의문사
            # ────────────────────────────────────
            if info.word_type == WordType.INTERROGATIVE:
                # '뭐' → '무엇'으로 변환
                if word == "뭐":
                    word = "무엇"
                
                # 다음이 동사면 목적격 조사
                if next_info and next_info.word_type == WordType.VERB:
                    particle = self._select_object_particle(word)
                    result_parts.append(f"{word}{particle}")
                else:
                    result_parts.append(word)
                continue
            
            # ────────────────────────────────────
            # Case 7: 장소
            # ────────────────────────────────────
            if info.word_type == WordType.PLACE:
                particle = self._select_place_particle(word, next_info)
                result_parts.append(f"{word}{particle}")
                continue
            
            # ────────────────────────────────────
            # Case 8: 동사
            # ────────────────────────────────────
            if info.word_type == WordType.VERB:
                conjugated = self._conjugate_verb(word, context, is_last)
                result_parts.append(conjugated)
                continue
            
            # ────────────────────────────────────
            # Case 9: 형용사
            # ────────────────────────────────────
            if info.word_type == WordType.ADJECTIVE:
                conjugated = self._conjugate_adjective(word, context, is_last)
                result_parts.append(conjugated)
                continue
            
            # ────────────────────────────────────
            # Case 10: 일반 명사 (목적어)
            # ────────────────────────────────────
            if info.word_type == WordType.NOUN:
                # 다음이 동사면 목적격 조사
                if next_info and next_info.word_type == WordType.VERB:
                    particle = self._select_object_particle(word)
                    result_parts.append(f"{word}{particle}")
                # 마지막이면 서술격 조사
                elif is_last:
                    result_parts.append(f"{word}입니다")
                else:
                    result_parts.append(word)
                continue
            
            # ────────────────────────────────────
            # Case 11: 기타
            # ────────────────────────────────────
            result_parts.append(word)
        
        return " ".join(result_parts)
    
    # ================================================================
    # - 조사 선택 헬퍼 함수
    # ================================================================
    
    def _has_batchim(self, text: str) -> bool:
        """
        받침(종성) 유무 확인
        
        한글 유니코드 계산:
        (글자코드 - '가'코드) % 28 == 0 이면 받침 없음
        
        Args:
            text: 확인할 문자열 (마지막 글자 기준)
        
        Returns:
            True: 받침 있음 (밥, 학, 집)
            False: 받침 없음 (사과, 나무, 커피)
        """
        if not text:
            return False
        
        last_char = text[-1]
        
        # 한글 범위 체크 (가 ~ 힣)
        if '가' <= last_char <= '힣':
            # 유니코드 계산
            # 한글 = 초성(19) × 중성(21) × 종성(28)
            # 종성 0이면 받침 없음
            return (ord(last_char) - ord('가')) % 28 != 0
        
        # 한글이 아니면 받침 없음으로 처리
        return False
    
    def _select_subject_particle(self, word: str) -> str:
        """
        주격 조사 선택 (이/가)
        
        받침 있으면 '이', 없으면 '가'
        """
        return "이" if self._has_batchim(word) else "가"
    
    def _select_topic_particle(self, word: str) -> str:
        """
        보조사 선택 (은/는)
        
        받침 있으면 '은', 없으면 '는'
        """
        return "은" if self._has_batchim(word) else "는"
    
    def _select_object_particle(self, word: str) -> str:
        """
        목적격 조사 선택 (을/를)
        
        받침 있으면 '을', 없으면 '를'
        """
        return "을" if self._has_batchim(word) else "를"
    
    def _select_place_particle(self, word: str, next_info: Optional[WordInfo]) -> str:
        """
        장소 조사 선택 (에/에서)
        
        동사에 따라 결정:
        - 이동 동사 (가다, 오다): 에
        - 상태 동사 (있다, 없다, 살다): 에
        - 행위 동사 (일하다, 먹다, 공부하다): 에서
        """
        if next_info and next_info.word_type == WordType.VERB:
            verb = next_info.word
            if verb in self.ling.PLACE_PARTICLE_BY_VERB:
                return self.ling.PLACE_PARTICLE_BY_VERB[verb]
            # 기본값: 에
            return "에"
        else:
            # 다음이 동사가 아니면 주격 조사
            return self._select_subject_particle(word)
    
    # ================================================================
    # - 대명사 변환
    # ================================================================
    
    def _convert_pronoun(self, word: str) -> str:
        """
        대명사를 높임말로 변환하고 조사 부착
        
        나 → 저는
        우리 → 저희는
        """
        pronoun_map = {
            "나": "저는",
            "저": "저는",
            "우리": "저희는",
            "저희": "저희는",
            "너": "당신은",
        }
        
        if word in pronoun_map:
            return pronoun_map[word]
        
        # 기타 대명사는 보조사 '는' 부착
        particle = self._select_topic_particle(word)
        return f"{word}{particle}"
    
    # ================================================================
    # - 동사 활용
    # ================================================================
    
    def _conjugate_verb(self, verb: str, context: SentenceContext, is_final: bool = True) -> str:
        """
        동사 활용
        
        시제, 문장유형, 요청 여부에 따라 적절한 활용형 선택
        
        Args:
            verb: 동사 기본형
            context: 문맥 정보
            is_final: 문장 끝 여부
        
        Returns:
            활용된 동사
        """
        # 사전에서 활용형 찾기
        if verb in self.ling.VERB_CONJUGATION:
            forms = self.ling.VERB_CONJUGATION[verb]
            
            # 요청문이면 request 형태
            if context.is_request and "request" in forms:
                return forms["request"]
            
            # 의문문이면 question 형태
            if context.is_question and "question" in forms:
                return forms["question"]
            
            # 시제에 따른 활용
            if context.tense == Tense.PAST and "past" in forms:
                return forms["past"]
            elif context.tense == Tense.FUTURE and "future" in forms:
                return forms["future"]
            else:
                return forms.get("present", verb)
        
        # 사전에 없으면 규칙 활용
        return self._conjugate_verb_by_rule(verb, context)
    
    def _conjugate_verb_by_rule(self, verb: str, context: SentenceContext) -> str:
        """
        규칙 기반 동사 활용
        
        ~하다 형태와 일반 동사 처리
        """
        if not verb.endswith("다"):
            return verb
        
        stem = verb[:-1]  # '다' 제거
        
        # ~하다 동사
        if stem.endswith("하"):
            if context.tense == Tense.PAST:
                return stem[:-1] + "했습니다"
            elif context.tense == Tense.FUTURE:
                return stem[:-1] + "할 것입니다"
            else:
                return stem[:-1] + "합니다"
        
        # 일반 동사
        if context.tense == Tense.PAST:
            # 받침에 따른 과거형
            if self._has_batchim(stem):
                return stem + "었습니다"
            else:
                return stem + "ㅆ습니다"
        elif context.tense == Tense.FUTURE:
            return stem + "ㄹ 것입니다"
        else:
            # 현재형
            if self._has_batchim(stem):
                return stem + "습니다"
            else:
                return stem + "ㅂ니다"
    
    # ================================================================
    # - 형용사 활용
    # ================================================================
    
    def _conjugate_adjective(self, adj: str, context: SentenceContext, is_final: bool = True) -> str:
        """
        형용사 활용
        
        시제에 따른 활용형 선택
        """
        # 사전에서 활용형 찾기
        if adj in self.ling.ADJECTIVE_CONJUGATION:
            forms = self.ling.ADJECTIVE_CONJUGATION[adj]
            
            if context.is_question and "question" in forms:
                return forms["question"]
            
            if context.tense == Tense.PAST and "past" in forms:
                return forms["past"]
            else:
                return forms.get("present", adj)
        
        # 사전에 없으면 규칙 활용
        return self._conjugate_adjective_by_rule(adj, context)
    
    def _conjugate_adjective_by_rule(self, adj: str, context: SentenceContext) -> str:
        """
        규칙 기반 형용사 활용
        
        ~하다 형태, ㅂ 불규칙, 르 불규칙 등 처리
        """
        if not adj.endswith("다"):
            return adj
        
        stem = adj[:-1]
        
        # ~하다 형용사
        if stem.endswith("하"):
            if context.tense == Tense.PAST:
                return stem[:-1] + "했습니다"
            else:
                return stem[:-1] + "합니다"
        
        # 르 불규칙 (빠르다, 다르다)
        if stem.endswith("르"):
            if context.tense == Tense.PAST:
                return stem[:-1] + "랐습니다"
            else:
                return stem[:-1] + "릅니다"
        
        # 일반 규칙
        if context.tense == Tense.PAST:
            if self._has_batchim(stem):
                return stem + "았습니다"
            else:
                return stem + "었습니다"
        else:
            if self._has_batchim(stem):
                return stem + "습니다"
            else:
                return stem + "ㅂ니다"
    
    # ================================================================
    # - 문장 마무리
    # ================================================================
    
    def _finalize_sentence(self, sentence: str, context: SentenceContext) -> str:
        """
        문장 유형에 따른 마무리 처리
        
        의문문이면 종결어미를 의문형으로 변환
        """
        if context.is_question:
            # 이미 물음표가 있으면 스킵
            if sentence.endswith("?"):
                return sentence
            
            # 종결어미 변환
            if sentence.endswith("입니다"):
                return sentence[:-3] + "인가요?"
            elif sentence.endswith("습니다"):
                return sentence[:-3] + "나요?"
            elif sentence.endswith("ㅂ니다"):
                return sentence[:-3] + "나요?"
            else:
                return sentence + "?"
        
        return sentence
    
    # ================================================================
    #  - 후처리
    # ================================================================
    
    def _postprocess(self, sentence: str) -> str:
        """
        문장 후처리
        
        1. 다중 공백 제거
        2. 조사 앞 공백 제거
        3. 문장 부호 정리
        """
        # 다중 공백 → 단일 공백
        sentence = re.sub(r'\s+', ' ', sentence)
        
        # 조사 앞 불필요한 공백 제거
        # "밥 을" → "밥을"
        particles = r'(은|는|이|가|을|를|에|에서|으로|로|와|과|의|도|만|까지|부터)'
        sentence = re.sub(rf'\s+({particles})\b', r'\1', sentence)
        
        # 앞뒤 공백 제거
        sentence = sentence.strip()
        
        # 중복 문장부호 제거
        sentence = re.sub(r'\.+', '.', sentence)
        sentence = re.sub(r'\?+', '?', sentence)
        sentence = re.sub(r'!+', '!', sentence)
        
        return sentence
    
    # ================================================================
    # - 캐싱
    # ================================================================
    
    def _check_cache(self, sentence: str) -> Optional[str]:
        """
        캐시된 TTS 파일 확인
        
        문장의 SHA-256 해시를 키로 사용
        """
        hash_key = hashlib.sha256(sentence.encode('utf-8')).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"{hash_key}.mp3")
        
        if os.path.exists(cache_path):
            return cache_path
        return None
    
    def get_cache_path(self, sentence: str) -> str:
        """
        TTS 파일 저장 경로 생성
        """
        hash_key = hashlib.sha256(sentence.encode('utf-8')).hexdigest()
        return os.path.join(CACHE_DIR, f"{hash_key}.mp3")


# ================================================================================
# 테스트
# ================================================================================

if __name__ == "__main__":
    manager = HyemiTextManager()
    
    # 종합 테스트 케이스
    test_cases = [
        # ── 기본 문장 (조사 테스트) ──
        (["나", "밥", "먹다"], "저는 밥을 먹습니다"),
        (["나", "사과", "먹다"], "저는 사과를 먹습니다"),
        
        # ── 시제 테스트 ──
        (["어제", "나", "친구", "만나다"], "어제 저는 친구를 만났습니다"),
        (["내일", "나", "병원", "가다"], "내일 저는 병원에 갈 것입니다"),
        (["지금", "나", "공부하다"], "지금 저는 공부합니다"),
        
        # ── 장소 + 동사 ──
        (["나", "학교", "가다"], "저는 학교에 갑니다"),
        (["나", "회사", "일하다"], "저는 회사에서 일합니다"),
        (["어디", "살다"], "어디에 사시나요?"),
        
        # ── 의문문 ──
        (["화장실", "어디"], "화장실이 어디인가요?"),
        (["이름이", "무엇이"], "성함이 어떻게 되시나요?"),
        (["나이가", "몇"], "연세가 어떻게 되시나요?"),
        (["언제", "오다"], "언제 오시나요?"),
        
        # ── 복합 표현 ──
        (["배", "고프다"], "배가 고픕니다"),
        (["머리", "아프다"], "머리가 아픕니다"),
        (["날씨", "좋다"], "날씨가 좋습니다"),
        (["시간", "없다"], "시간이 없습니다"),
        
        # ── 인사 ──
        (["안녕", "나", "학생"], "안녕하세요 저는 학생입니다"),
        (["처음", "만나다"], "처음 뵙겠습니다"),
        (["잘", "부탁"], "잘 부탁드립니다"),
        (["감사"], "감사합니다"),
        
        # ── 요청/부탁 ──
        (["다시", "말하다"], "다시 말씀해 주세요"),
        (["천천히", "말하다"], "천천히 말씀해 주세요"),
        (["도움", "주다"], "도와주세요"),
        
        # ── 부사 포함 ──
        (["빨리", "오다"], "빨리 옵니다"),
        (["같이", "밥", "먹다"], "같이 밥을 먹습니다"),
        (["많이", "피곤하다"], "많이 피곤합니다"),
        
        # ── 형용사 ──
        (["오늘", "날씨", "덥다"], "오늘 날씨가 덥습니다"),
        (["어제", "날씨", "춥다"], "어제 날씨가 추웠습니다"),
    ]
    
    print("\n" + "="*70)
    print("text_manager 종합 테스트")
    print("="*70 + "\n")
    
    passed = 0
    failed = 0
    
    for input_words, expected in test_cases:
        result, _ = manager.process_text(input_words)
        
        # 결과 비교 (공백 무시)
        result_normalized = result.replace(" ", "")
        expected_normalized = expected.replace(" ", "")
        
        if result_normalized == expected_normalized:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
        
        print(f"- 입력: {input_words}")
        print(f"- 결과: {result}")
        print(f"- 예상: {expected}")
        print(f"   {status}")
        print("-" * 50)
    
    print(f"\n📊 결과: {passed}/{passed+failed} 통과 ({passed/(passed+failed)*100:.1f}%)\n")