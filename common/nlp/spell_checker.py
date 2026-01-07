import re
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum, auto

# ================================================================================
# 로깅 설정
# ================================================================================

logger = logging.getLogger("SpellChecker") # 로거 생성: 교정 과정 추적용

# 콘솔 핸들러가 없으면 추가 (중복 방지)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)


# ================================================================================
# 데이터 타입 정의
# ================================================================================

class CorrectionType(Enum):
    """
    교정 유형 분류
    
    UI에서 교정 이유를 표시할 때 사용
    """
    SPELLING = auto()      # 맞춤법 (됬→됐)
    SPACING = auto()       # 띄어쓰기 (안해→안 해)
    PARTICLE = auto()      # 조사 (거에요→거예요)
    FOREIGN = auto()       # 외래어 (써비스→서비스)
    PUNCTUATION = auto()   # 문장부호 (??→?)
    HONORIFIC = auto()     # 높임법 (뭐에요→뭐예요)


@dataclass
class CorrectionResult:
    """
    교정 결과 데이터 클래스
    
    Attributes:
        original: 원본 텍스트
        corrected: 교정된 텍스트
        corrections: 개별 교정 내역 리스트
        has_errors: 오류 존재 여부
    """
    original: str                          # 원본 텍스트
    corrected: str                         # 교정된 텍스트
    corrections: List[Dict[str, str]]      # 교정 내역
    has_errors: bool                       # 오류 존재 여부
    
    def __repr__(self):
        return f"CorrectionResult(has_errors={self.has_errors}, corrections={len(self.corrections)})"


# ================================================================================
# 한국어 맞춤법 교정기
# ================================================================================

class KoreanSpellChecker:
    """
    한국어 맞춤법 교정기
    
    [처리 순서]
    1. 맞춤법 사전 기반 교정
    2. 정규식 패턴 기반 교정
    3. 띄어쓰기 규칙 적용
    4. 문장 부호 정리
    
    [사용 예시]
    >>> checker = KoreanSpellChecker()
    >>> checker.check("이거 할께요")
    '이거 할게요'
    """
    
    # ========================================
    #  - 맞춤법 교정 사전
    # ========================================
    # key: 틀린 표현, value: 올바른 표현
    # 
    # 분류:
    # - ㄱ/ㅋ 혼동
    # - ㄷ/ㅌ 혼동
    # - 된소리 혼동
    # - 사이시옷
    # - 불규칙 활용
    # - 기타 자주 틀리는 표현
    
    SPELLING_CORRECTIONS: Dict[str, str] = {
        
        # ────────────────────────────────────
        # [A] 되다/돼다 혼동 (가장 흔한 오류)
        # ────────────────────────────────────
        # 규칙: '되어'가 줄면 '돼', 그 외는 '되'
        # 예: 되어요 → 돼요, 되었다 → 됐다
        
        "됬": "됐",           # 됐 (되+었)
        "됫": "됐",
        "되요": "돼요",       # 돼요 (되+어요)
        "되서": "돼서",       # 돼서 (되+어서)
        "되도": "돼도",       # 돼도 (되+어도)
        "되든": "돼든",       # 돼든 (되+어든)
        "되면": "되면",       # 되면 (O) - 이건 맞음
        "안되": "안 돼",      # 띄어쓰기 + 돼
        "않되": "안 돼",      # 않 → 안
        "안돼요": "안 돼요",
        "않돼요": "안 돼요",
        "될께요": "될게요",   # ㄹ게 (ㄹ께 X)
        "될꺼": "될 거",      # 띄어쓰기 + 거
        "될거에요": "될 거예요",
        "될거야": "될 거야",
        
        # ────────────────────────────────────
        # [B] 해/헤 혼동
        # ────────────────────────────────────
        
        "햇": "했",           # 했 (하+었)
        "핬": "했",
        "해써": "했어",
        "햇어": "했어",
        "해쓰": "했어",
        "해요": "해요",       # 이건 맞음
        "해서": "해서",       # 이건 맞음
        
        # ────────────────────────────────────
        # [C] ㄹ게/ㄹ께 혼동
        # ────────────────────────────────────
        # 규칙: 항상 'ㄹ게' (ㄹ께는 없음)
        
        "할게": "할게",       # 맞음
        "할께": "할게",       # ㄹ께 → ㄹ게
        "갈께": "갈게",
        "볼께": "볼게",
        "먹을께": "먹을게",
        "올께": "올게",
        "줄께": "줄게",
        "할게요": "할게요",   # 맞음
        "할께요": "할게요",
        "해볼게": "해볼게",   # 맞음
        "해볼께": "해볼게",
        "갈게요": "갈게요",
        "갈께요": "갈게요",
        
        # ────────────────────────────────────
        # [D] 거에요/거예요 혼동
        # ────────────────────────────────────
        # 규칙: 받침 없으면 '예요', 있으면 '이에요'
        # '거'는 받침 없음 → '거예요'
        
        "거에요": "거예요",   # 거 + 예요
        "뭐에요": "뭐예요",   # 뭐 + 예요
        "게에요": "게예요",
        "애기": "아기",       # 아기 (O)
        "에기": "아기",
        
        # ────────────────────────────────────
        # [E] 안/않 혼동
        # ────────────────────────────────────
        # 규칙: 
        # - '안' = 부정 부사 (안 가다)
        # - '않' = 보조용언 (가지 않다)
        
        "않해": "안 해",      # 안 해 (O)
        "안해": "안 해",      # 띄어쓰기
        "않가": "안 가",
        "안가": "안 가",
        "않갔": "안 갔",
        "않먹": "안 먹",
        "않해요": "안 해요",
        "않았어": "않았어",   # 이건 맞음 (-지 않았어)
        "않는다": "않는다",   # 이건 맞음 (-지 않는다)
        
        # ────────────────────────────────────
        # [F] 어떻게/어떡해 혼동
        # ────────────────────────────────────
        # 규칙:
        # - '어떻게' = 부사 (어떻게 해요?)
        # - '어떡해' = 어떻게 해의 준말 (어떡해!)
        
        "어떻해": "어떡해",   # 어떡해 (어떻게 해)
        "어떻게": "어떻게",   # 맞음
        "어떻하지": "어떡하지",
        "어떻해요": "어떡해요",
        "어떻케": "어떻게",
        "어떄": "어때",       # 어때 (어떠해)
        "어떄요": "어때요",
        
        # ────────────────────────────────────
        # [G] 이/히 혼동 (부사 파생 접미사)
        # ────────────────────────────────────
        # 불규칙하여 암기 필요
        
        "깨끗히": "깨끗이",   # 깨끗이 (O)
        "깨끗하게": "깨끗하게", # 맞음
        "조용이": "조용히",   # 조용히 (O)
        "급하게": "급하게",   # 맞음
        "급히": "급히",       # 맞음
        "솔직이": "솔직히",
        "정확이": "정확히",
        "분명이": "분명히",
        
        # ────────────────────────────────────
        # [H] ㅎ 탈락
        # ────────────────────────────────────
        
        "괜찬아": "괜찮아",   # 괜찮다
        "괜챦아": "괜찮아",
        "괜찬": "괜찮",
        "않해": "안 해",
        "않되": "안 돼",
        "그래": "그래",       # 맞음
        "좋아": "좋아",       # 맞음
        "넣는": "넣는",       # 맞음
        
        # ────────────────────────────────────
        # [I] 사이시옷
        # ────────────────────────────────────
        # 규칙: 복합어에서 뒷말 첫소리가 된소리로 나면 사이시옷
        # 단, 외래어나 한자어+한자어는 제외
        
        "갯수": "개수",       # 한자어+한자어 = 사이시옷 X
        "댓가": "대가",
        "숫자": "숫자",       # 맞음 (순우리말)
        "횟수": "횟수",       # 맞음
        "잇몸": "잇몸",       # 맞음
        "곳간": "곳간",       # 맞음
        "셋방": "셋방",       # 맞음
        
        # ────────────────────────────────────
        # [J] 기타 자주 틀리는 표현
        # ────────────────────────────────────
        
        "왠지": "왠지",       # 왠지 (왜인지) - 맞음
        "웬지": "왠지",       # 웬 → 왠
        "왠만하면": "웬만하면", # 웬만하면 (O)
        "왠만큼": "웬만큼",
        "몇일": "며칠",       # 며칠 (O)
        "몇 일": "며칠",
        "설겆이": "설거지",   # 설거지 (O)
        "설겂이": "설거지",
        "금새": "금세",       # 금세 (O)
        "오랫만": "오랜만",   # 오랜만 (O)
        "오랫동안": "오랫동안", # 맞음
        "희안하다": "희한하다", # 희한하다 (O)
        "희안한": "희한한",
        "문안하다": "무난하다", # 무난하다 (O)
        "일부로": "일부러",   # 일부러 (O)
        "어의없다": "어이없다", # 어이없다 (O)
        "어의가": "어이가",
        "구지": "굳이",       # 굳이 (O)
        "곰곰히": "곰곰이",   # 곰곰이 (O)
        "바램": "바람",       # 바람 (希望의 의미)
        "예기": "얘기",       # 얘기 (이야기)
        "가르켜": "가리켜",   # 가리키다
        "가르키다": "가리키다",
        "가르쳐": "가르쳐",   # 맞음 (가르치다)
        "틀리다": "틀리다",   # 맞음 (wrong)
        "다르다": "다르다",   # 맞음 (different)
        "낳다": "낳다",       # 맞음 (출산)
        "낫다": "낫다",       # 맞음 (치유, 비교)
        "넣다": "넣다",       # 맞음 (삽입)
        "빛나다": "빛나다",   # 맞음
        "비취다": "비치다",   # 비치다 (O)
        "비추다": "비추다",   # 맞음
        "네": "네",           # 맞음
        "네": "네",           # 맞음 (응답)
        "데": "데",           # 맞음 (의존명사)
        "대": "대",           # 맞음 (의존명사)
        "뵙겠습니다": "뵙겠습니다",  # 맞음
        "봽겠습니다": "뵙겠습니다",
        "싶습니다": "싶습니다",      # 맞음
        "싶읍니다": "싶습니다",
        "합니다": "합니다",          # 맞음
        "함니다": "합니다",
        "입니다": "입니다",          # 맞음
        "임니다": "입니다",
        "습니다": "습니다",          # 맞음
        "슴니다": "습니다",
        
        # ────────────────────────────────────
        # [K] 줄임말/구어체
        # ────────────────────────────────────
        
        "머임": "뭐임",       # 뭐임
        "머야": "뭐야",       # 뭐야
        "뭐임": "뭐임",       # 맞음
        "머라고": "뭐라고",
        "없음": "없음",       # 맞음
        "있음": "있음",       # 맞음
        "됌": "됨",           # 됨 (O)
        "됨": "됨",           # 맞음
        "함": "함",           # 맞음
        "임": "임",           # 맞음
        
        # ────────────────────────────────────
        # [L] 외래어 표기 오류
        # ────────────────────────────────────
        
        "써비스": "서비스",   # 서비스 (service)
        "쎄일": "세일",       # 세일 (sale)
        "컴퓨타": "컴퓨터",   # 컴퓨터 (computer)
        "핸드폰": "휴대폰",   # 휴대폰 (표준어)
        "휴대전화": "휴대전화", # 맞음
        "셀프": "셀프",       # 맞음 (self)
        "쎌프": "셀프",
        "메세지": "메시지",   # 메시지 (message)
        "마싸지": "마사지",   # 마사지 (massage)
        "악세사리": "액세서리", # 액세서리 (accessory)
        "레포트": "리포트",   # 리포트 (report)
        "로보트": "로봇",     # 로봇 (robot)
        "포인트": "포인트",   # 맞음 (point)
        "퍼센트": "퍼센트",   # 맞음 (percent)
        "프로세스": "프로세스", # 맞음 (process)
        "프러세스": "프로세스",
        "라이센스": "라이선스", # 라이선스 (license)
        "라이센서": "라이선서",
        "비지니스": "비즈니스", # 비즈니스 (business)
        "비지네스": "비즈니스",
    }
    
    # ========================================
    # - 띄어쓰기 교정 규칙
    # ========================================
    
    # ── 붙여 써야 하는 패턴 (조사) ──
    # 조사는 앞 단어에 붙여 씀
    PARTICLES_TO_ATTACH: Set[str] = {
        "은", "는", "이", "가", "을", "를",
        "에", "에서", "에게", "에게서",
        "으로", "로", "으로서", "로서", "으로써", "로써",
        "와", "과", "하고", "이랑", "랑",
        "의", "도", "만", "까지", "부터", "마저", "조차",
        "처럼", "같이", "보다", "마냥",
        "이나", "나", "이든", "든", "이든지", "든지",
        "이야", "야", "아", "여",
        "요", "이요",
    }
    
    # ── 띄어 써야 하는 의존명사 ──
    # 의존명사는 앞말과 띄어 씀
    # 예: 할 수 있다, 먹을 것, 갈 때
    DEPENDENT_NOUNS: Dict[str, str] = {
        "것": "것",     # ~ㄹ 것이다
        "거": "거",     # ~ㄹ 거야 (구어)
        "수": "수",     # ~ㄹ 수 있다
        "줄": "줄",     # ~ㄹ 줄 알다
        "리": "리",     # ~ㄹ 리 없다
        "듯": "듯",     # ~ㄴ 듯하다
        "때": "때",     # ~ㄹ 때
        "데": "데",     # ~ㄴ 데
        "지": "지",     # ~ㄴ 지 (시간)
        "만큼": "만큼", # ~ㄴ 만큼
        "대로": "대로", # ~ㄴ 대로
        "뿐": "뿐",     # ~ㄹ 뿐이다
        "척": "척",     # ~ㄴ 척하다
        "체": "체",     # ~ㄴ 체하다
        "양": "양",     # ~ㄴ 양하다
    }
    
    # ── 보조용언 띄어쓰기 ──
    # 본용언 + 보조용언은 띄어 쓰는 것이 원칙
    # 예: 해 보다, 해 주다, 해 버리다
    AUXILIARY_VERBS: List[str] = [
        "보다", "주다", "버리다", "내다", "두다", "놓다",
        "오다", "가다", "싶다", "하다", "있다", "없다",
        "말다", "대다", "빠지다", "치우다",
    ]
    
    # ========================================
    # - 정규식 패턴
    # ========================================
    
    # 조사 앞 공백 제거 패턴
    PARTICLE_PATTERN = re.compile(
        r'(\S)\s+(은|는|이|가|을|를|에|에서|으로|로|와|과|의|도|만|까지|부터|처럼|보다|이나|나|요)(?=\s|$|[.?!,])'
    )
    
    # 의존명사 앞 공백 확보 패턴
    # 예: "할수있다" → "할 수 있다"
    DEPENDENT_NOUN_PATTERNS = [
        (r'([를을ㄹ])([수것거줄리듯때데지])', r'\1 \2'),  # 을수 → 을 수
        (r'([가-힣])(수)(있|없)', r'\1 \2 \3'),          # 할수있다 → 할 수 있다
        (r'([가-힣])(것)(이|을|도)', r'\1 \2\3'),        # 할것이다 → 할 것이다
    ]
    
    # 부정 부사 띄어쓰기
    # 예: "안해" → "안 해", "못해" → "못 해"
    NEGATION_PATTERNS = [
        (r'안([가-힣])', r'안 \1'),   # 안+동사
        (r'못([가-힣])', r'못 \1'),   # 못+동사
    ]
    
    # ========================================
    # - 초기화
    # ========================================
    
    def __init__(self, use_online: bool = False):
        """
        맞춤법 교정기 초기화
        
        Args:
            use_online: 온라인 API 사용 여부
                       True면 py_hanspell(네이버 맞춤법) 연동 시도
                       네이버 API는 더 정확하지만 네트워크 필요
        
        Example:
            >>> checker = KoreanSpellChecker()           # 오프라인 모드
            >>> checker = KoreanSpellChecker(True)       # 온라인 모드
        """
        self.use_online = use_online
        self.online_checker = None
        
        # 온라인 검사기 초기화 시도
        if use_online:
            try:
                from py_hanspell import spell_checker
                self.online_checker = spell_checker
                logger.info("온라인 맞춤법 검사기 (네이버) 활성화")
            except ImportError:
                logger.warning("py_hanspell 미설치. pip install py-hanspell")
                self.use_online = False
    
    # ========================================
    # - 메인 교정 함수
    # ========================================
    
    def check(self, text: str) -> str:
        """
        맞춤법 검사 및 교정 (간단 버전)
        
        Args:
            text: 검사할 문장
        
        Returns:
            교정된 문장
        
        Example:
            >>> checker.check("이거 할께요")
            '이거 할게요'
            >>> checker.check("됬어요")
            '됐어요'
        """
        if not text or not text.strip():
            return text
        
        # Step 1: 규칙 기반 교정
        corrected = self._apply_all_rules(text)
        
        # Step 2: 온라인 검사 (선택적)
        if self.use_online and self.online_checker:
            try:
                result = self.online_checker.check(corrected)
                corrected = result.checked
            except Exception as e:
                logger.warning(f"온라인 검사 실패: {e}")
        
        return corrected
    
    def check_detailed(self, text: str) -> CorrectionResult:
        """
        맞춤법 검사 및 교정 (상세 버전)
        
        교정 내역을 함께 반환하여 UI에서 활용 가능
        
        Args:
            text: 검사할 문장
        
        Returns:
            CorrectionResult: 교정 결과 객체
        
        Example:
            >>> result = checker.check_detailed("이거 할께요")
            >>> print(result.corrected)
            '이거 할게요'
            >>> print(result.corrections)
            [{'original': '할께요', 'corrected': '할게요', 'type': 'SPELLING'}]
        """
        if not text or not text.strip():
            return CorrectionResult(
                original=text,
                corrected=text,
                corrections=[],
                has_errors=False
            )
        
        corrections = []
        corrected = text
        
        # Step 1: 맞춤법 사전 적용
        for wrong, correct in self.SPELLING_CORRECTIONS.items():
            if wrong in corrected and wrong != correct:
                corrected = corrected.replace(wrong, correct)
                corrections.append({
                    "original": wrong,
                    "corrected": correct,
                    "type": CorrectionType.SPELLING.name
                })
        
        # Step 2: 띄어쓰기 교정
        corrected, spacing_corrections = self._apply_spacing_rules(corrected)
        corrections.extend(spacing_corrections)
        
        # Step 3: 문장 부호 정리
        corrected = self._normalize_punctuation(corrected)
        
        return CorrectionResult(
            original=text,
            corrected=corrected,
            corrections=corrections,
            has_errors=len(corrections) > 0
        )
    
    # ========================================
    # - 규칙 적용 함수들
    # ========================================
    
    def _apply_all_rules(self, text: str) -> str:
        """
        모든 교정 규칙 순차 적용
        
        적용 순서가 중요함:
        1. 맞춤법 사전 → 기본 오류 교정
        2. 부정 부사 띄어쓰기 → "안해" → "안 해"
        3. 조사 붙이기 → "밥 을" → "밥을"
        4. 의존명사 띄어쓰기 → "할수" → "할 수"
        5. 문장 부호 정리 → 중복 제거
        """
        result = text
        
        # 1. 맞춤법 사전 기반 교정
        for wrong, correct in self.SPELLING_CORRECTIONS.items():
            if wrong != correct:  # 동일한 경우 스킵
                result = result.replace(wrong, correct)
        
        # 2. 부정 부사 띄어쓰기 ("안해" → "안 해")
        # 단, "안녕", "안전" 등은 제외
        negation_exceptions = [
            "안녕", "안전", "안심", "안내", "안경", "안개", 
            "안방", "안쪽", "안면", "안과", "안주", "안건",
            "안락", "안정", "안부", "안색", "안식", "안타",
            "못생", "못난",  # 못 예외
        ]
        
        # 예외 단어를 임시 마커로 대체
        temp_result = result
        for i, exc in enumerate(negation_exceptions):
            temp_result = temp_result.replace(exc, f"§§{i}§§")
        
        # 부정 부사 패턴 적용
        temp_result = re.sub(r'안([가-힣])', r'안 \1', temp_result)
        temp_result = re.sub(r'못([가-힣])', r'못 \1', temp_result)
        
        # 예외 단어 복원
        for i, exc in enumerate(negation_exceptions):
            temp_result = temp_result.replace(f"§§{i}§§", exc)
        
        result = temp_result
        
        # 3. 조사 앞 공백 제거
        # "밥 을" → "밥을"
        result = self.PARTICLE_PATTERN.sub(r'\1\2', result)
        
        # 4. 의존명사 앞 공백 확보
        # "할수있다" → "할 수 있다"
        # 패턴: 동사어간 + 수/것/줄/리 + 있다/없다/알다
        result = re.sub(r'([가-힣])수(있|없|알)', r'\1 수 \2', result)
        result = re.sub(r'([가-힣])것(이|을|도|은)', r'\1 것\2', result)
        result = re.sub(r'([가-힣])줄(알|모르)', r'\1 줄 \2', result)
        
        for pattern, replacement in self.DEPENDENT_NOUN_PATTERNS:
            result = re.sub(pattern, replacement, result)
        
        # 5. 중복 공백 제거
        result = re.sub(r'\s+', ' ', result)
        
        # 6. 문장 부호 정리
        result = self._normalize_punctuation(result)
        
        return result.strip()
    
    def _apply_spacing_rules(self, text: str) -> Tuple[str, List[Dict]]:
        """
        띄어쓰기 규칙 적용 및 교정 내역 반환
        
        Returns:
            Tuple[str, List[Dict]]: (교정된 텍스트, 교정 내역)
        """
        result = text
        corrections = []
        
        # 부정 부사 띄어쓰기
        for pattern, replacement in self.NEGATION_PATTERNS:
            if re.search(pattern, result):
                new_result = re.sub(pattern, replacement, result)
                if new_result != result:
                    corrections.append({
                        "original": re.search(pattern, result).group(0),
                        "corrected": replacement.replace(r'\1', ''),
                        "type": CorrectionType.SPACING.name
                    })
                    result = new_result
        
        # 조사 붙이기
        result = self.PARTICLE_PATTERN.sub(r'\1\2', result)
        
        return result, corrections
    
    def _normalize_punctuation(self, text: str) -> str:
        """
        문장 부호 정리
        
        1. 문장 부호 앞 공백 제거
        2. 중복 부호 제거
        3. 문장 부호 뒤 공백 확보
        """
        result = text
        
        # 1. 문장 부호 앞 공백 제거
        # "안녕 ?" → "안녕?"
        result = re.sub(r'\s+([.?!,;:])', r'\1', result)
        
        # 2. 중복 부호 제거
        # "???" → "?", "!!!" → "!"
        result = re.sub(r'\.{2,}', '...', result)  # ... 은 유지
        result = re.sub(r'\?{2,}', '?', result)
        result = re.sub(r'!{2,}', '!', result)
        result = re.sub(r',{2,}', ',', result)
        
        # 3. 문장 부호 뒤 공백 확보 (문장 끝 제외)
        result = re.sub(r'([.?!,])([가-힣a-zA-Z])', r'\1 \2', result)
        
        return result
    
    # ========================================
    # - 유틸리티 함수
    # ========================================
    
    def get_suggestions(self, text: str) -> List[Dict[str, str]]:
        """
        교정 제안 목록 반환 (UI용)
        
        실제 교정은 하지 않고 제안만 반환
        사용자에게 선택권을 줄 때 사용
        
        Returns:
            List[Dict]: 교정 제안 목록
            
        Example:
            >>> checker.get_suggestions("이거 할께요")
            [{'original': '할께요', 'corrected': '할게요', 'type': 'SPELLING', 'reason': 'ㄹ게/ㄹ께 혼동'}]
        """
        suggestions = []
        
        for wrong, correct in self.SPELLING_CORRECTIONS.items():
            if wrong in text and wrong != correct:
                suggestions.append({
                    "original": wrong,
                    "corrected": correct,
                    "type": CorrectionType.SPELLING.name,
                    "reason": self._get_correction_reason(wrong)
                })
        
        return suggestions
    
    def _get_correction_reason(self, wrong: str) -> str:
        """
        교정 이유 반환 (사용자 설명용)
        """
        reasons = {
            "됬": "되다/돼다 혼동: '되었'의 준말은 '됐'",
            "할께": "ㄹ게/ㄹ께 혼동: 항상 'ㄹ게'로 씀",
            "안해": "띄어쓰기: '안'은 부사이므로 띄어 씀",
            "거에요": "조사 혼동: 받침 없으면 '예요'",
            "어떻해": "어떻게/어떡해: '어떻게 해'의 준말은 '어떡해'",
        }
        
        for key, reason in reasons.items():
            if key in wrong:
                return reason
        
        return "맞춤법 오류"
    
    def has_errors(self, text: str) -> bool:
        """
        오류 존재 여부만 빠르게 확인
        
        전체 교정 없이 오류 유무만 판단할 때 사용
        """
        for wrong in self.SPELLING_CORRECTIONS.keys():
            if wrong in text:
                return True
        return False


# ================================================================================
# 텍스트 정규화기
# ================================================================================

class TextNormalizer:
    """
    텍스트 정규화 유틸리티
    
    [기능]
    1. 숫자 → 한글 변환 (TTS용)
    2. 영어 → 한글 발음 변환
    3. 특수문자 정리
    4. 단위 표현 정규화
    
    [사용 예시]
    >>> normalizer = TextNormalizer()
    >>> normalizer.numbers_to_korean("3시 30분")
    '세 시 삼십 분'
    """
    
    # ========================================
    # - 숫자 읽기 사전
    # ========================================
    
    # 한자어 숫자 (일, 이, 삼...)
    # 사용: 분, 초, 원, 번, 호, 층, 년, 월, 일(날짜), 번째 등
    SINO_KOREAN_DIGITS: Dict[str, str] = {
        "0": "영",
        "1": "일",
        "2": "이",
        "3": "삼",
        "4": "사",
        "5": "오",
        "6": "육",
        "7": "칠",
        "8": "팔",
        "9": "구",
    }
    
    # 고유어 숫자 (하나, 둘, 셋...)
    # 사용: 시(시간), 개, 명, 살, 잔, 병, 마리, 권, 장 등
    NATIVE_KOREAN_DIGITS: Dict[str, str] = {
        "1": "한",      # 한 시, 한 개
        "2": "두",      # 두 시, 두 개
        "3": "세",      # 세 시, 세 개
        "4": "네",      # 네 시, 네 개
        "5": "다섯",
        "6": "여섯",
        "7": "일곱",
        "8": "여덟",
        "9": "아홉",
        "10": "열",
        "20": "스물",
    }
    
    # 단독 숫자 (하나, 둘, 셋... - 관형사가 아닌 명사형)
    NATIVE_KOREAN_STANDALONE: Dict[str, str] = {
        "1": "하나",
        "2": "둘",
        "3": "셋",
        "4": "넷",
        "5": "다섯",
        "6": "여섯",
        "7": "일곱",
        "8": "여덟",
        "9": "아홉",
        "10": "열",
    }
    
    # 큰 수 단위
    LARGE_UNITS: Dict[int, str] = {
        100000000: "억",
        10000: "만",
        1000: "천",
        100: "백",
        10: "십",
    }
    
    # 단위별 숫자 읽기 방식
    # True: 고유어 (한, 두, 세), False: 한자어 (일, 이, 삼)
    UNIT_READING_STYLE: Dict[str, bool] = {
        # 고유어 사용 단위
        "시": True,      # 한 시, 두 시
        "개": True,      # 한 개, 두 개
        "명": True,      # 한 명, 두 명
        "살": True,      # 한 살, 두 살
        "잔": True,      # 한 잔, 두 잔
        "병": True,      # 한 병, 두 병
        "마리": True,    # 한 마리
        "권": True,      # 한 권
        "장": True,      # 한 장
        "벌": True,      # 한 벌
        "대": True,      # 한 대 (차)
        "채": True,      # 한 채 (집)
        "그루": True,    # 한 그루
        "송이": True,    # 한 송이
        "켤레": True,    # 한 켤레
        
        # 한자어 사용 단위
        "분": False,     # 일 분, 이 분
        "초": False,     # 일 초, 이 초
        "원": False,     # 일 원, 만 원
        "번": False,     # 일 번, 이 번
        "호": False,     # 일 호, 이 호
        "층": False,     # 일 층, 이 층
        "년": False,     # 이천이십오 년
        "월": False,     # 일 월, 이 월
        "일": False,     # 일 일, 이 일 (날짜)
        "번째": False,   # 첫 번째는 예외
        "퍼센트": False, # 십 퍼센트
        "프로": False,   # 십 프로
        "도": False,     # 삼십 도 (온도)
        "킬로": False,   # 백 킬로
        "미터": False,   # 백 미터
        "그램": False,   # 백 그램
        "리터": False,   # 일 리터
    }
    
    # ========================================
    # - 외래어/영어 발음 사전
    # ========================================
    
    FOREIGN_WORDS: Dict[str, str] = {
        # IT/인터넷
        "ai": "에이아이",
        "gpt": "지피티",
        "chatgpt": "챗지피티",
        "api": "에이피아이",
        "url": "유알엘",
        "html": "에이치티엠엘",
        "css": "씨에스에스",
        "ui": "유아이",
        "ux": "유엑스",
        "app": "앱",
        "blog": "블로그",
        "email": "이메일",
        "wifi": "와이파이",
        "usb": "유에스비",
        "pc": "피씨",
        "tv": "티비",
        "dj": "디제이",
        "vr": "브이알",
        "ar": "에이알",
        "sns": "에스엔에스",
        "iot": "아이오티",
        
        # 회사/서비스
        "google": "구글",
        "youtube": "유튜브",
        "facebook": "페이스북",
        "instagram": "인스타그램",
        "twitter": "트위터",
        "naver": "네이버",
        "kakao": "카카오",
        "samsung": "삼성",
        "apple": "애플",
        "microsoft": "마이크로소프트",
        "amazon": "아마존",
        "netflix": "넷플릭스",
        "spotify": "스포티파이",
        
        # 일반 외래어
        "ok": "오케이",
        "yes": "예스",
        "no": "노",
        "hello": "헬로",
        "hi": "하이",
        "bye": "바이",
        "thank": "땡크",
        "sorry": "쏘리",
        "please": "플리즈",
    }
    
    # ========================================
    # - 초기화 및 메인 함수
    # ========================================
    
    def __init__(self):
        """텍스트 정규화기 초기화"""
        logger.info("✅ TextNormalizer 초기화 완료")
    
    def normalize(self, text: str) -> str:
        """
        텍스트 종합 정규화
        
        1. 영어 → 한글 발음
        2. 특수문자 정리
        3. 공백 정리
        
        Args:
            text: 정규화할 텍스트
        
        Returns:
            정규화된 텍스트
        """
        if not text:
            return text
        
        result = text
        
        # 1. 영어 → 한글 발음 변환
        for eng, kor in self.FOREIGN_WORDS.items():
            # 대소문자 무시하고 변환
            pattern = re.compile(rf'\b{eng}\b', re.IGNORECASE)
            result = pattern.sub(kor, result)
        
        # 2. 특수문자 정리 (한글, 숫자, 기본 부호만 유지)
        result = re.sub(r'[^\w\s가-힣0-9?!.,;:\-]', '', result)
        
        # 3. 다중 공백 정리
        result = re.sub(r'\s+', ' ', result)
        
        return result.strip()
    
    # ========================================
    # - 숫자 → 한글 변환
    # ========================================
    
    def numbers_to_korean(self, text: str) -> str:
        """
        숫자를 한글로 변환 (TTS용)
        
        단위에 따라 고유어/한자어 자동 선택
        
        Args:
            text: 변환할 텍스트
        
        Returns:
            숫자가 한글로 변환된 텍스트
        
        Examples:
            >>> normalizer.numbers_to_korean("3시 30분")
            '세 시 삼십 분'
            >>> normalizer.numbers_to_korean("5개")
            '다섯 개'
            >>> normalizer.numbers_to_korean("2023년 1월 15일")
            '이천이십삼 년 일 월 십오 일'
        """
        result = text
        
        # 1. 단위가 있는 숫자 처리
        # 패턴: 숫자 + 단위 (예: 3시, 30분, 5개)
        for unit, use_native in self.UNIT_READING_STYLE.items():
            pattern = rf'(\d+)\s*{unit}'
            matches = re.finditer(pattern, result)
            
            for match in matches:
                number = match.group(1)
                korean_number = self._convert_number(number, use_native)
                result = result.replace(
                    match.group(0),
                    f"{korean_number} {unit}"
                )
        
        # 2. 단위 없는 독립 숫자 처리 (기본: 한자어)
        result = re.sub(
            r'\b(\d+)\b',
            lambda m: self._convert_number(m.group(1), False),
            result
        )
        
        return result
    
    def _convert_number(self, number_str: str, use_native: bool = False) -> str:
        """
        숫자 문자열을 한글로 변환
        
        Args:
            number_str: 숫자 문자열 (예: "123")
            use_native: True면 고유어, False면 한자어
        
        Returns:
            한글 숫자 (예: "백이십삼" 또는 "한두세")
        """
        try:
            number = int(number_str)
        except ValueError:
            return number_str
        
        # 0 처리
        if number == 0:
            return "영"
        
        # 1-10 범위: 사전 직접 참조
        if 1 <= number <= 10:
            if use_native:
                return self.NATIVE_KOREAN_DIGITS.get(number_str, number_str)
            else:
                return self.SINO_KOREAN_DIGITS.get(number_str, number_str)
        
        # 11 이상: 단위별 분해
        if use_native and number <= 99:
            return self._convert_two_digit_native(number)
        else:
            return self._convert_sino_korean(number)
    
    def _convert_two_digit_native(self, number: int) -> str:
        """
        두 자리 고유어 숫자 변환 (11-99)
        
        예: 11 → 열한, 23 → 스물세
        """
        tens_map = {
            10: "열", 20: "스물", 30: "서른", 40: "마흔",
            50: "쉰", 60: "예순", 70: "일흔", 80: "여든", 90: "아흔"
        }
        
        ones_map = {
            1: "한", 2: "두", 3: "세", 4: "네",
            5: "다섯", 6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉"
        }
        
        tens = (number // 10) * 10
        ones = number % 10
        
        result = tens_map.get(tens, "")
        if ones > 0:
            result += ones_map.get(ones, "")
        
        return result
    
    def _convert_sino_korean(self, number: int) -> str:
        """
        한자어 숫자 변환 (모든 범위)
        
        예: 123 → 백이십삼, 4567 → 사천오백육십칠
        """
        if number == 0:
            return "영"
        
        result = ""
        
        # 큰 단위부터 처리
        for unit_value, unit_name in self.LARGE_UNITS.items():
            if number >= unit_value:
                unit_count = number // unit_value
                number = number % unit_value
                
                # 1인 경우 생략 (일백 → 백, 일천 → 천)
                # 단, 만/억은 "일만", "일억"으로 읽음
                if unit_count == 1 and unit_value < 10000:
                    result += unit_name
                else:
                    if unit_count >= 10:
                        result += self._convert_sino_korean(unit_count)
                    else:
                        result += self.SINO_KOREAN_DIGITS.get(str(unit_count), "")
                    result += unit_name
        
        # 나머지 1의 자리
        if number > 0:
            result += self.SINO_KOREAN_DIGITS.get(str(number), "")
        
        return result
    
    # ========================================
    # - 시간 표현 변환
    # ========================================
    
    def time_to_korean(self, text: str) -> str:
        """
        시간 표현을 자연스러운 한글로 변환
        
        시: 고유어 (한 시, 두 시)
        분: 한자어 (일 분, 이 분)
        
        Examples:
            >>> normalizer.time_to_korean("3:30")
            '세 시 삼십 분'
            >>> normalizer.time_to_korean("12시 45분")
            '열두 시 사십오 분'
        """
        result = text
        
        # HH:MM 형식
        pattern = r'(\d{1,2}):(\d{2})'
        matches = re.finditer(pattern, result)
        
        for match in matches:
            hour = int(match.group(1))
            minute = int(match.group(2))
            
            hour_korean = self._convert_number(str(hour), use_native=True)
            minute_korean = self._convert_number(str(minute), use_native=False)
            
            time_korean = f"{hour_korean} 시"
            if minute > 0:
                time_korean += f" {minute_korean} 분"
            
            result = result.replace(match.group(0), time_korean)
        
        return result
    
    # ========================================
    # - 금액 표현 변환
    # ========================================
    
    def money_to_korean(self, text: str) -> str:
        """
        금액 표현을 자연스러운 한글로 변환
        
        Examples:
            >>> normalizer.money_to_korean("50000원")
            '오만 원'
            >>> normalizer.money_to_korean("1,234,567원")
            '백이십삼만 사천오백육십칠 원'
        """
        result = text
        
        # 콤마 제거
        result = result.replace(",", "")
        
        # 숫자+원 패턴
        pattern = r'(\d+)\s*원'
        matches = re.finditer(pattern, result)
        
        for match in matches:
            number = int(match.group(1))
            korean = self._convert_sino_korean(number)
            result = result.replace(match.group(0), f"{korean} 원")
        
        return result


# ================================================================================
# 통합 테스트
# ================================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Korean Spell Checker 테스트")
    print("="*70)
    
    # 맞춤법 교정기 테스트
    checker = KoreanSpellChecker()
    
    spelling_tests = [
        # (입력, 예상 출력)
        ("이거 할께요", "이거 할게요"),
        ("됬어요", "됐어요"),
        ("안해요", "안 해요"),
        ("뭐 먹을 거에요?", "뭐 먹을 거예요?"),
        ("어떻해요", "어떡해요"),
        ("괜찬아요", "괜찮아요"),
        ("깨끗히 청소했어요", "깨끗이 청소했어요"),
        ("몇일 걸려요?", "며칠 걸려요?"),
        ("써비스가 좋아요", "서비스가 좋아요"),
        ("할수있어요", "할 수 있어요"),
    ]
    
    print("\n맞춤법 교정 테스트\n")
    passed = 0
    for input_text, expected in spelling_tests:
        result = checker.check(input_text)
        status = "O" if result == expected else "X"
        if result == expected:
            passed += 1
        print(f"{status} '{input_text}' → '{result}'")
        if result != expected:
            print(f"   예상: '{expected}'")
    
    print(f"\n결과: {passed}/{len(spelling_tests)} 통과\n")
    
    # 텍스트 정규화기 테스트
    normalizer = TextNormalizer()
    
    print("-"*70)
    print("\n숫자 변환 테스트\n")
    
    number_tests = [
        ("3시 30분", "세 시 삼십 분"),
        ("5개", "다섯 개"),
        ("100원", "백 원"),
        ("2023년", "이천이십삼 년"),
    ]
    
    for input_text, expected in number_tests:
        result = normalizer.numbers_to_korean(input_text)
        status = "O" if result == expected else "X"
        print(f"{status} '{input_text}' → '{result}'")
        if result != expected:
            print(f"   예상: '{expected}'")
    
    print("-"*70)
    print("\n외래어 변환 테스트\n")
    
    foreign_tests = [
        ("ChatGPT 사용해봤어요", "챗지피티 사용해봤어요"),
        ("YouTube 영상", "유튜브 영상"),
        ("wifi 비밀번호", "와이파이 비밀번호"),
    ]
    
    for input_text, expected in foreign_tests:
        result = normalizer.normalize(input_text)
        status = "O" if expected in result else "X"
        print(f"{status} '{input_text}' → '{result}'")
    
    print("\n" + "="*70)
    print("테스트 완료!")
    print("="*70 + "\n")