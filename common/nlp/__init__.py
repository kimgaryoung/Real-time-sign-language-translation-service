"""
NLP (Natural Language Processing) 모듈

수어 인식 결과를 자연스러운 한국어 문장으로 변환하고 맞춤법을 교정합니다.

주요 기능:
- 수어 단어 리스트 → 완성된 한국어 문장 변환
- 조사 자동 부착 (은/는, 이/가, 을/를)
- 시제 처리 (과거/현재/미래)
- 높임말 변환
- 맞춤법 교정 및 띄어쓰기 수정

사용 예시:
    >>> from common.nlp import HyemiTextManager, KoreanSpellChecker
    >>> 
    >>> # 텍스트 변환
    >>> manager = HyemiTextManager()
    >>> sentence, _ = manager.process_text(['나', '밥', '먹다'])
    >>> print(sentence)
    '저는 밥을 먹습니다'
    >>> 
    >>> # 맞춤법 교정
    >>> checker = KoreanSpellChecker()
    >>> corrected = checker.check('이거 할께요')
    >>> print(corrected)
    '이거 할게요'
"""

from .text_manager import HyemiTextManager
from .spell_checker import KoreanSpellChecker, TextNormalizer

__version__ = '1.0.0'
__all__ = [
    'HyemiTextManager',
    'KoreanSpellChecker',
    'TextNormalizer',
]