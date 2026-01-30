"""
TTS (Text-to-Speech) 모듈

지원하는 TTS 방법:
1. gTTS (Google Text-to-Speech) - 무료, 온라인 필요
2. pyttsx3 - 오프라인, 시스템 TTS 엔진 사용
3. edge-tts (Microsoft Edge TTS) - 무료, 고품질 한국어
4. Naver Clova TTS - API 키 필요, 고품질 한국어
5. Kakao TTS - API 키 필요
6. ElevenLabs TTS - API 키 필요, 초고품질 AI 음성
"""

from .gtts_tts import tts_gtts
from .pyttsx3_tts import tts_pyttsx3, tts_pyttsx3_play
from .edge_tts_module import tts_edge, list_korean_voices, tts_edge_play
from .naver_clova_tts import tts_naver_clova
from .kakao_tts import tts_kakao
from .elevenlabs_tts import tts_elevenlabs, list_elevenlabs_voices

__all__ = [
    'tts_gtts',
    'tts_pyttsx3',
    'tts_pyttsx3_play',
    'tts_edge',
    'list_korean_voices',
    'tts_naver_clova',
    'tts_kakao',
    'tts_elevenlabs',
    'list_elevenlabs_voices',
]
