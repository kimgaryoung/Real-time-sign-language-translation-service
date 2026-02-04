"""
TTS 공통 설정
"""

import os
from pathlib import Path

# 출력 디렉토리 설정
OUTPUT_DIR = Path(__file__).parent.parent.parent / "tts_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 테스트 문장
TEST_TEXT = "안녕하세요 김가령입니다. 제가 오늘 말씀 드릴 주제는 TTS 입니다."

# API 키 설정 (환경변수에서 가져오기)
NAVER_CLIENT_ID = os.environ.get('NAVER_CLIENT_ID', 'YOUR_CLIENT_ID')
NAVER_CLIENT_SECRET = os.environ.get('NAVER_CLIENT_SECRET', 'YOUR_CLIENT_SECRET')
KAKAO_API_KEY = os.environ.get('KAKAO_API_KEY', 'YOUR_API_KEY')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY', 'YOUR_API_KEY')
