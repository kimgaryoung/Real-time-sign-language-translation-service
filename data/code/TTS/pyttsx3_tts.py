"""
pyttsx3 + edge-tts TTS

설치: pip install pyttsx3 edge-tts
특징:
    - pyttsx3: 오프라인, 시스템 음성 사용 (한국어 제한적)
    - edge-tts: 온라인, 남성/여성 음성 지원 (추천)

실행 방법 (단독 실행):
    python /Users/garyeong/Desktop/Real-time-sign-language-translation-service/data/code/TTS/pyttsx3_tts.py

실행 방법 (모듈로 실행):
    cd /Users/garyeong/Desktop/Real-time-sign-language-translation-service/data/code
    python -m TTS.pyttsx3_tts
"""

import pyttsx3
import edge_tts
import asyncio
import subprocess
import tempfile
import os
from pathlib import Path

# 단독 실행 시 설정
try:
    from .config import OUTPUT_DIR, TEST_TEXT
except ImportError:
    OUTPUT_DIR = Path(__file__).parent.parent.parent / "tts_output"
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEST_TEXT = "안녕하세요 김가령입니다. 제가 오늘 말씀 드릴 주제는 TTS 입니다."

# 음성 설정
VOICE_FEMALE = "ko-KR-SunHiNeural"  # 여성
VOICE_MALE = "ko-KR-InJoonNeural"    # 남성


def tts_pyttsx3(text: str, output_file: str = "pyttsx3_output.mp3", rate: int = 150) -> str:
    """pyttsx3를 사용한 오프라인 TTS (파일 저장)"""
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)

    voices = engine.getProperty('voices')
    for voice in voices:
        if 'korean' in voice.name.lower() or 'ko' in voice.id.lower():
            engine.setProperty('voice', voice.id)
            break

    output_path = OUTPUT_DIR / output_file
    engine.save_to_file(text, str(output_path))
    engine.runAndWait()
    print(f"[pyttsx3] 저장 완료: {output_path}")
    return str(output_path)


def tts_pyttsx3_play(text: str, rate: int = 150) -> None:
    """pyttsx3 실시간 재생 (오프라인)"""
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    print(f"[pyttsx3] 재생 중: {text}")
    engine.say(text)
    engine.runAndWait()
    print("[pyttsx3] 재생 완료")


async def _play_async(text: str, voice: str):
    """edge-tts 실시간 재생 비동기 함수"""
    communicate = edge_tts.Communicate(text, voice)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    await communicate.save(tmp_path)

    try:
        subprocess.run(["afplay", tmp_path], check=True)
    except FileNotFoundError:
        try:
            subprocess.run(["mpv", "--no-video", tmp_path], check=True)
        except FileNotFoundError:
            subprocess.run(["ffplay", "-nodisp", "-autoexit", tmp_path], check=True)

    os.unlink(tmp_path)


def tts_play_female(text: str):
    """여성 음성으로 실시간 재생"""
    print(f"[여성] 재생 중: {text}")
    asyncio.run(_play_async(text, VOICE_FEMALE))
    print("[여성] 재생 완료")


def tts_play_male(text: str):
    """남성 음성으로 실시간 재생"""
    print(f"[남성] 재생 중: {text}")
    asyncio.run(_play_async(text, VOICE_MALE))
    print("[남성] 재생 완료")


def tts_play(text: str, gender: str = "female"):
    """
    실시간 음성 재생

    Args:
        text: 재생할 텍스트
        gender: "female" (여성) 또는 "male" (남성)
    """
    if gender.lower() in ["male", "m", "남", "남성", "남자"]:
        tts_play_male(text)
    else:
        tts_play_female(text)


if __name__ == "__main__":
    print("=" * 50)
    print("TTS 테스트 - 남성/여성 음성")
    print("=" * 50)

    # 여성 음성 테스트
    print("\n[1] 여성 음성 테스트:")
    tts_play_female(TEST_TEXT)

    # 남성 음성 테스트
    print("\n[2] 남성 음성 테스트:")
    tts_play_male(TEST_TEXT)
