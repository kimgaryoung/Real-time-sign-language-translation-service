"""
edge-tts (Microsoft Edge TTS)

설치: pip install edge-tts
특징: 무료, 고품질, 다양한 한국어 음성 지원

사용 가능한 한국어 음성:
- ko-KR-SunHiNeural (여성)
- ko-KR-InJoonNeural (남성)
- ko-KR-HyunsuNeural (남성)

실행 방법 (단독 실행):
    python /Users/garyeong/Desktop/Real-time-sign-language-translation-service/data/code/TTS/edge_tts_module.py

실행 방법 (모듈로 실행):
    cd /Users/garyeong/Desktop/Real-time-sign-language-translation-service/data/code
    python -m TTS.edge_tts_module
"""

import asyncio
import edge_tts
import subprocess
import tempfile
import os
from pathlib import Path

# 단독 실행 시 설정
try:
    from .config import OUTPUT_DIR, TEST_TEXT
except ImportError:
    # 단독 실행 시 직접 설정
    OUTPUT_DIR = Path(__file__).parent.parent.parent / "tts_output"
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEST_TEXT = "안녕하세요 김가령입니다. 제가 오늘 말씀 드릴 주제는 TTS 입니다."


async def _tts_edge_async(text: str, output_file: str, voice: str) -> str:
    """edge-tts 비동기 함수"""
    communicate = edge_tts.Communicate(text, voice)
    output_path = OUTPUT_DIR / output_file
    await communicate.save(str(output_path))
    return str(output_path)


def tts_edge(text: str, output_file: str = "edge_output.mp3", voice: str = "ko-KR-SunHiNeural") -> str:
    """
    Microsoft Edge TTS를 사용한 TTS

    Args:
        text: 변환할 텍스트
        output_file: 출력 파일명
        voice: 음성 선택 (기본값: ko-KR-SunHiNeural)
            - ko-KR-SunHiNeural: 여성
            - ko-KR-InJoonNeural: 남성
            - ko-KR-HyunsuNeural: 남성

    Returns:
        저장된 파일 경로
    """
    output_path = asyncio.run(_tts_edge_async(text, output_file, voice))
    print(f"[edge-tts] 저장 완료: {output_path}")
    return output_path


def list_korean_voices() -> list:
    """
    사용 가능한 한국어 음성 목록 출력

    Returns:
        한국어 음성 목록
    """
    korean_voices = [
        {"ShortName": "ko-KR-SunHiNeural", "Gender": "Female"},
        {"ShortName": "ko-KR-InJoonNeural", "Gender": "Male"},
        {"ShortName": "ko-KR-HyunsuNeural", "Gender": "Male"},
    ]
    print("사용 가능한 한국어 음성:")
    for v in korean_voices:
        print(f"  - {v['ShortName']}: {v['Gender']}")
    return korean_voices


async def _tts_edge_play_async(text: str, voice: str):
    """edge-tts 실시간 재생 비동기 함수"""
    communicate = edge_tts.Communicate(text, voice)

    # 임시 파일에 저장 후 재생
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    await communicate.save(tmp_path)

    # macOS: afplay 사용 (기본 내장)
    try:
        subprocess.run(["afplay", tmp_path], check=True)
    except FileNotFoundError:
        # Linux/Windows: mpv 또는 ffplay 시도
        try:
            subprocess.run(["mpv", "--no-video", tmp_path], check=True)
        except FileNotFoundError:
            try:
                subprocess.run(["ffplay", "-nodisp", "-autoexit", tmp_path], check=True)
            except FileNotFoundError:
                print("재생기를 찾을 수 없습니다. afplay, mpv, ffplay 중 하나를 설치해주세요.")

    # 임시 파일 삭제
    os.unlink(tmp_path)


def tts_edge_play(text: str, voice: str = "ko-KR-SunHiNeural"):
    """
    실시간 음성 재생 (파일 저장 없이)

    Args:
        text: 재생할 텍스트
        voice: 음성 선택 (기본값: ko-KR-SunHiNeural)
    """
    print(f"[edge-tts] 재생 중: {text}")
    asyncio.run(_tts_edge_play_async(text, voice))
    print("[edge-tts] 재생 완료")


if __name__ == "__main__":
    # 음성 목록 확인
    list_korean_voices()

    # 실시간 재생 테스트
    print("\n실시간 재생 테스트 (여성):")
    tts_edge_play(TEST_TEXT, "ko-KR-SunHiNeural")

    print("\n실시간 재생 테스트 (남성):")
    tts_edge_play(TEST_TEXT, "ko-KR-InJoonNeural")
