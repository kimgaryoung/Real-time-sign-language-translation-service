"""
ElevenLabs TTS

설치: pip install elevenlabs
API 키 필요: https://elevenlabs.io/
특징: 초고품질 AI 음성, 다국어 지원, 음성 복제 가능

사용 가능한 기본 음성:
- Rachel: 여성, 차분한 톤
- Adam: 남성, 깊은 목소리
- Bella: 여성, 부드러운 톤
- Antoni: 남성, 따뜻한 톤
- Elli: 여성, 젊은 톤
- Josh: 남성, 젊은 톤

실행 방법:
    # API 키 설정
    export ELEVENLABS_API_KEY="your_api_key"

    cd /Users/garyeong/Desktop/Real-time-sign-language-translation-service/data/code
    python -m TTS.elevenlabs_tts
"""

from elevenlabs import ElevenLabs
from .config import OUTPUT_DIR, TEST_TEXT, ELEVENLABS_API_KEY


def tts_elevenlabs(text: str, output_file: str = "elevenlabs_output.mp3", voice: str = "Rachel") -> str | None:
    """
    ElevenLabs TTS를 사용한 고품질 TTS

    Args:
        text: 변환할 텍스트
        output_file: 출력 파일명
        voice: 음성 ID 또는 이름 (기본값: Rachel)

    Returns:
        저장된 파일 경로 또는 None (실패 시)
    """
    if ELEVENLABS_API_KEY == 'YOUR_API_KEY':
        print("[ElevenLabs] API 키가 설정되지 않았습니다.")
        print("  ELEVENLABS_API_KEY 환경변수를 설정해주세요.")
        print("  API 키 발급: https://elevenlabs.io/")
        return None

    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

        # 음성 생성
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=voice,
            model_id="eleven_multilingual_v2",  # 다국어 지원 모델 (한국어 포함)
            output_format="mp3_44100_128"
        )

        # 파일 저장
        output_path = OUTPUT_DIR / output_file
        with open(output_path, 'wb') as f:
            for chunk in audio:
                f.write(chunk)

        print(f"[ElevenLabs] 저장 완료: {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"[ElevenLabs] 오류: {e}")
        return None


def list_elevenlabs_voices() -> list | None:
    """
    사용 가능한 ElevenLabs 음성 목록 조회

    Returns:
        음성 목록 또는 None (실패 시)
    """
    if ELEVENLABS_API_KEY == 'YOUR_API_KEY':
        print("[ElevenLabs] API 키가 설정되지 않았습니다.")
        return None

    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        voices = client.voices.get_all()

        print("사용 가능한 ElevenLabs 음성:")
        for voice in voices.voices:
            print(f"  - {voice.name} (ID: {voice.voice_id})")
        return voices.voices
    except Exception as e:
        print(f"[ElevenLabs] 오류: {e}")
        return None


if __name__ == "__main__":
    # 음성 목록 확인
    list_elevenlabs_voices()

    # 테스트
    result = tts_elevenlabs(TEST_TEXT)
    if result:
        print(f"생성된 파일: {result}")
