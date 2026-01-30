"""
Kakao TTS

API 키 필요: https://developers.kakao.com/
특징: 한국어 지원, REST API

실행 방법:
    # API 키 설정
    export KAKAO_API_KEY="your_api_key"

    cd /Users/garyeong/Desktop/Real-time-sign-language-translation-service/data/code
    python -m TTS.kakao_tts
"""

import urllib.request
from .config import OUTPUT_DIR, TEST_TEXT, KAKAO_API_KEY


def tts_kakao(text: str, output_file: str = "kakao_output.mp3") -> str | None:
    """
    Kakao TTS를 사용한 TTS

    Args:
        text: 변환할 텍스트
        output_file: 출력 파일명

    Returns:
        저장된 파일 경로 또는 None (실패 시)
    """
    if KAKAO_API_KEY == 'YOUR_API_KEY':
        print("[Kakao] API 키가 설정되지 않았습니다.")
        print("  KAKAO_API_KEY 환경변수를 설정해주세요.")
        return None

    url = "https://kakaoi-newtone-openapi.kakao.com/v1/synthesize"
    headers = {
        "Content-Type": "application/xml",
        "Authorization": f"KakaoAK {KAKAO_API_KEY}"
    }

    data = f'<speak>{text}</speak>'

    try:
        request = urllib.request.Request(url, data=data.encode('utf-8'), headers=headers)
        response = urllib.request.urlopen(request)

        if response.getcode() == 200:
            output_path = OUTPUT_DIR / output_file
            with open(output_path, 'wb') as f:
                f.write(response.read())
            print(f"[Kakao] 저장 완료: {output_path}")
            return str(output_path)
        else:
            print(f"[Kakao] API 오류: {response.getcode()}")
            return None
    except Exception as e:
        print(f"[Kakao] 오류: {e}")
        return None


if __name__ == "__main__":
    # 테스트
    result = tts_kakao(TEST_TEXT)
    if result:
        print(f"생성된 파일: {result}")
