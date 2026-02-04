"""
Naver Clova TTS

API 키 필요: https://www.ncloud.com/product/aiService/clovaSpeech
특징: 고품질 한국어, 다양한 음성 스타일

사용 가능한 음성:
- nara: 여성 (기본)
- njinho: 남성
- nminsang: 남성
- nsujin: 여성
- vyuna: 여성 (밝은 톤)
- vdain: 여성 (차분한 톤)

실행 방법:
    # API 키 설정
    export NAVER_CLIENT_ID="your_client_id"
    export NAVER_CLIENT_SECRET="your_client_secret"

    cd /Users/garyeong/Desktop/Real-time-sign-language-translation-service/data/code
    python -m TTS.naver_clova_tts
"""

import urllib.request
import urllib.parse
from .config import OUTPUT_DIR, TEST_TEXT, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET


def tts_naver_clova(text: str, output_file: str = "clova_output.mp3", speaker: str = "nara") -> str | None:
    """
    Naver Clova TTS를 사용한 TTS

    Args:
        text: 변환할 텍스트
        output_file: 출력 파일명
        speaker: 음성 선택 (기본값: nara)
            - nara: 여성 (기본)
            - njinho: 남성
            - nminsang: 남성
            - nsujin: 여성
            - vyuna: 여성 (밝은 톤)
            - vdain: 여성 (차분한 톤)

    Returns:
        저장된 파일 경로 또는 None (실패 시)
    """
    if NAVER_CLIENT_ID == 'YOUR_CLIENT_ID':
        print("[Naver Clova] API 키가 설정되지 않았습니다.")
        print("  NAVER_CLIENT_ID, NAVER_CLIENT_SECRET 환경변수를 설정해주세요.")
        return None

    encText = urllib.parse.quote(text)
    data = f"speaker={speaker}&volume=0&speed=0&pitch=0&format=mp3&text={encText}"

    url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
    request = urllib.request.Request(url)
    request.add_header("X-NCP-APIGW-API-KEY-ID", NAVER_CLIENT_ID)
    request.add_header("X-NCP-APIGW-API-KEY", NAVER_CLIENT_SECRET)

    try:
        response = urllib.request.urlopen(request, data=data.encode('utf-8'))

        if response.getcode() == 200:
            output_path = OUTPUT_DIR / output_file
            with open(output_path, 'wb') as f:
                f.write(response.read())
            print(f"[Naver Clova] 저장 완료: {output_path}")
            return str(output_path)
        else:
            print(f"[Naver Clova] API 오류: {response.getcode()}")
            return None
    except Exception as e:
        print(f"[Naver Clova] 오류: {e}")
        return None


if __name__ == "__main__":
    # 테스트
    result = tts_naver_clova(TEST_TEXT)
    if result:
        print(f"생성된 파일: {result}")
