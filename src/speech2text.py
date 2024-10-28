import io
import os

from google.cloud import speech_v1p1beta1 as speech

from custom_logging import logger

def query_speech2text_api(audio_file_path):
    # Google Cloud Speech-to-Text APIのクライアントを作成
    client = speech.SpeechClient()

    # 音声ファイルを読み込む
    with io.open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    # 音声ファイルのエンコーディングとサンプルレートを指定
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ja-JP",  # 日本語の言語コードを指定
    )

    # 音声認識のリクエストを送信
    response = client.recognize(config=config, audio=audio)

    # 認識結果を取得
    for result in response.results:
        logger.info("Speech-to-Text API: {}".format(result.alternatives[0].transcript))
    
    return result.alternatives[0].transcript


if __name__ == '__main__':
    # 音声認識を実行
    query_speech2text_api("./wav_files/recording.wav")