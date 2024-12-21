import queue
import json
import time
import sys

import sounddevice as sd
import soundfile as sf
import vosk
import numpy as np
import MeCab
import scipy.io.wavfile as wavfile

from custom_logging import logger


# マイクの設定
channels = 1  # モノラル録音
sample_rate = 16000  # サンプリングレート（Voskの推奨値）
blocksize = 1024  # バッファサイズ（サンプル数）
vosk_blocksize = 1024 * 8 # Voskのバッファサイズ（サンプル数）

# 入力の閾値
silence_threshold = 0.1  # 無音の閾値（適宜調整）
silence_duration = 2.0  # 無音と判断する連続秒数
silence_timeout = 10 # タイムアウトまでの連続秒数

# トリガーワード #TODO: 文字列で指定する必要あり
trigger_word = ["こんにちは"]

# 音声認識の設定
model = vosk.Model("./vosk_model")
recognizer = vosk.KaldiRecognizer(model, sample_rate)

# マイクからの入力を処理するためのキュー
audio_queue = queue.Queue()


def convert_to_katakana(text):
    tagger = MeCab.Tagger("-Oyomi -d 'C:\Program Files\MeCab\dic\ipadic'")  # ヨミ（読み）出力モードを指定
    result = tagger.parse(text)
    katakana_text = result.rstrip()
    katakana_text = katakana_text.replace(' ', '')
    return katakana_text


def callback(indata, frames, time, status):
    """マイクからのオーディオデータをキューに追加するコールバック関数"""
    if status:
        print(status)
    audio_queue.put(bytes(indata))


def wait_for_trigger(input_device):
    """特定の用語が音声として入力されるまで待機する関数"""
    logger.info("wait for trigger word(s)...")
    logger.info(sd.query_devices())
    logger.info(input_device)
    
    if len(trigger_word) == 0:
        logger.error("trigger_word does not exist.")
        sys.exit(1)

    with sd.InputStream(samplerate=sample_rate, blocksize=vosk_blocksize, device=input_device, dtype='int16', channels=channels, callback=callback):
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result['text']
                logger.info(text)
                text_katakana = convert_to_katakana(text)  # 認識結果をカタカナに変換
                if text_katakana == '':
                    continue
                logger.info(f"voice recognized: {text_katakana}")  # カタカナでの認識結果をログ出力
                if any(word in text_katakana for word in trigger_word):
                    logger.info(f"detected '{trigger_word}'")
                    break
                

def play_until_the_end(wav_path: str, device_id: int):
    # WAVファイルを読み込む
    data, samplerate = sf.read(wav_path)
    
    # 指定されたデバイスで音声を再生
    sd.play(data, samplerate=samplerate, device=device_id)
    sd.wait()  # 再生終了まで待機
    

def store_wav(recording, output_path):
    wavfile.write(output_path, sample_rate, (recording * 32767).astype(np.int16))
    logger.info(f"end of recording. file name: {output_path}")
    

# 録音を開始する関数
def start_recording(input_device: int, output_device: int):
    logger.info("start recording...")
    recording = []
    silence_start_time = time.time()
    is_recording = False

    # 録音開始の効果音を鳴らす
    play_until_the_end("./wav_files/start_sound.wav", device_id=output_device)
    
    # ストリームを開始
    with sd.InputStream(samplerate=sample_rate, channels=channels, blocksize=blocksize, dtype=np.float32, device=input_device) as stream:
        while True:   
            # ストリームからデータを読み込む
            data, overflowed = stream.read(blocksize)

            if overflowed:
                logger.info("overflow occurs")

            # データの最大振幅を計算
            max_amplitude = np.max(np.abs(data))
            
            # 最大振幅が無音の閾値を下回った場合、無音カウンターを増やす
            if max_amplitude < silence_threshold:
                # 入力がされず、無音の時間が閾値以上であれば処理を中止
                if not is_recording:
                    if silence_start_time:
                        silence_time = time.time() - silence_start_time
                        if silence_time > silence_timeout:
                            return False
                    continue
                
                # 無音であるが、録音に音声の適切な間を含める
                recording.append(data.copy())
                
                # 時間の計測を終了し開始との差分を取得
                if silence_start_time is None:
                    silence_start_time = time.time()
                    continue
                    
                # 差分の合計が指定の秒数を超えた場合、録音を停止
                silence_time = time.time() - silence_start_time
                if silence_time > silence_duration:
                    # 録音終了の効果音を鳴らす
                    play_until_the_end("./wav_files/end_sound.wav", device_id=output_device)
                    break
                
            else:
                recording.append(data.copy())
                if not is_recording:
                    is_recording = True
                    
                # 無音時間測定をリセット
                silence_start_time = time.time()
    
    # 録音データを結合
    recording = np.concatenate(recording, axis=0)

    # 録音データを保存
    store_wav(recording, output_path="./wav_files/recording.wav")

    return True