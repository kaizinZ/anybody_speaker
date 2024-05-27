import argparse
import os
from pathlib import Path
import sys
import cProfile

import torch
import simpleaudio as sa
import numpy as np
from scipy.io import wavfile

from constants import (
    DEFAULT_LENGTH,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    Languages,
)
from custom_logging import logger, log_memory_usage
from nlp import bert_models
from tts_model import TTSModel, TTSModelHolder
from llm import Claude, ChatGPT
from input_recording import wait_for_trigger, start_recording, play_until_the_end
from speech2text import query_speech2text_api


# 事前に BERT モデル/トークナイザーをロードしておく
bert_models.load_model(Languages.JP)
bert_models.load_tokenizer(Languages.JP)
#bert_models.load_model(Languages.EN)
#bert_models.load_tokenizer(Languages.EN)
#bert_models.load_model(Languages.ZH)
#bert_models.load_tokenizer(Languages.ZH)

# model_assets ディレクトリのパス
model_assets_dir = "./model_assets/"

# model_assets ディレクトリ内のディレクトリ名を取得
model_assets_names = [
    name for name in os.listdir(model_assets_dir)
    if os.path.isdir(os.path.join(model_assets_dir, name))
]
logger.info(f'available models: {model_assets_names}')

default_model = "jun_ver_0"

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU", default=True)
parser.add_argument(
    "--dir", "-d", type=str, help="Model directory", default=model_assets_dir
)
parser.add_argument(
    "--llm", "-l", type=str, help="llm assignment", default="claude", choices=["chatgpt", "claude"]
)
parser.add_argument(
    "--tts", "-t", type=str, help="tts model assignment", default=default_model, choices=model_assets_names
)

parser.add_argument(
    "--is_multi", "-m", help="use multiple speakers", action="store_true"
)

prefix_message = "あなたは加藤純一.タメ口で話す必要あり,敬語は禁止.返答は簡潔に短くまとめる必要あり.英単語は日本語の読み方に直すこと."
init_voice_path = "./wav_files/yaa.wav"
    
    
@log_memory_usage
def load_models(model_holder: TTSModelHolder):
    logger.info("Loading models...")
    loaded_models = []
    for model_name, model_paths in model_holder.model_files_dict.items():
        model = TTSModel(
            model_path=model_paths[0],
            config_path=model_holder.root_dir / model_name / "config.json",
            style_vec_path=model_holder.root_dir / model_name / "style_vectors.npy",
            device=model_holder.device,
        )
        loaded_models.append(model)
    logger.info([(i, m.get_model_name()) for (i, m) in enumerate(loaded_models)])
    return loaded_models


@log_memory_usage
def load_single_model(model: TTSModel):
    logger.info("Loading a model...")
    model_info = model.get_model_paths()
    model = TTSModel(
        model_path=model_info["model_path"],
        config_path=model_info["config_path"],
        style_vec_path=model_info["style_vec_path"],
        device=model_info["device"],
    )
    logger.info(f"model {model.get_model_name()} was loaded")
    return model


@log_memory_usage
def generate_audio_single_speaker(text, loaded_model, language=Languages.JP, **kwargs):
    """文章から単一の話者による音声を生成する関数

    Args:
        text (_type_): 音声合成するテキスト
        loaded_models (_type_): 音声モデル, 複数候補がある場合はmodel_idによって選択
        model_id (int, optional): Defaults to 0.
        language (_type_, optional): _description_. Defaults to Languages.JP.
        is_selected (bool, optional): 読み込んだモデルが1つの場合true, 複数の場合かモデルがディレクトリに1つしかない場合はfalse

    Returns:
        _type_: _description_
    """
    sr, audio = loaded_model.infer(text=text, language=language, **kwargs)
    return sr, audio


#TODO: 複数の場合にどの順番で発話するかのアルゴリズムを考える
@log_memory_usage
def generate_audio_multiple_speakers(text, loaded_models, model_id=0, language=Languages.JP, **kwargs):
    """文章から複数の話者による音声を生成する関数

    Args:
        text (_type_): 音声合成するテキスト
        loaded_models (_type_): 音声モデル, 複数候補がある場合はmodel_idによって選択
        model_id (int, optional): Defaults to 0.
        language (_type_, optional): _description_. Defaults to Languages.JP.
        is_selected (bool, optional): 読み込んだモデルが1つの場合true, 複数の場合かモデルがディレクトリに1つしかない場合はfalse

    Returns:
        _type_: _description_
    """
    model = loaded_models[model_id]
    sr, audio = model.infer(text=text, language=language, **kwargs)
    return sr, audio


def play_audio(sr: int, audio: np.ndarray):
    # オーディオデータをバイト列に変換する
    audio_bytes = audio.tobytes()
    audio_obj = sa.play_buffer(audio_bytes, num_channels=1, bytes_per_sample=2, sample_rate=sr)
    audio_obj.wait_done()


def main():
    args = parser.parse_args()

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # LLMのAPIを利用するためのインスタンス生成
    if args.llm == "chatgpt":
        llm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"), messages=None, stream=False)
    elif args.llm == "claude":
        llm = Claude(prefix_message=prefix_message, stream=False)

    # モデル読み込み
    model_dir = Path(args.dir)
    model_holder = TTSModelHolder(model_dir, device)
    if len(model_holder.model_names) == 0:
        logger.error(f"Models not found in {model_dir}.")
        sys.exit(1)
    
    if args.is_multi:
        loaded_models = load_models(model_holder)
        #models_info = [{i: m} for (i, m) in enumerate(model_holder.model_files_dict.keys())]
        #logger.info(models_info)
        model_id = int(input("choose model: "))
    else:
        model_dir_path = args.dir + args.tts + '/'
        safetensors_files = [
            file for file in os.listdir(model_dir_path)
            if file.endswith(".safetensors")
        ]
        model_path = model_dir_path + safetensors_files[0]
        loaded_model = load_single_model(model_holder.get_model(args.tts, model_path))
        
        # 起動音声 (モデル読み込みが単一の場合のみ)
        text = "やあ。"
        sr, audio = generate_audio_single_speaker(text, loaded_model)
        #wavfile.write(init_voice_path, sr, audio)
        play_until_the_end(init_voice_path)
    
    # ユーザーからの音声入力待機
    while True:
        wait_for_trigger()
        start_recording()
        
        # 音声をspeech-to-text APIで文章に変換
        prompt = query_speech2text_api("./wav_files/recording.wav")
        
        # 文章をLLM APIに投げる
        llm.chat(prompt)
        
        # LLMが生成した文章を取得
        ans_text = llm.get_messages()[-1]['content']

        # 取得した文章で音声合成
        if args.is_multi:
            sr, audio = generate_audio_multiple_speakers(ans_text, loaded_model, model_id=model_id) #, speaker_id=0)
        else:
            sr, audio = generate_audio_single_speaker(ans_text, loaded_model)
        play_audio(sr, audio)


    # 使用例
    text = "やっぱ若いっていいなあでも、なんかそんなあれがまかり通るんだもんな、あいつ25くらいだろ？30なんだ。あそっか、止まるんだよ。まあ極論言っちゃえばもう本人の。"
    sr, audio = generate_audio(text, loaded_models, model_id=1) #, speaker_id=0)

    play_audio(sr, audio)
    
    # 音声をファイルに保存する場合
    wavfile.write("output.wav", sr, audio)
    
    
if __name__ == "__main__":
    main()
    cProfile.run('main()')