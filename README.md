# anybody_speaker
音声合成を兼ね備えたスマートスピーカー。
学習すれば誰の声でもそっくりに、LLMを用いて会話できます。

注意事項: 

音声モデルは自身の声を学習データに推奨します。
本リポジトリの利用についてはご自身の責任でお願いいたします。

# Version
python3.9.13

## 動かし方

For macOS
```bash
# youtube動画から学習データを自動生成する場合
brew install ffmpeg
```

```bash
pip install -r requirements.txt
cd src
python3 initialize.py
python3 main.py
```

```initialize.py```では日本語以外のモデルもダウンロードされる可能性があります。
不要な場合は引数で指定するか、モデルを削除してください。

jiebaのインストールでhashに関するエラーが出たら以下を実行すると解決するかもしれません。

```bash
pip download jieba==0.42.1 -d .
shasum -a 256 jieba-0.42.1-py3-none-any.whl
```

[issue](https://github.com/pytorch/pytorch/issues/104598)にあるように、比較的大きいサイズのパッケージは失敗する可能性があるようです。


## download vosk_model
オフラインで動作する音声認識モデルのダウンロードが必要です。

日本語のモデルは以下のリンクから取得可能です。

https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip

ダウンロード後、ディレクトリ名をvosk_modelに変更し、```src```下に配置する必要があります。


## コードの引用

音声合成にあたり、以下のリポジトリから学習用コードと推論用コードを引用しました。

https://github.com/litagin02/Style-Bert-VITS2?tab=readme-ov-file

なお、学習は[colab](https://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb)上で行なったため、本リポジトリには含まれていません。
学習後のモデルを含むディレクトリを```src/model_assets/```に配置すると利用できるようになります。


こちらのリポジトリを引用するにあたり、本リポジトリの思想は[開発陣からのお願い](https://github.com/litagin02/Style-Bert-VITS2/blob/master/docs/TERMS_OF_USE.md)に準拠しております。本リポジトリにはデフォルトのモデルはございませんのでライセンスは存在しませんが、作成したモデルの利用につきましてはこちらのドキュメントに従ってください。

<!--
## 学習データの作成 (WIP)
学習に必要なデータは上記のリポジトリを参考に手動で用意するか、
以下の方法で生成したものを利用してください。

python your_script.py --model_name your_model_name --initial_prompt "こんにちは。元気、ですかー？ふふっ、私は……ちゃんと元気だよ！" --language ja --model large-v3 --device cpu --compute_type bfloat16 --use_hf_whisper --hf_repo_id your_repo_id --batch_size 16 --num_beams 1 --no_repeat_ngram_size 10
-->