# anybody_speaker
音声合成を兼ね備えたスマートスピーカー。
学習すれば誰の声でもそっくりに、LLMを用いて会話できます。

# Version
python3.9.7

## 動かし方
```
pip install -r requirements.txt
cd src
python3 initialize.py
python3 main.py
```

```initialize.py```では日本語以外のモデルもダウンロードされる可能性があります。
不要な場合は引数で指定するか、モデルを削除してください。

gradioのインストールでhashに関するエラーが出たら以下を実行すると解決するかもしれません。
```
pip download gradio==4.23.0 -d .
shasum -a 256 gradio-4.23.0-py3-none-any.whl
```

```
pip download jieba==0.42.1 -d .
shasum -a 256 jieba-0.42.1-py3-none-any.whl
```

[issue](https://github.com/pytorch/pytorch/issues/104598)にあるように、比較的大きいサイズのパッケージは失敗する可能性があるようです。


## download vosk_model
オフラインで動作する音声認識モデルのダウンロードが必要です。

日本語のモデルは以下のリンクから取得可能です。

https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip

ダウンロード後、名前をvosk_modelに変更し、src下に配置する必要があります。


## コードの引用

音声合成にあたり、以下のリポジトリから学習用コードと推論用コードを引用しました。

https://github.com/litagin02/Style-Bert-VITS2?tab=readme-ov-file

なお、学習は[colab](https://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb)上で行なったため、本リポジトリには含まれていません。
学習後のモデルを含むディレクトリを```src/model_assets/```に配置すると利用できるようになります。


