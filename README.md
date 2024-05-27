# anybody_speaker
音声合成を兼ね備えたスマートスピーカー。
学習すれば誰の声でもそっくりに、LLMを用いて会話できます。


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


