# anybody_speaker
音声合成を兼ね備えたスマートスピーカー。
学習すれば誰の声でもそっくりに、LLMを用いて会話できます。

# Version
python3.9.17

## 動かし方
```
pip install -r requirements.txt
cd src
python3 initialize.py
python3 main.py
```

## 環境構築について
simpleaudioのインストールにした場合は以下を試してみてください。
```
pip install simpleaudio --use-pep517
```

```initialize.py```では日本語以外のモデルもダウンロードされる可能性があります。
不要な場合は引数で指定するか、モデルを削除してください。


ラズパイ環境で動かすには以下のインストールが必要でした。
```
sudo apt install python3-dev build-essential default-libmysqlclient-dev
sudo apt install libncursesw5-dev libgdbm-dev libc6-dev libctypes-ocaml-dev zlib1g-dev libsqlite3-dev tk-dev 
sudo apt install libssl1.1 libssl1.1=1.1.1f-1ubuntu2 libssl-dev libmysqlclient-dev
sudo apt install librust-libsodium-sys-dev
sudo apt-get install libffi-dev
sudo apt-get install libasound2-dev
sudo apt-get install libportaudio2
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


