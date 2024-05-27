import torch

from models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)


# モデルの定義部分を修正する
class SynthesizerTrnONNX(SynthesizerTrnJPExtra):
    def forward(self, x, x_lengths, sid, tone, language, bert, style_vec):
        return super().forward(x, x_lengths, None, None, sid, tone, language, bert, style_vec)

model = SynthesizerTrnONNX(112,1025,32,1,True,True,False,False,True,192,192,768,2,6,3,0.1,1,[3, 7, 11],[[1, 3, 5], [1, 3, 5], [1, 3, 5]],[8, 8, 2, 2, 2],512,[16, 16, 8, 2, 2],3,False,512,model='./slm/wavlm-base-plus',sr=16000,hidden=768,nlayers=13,initial_channel=64)
model.eval()

# モデルの入力サンプルを準備する
x = torch.randn(1, 100, 256)
x_lengths = torch.tensor([100])
sid = torch.tensor([0])
tone = torch.tensor([0])
language = torch.tensor([0])
bert = torch.randn(1, 100, 256)
style_vec = torch.randn(1, 256)

# torch.onnx.exportを使用してモデルをONNXにエクスポートする
torch.onnx.export(
    model,
    (x, x_lengths, sid, tone, language, bert, style_vec),
    "output_model.onnx",
    input_names=["x", "x_lengths", "sid", "tone", "language", "bert", "style_vec"],
    output_names=["output", "attn", "mask", "z_info"],
    opset_version=11,
    dynamic_axes={
        "x": {0: "batch_size", 1: "seq_len"},
        "x_lengths": {0: "batch_size"},
        "output": {0: "batch_size", 2: "out_seq_len"},
    },
)