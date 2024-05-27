from typing import Any, Optional, Union, cast
import os
import sys

import torch
from numpy.typing import NDArray
import onnxruntime as rt

from constants import Languages
from custom_logging import logger
from models import commons, utils
from models.hyper_parameters import HyperParameters
from models.models import SynthesizerTrn
from models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)
from nlp import (
    clean_text,
    cleaned_text_to_sequence,
    extract_bert_feature,
)
from nlp.symbols import SYMBOLS

onnx_model_path = "../onnx_models/model.onnx"


def get_net_g(model_path: str, version: str, device: str, hps: HyperParameters):
    if version.endswith("JP-Extra"):
        logger.info("Using JP-Extra model")
        net_g = SynthesizerTrnJPExtra(
            n_vocab=len(SYMBOLS),
            spec_channels=hps.data.filter_length // 2 + 1,
            segment_size=hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            # hps.model 以下のすべての値を引数に渡す
            use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
            use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
            use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
            use_duration_discriminator=hps.model.use_duration_discriminator,
            use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
            inter_channels=hps.model.inter_channels,
            hidden_channels=hps.model.hidden_channels,
            filter_channels=hps.model.filter_channels,
            n_heads=hps.model.n_heads,
            n_layers=hps.model.n_layers,
            kernel_size=hps.model.kernel_size,
            p_dropout=hps.model.p_dropout,
            resblock=hps.model.resblock,
            resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
            resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
            upsample_rates=hps.model.upsample_rates,
            upsample_initial_channel=hps.model.upsample_initial_channel,
            upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
            n_layers_q=hps.model.n_layers_q,
            use_spectral_norm=hps.model.use_spectral_norm,
            gin_channels=hps.model.gin_channels,
            slm=hps.model.slm,
        ).to(device)
        '''
        print(len(SYMBOLS), hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, hps.data.n_speakers,
            hps.model.use_spk_conditioned_encoder, hps.model.use_noise_scaled_mas, hps.model.use_mel_posterior_encoder,
            hps.model.use_duration_discriminator, hps.model.use_wavlm_discriminator, hps.model.inter_channels,hps.model.hidden_channels,
            hps.model.filter_channels, hps.model.n_heads,hps.model.n_layers,hps.model.kernel_size,hps.model.p_dropout,hps.model.resblock,
            hps.model.resblock_kernel_sizes,hps.model.resblock_dilation_sizes,hps.model.upsample_rates,hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,hps.model.n_layers_q,hps.model.use_spectral_norm,hps.model.gin_channels, hps.model.slm)
        '''
    else:
        logger.info("Using normal model")
        net_g = SynthesizerTrn(
            n_vocab=len(SYMBOLS),
            spec_channels=hps.data.filter_length // 2 + 1,
            segment_size=hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            # hps.model 以下のすべての値を引数に渡す
            use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
            use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
            use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
            use_duration_discriminator=hps.model.use_duration_discriminator,
            use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
            inter_channels=hps.model.inter_channels,
            hidden_channels=hps.model.hidden_channels,
            filter_channels=hps.model.filter_channels,
            n_heads=hps.model.n_heads,
            n_layers=hps.model.n_layers,
            kernel_size=hps.model.kernel_size,
            p_dropout=hps.model.p_dropout,
            resblock=hps.model.resblock,
            resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
            resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
            upsample_rates=hps.model.upsample_rates,
            upsample_initial_channel=hps.model.upsample_initial_channel,
            upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
            n_layers_q=hps.model.n_layers_q,
            use_spectral_norm=hps.model.use_spectral_norm,
            gin_channels=hps.model.gin_channels,
            slm=hps.model.slm,
        ).to(device)
    net_g.state_dict()
    _ = net_g.eval()
    if model_path.endswith(".pth") or model_path.endswith(".pt"):
        _ = utils.checkpoints.load_checkpoint(
            model_path, net_g, None, skip_optimizer=True
        )
    elif model_path.endswith(".safetensors"):
        _ = utils.safetensors.load_safetensors(model_path, net_g, True)
    else:
        raise ValueError(f"Unknown model format: {model_path}")
    return net_g


def get_text(
    text: str,
    language_str: Languages,
    hps: HyperParameters,
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_tone: 'Optional[list[int]]' = None,
):
    use_jp_extra = hps.version.endswith("JP-Extra")
    # 推論時のみ呼び出されるので、raise_yomi_error は False に設定
    norm_text, phone, tone, word2ph = clean_text(
        text,
        language_str,
        use_jp_extra=use_jp_extra,
        raise_yomi_error=False,
    )
    if given_tone is not None:
        if len(given_tone) != len(phone):
            raise InvalidToneError(
                f"Length of given_tone ({len(given_tone)}) != length of phone ({len(phone)})"
            )
        tone = given_tone
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert_ori = extract_bert_feature(
        norm_text,
        word2ph,
        language_str,
        device,
        assist_text,
        assist_text_weight,
    )
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    if language_str == Languages.ZH:
        bert = bert_ori
        ja_bert = torch.zeros(1024, len(phone))
        en_bert = torch.zeros(1024, len(phone))
    elif language_str == Languages.JP:
        bert = torch.zeros(1024, len(phone))
        ja_bert = bert_ori
        en_bert = torch.zeros(1024, len(phone))
    elif language_str == Languages.EN:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(1024, len(phone))
        en_bert = bert_ori
    else:
        raise ValueError("language_str should be ZH, JP or EN")

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, en_bert, phone, tone, language


def infer(
    text: str,
    style_vec: NDArray[Any],
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    sid: int,  # In the original Bert-VITS2, its speaker_name: str, but here it's id
    language: Languages,
    hps: HyperParameters,
    net_g: Union[SynthesizerTrn, SynthesizerTrnJPExtra],
    device: str,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_tone: 'Optional[list[int]]' = None,
):
    is_jp_extra = hps.version.endswith("JP-Extra")
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text,
        language,
        hps,
        device,
        assist_text=assist_text,
        assist_text_weight=assist_text_weight,
        given_tone=given_tone,
    )
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        bert = bert[:, 3:]
        ja_bert = ja_bert[:, 3:]
        en_bert = en_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        bert = bert[:, :-2]
        ja_bert = ja_bert[:, :-2]
        en_bert = en_bert[:, :-2]
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        en_bert = en_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        style_vec_tensor = torch.from_numpy(style_vec).to(device).unsqueeze(0)
        del phones
        sid_tensor = torch.LongTensor([sid]).to(device)
        
        if is_jp_extra:
            """
            onnx_model_path = "./onnx_models/model.onnx"
            print(os.path.exists(onnx_model_path))
            if os.path.exists(onnx_model_path):
                logger.info(f"ONNXモデルファイル {onnx_model_path} を作成します")

                dummy_input = (x_tst,x_tst_lengths,sid_tensor,tones,lang_ids,ja_bert,style_vec_tensor)
                torch.onnx.export(
                    cast(SynthesizerTrnJPExtra, net_g),
                    dummy_input,
                    onnx_model_path,
                    input_names=['x', 'x_lengths', 'sid', 'tone', 'language', 'bert', 'style_vec'],
                    output_names=['output'],
                    opset_version=11,
                    dynamic_axes={
                        'x': {0: 'batch_size', 1: 'text_seq_len'},
                        'x_lengths': {0: 'batch_size'},
                        'sid': {0: 'batch_size'},
                        'tone': {0: 'batch_size', 1: 'tone_seq_len'},
                        'language': {0: 'batch_size'},
                        'bert': {0: 'batch_size', 1: 'bert_seq_len', 2: 'bert_hidden_size'},
                        'style_vec': {0: 'batch_size', 1: 'style_vec_dim'},
                        'output': {0: 'batch_size', 1: 'output_seq_len'}
                    }
                )
            sys.exit(0)
            input_data = (x_tst,x_tst_lengths,sid_tensor,tones,lang_ids,ja_bert,style_vec_tensor)
            #input_data = tuple(x.to(torch.float) for x in input_data),
            session = rt.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name        
            output = session.run([output_name], {input_name: input_data})
            """
            output = cast(SynthesizerTrnJPExtra, net_g).infer(
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                ja_bert,
                style_vec=style_vec_tensor,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )
            
            '''
            output = cast(SynthesizerTrnJPExtra, net_g).infer(
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                ja_bert,
                style_vec=style_vec_tensor,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )
            '''

        else:
            output = cast(SynthesizerTrn, net_g).infer(
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                bert,
                ja_bert,
                en_bert,
                style_vec=style_vec_tensor,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )
        audio = output[0][0, 0].data.cpu().float().numpy()
        del (
            x_tst,
            tones,
            lang_ids,
            bert,
            x_tst_lengths,
            sid_tensor,
            ja_bert,
            en_bert,
            style_vec,
        )  # , emo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio


class InvalidToneError(ValueError):
    pass
