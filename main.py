import os
import queue
from huggingface_hub import snapshot_download
import hydra
import numpy as np
import wave
import io
import pyrootutils
import gc

# Download if not exists
os.makedirs("checkpoints", exist_ok=True)
snapshot_download(repo_id="fishaudio/fish-speech-1.5", local_dir="./checkpoints/fish-speech-1.5")

print("All checkpoints downloaded")

import html
import os
import threading
from argparse import ArgumentParser
from pathlib import Path
from functools import partial

import gradio as gr
import librosa
import torch
import torchaudio

torchaudio.set_audio_backend("soundfile")

from loguru import logger
from transformers import AutoTokenizer

from fish_speech.i18n import i18n
from fish_speech.text.chn_text_norm.text import Text as ChnNormedText
from fish_speech.utils import autocast_exclude_mps, set_seed
from tools.api import decode_vq_tokens, encode_reference
from tools.file import AUDIO_EXTENSIONS, list_files
from tools.llama.generate import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
    launch_thread_safe_queue,
)
from tools.vqgan.inference import load_model as load_decoder_model

from tools.schema import (
    GLOBAL_NUM_SAMPLES,
    ASRPackRequest,
    ServeASRRequest,
    ServeASRResponse,
    ServeASRSegment,
    ServeAudioPart,
    ServeForwardMessage,
    ServeMessage,
    ServeRequest,
    ServeResponse,
    ServeStreamDelta,
    ServeStreamResponse,
    ServeTextPart,
    ServeTimedASRResponse,
    ServeTTSRequest,
    ServeVQGANDecodeRequest,
    ServeVQGANDecodeResponse,
    ServeVQGANEncodeRequest,
    ServeVQGANEncodeResponse,
    ServeVQPart,
    ServeReferenceAudio
)
# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"


HEADER_MD = """# Fish Speech

## The demo in this space is version 1.5, Please check [Fish Audio](https://fish.audio) for the best model.
## 该 Demo 为 Fish Speech 1.5 版本, 请在 [Fish Audio](https://fish.audio) 体验最新 DEMO.

A text-to-speech model based on VQ-GAN and Llama developed by [Fish Audio](https://fish.audio).  
由 [Fish Audio](https://fish.audio) 研发的基于 VQ-GAN 和 Llama 的多语种语音合成. 

You can find the source code [here](https://github.com/fishaudio/fish-speech) and models [here](https://huggingface.co/fishaudio/fish-speech-1.5).  
你可以在 [这里](https://github.com/fishaudio/fish-speech) 找到源代码和 [这里](https://huggingface.co/fishaudio/fish-speech-1.5) 找到模型.  

Related code and weights are released under CC BY-NC-SA 4.0 License.  
相关代码，权重使用 CC BY-NC-SA 4.0 许可证发布.

We are not responsible for any misuse of the model, please consider your local laws and regulations before using it.  
我们不对模型的任何滥用负责，请在使用之前考虑您当地的法律法规.

The model running in this WebUI is Fish Speech V1.5 Medium.
在此 WebUI 中运行的模型是 Fish Speech V1.5 Medium.
"""

TEXTBOX_PLACEHOLDER = """Put your text here. 在此处输入文本."""

try:
    import spaces

    GPU_DECORATOR = spaces.GPU
except ImportError:

    def GPU_DECORATOR(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper


def build_html_error_message(error):
    return f"""
    <div style="color: red; 
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """


@GPU_DECORATOR
@torch.inference_mode()
def inference(req: ServeTTSRequest):
    # Parse reference audio aka prompt
    refs = req.references

    prompt_tokens = [
        encode_reference(
            decoder_model=decoder_model,
            reference_audio=ref.audio,
            enable_reference_audio=True,
        )
        for ref in refs
    ]
    prompt_texts = [ref.text for ref in refs]

    if req.seed is not None:
        set_seed(req.seed)
        logger.warning(f"set seed: {req.seed}")

    # LLAMA Inference
    request = dict(
        device=decoder_model.device,
        max_new_tokens=req.max_new_tokens,
        text=(
            req.text
            if not req.normalize
            else ChnNormedText(raw_text=req.text).normalize()
        ),
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        temperature=req.temperature,
        compile=args.compile,
        iterative_prompt=req.chunk_length > 0,
        chunk_length=req.chunk_length,
        max_length=4096,
        prompt_tokens=prompt_tokens,
        prompt_text=prompt_texts,
    )

    response_queue = queue.Queue()
    llama_queue.put(
        GenerateRequest(
            request=request,
            response_queue=response_queue,
        )
    )

    segments = []

    while True:
        result: WrappedGenerateResponse = response_queue.get()
        if result.status == "error":
            yield None, None, build_html_error_message(result.response)
            break

        result: GenerateResponse = result.response
        if result.action == "next":
            break

        with autocast_exclude_mps(
            device_type=decoder_model.device.type, dtype=args.precision
        ):
            fake_audios = decode_vq_tokens(
                decoder_model=decoder_model,
                codes=result.codes,
            )

        fake_audios = fake_audios.float().cpu().numpy()
        segments.append(fake_audios)

    if len(segments) == 0:
        return (
            None,
            None,
            build_html_error_message(
                i18n("No audio generated, please check the input text.")
            ),
        )

    # No matter streaming or not, we need to return the final audio
    audio = np.concatenate(segments, axis=0)
    yield None, (decoder_model.spec_transform.sample_rate, audio), None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

n_audios = 4

global_audio_list = []
global_error_list = []


def wav_chunk_header(sample_rate=44100, bit_depth=16, channels=1):
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()
    return wav_header_bytes

def normalize_text(user_input, use_normalization):
    if use_normalization:
        return ChnNormedText(raw_text=user_input).normalize()
    else:
        return user_input

def build_app():
    with gr.Blocks(theme=gr.themes.Base()) as app:
        gr.Markdown(HEADER_MD)

        # Use light theme by default
        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', '%s');window.location.search = params.toString();}}"
            % args.theme,
        )

        # Inference
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label=i18n("Input Text"), placeholder=TEXTBOX_PLACEHOLDER, lines=10
                )
                refined_text = gr.Textbox(
                    label=i18n("Realtime Transform Text"),
                    placeholder=i18n(
                        "Normalization Result Preview (Currently Only Chinese)"
                    ),
                    lines=5,
                    interactive=False,
                )

                with gr.Row():
                    normalize = gr.Checkbox(
                        label=i18n("Text Normalization"),
                        value=False,
                    )

                with gr.Row():
                    with gr.Column():
                        with gr.Tab(label=i18n("Advanced Config")):
                            with gr.Row():
                                chunk_length = gr.Slider(
                                    label=i18n("Iterative Prompt Length, 0 means off"),
                                    minimum=0,
                                    maximum=300,
                                    value=200,
                                    step=8,
                                )

                                max_new_tokens = gr.Slider(
                                    label=i18n(
                                        "Maximum tokens per batch"
                                    ),
                                    minimum=512,
                                    maximum=2048,
                                    value=1024,
                                    step=64,
                                )

                            with gr.Row():
                                top_p = gr.Slider(
                                    label="Top-P",
                                    minimum=0.6,
                                    maximum=0.9,
                                    value=0.7,
                                    step=0.01,
                                )

                                repetition_penalty = gr.Slider(
                                    label=i18n("Repetition Penalty"),
                                    minimum=1,
                                    maximum=1.5,
                                    value=1.2,
                                    step=0.01,
                                )

                            with gr.Row():
                                temperature = gr.Slider(
                                    label="Temperature",
                                    minimum=0.6,
                                    maximum=0.9,
                                    value=0.7,
                                    step=0.01,
                                )
                                seed = gr.Number(
                                    label="Seed",
                                    info="0 means randomized inference, otherwise deterministic",
                                    value=0,
                                )

                        with gr.Tab(label=i18n("Reference Audio")):
                            with gr.Row():
                                gr.Markdown(
                                    i18n(
                                        "15 to 60 seconds of reference audio, useful for specifying speaker."
                                    )
                                )

                            with gr.Row():
                                # Add dropdown for selecting example audio files
                                example_audio_files = [f for f in os.listdir("examples") if f.endswith(".wav")]
                                example_audio_dropdown = gr.Dropdown(
                                    label="Select Example Audio",
                                    choices=[""] + example_audio_files,
                                    value=""
                                )

                            with gr.Row():
                                use_memory_cache = gr.Radio(
                                    label=i18n("Use Memory Cache"),
                                    choices=["never"],
                                    value="never",
                                )

                            with gr.Row():
                                reference_audio = gr.Audio(
                                    label=i18n("Reference Audio"),
                                    type="filepath",
                                )

                            with gr.Row():
                                reference_text = gr.Textbox(
                                    label=i18n("Reference Text"),
                                    lines=1,
                                    placeholder="在一无所知中，梦里的一天结束了，一个新的「轮回」便会开始。",
                                    value="",
                                )

            with gr.Column(scale=3):
                with gr.Row():
                    error = gr.HTML(
                        label=i18n("Error Message"),
                        visible=True,
                    )
                with gr.Row():
                    audio = gr.Audio(
                        label=i18n("Generated Audio"),
                        type="numpy",
                        interactive=False,
                        visible=True,
                    )

                with gr.Row():
                    with gr.Column(scale=3):
                        generate = gr.Button(
                            value="\U0001F3A7 " + i18n("Generate"), variant="primary"
                        )

        text.input(
            fn=normalize_text, inputs=[text, normalize], outputs=[refined_text]
        )

        def inference_wrapper(
            text,
            normalize,
            reference_audio,
            reference_text,
            max_new_tokens,
            chunk_length,
            top_p,
            repetition_penalty,
            temperature,
            seed,
            use_memory_cache,
        ):
            print(
                "call inference wrapper", 
                text,
                normalize,
                reference_audio,
                reference_text,
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
                seed,
                use_memory_cache
            )

            references = []
            if reference_audio:
                # 将文件路径转换为字节
                with open(reference_audio, 'rb') as audio_file:
                    audio_bytes = audio_file.read()

                references = [
                    ServeReferenceAudio(audio=audio_bytes, text=reference_text)
                ]

            req = ServeTTSRequest(
                text=text,
                normalize=normalize,
                reference_id=None,
                references=references,
                max_new_tokens=max_new_tokens,
                chunk_length=chunk_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                seed=int(seed) if seed else None,
                use_memory_cache=use_memory_cache,
            )
            
            for result in inference(req):
                if result[2]:  # Error message
                    return None, result[2]
                elif result[1]:  # Audio data
                    return result[1], None
            
            return None, i18n("No audio generated")

        def select_example_audio(audio_file):
            if audio_file:
                audio_path = os.path.join("examples", audio_file)
                lab_file = os.path.splitext(audio_file)[0] + ".lab"
                lab_path = os.path.join("examples", lab_file)
                
                if os.path.exists(lab_path):
                    with open(lab_path, "r", encoding="utf-8") as f:
                        lab_content = f.read().strip()
                else:
                    lab_content = ""
                
                return audio_path, lab_content
            return None, ""

        # Connect the dropdown to update reference audio and text
        example_audio_dropdown.change(
            fn=select_example_audio,
            inputs=[example_audio_dropdown],
            outputs=[reference_audio, reference_text]
        )

        # Submit
        generate.click(
            inference_wrapper,
            [
                refined_text,
                normalize,
                reference_audio,
                reference_text,
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
                seed,
                use_memory_cache,
            ],
            [audio, error],
            concurrency_limit=1,
        )

    return app



def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/fish-speech-1.5",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default="checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="firefly_gan_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true",default=True)
    parser.add_argument("--max-gradio-length", type=int, default=0)
    parser.add_argument("--theme", type=str, default="light")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16

    logger.info("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=args.precision,
        compile=args.compile,
    )
    logger.info("Llama model loaded, loading VQ-GAN model...")

    decoder_model = load_decoder_model(
        config_name=args.decoder_config_name,
        checkpoint_path=args.decoder_checkpoint_path,
        device=args.device,
    )

    logger.info("Decoder model loaded, warming up...")

    # Dry run to check if the model is loaded correctly and avoid the first-time latency
    list(
            inference(
                ServeTTSRequest(
                    text="Hello world.",
                    references=[],
                    reference_id=None,
                    max_new_tokens=0,
                    chunk_length=200,
                    top_p=0.7,
                    repetition_penalty=1.5,
                    temperature=0.7,
                    emotion=None,
                    format="wav",
                )
            )
    )

    logger.info("Warming up done, launching the web UI...")

    app = build_app()
    app.queue(api_open=True).launch(show_error=True, show_api=True)
