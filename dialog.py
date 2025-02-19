import os
import json
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
root_dir = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.join(root_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)
temp_dir = os.path.join(root_dir, "temp")
os.makedirs(temp_dir, exist_ok=True)

import shutil

for filename in os.listdir(temp_dir):
    file_path = os.path.join(temp_dir, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'Error delete in {file_path}. {e}')

os.environ["GRADIO_TEMP_DIR"] = temp_dir
import queue
from huggingface_hub import snapshot_download
import numpy as np
import wave
import io
import gc
from datetime import datetime
import html
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
    ServeTTSRequest,
    ServeReferenceAudio
)
file_list = sorted([f for f in os.listdir("examples") if f.lower().endswith(('.wav', '.mp3'))])
HEADER_MD = """# 🎭 Fish Speech Dialogue

<div class="container" style="display: flex; width: 100%;">
<div style="flex: 1; padding-right: 20px;">
<h2 style="font-size: 1.5em; margin-bottom: 10px;">Система для озвучивания диалогов различными голосами</h2>
<p>✏️ Вставтье диалог (реплику), где каждая реплика начинается с имени говорящего и двоеточия</p>
<p>🎤 Укажите голоса, нажмите кнопку генерации</p>
</div>

<div style="flex: 1; padding-left: 20px;">
<h2 style="font-size: 1.5em; margin-bottom: 10px;">Авторы:</h2>
<p><a href="https://t.me/neuro_art0" style="color: #2196F3; text-decoration: none;">Nerual Dreming</a> — основной код обработки диалогов - основатель <a href="https://artgeneration.me" style="color: #2196F3; text-decoration: none;">ArtGeneration.me</a>, техноблогер и нейро-евангелист</p>
<p><a href="https://t.me/FooocusExtend_Support" style="color: #2196F3; text-decoration: none;">Shahmatist^RMDA</a> — некоторые правки, мультиязычный интерфейс, портативный репак</p>
<p><a href="https://t.me/neuroport" style="color: #2196F3; text-decoration: none;">👾 НЕЙРО-СОФТ</a> — Репаки и портативные версии полезных нейросетей</p>
<p><a href="https://t.me/FooocusExtend_Support" style="color: #2196F3; text-decoration: none;">👾 FooocusExtend</a> — Форк Fooocus c расширенными и дополнительными функциями</p>
</div>
</div>
"""

try:
    import spaces
    GPU_DECORATOR = spaces.GPU
except ImportError:
    def GPU_DECORATOR(func):
        def wrapper(*args, **kwargs):
            with torch.inference_mode():
                return func(*args, **kwargs)
        return wrapper

def normalize_audio_rms(audio, target_db=-20.0):
    current_rms = np.sqrt(np.mean(audio ** 2))
    current_db = 20 * np.log10(current_rms) if current_rms > 0 else -80.0
    gain = 10 ** ((target_db - current_db) / 20)
    return np.clip(audio * gain, -1.0, 1.0)

def build_html_error_message(error):
    return f"""<div style="color: red; font-weight: bold;">{html.escape(str(error))}</div>"""

def get_audio_transcription(audio_path):
    if not audio_path:
        return ""
        
    base_name = os.path.splitext(audio_path)[0]
    
    for ext in ['.txt', '.lab']:
        text_path = base_name + ext
        if os.path.exists(text_path):
            try:
                with open(text_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except:
                continue
    return ""

def parse_dialogue(text):
    dialogue_parts = []
    speakers = set()
    current_speaker = None
    current_text = []
    phrases_count = 0
    total_chars = len(text.strip())
    
    if not text or not text.strip():
        return [], 0, 0, 0
        
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if ':' in line:
            if current_speaker and current_text:
                dialogue_parts.append((current_speaker, ' '.join(current_text)))
                phrases_count += 1
            
            speaker, text = line.split(':', 1)
            current_speaker = speaker.strip()
            current_text = [text.strip()]
            speakers.add(current_speaker)
        elif current_speaker:
            current_text.append(line)
    
    if current_speaker and current_text:
        dialogue_parts.append((current_speaker, ' '.join(current_text)))
        phrases_count += 1
        
    return dialogue_parts, len(speakers), phrases_count, total_chars

def update_dialogue_stats(text):
    _, num_speakers, phrases_count, chars_count = parse_dialogue(text)
    return f"Говорящих: {num_speakers} | Реплик: {phrases_count} | Символов: {chars_count}"

def update_accordion_label(speaker_name, voice_file, index):
    if not speaker_name:
        return f"Говорящий {index+1}"
    
    voice_name = os.path.basename(voice_file) if voice_file else "Нет голоса"
    return f"Говорящий {index+1} - {speaker_name} - {voice_name}"

@GPU_DECORATOR
@torch.inference_mode()
def inference(req: ServeTTSRequest, selected_formats):
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

    request = dict(
        device=decoder_model.device,
        max_new_tokens=req.max_new_tokens,
        text=req.text,
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
            yield None, None, None, None, None
            return

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
        yield None, None, build_html_error_message("Аудио не сгенерировано"), None, None, None
        return

    audio = np.concatenate(segments, axis=0)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    
    audio_paths = {'wav': None, 'mp3': None, 'flac': None}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for fmt in selected_formats:
        path = os.path.join(outputs_dir, f"output_{timestamp}.{fmt}")
        torchaudio.save(path, audio_tensor, decoder_model.spec_transform.sample_rate)
        audio_paths[fmt] = path

    yield (None, (decoder_model.spec_transform.sample_rate, audio), None, 
            audio_paths['wav'], audio_paths['mp3'], audio_paths['flac'])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def select_example_audio(audio_file, speaker_name, accordion_index):
    if audio_file:
        audio_path = os.path.join("examples", audio_file)
        transcription = get_audio_transcription(audio_path)
        # Обновляем только нужный аккордеон
        accordion_updates = [gr.update() for _ in range(10)]
        # Обновляем только нужный аккордеон
        accordion_updates[accordion_index] = gr.update(
            label=update_accordion_label(speaker_name, audio_path, accordion_index)
        )
        return [
            audio_path,
            transcription,
            *accordion_updates
        ]
    return [None, "", *[gr.update() for _ in range(10)]]

def on_dialogue_change(text):
    global file_list
    dialogue_parts, num_speakers, phrases_count, chars_count = parse_dialogue(text)
    num_to_show = min(max(num_speakers, 1), 10)
    stats = update_dialogue_stats(text)
    example_audio_files = file_list
    
    updates = []
    
    # Обновления для всех 10 аккордеонов
    for i in range(10):
        updates.append(gr.update(visible=(i < num_to_show)))
    
    updates.append(stats)
    updates.append(gr.update(value=num_to_show))
    
    # Обновления для компонентов каждого спикера
    for i in range(10):
        if i < len(dialogue_parts):
            # Для существующих строк диалога берем имя прямо из dialogue_parts
            name = dialogue_parts[i][0]
            updates.extend([
                gr.update(value=name),
                gr.update(),
                gr.update(),
                gr.update(choices=[""] + example_audio_files)
            ])
        else:
            # Для дополнительных слотов создаем нового пользователя
            name = f"Пользователь {i+1}"
            initial_audio = np.random.choice(example_audio_files) if example_audio_files else ""
            initial_audio_path = os.path.join("examples", initial_audio) if initial_audio else None
            initial_transcript = get_audio_transcription(initial_audio_path) if initial_audio_path else ""
            
            updates.extend([
                gr.update(value=name),
                gr.update(value=initial_audio_path),
                gr.update(value=initial_transcript),
                gr.update(value=initial_audio, choices=[""] + example_audio_files)
            ])
    
    return updates

def update_speaker_visibility(num):
    global file_list
    updates = []
    example_audio_files = file_list
    
    for i in range(10):
        visible = i < num
        
        if visible:
            voice = np.random.choice(example_audio_files) if example_audio_files else ""
            voice_path = os.path.join("examples", voice) if voice else None
            transcript = get_audio_transcription(voice_path) if voice_path else ""
            
            updates.extend([
                gr.update(visible=True),
                gr.update(value=f"Пользователь {i+1}"),
                gr.update(value=voice_path),
                gr.update(value=transcript),
                gr.update(value=voice, choices=[""] + example_audio_files)
            ])
        else:
            updates.extend([
                gr.update(visible=False),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update()
            ])
    
    return updates

@GPU_DECORATOR
@torch.inference_mode()
def generate_dialogue_audio(
    text_parts,
    voice_files,
    voice_transcripts,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    seed,
    selected_formats
):
    audio_parts = []
    
    for speaker, text in text_parts:
        mapped_speaker = speaker
        if speaker not in voice_files:
            try:
                num = int(''.join(filter(str.isdigit, speaker)))
                mapped_index = ((num - 1) % 10)
                visible_speakers = list(voice_files.keys())
                mapped_speaker = visible_speakers[mapped_index]
            except:
                mapped_speaker = list(voice_files.keys())[0]
        
        audio_path = voice_files[mapped_speaker]
        transcription = voice_transcripts.get(mapped_speaker, '') or get_audio_transcription(audio_path)
        
        with open(audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            
        reference = ServeReferenceAudio(
            audio=audio_bytes,
            text=transcription
        )
        
        req = ServeTTSRequest(
            text=text,
            normalize=False,
            reference_id=None,
            references=[reference],
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=int(seed) if seed else None,
            use_memory_cache="never",
        )

        for result in inference(req, selected_formats):
            _, (sample_rate, audio_data), error_msg, wav_path, mp3_path, flac_path = result
            
            if error_msg:
                yield None, None, None
                return
                
            normalized_audio = normalize_audio_rms(audio_data)
            audio_parts.append((sample_rate, normalized_audio))

    if not audio_parts:
        yield None, None, None
        return

    target_sr = audio_parts[0][0]
    combined_audio = []
    
    for sr, audio in audio_parts:
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        combined_audio.append(audio)
    
    final_audio = np.concatenate(combined_audio)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    audio_tensor = torch.from_numpy(final_audio).unsqueeze(0)
    
    wav_path = mp3_path = flac_path = None
    
    if 'wav' in selected_formats:
        wav_path = os.path.join(outputs_dir, f"dialogue_{timestamp}.wav")
        torchaudio.save(wav_path, audio_tensor, target_sr)
        
    if 'mp3' in selected_formats:
        mp3_path = os.path.join(outputs_dir, f"dialogue_{timestamp}.mp3")
        torchaudio.save(mp3_path, audio_tensor, target_sr)
        
    if 'flac' in selected_formats:
        flac_path = os.path.join(outputs_dir, f"dialogue_{timestamp}.flac")
        torchaudio.save(flac_path, audio_tensor, target_sr)

    yield wav_path, mp3_path, flac_path, None

def load_translations(lang):
        file_path = os.path.join('', f'{lang}.json')
        print('---------------------',file_path)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        else:
            raise ValueError(f"Translation file for language '{lang}' not found")

current_language = 'ru'
translations = load_translations(current_language)

def set_language(lang):
        global current_language, translations
        translations = load_translations(lang)
        current_language = lang
def _(key):
        return translations.get(key, key)
def change_lang():	
       global current_language
       if current_language=='en':
		      set_language('ru')
       else:
          set_language('en')

with gr.Blocks(theme=gr.themes.Base()) as app:

        gr.Markdown(HEADER_MD)
        lang=gr.Button("Change Language")

        example_audio_files = file_list
        
        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', 'dark');window.location.search = params.toString();}}"
        )

        with gr.Row():
            with gr.Column(scale=3):
                initial_text = "Пользователь 1: Ребята, у меня проблема: мой кот постоянно будит меня в 5 утра.\nПользователь 2: Может, он хочет есть? Попробуй кормить его перед сном.\nПользователь 3: Или заведи будильник на 4:30 и разбуди его первым. Пусть знает, каково это!"
                
                dialogue_stats = gr.Textbox(
                    label=i18n('statis_dialog'),
                    value=update_dialogue_stats(initial_text),
                    interactive=False
                    )
                
                dialogue_text = gr.Textbox(
                    label="Текст диалога",
                    value=initial_text,
                    placeholder="Пользователь 1: Привет!\nПользователь 2: Здравствуйте!",
                    lines=10
                )

                with gr.Row():
                    num_speakers = gr.Slider(
                        label="Количество говорящих",
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        interactive=False
                    )

                speaker_boxes = []
                initial_parts, _, _, _ = parse_dialogue(initial_text)
                
                for i in range(10):
                    initial_name = f"Пользователь {i+1}"
                    if i < len(initial_parts):
                        initial_name = initial_parts[i][0]
                        
                    initial_audio = np.random.choice(example_audio_files) if example_audio_files else ""
                    initial_audio_path = os.path.join("examples", initial_audio) if initial_audio else None
                    initial_transcript = get_audio_transcription(initial_audio_path) if initial_audio_path else ""
                    
                    with gr.Accordion(
                        label=update_accordion_label(initial_name, initial_audio_path, i),
                        open=False,
                        visible=(i < 3)
                    ) as speaker_accordion:
                        speaker_name = gr.Textbox(
                            label=f"Имя говорящего {i+1}",
                            value=initial_name
                        )
                        
                        example_audio = gr.Dropdown(
                            label=f"Пример голоса {i+1}",
                            choices=[""] + example_audio_files,
                            value=initial_audio
                        )
                        
                        speaker_voice = gr.Audio(
                            label=f"Голос говорящего {i+1}",
                            type="filepath",
                            value=initial_audio_path
                        )
                        
                        voice_transcript = gr.Textbox(
                            label="Транскрипция",
                            lines=3,
                            interactive=True,
                            value=initial_transcript
                        )
                        
                    speaker_boxes.append((speaker_accordion, speaker_name, speaker_voice, voice_transcript, example_audio))

                with gr.Row():
                    with gr.Column():
                        with gr.Accordion(label="Расширенные настройки", open=False):
                            with gr.Row():
                                chunk_length = gr.Slider(
                                    label="Длина итеративного промпта",
                                    minimum=0,
                                    maximum=300,
                                    value=200,
                                    step=8,
                                )
                                
                                max_new_tokens = gr.Slider(
                                    label="Максимальное количество токенов",
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
                                    label="Штраф за повторение",
                                    minimum=1,
                                    maximum=1.5,
                                    value=1.2,
                                    step=0.01,
                                )

                            with gr.Row():
                                temperature = gr.Slider(
                                    label="Температура",
                                    minimum=0.6,
                                    maximum=0.9,
                                    value=0.7,
                                    step=0.01,
                                )
                                seed = gr.Number(
                                    label="Сид",
                                    value=0,
                                    info="0 означает случайную генерацию"
                                )

                            with gr.Row():
                                gr.Markdown("### Форматы сохранения")
                            with gr.Row():
                                wav_format = gr.Checkbox(label="WAV", value=True)
                                mp3_format = gr.Checkbox(label="MP3", value=False)
                                flac_format = gr.Checkbox(label="FLAC", value=False)

            with gr.Column(scale=3):

                
                with gr.Row(visible=wav_format.value) as wav_panel:
                    audio_wav = gr.Audio(
                        label="WAV",
                        type="filepath",
                        interactive=False,
                        visible=True,
                    )
                with gr.Row(visible=mp3_format.value) as mp3_panel:
                    audio_mp3 = gr.Audio(
                        label="MP3",
                        type="filepath",
                        interactive=False,
                        visible=True,
                    )
                with gr.Row(visible=flac_format.value) as flac_panel:
                    audio_flac = gr.Audio(
                        label="FLAC",
                        type="filepath",
                        interactive=False,
                        visible=True,
                    )

                with gr.Row():
                    generate_button = gr.Button(
                        value="🎭 Сгенерировать диалог (реплику)",
                        variant="primary"
                    )
        mp3_format.change(lambda x: gr.update(visible=x), inputs=mp3_format, outputs=mp3_panel, queue=False)
        wav_format.change(lambda x: gr.update(visible=x), inputs=wav_format, outputs=wav_panel, queue=False)
        flac_format.change(lambda x: gr.update(visible=x), inputs=flac_format, outputs=flac_panel, queue=False)
        # Обновление при изменении текста диалога
        dialogue_text.change(
                fn=on_dialogue_change,
                inputs=[dialogue_text],
                outputs=[
                    *[box[0] for box in speaker_boxes],  # 10 аккордеонов
                    dialogue_stats,  # статистика
                    num_speakers,   # количество спикеров
                    *[item for box in speaker_boxes for item in box[1:]]  # компоненты спикеров
                ]
            )           

        # Обработка выбора примера аудио
        for i, (accordion, name, voice, transcript, example) in enumerate(speaker_boxes):
            example.change(
        fn=lambda audio, name, i=i: select_example_audio(audio, name, i),
        inputs=[example, name],
        outputs=[
            voice, 
            transcript,
            *[box[0] for box in speaker_boxes]  # возвращаем обновление аккордеонов
            ]
        )

            # Обновление заголовка при изменении имени
            name.blur(
                fn=lambda n, v, i=i: gr.update(label=update_accordion_label(n, v, i)),
                inputs=[name, voice],
                outputs=[accordion],show_progress=True,queue=False
            )
            name.submit(
                fn=lambda n, v, i=i: gr.update(label=update_accordion_label(n, v, i)),
                inputs=[name, voice],
                outputs=[accordion],show_progress=True,queue=False
            )

        def generate_dialogue(*args):
            dialogue_text_value = args[0]
            num_speakers_value = int(args[1])
            
            speaker_voices = {}
            speaker_transcripts = {}
            
            for i in range(num_speakers_value):
                name = args[2 + i*4]
                voice = args[2 + i*4 + 1]
                transcript = args[2 + i*4 + 2]
                if name and voice:
                    name = name.strip()
                    speaker_voices[name] = voice
                    if transcript:
                        speaker_transcripts[name] = transcript
            
            max_new_tokens_value = args[42]  # 2 + 10*4 = 42 (начало параметров после спикеров)
            chunk_length_value = args[43]
            top_p_value = args[44]
            repetition_penalty_value = args[45]
            temperature_value = args[46]
            seed_value = args[47]
            wav_format_value = args[48]
            mp3_format_value = args[49]
            flac_format_value = args[50]

            selected_formats = []
            if wav_format_value:
                selected_formats.append('wav')
            if mp3_format_value:
                selected_formats.append('mp3')
            if flac_format_value:
                selected_formats.append('flac')
            
            if not selected_formats:
                selected_formats = ['wav']

            dialogue_parts, _, _, _ = parse_dialogue(dialogue_text_value)
            
            if not dialogue_parts:
                return None, None, None, "Не удалось разобрать текст диалога. Убедитесь, что каждая реплика начинается с имени говорящего и двоеточия."
            
            for result in generate_dialogue_audio(
                dialogue_parts,
                speaker_voices,
                speaker_transcripts,
                max_new_tokens_value,
                chunk_length_value,
                top_p_value,
                repetition_penalty_value,
                temperature_value,
                seed_value,
                selected_formats
            ):
                return result

        generate_button.click(
            fn=generate_dialogue,
            inputs=[
                dialogue_text,
                num_speakers,
                *[item for box in speaker_boxes for item in (box[1], box[2], box[3], box[4])],
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
                seed,
                wav_format,
                mp3_format,
                flac_format,
            ],
            outputs=[
                audio_wav,
                audio_mp3,
                audio_flac
            ],
            concurrency_limit=1,
        )
        lang.click(change_lang) \
		            .then(lambda: (gr.update()),outputs=dialogue_stats)


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
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--max_gradio_length", type=int, default=0)
    parser.add_argument("--theme", type=str, default="dark")
    parser.add_argument("--share", type=bool, default=False)

    return parser.parse_args()

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    snapshot_download(repo_id="fishaudio/fish-speech-1.5", local_dir="./checkpoints/fish-speech-1.5")
    print("All checkpoints downloaded")

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



    logger.info("Warming up done, launching the web UI...")

    
    app.launch(show_error=True, show_api=True, inbrowser=True, share=args.share)
