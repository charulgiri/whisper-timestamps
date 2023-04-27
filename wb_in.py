import numpy as np
import torch
import pandas as pd
import whisper
import gradio as gr
import string
import itertools


# from IPython.display import display, HTML
from whisper.tokenizer import get_tokenizer
from dtw import dtw
from scipy.ndimage import median_filter

import librosa
from transformers import WEIGHTS_NAME, CONFIG_NAME, AutoModelForSpeechSeq2Seq, AutoFeatureExtractor, WhisperTokenizer, AutoProcessor

pd.options.display.max_rows = 100
pd.options.display.max_colwidth = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE

medfilt_width = 7
qk_scale = 1.0

def split_tokens_on_unicode(tokens: torch.Tensor):
    words = []
    word_tokens = []
    current_tokens = []
    
    for token in tokens.tolist():
        current_tokens.append(token)
        decoded = tokenizer_.decode(current_tokens, skip_special_tokens=True)
#         decoded = tokenizer.decode_with_timestamps(current_tokens)
        if "\ufffd" not in decoded:
            words.append(decoded)
            word_tokens.append(current_tokens)
            current_tokens = []
#     print(f"\n\n\nWords{words}\nWord_tokens:{word_tokens}\n\n\n")
    
    return words, word_tokens

def split_tokens_on_spaces(tokens: torch.Tensor):
    subwords, subword_tokens_list = split_tokens_on_unicode(tokens)
    words = []
    word_tokens = []
    
    for subword, subword_tokens in zip(subwords, subword_tokens_list):
        special = subword_tokens[0] >= tokenizer.eot
        with_space = subword.startswith(" ")
        punctuation = subword.strip() in string.punctuation
        if special or with_space or punctuation:
            words.append(subword)
            word_tokens.append(subword_tokens)
        else:
            words[-1] = words[-1] + subword
            word_tokens[-1].extend(subword_tokens)
    
    return words, word_tokens

def load_audio(audio_file_path):
    audio, r = librosa.load(audio_file_path)
    audio = librosa.resample(audio, orig_sr=r, target_sr=16000, res_type="kaiser_best")
    if len(audio)>48000:
        audio_chunks = split_audio(audio)
    else:
        audio_chunks = [audio]
    return audio_chunks

def split_audio(audio, sample_rate=16000):
    """
    Split an audio time series array into chunks of 30 seconds.

    :param audio: 1D numpy array representing the audio time series
    :param sample_rate: sampling rate of the audio (default 44100 Hz)
    :return: a list of 1D numpy arrays, each representing a 30-second chunk of audio
    """
    chunk_size = sample_rate * 30  # number of samples in a 30-second chunk
    num_chunks = (len(audio) // chunk_size)+1  # number of 30-second chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunks.append(audio[start:end])
    return chunks





# dir = "/home/coder/whisper/whisper_processor"
dir = "/Users/charug18/Drive/Work/PhD/Research Visit/Code/web_interface/whisper_processor"
tokenizer = get_tokenizer(True, language="Norwegian")
tokenizer_ = WhisperTokenizer.from_pretrained(dir, language="Norwegian", task="transcribe")
# tokenizer_.save_pretrained("whisper_tokenizer")
tokenizer_.predict_timestamps = True
feature_extractor = AutoFeatureExtractor.from_pretrained(dir)
processor = AutoProcessor.from_pretrained(dir)
model = AutoModelForSpeechSeq2Seq.from_pretrained(dir)
# model.save_pretrained("whisper_model")
QKs = [None] * model.config.num_hidden_layers
decoder = model.get_decoder()
for i, layer in enumerate(decoder.layers):
    layer.encoder_attn.register_forward_hook(
            lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1])
        )

def transcribe(microphone, file_upload):
    warn_output = ""
    if (microphone is not None) and (file_upload is not None):
        warn_output = (
            "WARNING: You've uploaded an audio file and used the microphone. "
            "The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
        )

    elif (microphone is None) and (file_upload is None):
        return "ERROR: You have to either use the microphone or upload an audio file"

    file = microphone if microphone is not None else file_upload

    audio_chunks = load_audio(file)

    # text = get_transcription_frompipe(file)
    words = []
    start_times =[]
    end_times=[]
    text = ""
    for batch_number, audio in enumerate(audio_chunks):
    #     print(batch_number)
        words_, st, et, transcription = get_ts(audio, batch_number)
        words.append(words_)
        start_times.append(st)
        end_times.append(et)
        text = " ".join([text, transcription])

    words = list(itertools.chain.from_iterable(words))
    start_times = list(itertools.chain.from_iterable(start_times))
    end_times = list(itertools.chain.from_iterable(end_times))

    data = [
        dict(word=word, begin=begin, end=end)
        for word, begin, end in zip(words, start_times, end_times)
        if not word.endswith("|>") and word.strip() not in ".,!?、。"
    ]

    # data = prettify_data(data)

    ts = pd.DataFrame(data)
    # ts = pd.DataFrame({'A' : []})
    print(f"Data_frame: {ts}")
    return {txtbox: text, df: ts}

# def prettify_data(data):
#     for i, point in enumerate(data):
#         if point["word"].endswith("-"):
#             point["word"] = point["word"] + data[i+1]["word"]
#             point["end"]

#         elif point["word"].startswith("-"):

#         else:
#             continue

def load_model(model_name="NbAiLab/whisper-tiny-nob"):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    tokenizer_hf = WhisperTokenizer.from_pretrained(model_name, language="Norwegian", task="transcribe")
    tokenizer_hf.predict_timestamps = True
    tokenizer_oai = get_tokenizer(True, language="Norwegian")
    processor = AutoProcessor.from_pretrained(model_name)
#     feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    return model, tokenizer_hf, tokenizer_oai, processor

def get_transcription(model, processor,audio):
    input_features = processor(audio, return_tensors="pt").input_features
    generated_ids = model.generate(inputs=input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return input_features, transcription

def get_ts(audio, batch_number):
    duration = len(audio)
    model, tokenizer_hf, tokenizer_oai, processor = load_model()
    input_features, transcription = get_transcription(model, processor,audio)
    decoder = model.get_decoder()
    
    #install attention hooks
    mel = input_features
    QKs = [None] * model.config.num_hidden_layers
    hooks = [
        block.encoder_attn.register_forward_hook(
            lambda _, ins, outs, index=i: QKs.__setitem__(index, outs)
        )
    for i, block in enumerate(decoder.layers)]
    
    #create tokes
    text_tokens = torch.tensor(tokenizer_hf.encode(transcription, add_special_tokens=True)) 
    tokens = torch.tensor(
        [
            *tokenizer_oai.sot_sequence,
            tokenizer_oai.timestamp_begin,
        ] + tokenizer_hf.encode(transcription, add_special_tokens=False) + [
            tokenizer_oai.eot
        ]
    )
    
    #extract logits
    with torch.no_grad():
        logits = model(mel, tokens.unsqueeze(0), output_attentions=True)[0]
        logits = logits.squeeze(0)
        token_probs = logits.softmax(dim=-1)
        text_token_probs = token_probs[np.arange(len(text_tokens)), text_tokens].tolist()

    for hook in hooks:
        hook.remove()

        #get weights
    QKs_=[]
    for qk in QKs:
        temp = qk[1]
        QKs_.append(temp)

    weights = torch.cat(QKs_)

    weights = weights[ :, :, :, : duration // AUDIO_SAMPLES_PER_TOKEN]
    weights = median_filter(weights, (1, 1, 1, medfilt_width))
    weights = torch.tensor(weights * qk_scale).softmax(dim=-1)
    weights = weights / weights.norm(dim=-2, keepdim=True)
    
    #calculate matrix
    matrix = weights.mean(axis=(0, 1))
#     matrix = matrix[len(tokenizer_oai.sot_sequence)-1: ]
    matrix = torch.tensor(matrix)
    
    # text_indices, time_indices = dtw_oai(-matrix.double().numpy())
    alignments = dtw(-matrix.double().numpy())

    #calculate word tokens and word boundries
    words, word_tokens = split_tokens_on_spaces(text_tokens)
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens]), (1, 0))
    
    jumps = np.pad(np.diff(alignments.index1s), (1, 0), constant_values=1).astype(bool)
    jump_times = alignments.index2s[jumps] * AUDIO_TIME_PER_TOKEN
    
    start_times = jump_times[word_boundaries[:-1]]+(batch_number*30)
    end_times = jump_times[word_boundaries[1:]]+(batch_number*30)
    
    
    return words, start_times, end_times, transcription


def Insert_row_(row_number, df, row_value):
    # Slice the upper half of the dataframe
    df1 = df[0:row_number]
    df2 = df[row_number:]
    df1.loc[row_number] = row_value
    df = pd.concat([df1, df2])
    df.index = [*range(df.shape[0])]
    return df

def add_new_words(new_words, indices, df):
    for i, w in enumerate(new_words):
        df = Insert_row_(indices[i], df, [w, "NaN", "NaN"])
    return df

def correct_word(tokens, words, df):
    for i in range(len(tokens)):
        if tokens[i]!=words[i]:
            df.at[i, "word"] = tokens[i]
    return df

def edit_text(txt, df):
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    tokens =  txt.split()
    words = list(df.iloc[:, 0])
    words = [word.strip() for word in words]
    new_words = [word for word in tokens if word not in words]
    print(f"Printing from edit_text: {len(tokens), len(words), len(new_words)}")
    return df
    if len(new_words) == 0:
        return df
    else:
        if len(tokens) == len(words):
            correct_word(tokens, words, df)
        else:
            for i in range(len(new_words)):
                inde = tokens.index(new_words[i])
                # if tokens[i]!=words[i]:
                df = Insert_row_(inde, df, [new_words[i] , "NaN", "NaN"])

            old_words = [word for word in words if word not in tokens]
            print(f"words: {words}\ntokens: {tokens}\nold_words: {old_words}")
            for i in range(len(old_words)):
                inde = words.index(old_words[i]) + len(new_words)
                df.drop([df.index[inde]], inplace=True)
                # words = df.iloc[:, 0]
                
        return {df: df }

def download_file(df):
    file_name = "timestamped_transcription.json"
    df.to_json(file_name, orient="split", index=False)
    return {fi: gr.update(visible=True, value=file_name) }


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input=[
            gr.inputs.Audio(source="microphone", type="filepath", optional=True),
            gr.inputs.Audio(source="upload", type="filepath", optional=True),
        ]
            with gr.Row():
                button_clr = gr.Button("Clear")
                button_smt = gr.Button("Submit", variant="primary")
        with gr.Column():
            txtbox = gr.Textbox(lines=10, interactive=True)
            # button_update_df = gr.Button("Update Timestamps").style(full_width=True)
            df =        gr.Dataframe(interactive=False, overflow_row_behaviour="show_ends", wrap=True)
            

            with gr.Column():
            # with gr.Row():
                button_dwnld = gr.Button("Download TS").style(full_width=True)
                fi = gr.File(label="timestamped_transcription.json", visible=False)
                    # output_file = [gr.File(label="Output File", file_count="single", file_types=["", ".json"])]
    # button_update_df.click(edit_text, [txtbox, df], df)
    button_smt.click(transcribe, input, outputs=[txtbox, df])
    button_dwnld.click(download_file, df, fi)

demo.launch(enable_queue=True)