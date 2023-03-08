import numpy as np
import torch
import pandas as pd
import whisper
import gradio as gr
import string


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

def get_ts(audio, transcription):
    inputs = feature_extractor(audio, return_tensors="pt")
    input_features = inputs.input_features
#     print(len(audio))
    print(transcription)
  
    duration = len(audio)
    mel = input_features
#     print(mel.shape)
    tokens = torch.tensor(
        [
            *tokenizer.sot_sequence,
            tokenizer.timestamp_begin,
        ] + tokenizer_.encode(transcription) + [
            tokenizer.timestamp_begin + duration // AUDIO_SAMPLES_PER_TOKEN,
            tokenizer.eot,
        ]
    )
    with torch.no_grad():
        logits = model(mel, tokens.unsqueeze(0))
    QKs_= []
    for qk in QKs:
        sh = qk[0].shape
        temp = qk[0].reshape([sh[0],sh[1], sh[3], sh[2]])
#         print(len(qk))
        QKs_.append(temp)
    weights = torch.cat(QKs_)  # layers * heads * tokens * frames    
    weights = weights[:, :, :, : duration // AUDIO_SAMPLES_PER_TOKEN].cpu()
    weights = median_filter(weights, (1, 1, 1, medfilt_width))
    weights = torch.tensor(weights * qk_scale).softmax(dim=-1)
    
    w = weights / weights.norm(dim=-2, keepdim=True)
    matrix = w[-6:].mean(axis=(0, 1))

    alignment = dtw(-matrix.double().numpy())

    jumps = np.pad(np.diff(alignment.index1s), (1, 0), constant_values=1).astype(bool)
    jump_times = alignment.index2s[jumps] * AUDIO_TIME_PER_TOKEN
    words, word_tokens = split_tokens_on_spaces(tokens)
#     print(words,word_tokens)

    # display the word-level timestamps in a table
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
#     print(f"word boundaries: {words[:-1]}")
    begin_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]

    data = [
        dict(word=word, begin=begin, end=end)
        for word, begin, end in zip(words[:-1], begin_times, end_times)
        if not word.startswith("<|") and word.strip() not in ".,!?、。"
    ]

    data = pd.DataFrame(data)
    # display(data)
    return data



# def get_transcription_frompipe(y):
#     transcriber = pipeline(model="NbAiLab/whisper-tiny-nob")
#     transcriptions = transcriber(y)["text"]
#     return transcriptions

def get_transcription_fromlocal(y, model) :
    # model = AutoModelForSpeechSeq2Seq.from_pretrained("NbAiLab/whisper-tiny-nob")
    inputs = feature_extractor(y, return_tensors="pt")
    input_features = inputs.input_features
    generated_ids  = model.generate(inputs=input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

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

    y, r = librosa.load(file)
    y = librosa.resample(y, orig_sr=r, target_sr=16000, res_type="kaiser_best")

    # text = get_transcription_frompipe(file)
    text = get_transcription_fromlocal(y, model)
    text = text.translate(str.maketrans('', '', string.punctuation))
    print(text)
    ts = get_ts(y, text)
    
    return {txtbox: text, df: gr.update(max_rows=5, value=ts)}

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
    tokens =  txt.split()
    words = list(df.iloc[:, 0])
    words = [word.strip() for word in words]
    new_words = [word for word in tokens if word not in words]
    
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
            
    return df

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
            df =        gr.Dataframe(interactive=False, max_rows=5, overflow_row_behaviour="show_ends", wrap=True)

            with gr.Column():
            # with gr.Row():
                button_dwnld = gr.Button("Download TS").style(full_width=True)
                fi = gr.File(label="timestamped_transcription.json", visible=False)
                    # output_file = [gr.File(label="Output File", file_count="single", file_types=["", ".json"])]
    txtbox.change(edit_text, [txtbox, df], df)
    button_smt.click(transcribe, input, outputs=[txtbox, df])
    button_dwnld.click(download_file, df, fi)

demo.launch(enable_queue=True)