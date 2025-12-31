import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Bidirectional, LSTM, Reshape, TimeDistributed, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import cv2
import numpy as np
import tempfile
from moviepy.editor import VideoFileClip
import whisper
from deep_translator import GoogleTranslator
import yt_dlp
import os

# Initialize Whisper Model
whisper_model = whisper.load_model("base")

# CTC Loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# Build Model
def build_lip_reading_model(input_shape, output_dim):
    input_data = Input(name='input', shape=input_shape, dtype='float32')

    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(input_data)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(Flatten())(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)

    y_pred = Dense(output_dim + 1, activation='softmax', name='y_pred')(x)

    labels = Input(name='labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = tf.keras.layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    prediction_model = Model(inputs=input_data, outputs=y_pred)

    return model, prediction_model

# Preprocess Video
def preprocess_video(video_path, frame_size=(100, 50), max_frames=75):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, frame_size)
        frames.append(resized)

    cap.release()
    frames = np.array(frames) / 255.0
    frames = np.expand_dims(frames, axis=-1)

    if len(frames) < max_frames:
        pad_width = ((0, max_frames - len(frames)), (0, 0), (0, 0), (0, 0))
        frames = np.pad(frames, pad_width, mode='constant')

    return np.expand_dims(frames, axis=0)

# Decode Predictions
def decode_predictions(y_pred):
    decoded = K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], greedy=False)[0][0]
    decoded = K.get_value(decoded)
    char_map = {i: chr(65 + i) for i in range(26)}
    char_map[26] = ' '

    results = []
    for seq in decoded:
        text = ''.join([char_map.get(i, '') for i in seq if i != -1])
        results.append(text)
    return results

# Translate to Tamil
def translate_to_tamil(text):
    if not text or not text.strip():
        return "No text to translate."
    try:
        translated = GoogleTranslator(source='auto', target='ta').translate(text)
        return translated
    except Exception as e:
        return f"Translation error: {str(e)}"

# Audio Extraction
def extract_audio_from_video(video_path, output_audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path, logger=None)

# Transcription
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result['text']

# Download YouTube Video
def download_video_with_ytdlp(youtube_url):
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, '%(title).200s.%(ext)s')

    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'merge_output_format': 'mp4',
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        downloaded_path = ydl.prepare_filename(info_dict)
        if not downloaded_path.endswith(".mp4"):
            downloaded_path = downloaded_path.replace('.webm', '.mp4')
        return downloaded_path

# Main Streamlit App
def main():
    st.title("🧠 AI Lip Reader with Tamil Translation 📹")
    st.write("Upload a muted/speaking video or paste a YouTube URL.")

    mode = st.radio("Choose Mode:", ["Lip Reading Model", "Wav2Lip + Whisper"], index=0)
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    youtube_url = st.text_input("Or enter a YouTube URL")

    video_path = None

    if youtube_url:
        try:
            st.write("📥 Downloading video using yt-dlp...")
            video_path = download_video_with_ytdlp(youtube_url)
            uploaded_file = open(video_path, 'rb')
            st.success("✅ Downloaded successfully.")
        except Exception as e:
            st.error(f"❌ yt-dlp download failed: {e}")

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        st.video(tfile.name)
        st.write("⏳ Processing...")

        try:
            if mode == "Lip Reading Model":
                input_shape = (75, 50, 100, 1)
                output_dim = 26
                model, prediction_model = build_lip_reading_model(input_shape, output_dim)
                video_input = preprocess_video(tfile.name)
                y_pred = prediction_model.predict(video_input)
                decoded_text = decode_predictions(y_pred)
                english_text = decoded_text[0] if decoded_text else "No text detected."
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                    extract_audio_from_video(tfile.name, temp_audio.name)
                    english_text = transcribe_audio(temp_audio.name)

            st.subheader("📜 Predicted English Text:")
            st.write(english_text)

            st.subheader("🌐 Tamil Translation:")
            tamil_translation = translate_to_tamil(english_text)
            st.write(tamil_translation)

        except Exception as e:
            st.error(f"❌ Error: {e}")

if __name__ == '__main__':
    main()
