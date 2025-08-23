
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import pickle
import datetime
import pandas as pd

SR = 16000
DURATION = 4.0  # seconds
MODEL_PATH = os.path.join('models', 'voice_emotion_model.pkl')
LOG_CSV = 'voice_emotion_logs.csv'

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train and save from the notebook.")

with open(MODEL_PATH, 'rb') as f:
    payload = pickle.load(f)
pipe = payload['pipeline']
EMOTIONS = payload['emotions']
SR = payload.get('sr', SR)

# Ensure log exists
if not os.path.exists(LOG_CSV):
    pd.DataFrame(columns=['timestamp','source','emotion','female_gate','note']).to_csv(LOG_CSV, index=False)

def log_entry(source, emotion, female_gate, note=''):
    t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = {'timestamp': t, 'source': source, 'emotion': emotion, 'female_gate': int(female_gate), 'note': note}
    pd.DataFrame([row]).to_csv(LOG_CSV, mode='a', header=False, index=False)

def extract_features(y, sr=SR):
    y = librosa.util.fix_length(y, int(sr*max(1.0, len(y)/sr)))
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    try:
        f0 = librosa.yin(y, fmin=80, fmax=400, sr=sr)
        f0_stats = [np.nanmean(f0), np.nanstd(f0)]
    except Exception:
        f0_stats = [0.0, 0.0]
    feat = np.hstack([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        sc.mean(), sc.std(),
        roll.mean(), roll.std(),
        zcr.mean(), zcr.std(),
        f0_stats
    ]).astype(np.float32)
    return feat.reshape(1, -1)

def is_female_voice(y, sr=SR):
    try:
        f0 = librosa.yin(y, fmin=80, fmax=400, sr=sr)
        f0_med = np.nanmedian(f0)
        return (165 <= f0_med <= 255), f0_med
    except Exception:
        return False, np.nan

def predict_emotion(y, sr=SR):
    X = extract_features(y, sr)
    idx = int(pipe.predict(X)[0])
    return EMOTIONS[idx]

# GUI actions
def record_voice():
    try:
        messagebox.showinfo("Recording", f"Recording for {DURATION} seconds. Please speak...")
        y = sd.rec(int(DURATION*SR), samplerate=SR, channels=1, dtype='float32')
        sd.wait()
        y = y.flatten()
        female, f0 = is_female_voice(y, SR)
        if not female:
            log_entry('record', None, False, note=f'f0_med={f0:.1f}Hz')
            messagebox.showwarning("Female-only", "Non-female voice detected. Please upload/record a female voice.")
            return
        emo = predict_emotion(y, SR)
        log_entry('record', emo, True, note=f'f0_med={f0:.1f}Hz')
        messagebox.showinfo("Emotion", f"Detected emotion: {emo}\n(f0≈{f0:.0f} Hz)")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def upload_file():
    path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav;*.mp3;*.flac;*.ogg")])
    if not path: return
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
        female, f0 = is_female_voice(y, sr)
        if not female:
            log_entry(os.path.basename(path), None, False, note=f'f0_med={f0:.1f}Hz')
            messagebox.showwarning("Female-only", "Non-female voice detected. Please upload a female voice.")
            return
        emo = predict_emotion(y, sr)
        log_entry(os.path.basename(path), emo, True, note=f'f0_med={f0:.1f}Hz')
        messagebox.showinfo("Emotion", f"Detected emotion: {emo}\n(f0≈{f0:.0f} Hz)")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Build GUI
root = tk.Tk()
root.title("Voice Emotion Detection (Female-only)")
root.geometry("420x220")

title = tk.Label(root, text="Emotion Detection through Voice", font=("Arial", 14, "bold"))
title.pack(pady=10)

btn_rec = tk.Button(root, text="🎙️ Record Voice", command=record_voice, width=25, height=2)
btn_rec.pack(pady=10)

btn_upload = tk.Button(root, text="⬆️ Upload Voice Note", command=upload_file, width=25, height=2)
btn_upload.pack(pady=5)

note = tk.Label(root, text="Only female voices are accepted.\nLogs: voice_emotion_logs.csv", font=("Arial", 10))
note.pack(pady=5)

root.mainloop()
