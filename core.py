import os
import re
import numpy as np
import scipy.io.wavfile as wav
from transformers import pipeline
from pydub import AudioSegment
from gtts import gTTS

# Initialize model
lyric_gen = pipeline("text-generation", model="gpt2")

def generate_lyrics(prompt):
    full_prompt = (
        f"Write a full emotional and poetic English song about: '{prompt}'.\n"
        "The song should contain:\n"
        "- Meaningful verses\n"
        "- Rhyming lines\n"
        "- A chorus (repeat once)\n"
        "- Total at least 10 lines\n\n"
        "Song:\n"
    )
    output = lyric_gen(
        full_prompt,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1
    )
    lyrics = output[0]["generated_text"]
    lyrics = lyrics.replace(full_prompt, "")
    lyrics = re.sub(r"\n{2,}", "\n", lyrics).strip()
    return lyrics

def generate_melodic_background(duration=60):
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    notes = [220.00, 261.63, 329.63, 440.00]
    melody = np.zeros_like(t)
    note_duration = 0.5
    samples_per_note = int(note_duration * sr)

    for i in range(0, len(t), samples_per_note):
        note = notes[(i // samples_per_note) % len(notes)]
        melody[i:i+samples_per_note] = 0.5 * np.sin(2 * np.pi * note * t[i:i+samples_per_note])

    fade_len = int(sr * 2)
    melody[:fade_len] *= np.linspace(0, 1, fade_len)
    melody[-fade_len:] *= np.linspace(1, 0, fade_len)

    os.makedirs("output", exist_ok=True)
    melody_file = "output/melodic_background.wav"
    wav.write(melody_file, sr, melody.astype(np.float32))
    return melody_file

def generate_singing_voice(lyrics):
    os.makedirs("output", exist_ok=True)
    tts = gTTS(lyrics)
    tts.save("output/voice.mp3")
    sound = AudioSegment.from_mp3("output/voice.mp3")
    voice_path = "output/voice.wav"
    sound.export(voice_path, format="wav")
    return voice_path

def combine_audio(melody_path, voice_path):
    melody = AudioSegment.from_file(melody_path)
    voice = AudioSegment.from_file(voice_path)

    if len(melody) < len(voice):
        melody = melody * ((len(voice) // len(melody)) + 1)
    melody = melody[:len(voice)]
    voice = voice[:len(melody)]

    final = melody + 4
    voice = voice + 2
    final_audio = final.overlay(voice)

    final_path = "output/final_song.wav"
    final_audio.export(final_path, format="wav")
    return final_path
