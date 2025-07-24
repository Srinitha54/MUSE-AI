# src/run.py

import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import Audio, display
from src.core import generate_lyrics, generate_melodic_background, generate_singing_voice, combine_audio

# 🎤 Prompt
prompt = input("🎤 Enter your song prompt (e.g., 'a journey of hope'): ")

# Step 1: Lyrics
print("\n🎼 Generating lyrics...")
lyrics = generate_lyrics(prompt)
print("\n🎶 Lyrics:\n", lyrics)

# Step 2: Background + Voice
melody_path = generate_melodic_background()
voice_path = generate_singing_voice(lyrics)

# Step 3: Final Song
final_song = combine_audio(melody_path, voice_path)

print("\n✅ Final Song Created!")
display(Audio(final_song))

# Step 4: Visualize waveform
sr, audio_data = wavfile.read(final_song)
if len(audio_data.shape) == 2:
    audio_data = audio_data.mean(axis=1)

plt.figure(figsize=(12, 3))
plt.plot(audio_data[:sr * 10])
plt.title("🎵 Music Waveform (Heartbeat Style)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
