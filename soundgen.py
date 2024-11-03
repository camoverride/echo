from pydub import AudioSegment
from pydub.generators import Sine, Square, WhiteNoise
import os
import random

# Create a directory for audio files if it doesn't exist
os.makedirs("audio", exist_ok=True)

def generate_sound(filename, duration_ms=1000):
    # Start with white noise
    noise = WhiteNoise().to_audio_segment(duration=duration_ms)
    
    # Modulate with a randomly pitched sine or square wave
    modulation_freq = random.randint(100, 1000)
    modulator_type = random.choice([Sine, Square])
    modulator = modulator_type(modulation_freq).to_audio_segment(duration=duration_ms)

    # Mix the noise with the modulator at various random intensities
    sound = noise.overlay(modulator, gain_during_overlay=random.uniform(-10, 10))
    
    # Add random distortion and reverb effects
    sound = sound.high_pass_filter(random.randint(500, 3000)).low_pass_filter(random.randint(2000, 5000))
    sound = sound + random.randint(-5, 5)  # Random volume adjustment

    # Export the sound file
    sound.export(filename, format="wav")

# Generate 6x6 sounds
for i in range(10):
    for j in range(10):
        filename = f"audio/sound_{i}_{j}.wav"
        generate_sound(filename)
        print(f"Generated {filename}")
