import os
import numpy as np
import argparse
from pydub import AudioSegment
from scipy.signal import get_window
from scipy.io.wavfile import write as wav_write

def segment_audio(file_path, output_folder, num_segments=100, overlap=0.1, smooth=True, duration=None):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load audio file and convert to consistent format (int16) if necessary
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_sample_width(2)  # 16-bit audio (int16)
    sample_rate = audio.frame_rate
    data = np.array(audio.get_array_of_samples(), dtype=np.int16)
    
    # Calculate total samples and determine segment parameters
    total_samples = len(data)
    if duration:
        # Calculate num_segments based on duration if provided
        segment_length_in_samples = int(duration * sample_rate)
        num_segments = max(total_samples // segment_length_in_samples, 1)
    else:
        # Calculate segment length if only num_segments is provided
        segment_length_in_samples = int(total_samples / num_segments * (1 + overlap))

    step_size = int(total_samples / num_segments)

    for i in range(num_segments):
        # Extract segment with optional overlap
        start_idx = i * step_size
        end_idx = min(start_idx + segment_length_in_samples, total_samples)
        segment = data[start_idx:end_idx]
        
        # Optional smoothing
        if smooth:
            fade_length = int(0.05 * segment_length_in_samples)  # 5% fade in/out
            window = get_window("hann", fade_length * 2)
            fade_in = window[:fade_length]
            fade_out = window[-fade_length:]
            segment[:fade_length] = (segment[:fade_length] * fade_in).astype(np.int16)
            segment[-fade_length:] = (segment[-fade_length:] * fade_out).astype(np.int16)
        
        # Write segment to file
        segment_path = os.path.join(output_folder, f"segment_{i + 1}.wav")
        wav_write(segment_path, sample_rate, segment)

    print(f"Audio segmented into {num_segments} parts in {output_folder}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment audio file into overlapping parts with optional smoothing.")
    parser.add_argument("file_path", type=str, help="Path to the input audio file.")
    parser.add_argument("output_folder", type=str, help="Directory to save the segments.")
    parser.add_argument("--num_segments", type=int, default=100, help="Number of segments to create.")
    parser.add_argument("--overlap", type=float, default=0.1, help="Fraction of overlap between segments.")
    parser.add_argument("--smooth", action="store_true", help="Apply smoothing to the segment edges.")
    parser.add_argument("--duration", type=float, help="Approximate duration of output in seconds.")
    args = parser.parse_args()
    
    segment_audio(args.file_path, args.output_folder, args.num_segments, args.overlap, args.smooth, args.duration)
