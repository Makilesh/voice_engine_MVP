from typing import Any, Dict
import os
import wave
import numpy as np

def save_audio_file(file_path: str, audio_data: np.ndarray, sample_rate: int):
    """Save audio data to a WAV file."""
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

def load_audio_file(file_path: str) -> Dict[str, Any]:
    """Load audio data from a WAV file."""
    with wave.open(file_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
    return {
        'audio_data': audio_array,
        'sample_rate': sample_rate
    }

def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """Normalize audio data to the range [-1.0, 1.0]."""
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        return audio_data / max_val
    return audio_data

def trim_silence(audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Trim silence from the beginning and end of audio data."""
    non_silent_indices = np.where(np.abs(audio_data) > threshold)[0]
    if non_silent_indices.size > 0:
        start_index = non_silent_indices[0]
        end_index = non_silent_indices[-1] + 1
        return audio_data[start_index:end_index]
    return audio_data  # Return original if no non-silent parts found

def audio_to_mono(audio_data: np.ndarray) -> np.ndarray:
    """Convert stereo audio data to mono by averaging the channels."""
    if audio_data.ndim == 2 and audio_data.shape[1] == 2:
        return np.mean(audio_data, axis=1)
    return audio_data  # Return original if already mono