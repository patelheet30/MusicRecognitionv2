import logging
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(self, sample_rate: int = 44100, mono: bool = True):
        self.sample_rate = sample_rate
        self.mono = mono
        logger.info(
            "Initialised AudioProcessor with sample_rate: %d, mono: %s",
            sample_rate,
            mono,
        )

    def load_audio(self, audio_path: str | Path) -> np.ndarray:
        try:
            audio, sr = librosa.load(
                str(audio_path),
                sr=self.sample_rate,
                mono=self.mono,
            )
            logger.debug(f"Loaded audio from {audio_path}")
            return audio
        except Exception as e:
            logger.error(f"Error loading audio from {audio_path}: {e}")
            raise
