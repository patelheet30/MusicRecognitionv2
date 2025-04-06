import sys
from pathlib import Path

backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

import config  # noqa: E402
from src.audio.processor import AudioProcessor  # noqa: E402
from src.fingerprinting.peaks import PeakFinder  # noqa: E402
from src.utils.visualisation import SpectrogramVisualiser  # noqa: E402


def plot_peaks():
    processor = AudioProcessor(config.SAMPLE_RATE, config.MONO)

    audio = processor.load_audio(
        "../data/songs/The Weeknd - Starboy (Audio) ft. Daft Punk.wav"
    )

    peaks = PeakFinder(
        config.N_FFT,
        config.HOP_LENGTH,
        config.WINDOW,
        config.NEIGHBOURHOOD_SIZE,
        config.THRESHOLD_ABS,
        config.MIN_PEAK_DISTANCE,
        config.MAX_PEAKS_TOTAL,
        config.MAX_PEAKS_PER_FRAME,
        config.MIN_FREQ,
        config.MAX_FREQ,
        config.FREQ_BINS,
    )

    audio = audio[: int(config.SAMPLE_RATE * 30)]

    spectrogram, freqs, times, peaks = peaks.process_audio(audio)

    visualiser = SpectrogramVisualiser()

    visualiser.plot_peaks(spectrogram, freqs, times, peaks, "Peaks", limit=True)
    visualiser.show_plot()


if __name__ == "__main__":
    plot_peaks()
