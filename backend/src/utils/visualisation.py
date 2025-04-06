import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class SpectrogramVisualiser:
    def __init__(self, figsize: tuple[int, int] = (12, 8)):
        self.figsize = figsize

    def plot_spectrogram(self, spectrogram, freqs, times, title="Spectrogram"):
        plt.figure(figsize=self.figsize)
        plt.pcolormesh(times, freqs, spectrogram, shading="gouraud")
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [sec]")
        plt.title(title)
        plt.colorbar(label="Amplitude [dB]")

    def plot_peaks(self, spectrogram, freqs, times, peaks, title="Peaks", limit=False):
        plt.figure(figsize=self.figsize)

        if limit:
            mask = freqs <= 6000
            spectrogram = spectrogram[mask, :]
            freqs = freqs[mask]

        plt.pcolormesh(times, freqs, spectrogram, shading="gouraud")

        if len(peaks) > 0:
            peak_times = times[peaks[:, 1]]
            peak_freqs = freqs[peaks[:, 0]]
            plt.scatter(peak_times, peak_freqs, color="red", s=30, label="Peaks")
            plt.legend()
        else:
            logger.warning("No peaks to plot in analysis")

        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [sec]")
        plt.title(title)
        plt.colorbar(label="Amplitude [dB]")

    def plot_analysis(
        self,
        spectrogram: np.ndarray,
        freqs: np.ndarray,
        times: np.ndarray,
        peaks: np.ndarray,
    ) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        pcm = ax1.pcolormesh(times, freqs, spectrogram, shading="gouraud")
        fig.colorbar(pcm, ax=ax1, label="Amplitude (dB)")
        ax1.set_ylabel("Frequency (Hz)")
        ax1.set_title("Spectrogram")

        pcm_2 = ax2.pcolormesh(times, freqs, spectrogram, shading="gouraud")

        if len(peaks) > 0:
            peak_times = times[peaks[:, 1]]
            peak_freqs = freqs[peaks[:, 0]]
            ax2.scatter(peak_times, peak_freqs, color="red", s=30, label="Peaks")
            ax2.legend()
        else:
            logger.warning("No peaks to plot in analysis")

        fig.colorbar(pcm_2, ax=ax2, label="Amplitude (dB)")
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_xlabel("Time (s)")
        ax2.set_title("Spectrogram with Peaks")

        plt.tight_layout()

    def save_plot(self, filename: str | Path) -> None:
        try:
            plt.savefig(filename)
            logger.info(f"Saved plot to {filename}")
        except Exception as e:
            logger.error(f"Error saving plot to {filename}: {e}")
            raise

    def show_plot(self) -> None:
        plt.show()
