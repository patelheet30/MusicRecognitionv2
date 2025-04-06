import logging

import librosa
import numpy as np
from scipy.ndimage import generate_binary_structure, iterate_structure, maximum_filter

logger = logging.getLogger(__name__)


class PeakFinder:
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        window: str,
        neighborhood_size: int,
        threshold_abs: int,
        min_peak_distance: int,
        max_peaks_total: int,
        max_peaks_per_frame: int,
        min_frequency: int,
        max_frequency: int,
        freq_bins: int,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.neighborhood_size = neighborhood_size
        self.threshold_abs = threshold_abs
        self.min_peak_distance = min_peak_distance
        self.max_peaks_total = max_peaks_total
        self.max_peaks_per_frame = max_peaks_per_frame
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.freq_bins = freq_bins

        logger.info(
            f"Initialized PeakFinder with n_fft: {n_fft}, hop_length: {hop_length}, "
            f"window: {window}, neighborhood_size: {neighborhood_size}, "
            f"threshold_abs: {threshold_abs:.2f}, min_peak_distance: {min_peak_distance}, "
            f"max_peaks_total: {max_peaks_total}, max_peaks_per_frame: {max_peaks_per_frame}, "
            f"freq_range: {min_frequency}-{max_frequency}Hz, freq_bins: {freq_bins}"
        )

    def generate_spectrogram(self, audio_signal):
        try:
            D = librosa.stft(
                audio_signal,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
            )

            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            freqs = librosa.fft_frequencies(sr=44100, n_fft=self.n_fft)
            times = librosa.times_like(S_db, sr=44100, hop_length=self.hop_length)

            logger.info(f"Generated spectrogram with shape {S_db.shape}")

            return S_db, freqs, times

        except Exception as e:
            logger.error(f"Error generating spectrogram: {str(e)}")
            raise

    def find_peaks(self, spectrogram, freqs):
        try:
            struct = generate_binary_structure(2, 1)
            neighborhood = iterate_structure(struct, self.neighborhood_size)

            local_max = (
                maximum_filter(spectrogram, footprint=neighborhood) == spectrogram
            )

            is_peak = (local_max) & (spectrogram > self.threshold_abs)

            peak_indices = np.where(is_peak)

            if len(peak_indices[0]) == 0:
                logger.warning("No peaks found!")
                return np.empty((0, 2), dtype=int)

            peak_coords = np.column_stack(peak_indices)

            min_freq_idx = np.searchsorted(freqs, self.min_frequency)
            max_freq_idx = np.searchsorted(freqs, self.max_frequency, side="right")
            freq_mask = (peak_coords[:, 0] >= min_freq_idx) & (
                peak_coords[:, 0] < max_freq_idx
            )
            peak_coords = peak_coords[freq_mask]

            if len(peak_coords) == 0:
                logger.warning("No peaks found in the specified frequency range!")
                return np.empty((0, 2), dtype=int)

            peak_amplitudes = spectrogram[peak_coords[:, 0], peak_coords[:, 1]]

            peak_coords = self._apply_freq_binning(peak_coords, peak_amplitudes, freqs)

            peak_coords = self._limit_peaks_per_frame(peak_coords, spectrogram)

            if len(peak_coords) > self.max_peaks_total:
                logger.info(
                    f"Limiting peaks from {len(peak_coords)} to {self.max_peaks_total}"
                )
                peak_coords = peak_coords[: self.max_peaks_total]

            logger.info(f"Final peak count: {len(peak_coords)}")
            return peak_coords

        except Exception as e:
            logger.error(f"Error finding peaks: {str(e)}")
            raise

    def _apply_freq_binning(self, peaks, amplitudes, freqs):
        min_log_freq = np.log10(max(self.min_frequency, 20))
        max_log_freq = np.log10(self.max_frequency)
        freq_bin_edges = np.logspace(min_log_freq, max_log_freq, self.freq_bins + 1)

        binned_peaks = []

        peak_freqs = freqs[peaks[:, 0]]

        bin_indices = np.digitize(peak_freqs, freq_bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.freq_bins - 1)

        for bin_idx in range(self.freq_bins):
            bin_mask = bin_indices == bin_idx
            if not np.any(bin_mask):
                continue

            bin_peaks = peaks[bin_mask]
            bin_amplitudes = amplitudes[bin_mask]

            sorted_indices = np.argsort(bin_amplitudes)[::-1]
            max_per_bin = max(5, self.max_peaks_total // self.freq_bins)
            bin_selected = bin_peaks[sorted_indices[:max_per_bin]]

            binned_peaks.append(bin_selected)

        if binned_peaks:
            return np.vstack(binned_peaks)
        else:
            return peaks

    def _limit_peaks_per_frame(self, peaks, spectrogram):
        if len(peaks) == 0 or self.max_peaks_per_frame <= 0:
            return peaks

        time_frames = np.unique(peaks[:, 1])

        selected_peaks = []

        for t in time_frames:
            frame_mask = peaks[:, 1] == t
            frame_peaks = peaks[frame_mask]

            if len(frame_peaks) <= self.max_peaks_per_frame:
                selected_peaks.append(frame_peaks)
                continue

            frame_amplitudes = spectrogram[frame_peaks[:, 0], frame_peaks[:, 1]]

            sorted_indices = np.argsort(frame_amplitudes)[::-1]
            frame_selected = frame_peaks[sorted_indices[: self.max_peaks_per_frame]]

            selected_peaks.append(frame_selected)

        if selected_peaks:
            return np.vstack(selected_peaks)
        else:
            return np.empty((0, 2), dtype=int)

    def _filter_by_distance_grid(self, peaks, spectrogram):
        n_freq, n_time = spectrogram.shape

        amplitudes = spectrogram[peaks[:, 0], peaks[:, 1]]
        sorted_idx = np.argsort(amplitudes)[::-1]
        peaks = peaks[sorted_idx]

        grid = np.zeros((n_freq, n_time), dtype=bool)

        min_dist = max(1, self.min_peak_distance)

        selected_peaks = []

        for peak in peaks:
            f_idx, t_idx = peak

            f_min = max(0, f_idx - min_dist)
            f_max = min(n_freq, f_idx + min_dist + 1)
            t_min = max(0, t_idx - min_dist)
            t_max = min(n_time, t_idx + min_dist + 1)

            if np.any(grid[f_min:f_max, t_min:t_max]):
                continue

            grid[f_min:f_max, t_min:t_max] = True

            selected_peaks.append(peak)

            if len(selected_peaks) >= self.max_peaks_total:
                break

        if selected_peaks:
            return np.array(selected_peaks)
        else:
            return np.empty((0, 2), dtype=int)

    def process_audio(self, audio_signal):
        spectrogram, freqs, times = self.generate_spectrogram(audio_signal)

        peaks = self.find_peaks(spectrogram, freqs)

        return spectrogram, freqs, times, peaks
