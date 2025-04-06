"""Fingerprint generation module for audio fingerprinting."""

import logging
import random
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Fingerprint:
    """Fingerprint dataclass to store hash and time offset."""

    hash: int
    time_offset: float


class FingerprintGenerator:
    """Generates fingerprints from peaks."""

    def __init__(
        self,
        fan_value: int = 15,
        min_time_delta: float = 0.0,
        max_time_delta: float = 50.0,
        hash_bits: int = 32,
        freq_bin_count: int = 32,
        sample_rate: int = 44100,
        hop_length: int = 512,
    ):
        """
        Initialize fingerprint generator.

        Args:
            fan_value: Number of target points to consider for each anchor point
            min_time_delta: Minimum time difference between anchor and target (ms)
            max_time_delta: Maximum time difference between anchor and target (ms)
            hash_bits: Number of bits for the hash value
            freq_bin_count: Number of frequency bins for quantization
            sample_rate: Audio sample rate
            hop_length: Hop length used in spectrogram generation
        """
        self.fan_value = fan_value
        self.min_time_delta = min_time_delta
        self.max_time_delta = max_time_delta
        self.hash_bits = hash_bits
        self.freq_bin_count = freq_bin_count
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        logger.info(
            f"Initialized FingerprintGenerator with fan_value={fan_value}, "
            f"min_time_delta={min_time_delta}, max_time_delta={max_time_delta}, "
            f"hash_bits={hash_bits}, freq_bin_count={freq_bin_count}, "
            f"sample_rate={sample_rate}, hop_length={hop_length}"
        )

    def generate_fingerprint(
        self, peaks: np.ndarray, freqs: np.ndarray, times: np.ndarray
    ) -> List[Fingerprint]:
        """
        Generate fingerprints from peaks using an improved algorithm with better
        hash distribution and more robust matching properties.

        Args:
            peaks: Peak coordinates (freq_idx, time_idx)
            freqs: Frequency array
            times: Time array

        Returns:
            List of fingerprints
        """
        if len(peaks) == 0:
            logger.warning("No peaks provided, returning empty fingerprint list")
            return []

        logger.info(f"Generating fingerprints from {len(peaks)} peaks")

        peak_freqs = freqs[peaks[:, 0]]
        peak_times = times[peaks[:, 1]]

        freq_bins = self._freq_to_bin(peak_freqs, self.freq_bin_count)

        fingerprints = []

        for i, (anchor_time, anchor_freq, anchor_freq_bin) in enumerate(
            zip(peak_times, peak_freqs, freq_bins)
        ):
            min_time = anchor_time + (self.min_time_delta / 1000.0)
            max_time = anchor_time + (self.max_time_delta / 1000.0)

            target_indices = np.where(
                (peak_times > min_time) & (peak_times < max_time)
            )[0]

            if len(target_indices) == 0:
                continue

            if len(target_indices) > self.fan_value:
                sorted_by_time = sorted(
                    target_indices, key=lambda idx: peak_times[idx] - anchor_time
                )[: self.fan_value // 2]

                remaining = [idx for idx in target_indices if idx not in sorted_by_time]
                if remaining:
                    random_targets = random.sample(
                        remaining,
                        min(len(remaining), self.fan_value - len(sorted_by_time)),
                    )
                    target_indices = sorted_by_time + random_targets

            for target_idx in target_indices:
                target_time = peak_times[target_idx]
                target_freq = peak_freqs[target_idx]
                target_freq_bin = freq_bins[target_idx]

                time_delta = (target_time - anchor_time) * 1000
                time_delta_bin = min(
                    2**10 - 1, int(time_delta / (self.max_time_delta / (2**10)))
                )

                freq_delta = abs(target_freq - anchor_freq)
                freq_delta_bin = min(2**10 - 1, int(freq_delta / 50))

                hash1 = self._generate_hash_v1(
                    anchor_freq_bin,
                    target_freq_bin,  # type: ignore
                    time_delta_bin,
                )

                hash2 = self._generate_hash_v2(
                    anchor_freq_bin, freq_delta_bin, time_delta_bin
                )

                fingerprints.append(Fingerprint(hash=hash1, time_offset=anchor_time))
                fingerprints.append(Fingerprint(hash=hash2, time_offset=anchor_time))

        logger.info(f"Generated {len(fingerprints)} fingerprints")
        return fingerprints

    def _freq_to_bin(self, freqs: np.ndarray, num_bins: int) -> np.ndarray:
        """
        Convert frequencies to logarithmic bins.

        Args:
            freqs: Frequency values
            num_bins: Number of bins

        Returns:
            Binned frequency values
        """
        min_freq = 20.0  # Hz
        max_freq = 20000.0  # Hz
        bounded_freqs = np.clip(freqs, min_freq, max_freq)

        # Convert to log scale
        log_min = np.log(min_freq)
        log_max = np.log(max_freq)

        log_freqs = np.log(bounded_freqs)
        bin_values = (
            (log_freqs - log_min) / (log_max - log_min) * (num_bins - 1)
        ).astype(int)

        return np.clip(bin_values, 0, num_bins - 1)

    def _generate_hash_v1(
        self, anchor_freq_bin: int, target_freq_bin: int, time_delta_bin: int
    ) -> int:
        """
        Generate hash from anchor and target frequency bins and time delta.
        Uses absolute frequency bin values.

        Args:
            anchor_freq_bin: Frequency bin of anchor point
            target_freq_bin: Frequency bin of target point
            time_delta_bin: Time delta bin

        Returns:
            Integer hash value
        """
        hash_value = (
            (anchor_freq_bin & 0xFFF) << 22
            | (target_freq_bin & 0xFFF) << 10
            | (time_delta_bin & 0x3FF)
        )

        return hash_value

    def _generate_hash_v2(
        self, anchor_freq_bin: int, freq_delta_bin: int, time_delta_bin: int
    ) -> int:
        """
        Generate hash from anchor frequency, frequency delta, and time delta.
        Uses frequency difference rather than absolute target frequency.

        Args:
            anchor_freq_bin: Frequency bin of anchor point
            freq_delta_bin: Bin of frequency difference between anchor and target
            time_delta_bin: Time delta bin

        Returns:
            Integer hash value
        """
        hash_value = (
            (anchor_freq_bin & 0xFFF) << 22
            | (freq_delta_bin & 0xFFF) << 10
            | (time_delta_bin & 0x3FF)
        )

        hash_value |= 1 << 31

        return hash_value
