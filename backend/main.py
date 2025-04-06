import argparse
import glob
import logging
import multiprocessing
import os
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Optional

backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

import config as config  # noqa: E402
from src.audio.processor import AudioProcessor  # noqa: E402
from src.database.manager import DatabaseManager, Match  # noqa: E402
from src.fingerprinting.generator import Fingerprint, FingerprintGenerator  # noqa: E402
from src.fingerprinting.peaks import PeakFinder  # noqa: E402


def setup_logging(log_file=None):
    handlers = [logging.StreamHandler()]

    if log_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)  # type: ignore

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def get_db_path(base_path: str, db_index: int) -> str:
    """Generate a database path with index"""
    if base_path is None:
        db_dir = config.DATABASE_DIR
        db_name = config.DB_FILENAME
        # Split the filename and extension
        name_parts = os.path.splitext(db_name)
        return str(db_dir / f"{name_parts[0]}_{db_index}{name_parts[1]}")
    else:
        # If custom path provided, add index to it
        name_parts = os.path.splitext(base_path)
        return f"{name_parts[0]}_{db_index}{name_parts[1]}"


def get_all_db_paths(base_path: str) -> List[str]:
    """Get all database paths matching the pattern"""
    if base_path is None:
        db_dir = config.DATABASE_DIR
        db_name = config.DB_FILENAME
        name_parts = os.path.splitext(db_name)
        pattern = str(db_dir / f"{name_parts[0]}_*{name_parts[1]}")
    else:
        name_parts = os.path.splitext(base_path)
        pattern = f"{name_parts[0]}_*{name_parts[1]}"

    return sorted(glob.glob(pattern))


def fingerprint_song(
    file_path: str,
    song_name: Optional[str],
    db_path: str,
    chunk_size: int = 30,
) -> int:
    logger = logging.getLogger(__name__)
    file_path = Path(file_path)  # type: ignore

    if not file_path.exists():  # type: ignore
        logger.error(f"File not found: {file_path}")
        return -1

    if song_name is None:
        song_name = file_path.stem  # type: ignore

    audio_proc = AudioProcessor(sample_rate=config.SAMPLE_RATE, mono=config.MONO)

    peak_finder = PeakFinder(
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        window=config.WINDOW,
        neighborhood_size=config.NEIGHBOURHOOD_SIZE,
        threshold_abs=config.THRESHOLD_ABS,
        min_peak_distance=config.MIN_PEAK_DISTANCE,
        max_peaks_total=config.MAX_PEAKS_TOTAL,
        max_peaks_per_frame=config.MAX_PEAKS_PER_FRAME,
        min_frequency=config.MIN_FREQ,
        max_frequency=config.MAX_FREQ,
        freq_bins=config.FREQ_BINS,
    )

    fingerprint_gen = FingerprintGenerator(
        fan_value=config.FAN_VALUE,
        min_time_delta=config.MIN_TIME_DELTA,
        max_time_delta=config.MAX_TIME_DELTA,
        hash_bits=config.HASH_BITS,
        freq_bin_count=config.FREQ_BIN_COUNT,
        sample_rate=config.SAMPLE_RATE,
        hop_length=config.HOP_LENGTH,
    )

    # Find the database with the least songs
    db_paths = get_all_db_paths(db_path)
    db_to_use = None
    min_songs = float("inf")

    # If no databases exist yet, create the first one
    if not db_paths:
        db_to_use = get_db_path(db_path, 1)
        logger.info(f"Creating new database: {db_to_use}")
    else:
        # Check each database's song count
        for db in db_paths:
            try:
                db_manager = DatabaseManager(database_path=db)
                stats = db_manager.get_database_stats()
                if hasattr(db_manager, "close"):
                    db_manager.close()

                song_count = stats.get("songs", 0)
                if (
                    song_count < min_songs and song_count < 35
                ):  # Limit to 35 songs per DB
                    min_songs = song_count
                    db_to_use = db
            except Exception as e:
                logger.error(f"Error accessing database {db}: {e}")

        # If all databases are full, create a new one
        if db_to_use is None or min_songs >= 35:
            next_index = len(db_paths) + 1
            db_to_use = get_db_path(db_path, next_index)
            logger.info(f"Creating new database: {db_to_use}")

    logger.info(f"Using database: {db_to_use}")
    db_manager = DatabaseManager(database_path=db_to_use)

    logger.info(f"Processing song: {song_name} from {file_path}")
    start_time = time.time()
    audio = audio_proc.load_audio(file_path)
    logger.info(f"Audio loaded in {time.time() - start_time:.2f} seconds")

    duration = len(audio) / config.SAMPLE_RATE
    logger.info(f"Audio duration: {duration:.2f} seconds")

    all_fingerprints: List[Fingerprint] = []

    num_chunks = int(duration / chunk_size) + 1

    for i in range(num_chunks):
        start = i * chunk_size * config.SAMPLE_RATE
        end = min(len(audio), (i + 1) * chunk_size * config.SAMPLE_RATE)

        if start >= len(audio):
            break

        chunk = audio[start:end]
        chunk_duration = len(chunk) / config.SAMPLE_RATE

        logger.info(f"Processing chunk {i + 1}/{num_chunks} ({chunk_duration:.2f}s)")

        chunk_start_time = time.time()
        spectrogram, freqs, times, peaks = peak_finder.process_audio(chunk)
        logger.info(
            f"Peak finding completed in {time.time() - chunk_start_time:.2f} seconds"
        )
        logger.info(f"Found {len(peaks)} peaks")

        fp_start_time = time.time()

        chunk_offset = i * chunk_size
        adjusted_times = times + chunk_offset

        fingerprints = fingerprint_gen.generate_fingerprint(
            peaks, freqs, adjusted_times
        )
        logger.info(
            f"Fingerprint generation completed in {time.time() - fp_start_time:.2f} seconds"
        )
        logger.info(f"Generated {len(fingerprints)} fingerprints")

        all_fingerprints.extend(fingerprints)

    logger.info(f"Adding {len(all_fingerprints)} fingerprints to database")
    song_id = db_manager.add_song(song_name, str(file_path), all_fingerprints)  # type: ignore
    logger.info(f"Song added to database with ID: {song_id}")

    if hasattr(db_manager, "close"):
        db_manager.close()

    return song_id


def process_song_worker(file_path, chunk_size):
    logger = logging.getLogger(__name__)
    try:
        song_name = file_path.stem
        logger.info(f"Processing file: {file_path.name}")

        audio_proc = AudioProcessor(sample_rate=config.SAMPLE_RATE, mono=config.MONO)
        peak_finder = PeakFinder(
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            window=config.WINDOW,
            neighborhood_size=config.NEIGHBOURHOOD_SIZE,
            threshold_abs=config.THRESHOLD_ABS,
            min_peak_distance=config.MIN_PEAK_DISTANCE,
            max_peaks_total=config.MAX_PEAKS_TOTAL,
            max_peaks_per_frame=config.MAX_PEAKS_PER_FRAME,
            min_frequency=config.MIN_FREQ,
            max_frequency=config.MAX_FREQ,
            freq_bins=config.FREQ_BINS,
        )
        fingerprint_gen = FingerprintGenerator(
            fan_value=config.FAN_VALUE,
            min_time_delta=config.MIN_TIME_DELTA,
            max_time_delta=config.MAX_TIME_DELTA,
            hash_bits=config.HASH_BITS,
            freq_bin_count=config.FREQ_BIN_COUNT,
            sample_rate=config.SAMPLE_RATE,
            hop_length=config.HOP_LENGTH,
        )

        start_time = time.time()
        audio = audio_proc.load_audio(file_path)
        logger.info(f"Audio loaded in {time.time() - start_time:.2f} seconds")

        duration = len(audio) / config.SAMPLE_RATE
        logger.info(f"Audio duration: {duration:.2f} seconds")

        all_fingerprints = []
        num_chunks = int(duration / chunk_size) + 1

        for j in range(num_chunks):
            start = j * chunk_size * config.SAMPLE_RATE
            end = min(len(audio), (j + 1) * chunk_size * config.SAMPLE_RATE)

            if start >= len(audio):
                break

            chunk = audio[start:end]
            chunk_duration = len(chunk) / config.SAMPLE_RATE
            logger.info(
                f"Processing chunk {j + 1}/{num_chunks} ({chunk_duration:.2f}s)"
            )

            spectrogram, freqs, times, peaks = peak_finder.process_audio(chunk)
            logger.info(f"Found {len(peaks)} peaks")

            chunk_offset = j * chunk_size
            adjusted_times = times + chunk_offset

            fingerprints = fingerprint_gen.generate_fingerprint(
                peaks, freqs, adjusted_times
            )
            logger.info(f"Generated {len(fingerprints)} fingerprints")

            all_fingerprints.extend(fingerprints)

        logger.info(
            f"Finished processing '{song_name}' with {len(all_fingerprints)} fingerprints"
        )

        return (str(file_path), song_name, all_fingerprints)

    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")
        return None


def search_database_worker(args):
    """Worker function for parallel database searching"""
    db_path, fingerprints, threshold = args
    logger = logging.getLogger(__name__)

    try:
        db_manager = DatabaseManager(database_path=db_path)

        match_start_time = time.time()
        matches = db_manager.find_matches(fingerprints, threshold=threshold)
        search_time = time.time() - match_start_time

        if hasattr(db_manager, "close"):
            db_manager.close()

        return db_path, matches, search_time
    except Exception as e:
        logger.error(f"Error searching database {db_path}: {e}")
        return db_path, [], 0


def identify_song(
    file_path: str,
    db_path: str = None,  # type: ignore
    duration: float = 10.0,
    threshold: float = None,  # type: ignore
    verbose: bool = False,
    max_workers: int = None,  # type: ignore
) -> List[Match] | None:
    logger = logging.getLogger(__name__)
    file_path = Path(file_path)  # type: ignore

    if not file_path.exists():  # type: ignore
        logger.error(f"File not found: {file_path}")
        return

    # Set default max_workers if not specified
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 4)

    audio_proc = AudioProcessor(sample_rate=config.SAMPLE_RATE, mono=config.MONO)

    peak_finder = PeakFinder(
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        window=config.WINDOW,
        neighborhood_size=config.NEIGHBOURHOOD_SIZE,
        threshold_abs=config.THRESHOLD_ABS,
        min_peak_distance=config.MIN_PEAK_DISTANCE,
        max_peaks_total=config.MAX_PEAKS_TOTAL,
        max_peaks_per_frame=config.MAX_PEAKS_PER_FRAME,
        min_frequency=config.MIN_FREQ,
        max_frequency=config.MAX_FREQ,
        freq_bins=config.FREQ_BINS,
    )

    fingerprint_gen = FingerprintGenerator(
        fan_value=config.FAN_VALUE,
        min_time_delta=config.MIN_TIME_DELTA,
        max_time_delta=config.MAX_TIME_DELTA,
        hash_bits=config.HASH_BITS,
        freq_bin_count=config.FREQ_BIN_COUNT,
        sample_rate=config.SAMPLE_RATE,
        hop_length=config.HOP_LENGTH,
    )

    # Get list of all database files
    db_paths = get_all_db_paths(db_path)
    if not db_paths:
        logger.error("No database files found")
        return None

    logger.info(f"Found {len(db_paths)} database files to search")

    if threshold is None:
        threshold = config.MATCH_THRESHOLD

    logger.info(f"Loading query audio: {file_path}")
    start_time = time.time()
    audio = audio_proc.load_audio(file_path)
    logger.info(f"Audio loaded in {time.time() - start_time:.2f} seconds")

    if duration is not None and duration > 0:
        samples = int(duration * config.SAMPLE_RATE)
        if samples < len(audio):
            logger.info(f"Using first {duration:.2f} seconds of audio")
            audio = audio[:samples]

    logger.info("Extracting peaks")
    peak_start_time = time.time()
    spectrogram, freqs, times, peaks = peak_finder.process_audio(audio)
    logger.info(
        f"Peak finding completed in {time.time() - peak_start_time:.2f} seconds"
    )
    logger.info(f"Found {len(peaks)} peaks")

    logger.info("Generating fingerprints")
    fp_start_time = time.time()
    fingerprints = fingerprint_gen.generate_fingerprint(peaks, freqs, times)
    logger.info(
        f"Fingerprint generation completed in {time.time() - fp_start_time:.2f} seconds"
    )
    logger.info(f"Generated {len(fingerprints)} fingerprints")

    # Search across all databases in parallel
    all_matches = []
    logger.info(
        f"Searching across {len(db_paths)} databases using {max_workers} workers"
    )

    search_start_time = time.time()

    # Prepare arguments for parallel processing
    search_args = [(db, fingerprints, threshold) for db in db_paths]

    with multiprocessing.Pool(processes=max_workers) as pool:
        results = pool.map(search_database_worker, search_args)

    logger.info(
        f"All database searches completed in {time.time() - search_start_time:.2f} seconds"
    )

    # Process results
    for db_path, matches, search_time in results:
        if matches:
            db_name = os.path.basename(db_path)
            logger.info(
                f"Found {len(matches)} potential matches in {db_name} (search time: {search_time:.2f}s)"
            )
            all_matches.extend(matches)

    # Early termination if we have a very strong match
    # This can speed up identification by returning as soon as we find a confident match
    fast_match_threshold = 0.90  # 90% confidence threshold for early return
    if all_matches:
        best_match = max(all_matches, key=lambda m: m.confidence)
        if best_match.confidence >= fast_match_threshold:
            logger.info(
                f"Found high-confidence match ({best_match.confidence:.2%}), terminating search early"
            )
            return [best_match]

    # Sort all matches by confidence
    if all_matches:
        all_matches.sort(key=lambda m: m.confidence, reverse=True)

        # Take top matches (or all if fewer)
        top_matches = all_matches[:10]

        logger.info(f"Found {len(all_matches)} matches across all databases")
        logger.info("Top matches:")

        for i, match in enumerate(top_matches):
            match_info = (
                f"{i + 1}. {match.song_name} "
                f"(Confidence: {match.confidence:.2%}, Offset: {match.offset:.2f}s"
            )

            if hasattr(match, "match_count"):
                match_info += f", Matches: {match.match_count}"

            match_info += ")"
            print(match_info)

        return top_matches
    else:
        logger.info("No matches found in any database")
        return None


def fingerprint_directory(
    directory: str,
    db_path: str,
    extensions: List[str],
    chunk_size: int = 30,
    max_workers: int = 1,  # type: ignore
    songs_per_db: int = 35,
) -> None:
    logger = logging.getLogger(__name__)
    directory = Path(directory)  # type: ignore

    if not directory.exists() or not directory.is_dir():  # type: ignore
        logger.error(f"Directory not found: {directory}")
        return

    if extensions is None:
        extensions = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]

    audio_files = []
    for ext in extensions:
        # Recursively find files in all subdirectories
        audio_files.extend(directory.glob(f"**/*{ext}"))  # type: ignore

    if not audio_files:
        logger.error(f"No audio files found in {directory}")
        return

    logger.info(f"Found {len(audio_files)} audio files")

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 4)

    logger.info(f"Processing with {max_workers} parallel workers")

    start_time = time.time()
    all_results = []

    with multiprocessing.Pool(processes=max_workers) as pool:
        process_func = partial(process_song_worker, chunk_size=chunk_size)

        for i, result in enumerate(
            pool.imap_unordered(process_func, sorted(audio_files))
        ):
            if result:
                all_results.append(result)

            if (i + 1) % 10 == 0 or (i + 1) == len(audio_files):
                logger.info(
                    f"Processed {i + 1}/{len(audio_files)} files ({(i + 1) / len(audio_files) * 100:.1f}%)"
                )

    processing_time = time.time() - start_time
    logger.info(f"Parallel processing completed in {processing_time:.2f} seconds")
    logger.info(f"Successfully processed {len(all_results)}/{len(audio_files)} files")

    # Calculate how many databases we need
    total_songs = len(all_results)
    num_databases = (total_songs + songs_per_db - 1) // songs_per_db  # Ceiling division
    logger.info(
        f"Distributing {total_songs} songs across {num_databases} databases ({songs_per_db} songs per database)"
    )

    # Group songs into database batches
    db_batches = {}
    for i, result in enumerate(all_results):
        db_index = (i // songs_per_db) + 1
        if db_index not in db_batches:
            db_batches[db_index] = []
        db_batches[db_index].append(result)

    # Process each database
    for db_index, songs in db_batches.items():
        actual_db_path = get_db_path(db_path, db_index)
        logger.info(f"Creating database {db_index}/{num_databases}: {actual_db_path}")

        db_manager = DatabaseManager(database_path=actual_db_path)
        db_start_time = time.time()

        for file_path, song_name, fingerprints in songs:
            try:
                logger.info(
                    f"Adding '{song_name}' with {len(fingerprints)} fingerprints to database {db_index}"
                )
                db_manager.add_song(song_name, file_path, fingerprints)
            except Exception as e:
                logger.error(f"Error adding {song_name} to database {db_index}: {e}")

        db_time = time.time() - db_start_time
        logger.info(
            f"Database {db_index} completed in {db_time:.2f} seconds with {len(songs)} songs"
        )

        if hasattr(db_manager, "close"):
            db_manager.close()

    logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")


def show_db_stats(db_path: str) -> None:
    logger = logging.getLogger(__name__)

    # Get all database files
    db_paths = get_all_db_paths(db_path)

    if not db_paths:
        logger.error("No database files found")
        return

    logger.info(f"Found {len(db_paths)} database files")

    total_songs = 0
    total_fingerprints = 0

    for i, db in enumerate(db_paths):
        logger.info(f"Database {i + 1}: {db}")
        db_manager = DatabaseManager(database_path=db)

        stats = db_manager.get_database_stats()
        logger.info(f"  Statistics: {stats}")

        total_songs += stats.get("num_songs", 0)
        total_fingerprints += stats.get("num_fingerprints", 0)

        if hasattr(db_manager, "close"):
            db_manager.close()

    logger.info(
        f"Total across all databases: {total_songs} songs, {total_fingerprints} fingerprints"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Fingerprinting System")
    subparsers = parser.add_subparsers(dest="command")

    fp_parser = subparsers.add_parser("fingerprint", help="Fingerprint a song")
    fp_parser.add_argument("path", help="Path to audio file or directory")
    fp_parser.add_argument("--name", help="Song name (for single files)")
    fp_parser.add_argument("--db", help="Path to database file")
    fp_parser.add_argument("--dir", action="store_true", help="Process directory")
    fp_parser.add_argument(
        "--chunk-size", type=int, default=30, help="Chunk size in seconds"
    )
    fp_parser.add_argument("--log", "-l", action="store_true", help="Save logs to file")
    fp_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for directory processing",
    )
    fp_parser.add_argument(
        "--songs-per-db",
        type=int,
        default=35,
        help="Maximum number of songs per database file",
    )

    id_parser = subparsers.add_parser("identify", help="Identify a song")
    id_parser.add_argument("path", help="Path to audio file")
    id_parser.add_argument("--db", help="Base path to database files")
    id_parser.add_argument(
        "--duration", type=float, default=10.0, help="Duration to analyze in seconds"
    )
    id_parser.add_argument("--threshold", type=float, help="Matching threshold")
    id_parser.add_argument("--log", "-l", action="store_true", help="Save logs to file")
    id_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed match information"
    )
    id_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for database searching",
    )

    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.add_argument("--db", help="Base path to database files")
    stats_parser.add_argument(
        "--log", "-l", action="store_true", help="Save logs to file"
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = None

    if args.command and getattr(args, "log", False):
        log_file = f"logs/{args.command}_{timestamp}.log"

    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    if args.command == "fingerprint":
        if args.dir:
            fingerprint_directory(
                args.path,
                args.db,
                None,  # type: ignore
                args.chunk_size,
                args.workers,
                getattr(args, "songs_per_db", 35),
            )
        else:
            fingerprint_song(args.path, args.name, args.db, args.chunk_size)
    elif args.command == "identify":
        identify_song(
            args.path,
            args.db,
            args.duration,
            args.threshold,
            getattr(args, "verbose", False),
            getattr(args, "workers", None),  # type: ignore
        )
    elif args.command == "stats":
        show_db_stats(args.db)
    else:
        parser.print_help()
