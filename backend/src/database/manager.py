import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.fingerprinting.generator import Fingerprint

logger = logging.getLogger(__name__)


@dataclass
class Song:
    id: int
    name: str
    path: str


@dataclass
class Match:
    song_id: int
    song_name: str
    confidence: float
    offset: float
    match_count: int


class DatabaseManager:
    def __init__(self, database_path: str | Path = "fingerprints.db"):
        self.database_path = Path(database_path)
        self.database_dir = self.database_path.parent

        self.database_dir.mkdir(exist_ok=True, parents=True)

        try:
            # Connect to SQLite database
            self.conn = sqlite3.connect(str(self.database_path))
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()

            # Create tables if they don't exist
            self._create_tables()

            # Get next song ID
            self.cursor.execute("SELECT MAX(id) FROM songs")
            result = self.cursor.fetchone()
            self.next_song_id = (result[0] or 0) + 1

            # Create indices for faster lookups if they don't exist
            self._create_indices()

            logger.info(
                f"Initialized DatabaseManager with database_path: {database_path}"
            )

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def _create_tables(self):
        """Create database tables if they don't exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT NOT NULL
            )
        """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS fingerprints (
                hash INTEGER NOT NULL,
                song_id INTEGER NOT NULL,
                time_offset REAL NOT NULL,
                FOREIGN KEY (song_id) REFERENCES songs(id)
            )
        """)

        self.conn.commit()

    def _create_indices(self):
        """Create indices for faster lookups."""
        # Check if indices exist
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_fingerprints_hash'"
        )
        if not self.cursor.fetchone():
            self.cursor.execute(
                "CREATE INDEX idx_fingerprints_hash ON fingerprints(hash)"
            )
            logger.info("Created index on fingerprints.hash")

        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_fingerprints_song_id'"
        )
        if not self.cursor.fetchone():
            self.cursor.execute(
                "CREATE INDEX idx_fingerprints_song_id ON fingerprints(song_id)"
            )
            logger.info("Created index on fingerprints.song_id")

        self.conn.commit()

    def add_song(
        self, song_name: str, song_path: str, fingerprints: List[Fingerprint]
    ) -> int:
        """
        Add a song to the database.

        Args:
            song_name: Name of the song
            song_path: Path to the song file
            fingerprints: List of fingerprints

        Returns:
            Song ID
        """
        # Get next song ID
        song_id = self.next_song_id
        self.next_song_id += 1

        try:
            # Begin transaction
            self.conn.execute("BEGIN TRANSACTION")

            # Add song to songs table
            self.cursor.execute(
                "INSERT INTO songs (id, name, path) VALUES (?, ?, ?)",
                (song_id, song_name, song_path),
            )

            # Add fingerprints to fingerprints table
            fp_batch = [(fp.hash, song_id, fp.time_offset) for fp in fingerprints]
            self.cursor.executemany(
                "INSERT INTO fingerprints (hash, song_id, time_offset) VALUES (?, ?, ?)",
                fp_batch,
            )

            # Commit transaction
            self.conn.commit()

            logger.info(
                f"Added song '{song_name}' (ID: {song_id}) with {len(fingerprints)} fingerprints"
            )
            return song_id

        except Exception as e:
            # Rollback transaction in case of error
            self.conn.rollback()
            logger.error(f"Error adding song '{song_name}' to database: {e}")
            raise

    def find_matches(
        self, fingerprints: List[Fingerprint], threshold: float = 0.05
    ) -> List[Match]:
        """
        Find matches for a set of fingerprints using improved matching algorithm.

        Args:
            fingerprints: List of fingerprints to match
            threshold: Minimum confidence threshold

        Returns:
            List of matches
        """
        if not fingerprints:
            logger.warning("No fingerprints provided for matching")
            return []

        logger.info(f"Finding matches for {len(fingerprints)} fingerprints")
        logger.info(f"Threshold: {threshold}")

        try:
            # Extract hashes from query fingerprints
            query_hashes = [(fp.hash, fp.time_offset) for fp in fingerprints]

            # Create a temporary table for query fingerprints
            self.cursor.execute("DROP TABLE IF EXISTS temp_query")
            self.cursor.execute("""
                CREATE TEMPORARY TABLE temp_query (
                    hash INTEGER NOT NULL,
                    time_offset REAL NOT NULL
                )
            """)

            self.cursor.executemany(
                "INSERT INTO temp_query (hash, time_offset) VALUES (?, ?)", query_hashes
            )

            # Join temporary table with fingerprints table to find matches
            # Calculate time offsets between query and database fingerprints
            # Group by song_id and time_offset bin
            self.cursor.execute("""
                SELECT 
                    f.song_id,
                    ROUND((f.time_offset - q.time_offset) * 10) / 10 AS time_delta,
                    COUNT(*) AS match_count
                FROM 
                    temp_query q
                JOIN 
                    fingerprints f ON q.hash = f.hash
                GROUP BY 
                    f.song_id, time_delta
                ORDER BY 
                    match_count DESC
                LIMIT 100
            """)

            matches_raw = self.cursor.fetchall()

            # No matches found
            if not matches_raw:
                logger.info("No matches found in database")
                return []

            # Calculate match confidence and compile results
            song_matches = {}
            query_fingerprint_count = len(fingerprints)

            for match in matches_raw:
                song_id = match["song_id"]
                time_delta = match["time_delta"]
                match_count = match["match_count"]

                # Calculate confidence score based on match count
                confidence = match_count / query_fingerprint_count

                # Only consider matches above threshold
                if confidence >= threshold:
                    key = (song_id, time_delta)
                    song_matches[key] = (confidence, match_count)

            # Get song details for matches
            if song_matches:
                results = []
                song_ids = {song_id for (song_id, _), _ in song_matches.items()}

                # Get song information
                placeholders = ",".join(["?"] * len(song_ids))
                self.cursor.execute(
                    f"SELECT id, name FROM songs WHERE id IN ({placeholders})",
                    tuple(song_ids),
                )
                songs = {row["id"]: row["name"] for row in self.cursor.fetchall()}

                # Compile match results
                for (song_id, time_delta), (confidence, match_count) in sorted(
                    song_matches.items(), key=lambda x: x[1][0], reverse=True
                ):
                    if song_id in songs:
                        results.append(
                            Match(
                                song_id=song_id,
                                song_name=songs[song_id],
                                confidence=confidence,
                                offset=time_delta,
                                match_count=match_count,
                            )
                        )

                logger.info(f"Found {len(results)} matches")
                return results

            logger.info("No matches found above threshold")
            return []

        except Exception as e:
            logger.error(f"Error finding matches: {e}")
            return []

    def get_song(self, song_id: int) -> Optional[Song]:
        """
        Get a song by ID.

        Args:
            song_id: Song ID

        Returns:
            Song or None if not found
        """
        try:
            self.cursor.execute(
                "SELECT id, name, path FROM songs WHERE id = ?", (song_id,)
            )
            row = self.cursor.fetchone()

            if row:
                return Song(id=row["id"], name=row["name"], path=row["path"])

            return None

        except Exception as e:
            logger.error(f"Error getting song: {e}")
            return None

    def get_database_stats(self) -> Dict:
        """
        Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        try:
            # Get song count
            self.cursor.execute("SELECT COUNT(*) FROM songs")
            song_count = self.cursor.fetchone()[0]

            # Get fingerprint count
            self.cursor.execute("SELECT COUNT(*) FROM fingerprints")
            fingerprint_count = self.cursor.fetchone()[0]

            # Get unique hash count
            self.cursor.execute("SELECT COUNT(DISTINCT hash) FROM fingerprints")
            unique_hash_count = self.cursor.fetchone()[0]

            # Get avg fingerprints per song
            avg_fps_per_song = fingerprint_count / max(1, song_count)

            # Get database file size
            db_size_mb = (
                os.path.getsize(self.database_path) / (1024 * 1024)
                if self.database_path.exists()
                else 0
            )

            return {
                "num_songs": song_count,
                "num_fingerprints": fingerprint_count,
                "num_unique_hashes": unique_hash_count,
                "avg_fingerprints_per_song": avg_fps_per_song,
                "database_size_mb": db_size_mb,
            }

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}

    def clear_database(self) -> None:
        """Clear the database."""
        try:
            self.conn.execute("BEGIN TRANSACTION")
            self.cursor.execute("DELETE FROM fingerprints")
            self.cursor.execute("DELETE FROM songs")
            self.conn.commit()

            # Reset song ID counter
            self.next_song_id = 1

            logger.info("Database cleared")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error clearing database: {e}")

    def close(self):
        """Close database connection."""
        if hasattr(self, "conn") and self.conn:
            try:
                self.conn.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")

    def __del__(self):
        """Destructor to ensure database connection is closed."""
        self.close()
