import argparse
import json
import os
import shutil
import subprocess

from pydub import AudioSegment
from tqdm import tqdm

# Change this constant to filter by a different keyword (including brackets)
FILTER_KEYWORD = "(Official Audio)"


class YouTubeMusicDownloader:
    def __init__(self, output_dir="data/songs"):
        """
        Initialize the YouTube Music Downloader using yt-dlp.

        Args:
            output_dir (str): Directory to save downloaded files
        """
        self.output_dir = output_dir

        # Check if yt-dlp is installed
        try:
            subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
            print("yt-dlp is installed and ready to use")
        except (subprocess.SubprocessError, FileNotFoundError):
            print("yt-dlp is not installed. Please install it with: pip install yt-dlp")
            exit(1)

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Create temp directory for downloads
        self.temp_dir = os.path.join(output_dir, "temp")
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def get_channel_videos(self, channel_url):
        """
        Get videos from a YouTube channel using yt-dlp.

        Args:
            channel_url (str): URL of the YouTube channel

        Returns:
            list: List of video information dictionaries
        """
        print(f"Fetching videos from: {channel_url}")

        try:
            # Use yt-dlp to get channel video information in JSON format
            # --flat-playlist gets just the video info without downloading
            # -j outputs in JSON format
            cmd = ["yt-dlp", "--flat-playlist", "-j", "--no-warnings", channel_url]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error fetching channel videos: {result.stderr}")
                return []

            # Parse JSON data (one JSON object per line)
            videos = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    try:
                        video_info = json.loads(line)
                        videos.append(video_info)
                    except json.JSONDecodeError:
                        continue

            print(f"Found {len(videos)} videos")
            return videos

        except Exception as e:
            print(f"Error accessing channel: {str(e)}")
            return []

    def get_video_details(self, video_id):
        """
        Get detailed information about a specific video.

        Args:
            video_id (str): YouTube video ID

        Returns:
            dict: Video details
        """
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            cmd = ["yt-dlp", "-j", "--no-warnings", video_url]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error getting video details: {result.stderr}")
                return None

            video_info = json.loads(result.stdout)
            return video_info

        except Exception as e:
            print(f"Error getting video details: {str(e)}")
            return None

    def download_audio(self, video_id, output_path):
        """
        Download the audio from a YouTube video and convert to WAV.

        Args:
            video_id (str): YouTube video ID
            output_path (str): Path to save the WAV file

        Returns:
            bool: True if successful, False otherwise
        """
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            # Create a temporary filename for the download
            temp_file = os.path.join(self.temp_dir, f"{video_id}.m4a")

            # Download audio using yt-dlp
            cmd = [
                "yt-dlp",
                "-f",
                "bestaudio[ext=m4a]",  # Get best audio quality in m4a format
                "-o",
                temp_file,
                "--no-warnings",
                video_url,
            ]

            print(f"Downloading audio for video {video_id}...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error downloading audio: {result.stderr}")
                return False

            # Check if the file exists
            if not os.path.exists(temp_file):
                print(f"Download failed: File {temp_file} not found")
                return False

            # Convert to WAV using pydub
            print(f"Converting to WAV for video {video_id}...")
            audio = AudioSegment.from_file(temp_file)
            audio.export(output_path, format="wav")

            # Remove the temporary file
            os.remove(temp_file)

            print(f"Successfully downloaded and converted to WAV: {output_path}")
            return True

        except Exception as e:
            print(f"Error in download process: {str(e)}")
            return False

    def interactive_download(self, channel_url):
        """
        Interactive mode to select videos from a channel and then download them in batch.
        Only videos with the FILTER_KEYWORD in the title are presented.

        Args:
            channel_url (str): URL of the YouTube channel
        """
        videos = self.get_channel_videos(channel_url)

        if not videos:
            print("No videos found or error accessing channel.")
            return

        # Collect videos that you want to download
        download_tasks = []
        for i, video in enumerate(videos):
            try:
                video_id = video.get("id")
                title = video.get("title", "Unknown Title")

                # Filter videos by the keyword in the title
                if FILTER_KEYWORD not in title:
                    continue

                duration_str = video.get("duration_string", "Unknown duration")

                print(f"\n[{i + 1}/{len(videos)}] {title}")
                print(
                    f"Duration: {duration_str} | URL: https://youtube.com/watch?v={video_id}"
                )

                choice = input("Download this video? (y/n/q to quit): ").strip().lower()

                if choice == "q":
                    print("Quitting selection...")
                    break

                if choice == "y":
                    download_tasks.append((video_id, title))
            except Exception as e:
                print(f"Error processing video: {str(e)}")

        if not download_tasks:
            print("No videos selected for download.")
            return

        print(f"\nStarting download for {len(download_tasks)} selected video(s)...")
        downloaded_count = 0
        for video_id, title in download_tasks:
            # Use the original title for the filename (no sanitization)
            output_path = os.path.join(self.output_dir, f"{title}.wav")
            if self.download_audio(video_id, output_path):
                downloaded_count += 1

        # Cleanup temp directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        print(
            f"\nDownload completed. {downloaded_count} files were downloaded to {self.output_dir}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="YouTube Music Downloader for Audio Fingerprinting Dataset"
    )
    parser.add_argument("--channel", type=str, help="YouTube channel URL")
    parser.add_argument(
        "--output",
        type=str,
        default="data/songs",
        help="Output directory for downloaded files",
    )

    args = parser.parse_args()

    downloader = YouTubeMusicDownloader(output_dir=args.output)

    if args.channel:
        channel_url = args.channel
    else:
        channel_url = input("Enter YouTube channel URL: ")

    # Ensure proper URL format
    if not channel_url.startswith("http"):
        if "@" in channel_url:
            channel_url = f"https://www.youtube.com/{channel_url}"
        else:
            channel_url = f"https://www.youtube.com/@{channel_url}"

    print(f"Using channel URL: {channel_url}")
    downloader.interactive_download(channel_url)


if __name__ == "__main__":
    main()
