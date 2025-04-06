import os
import subprocess
import time


def main():
    input_file = "artist_albums.txt"
    download_dir = "/Volumes/Heet's Passport/Songs"

    os.makedirs(download_dir, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f_in:
        for line_number, line in enumerate(f_in, 1):
            line = line.strip()
            parts = line.split(" - ")
            if len(parts) >= 3:
                artist_name = parts[0]
                album_name = parts[1]
                album_id = parts[2]

                safe_filename = (
                    f"{artist_name} - {album_name}".replace("/", "-")
                    .replace("\\", "-")
                    .replace(":", "-")
                    .replace("*", "")
                    .replace("?", "")
                    .replace('"', "")
                    .replace("<", "")
                    .replace(">", "")
                    .replace("|", "")
                )

                print(
                    f"Processing album {line_number}: {album_name} by {artist_name} (ID: {album_id})"
                )

                album_url = f"https://open.spotify.com/album/{album_id}"
                save_file_path = f"{safe_filename}.spotdl"

                try:
                    cmd = [
                        "spotdl",
                        "download",
                        album_url,
                        "--bitrate",
                        "320k",
                        "--output",
                        f"{download_dir}/{{artists}} - {{title}}",
                        "--save-file",
                        save_file_path,
                    ]

                    print(f"Running command: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)

                    print(f"Successfully downloaded: {album_name} by {artist_name}")
                    print(f"Album info saved to: {save_file_path}")

                    time.sleep(1)

                except Exception as e:
                    print(f"Error downloading album {album_name}: {str(e)}")
            else:
                print(f"Invalid line format (line {line_number}): {line}")


if __name__ == "__main__":
    main()
