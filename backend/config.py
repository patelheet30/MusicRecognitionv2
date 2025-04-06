from pathlib import Path

# Define the root directory of the project
BACKEND_ROOT = Path(__file__).parent
PROJECT_ROOT = BACKEND_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
SONG_DIR = DATA_DIR / "songs"
DATABASE_DIR = DATA_DIR / "database"

for directory in [DATA_DIR, SONG_DIR, DATABASE_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Define the audio parameters
SAMPLE_RATE = 44100
MONO = True

# Define the parameters for the spectrogram
N_FFT = 2048
HOP_LENGTH = 512
WINDOW = "hamming"

# Define the parameters for the fingerprinting
NEIGHBOURHOOD_SIZE = 5
THRESHOLD_ABS = -30
MIN_PEAK_DISTANCE = 1

# Peak Selection Parameters
MAX_PEAKS_PER_FRAME = 3
MAX_PEAKS_TOTAL = 1000
MIN_FREQ = 20
MAX_FREQ = 5000
FREQ_BINS = 32

# Define the parameters for the matching
FAN_VALUE = 15
MIN_TIME_DELTA = 0
MAX_TIME_DELTA = 200
HASH_BITS = 32
FREQ_BIN_COUNT = 32
MATCH_THRESHOLD = 0.05

# Define the database name
DB_FILENAME = "fingerprints.db"
API_FILENAME = "new_fingerprints.db"

# Define the API parameters
API_HOST = "localhost"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}"

# Define the frontend URL
FRONTEND_URL = "http://localhost:3000"
