import os
from dotenv import load_dotenv

load_dotenv()

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Video Processing Configuration
MAX_CLIP_DURATION = 60  # Maximum 60 seconds per clip
MIN_CLIP_DURATION = 10  # Minimum 10 seconds per clip
OUTPUT_RESOLUTION = (1920, 1080)  # 1080p
OUTPUT_FORMAT = 'mp4'
OUTPUT_CODEC = 'libx264'
AUDIO_CODEC = 'aac'
VIDEO_BITRATE = '8000k'  # High quality for 1080p
AUDIO_BITRATE = '192k'

# AI Detection Configuration
WHISPER_MODEL = 'base'  # Options: tiny, base, small, medium, large
EXCITEMENT_KEYWORDS = [
    'gila', 'wow', 'anjir', 'mantap', 'keren', 'bagus', 
    'savage', 'legendary', 'godlike', 'rampage', 'monster kill',
    'double kill', 'triple kill', 'ultra kill', 'mega kill',
    'killing spree', 'unstoppable', 'wipeout', 'ace',
    'headshot', 'kill', 'nice', 'gg', 'yes', 'yeah'
]

# Audio Analysis Configuration
VOLUME_THRESHOLD = 0.5  # Threshold for detecting loud moments
VOLUME_SPIKE_MULTIPLIER = 1.5  # How much louder than average
BUFFER_BEFORE = 3  # Seconds before the moment
BUFFER_AFTER = 5  # Seconds after the moment

# Paths
DOWNLOAD_PATH = 'downloads'
OUTPUT_PATH = 'outputs'
TEMP_PATH = 'temp'
