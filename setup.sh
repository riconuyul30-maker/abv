#!/bin/bash

echo "================================================"
echo "   Gaming Clipper Bot - Installation Script"
echo "================================================"
echo ""

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -qq

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get install -y ffmpeg python3-pip python3-venv libsndfile1 -qq

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip -q

# Install Python dependencies
echo "📚 Installing Python packages (this may take 5-10 minutes)..."
echo "   - Installing telegram bot library..."
pip install python-telegram-bot==20.7 -q

echo "   - Installing Whisper AI (speech recognition)..."
pip install openai-whisper -q

echo "   - Installing video processing libraries..."
pip install moviepy opencv-python -q

echo "   - Installing audio analysis libraries..."
pip install librosa soundfile -q

echo "   - Installing PyTorch (CPU version)..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu -q

echo "   - Installing other dependencies..."
pip install numpy scipy pillow python-dotenv -q

echo ""
echo "================================================"
echo "   ✅ Installation Complete!"
echo "================================================"
echo ""
echo "📋 Next steps:"
echo "   1. Make sure your .env file has correct bot token"
echo "   2. Run: source venv/bin/activate"
echo "   3. Run: python bot.py"
echo ""
echo "🚀 Ready to process gaming videos!"
echo ""
