#!/usr/bin/env python3
"""Test if all required packages are installed"""

import sys

def test_imports():
    results = []
    packages = {
        'telegram': 'python-telegram-bot',
        'whisper': 'openai-whisper',
        'moviepy.editor': 'moviepy',
        'cv2': 'opencv-python',
        'librosa': 'librosa',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'PIL': 'pillow',
        'dotenv': 'python-dotenv',
        'torch': 'torch',
        'torchaudio': 'torchaudio'
    }
    
    print("=" * 50)
    print("Testing Package Installation")
    print("=" * 50)
    print()
    
    all_ok = True
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"✓ {package:25s} OK")
            results.append((package, True))
        except ImportError:
            print(f"✗ {package:25s} MISSING")
            results.append((package, False))
            all_ok = False
    
    print()
    print("=" * 50)
    
    if all_ok:
        print("✅ All packages installed successfully!")
        print()
        print("You can now run: python bot.py")
    else:
        print("❌ Some packages are missing.")
        print()
        print("Install missing packages with:")
        for package, installed in results:
            if not installed:
                print(f"  pip install --break-system-packages {package}")
    
    print("=" * 50)
    return all_ok

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
