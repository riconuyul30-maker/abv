import whisper
import librosa
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter
import config

class MomentDetector:
    def __init__(self):
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model(config.WHISPER_MODEL)
    
    def preprocess_audio_for_speech(self, audio_path, output_path=None):
        """Preprocess audio to isolate human voice and reduce game sounds"""
        print("Preprocessing audio to isolate human voice...")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)  # 16kHz for Whisper
        
        # 1. Apply high-pass filter (remove low frequency game rumble/bass)
        # Human voice is typically 85-255 Hz for males, 165-255 Hz for females
        # We'll keep frequencies above 80 Hz
        from scipy.signal import butter, filtfilt
        nyquist = sr / 2
        high_cutoff = 80 / nyquist
        b, a = butter(5, high_cutoff, btype='high')
        y_filtered = filtfilt(b, a, y)
        
        # 2. Apply low-pass filter (remove high frequency game effects)
        # Human voice rarely goes above 4000 Hz
        low_cutoff = 4000 / nyquist
        b, a = butter(5, low_cutoff, btype='low')
        y_filtered = filtfilt(b, a, y_filtered)
        
        # 3. Noise reduction using spectral gating
        # Calculate spectral profile
        D = librosa.stft(y_filtered)
        magnitude = np.abs(D)
        
        # Estimate noise profile from quieter parts
        noise_profile = np.percentile(magnitude, 10, axis=1, keepdims=True)
        
        # Spectral gating - reduce frequencies below threshold
        threshold = noise_profile * 2.0  # 2x noise floor
        mask = magnitude > threshold
        D_cleaned = D * mask
        
        # Reconstruct audio
        y_denoised = librosa.istft(D_cleaned)
        
        # 4. Normalize volume
        y_denoised = librosa.util.normalize(y_denoised)
        
        # Save if output path provided
        if output_path:
            import soundfile as sf
            sf.write(output_path, y_denoised, sr)
            return output_path
        
        return y_denoised, sr
        
    def transcribe_audio(self, audio_path):
        """Transkripsi audio menggunakan Whisper dengan word-level timestamps"""
        print("Transcribing audio with Whisper (word-level timing)...")
        
        # Transcribe dengan word-level timestamps untuk timing presisi
        result = self.whisper_model.transcribe(
            audio_path,
            language='id',  # Bahasa Indonesia
            task='transcribe',
            fp16=False,
            # Word-level timestamps - CRITICAL untuk subtitle timing!
            word_timestamps=True,  # Enable word-level timing
            # Improve accuracy
            condition_on_previous_text=True,
            initial_prompt="Indonesian gaming commentary. Clear speech only.",
            temperature=0.0,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        
        # Manual filtering - buang segments dengan quality rendah
        filtered_segments = []
        for segment in result['segments']:
            # Check confidence (avg_logprob)
            confidence = segment.get('avg_logprob', -1)
            duration = segment['end'] - segment['start']
            no_speech_prob = segment.get('no_speech_prob', 0)
            
            # Get text and check if it's meaningful
            text = segment['text'].strip()
            
            # Filter rules:
            if (duration >= 0.3 and 
                confidence > -1.0 and 
                no_speech_prob < 0.6 and
                len(text) >= 2 and
                not self._is_noise_text(text)):
                
                # Also filter words if available
                if 'words' in segment and segment['words']:
                    filtered_words = []
                    for word_info in segment['words']:
                        word_text = word_info.get('word', '').strip()
                        word_prob = word_info.get('probability', 1.0)
                        
                        # Keep high-confidence words only
                        if word_text and word_prob > 0.5:
                            filtered_words.append(word_info)
                    
                    segment['words'] = filtered_words
                
                filtered_segments.append(segment)
                print(f"  ✅ Kept: '{text}' (conf={confidence:.2f}, dur={duration:.1f}s, {len(segment.get('words', []))} words)")
            else:
                print(f"  ❌ Filtered: '{text}' (conf={confidence:.2f}, no_speech={no_speech_prob:.2f}, dur={duration:.1f}s)")
        
        result['segments'] = filtered_segments
        print(f"\n  📊 Result: {len(filtered_segments)} high-quality speech segments")
        
        return result
    
    def _is_noise_text(self, text):
        """Check if text is likely noise/gibberish rather than real speech"""
        text_lower = text.lower().strip()
        
        # Empty or very short
        if len(text_lower) < 2:
            return True
        
        # Common noise patterns
        noise_patterns = [
            # Repetitive single chars
            'a a a', 'e e e', 'i i i', 'o o o', 'u u u',
            # Common filler sounds that might be game audio
            'uh uh uh', 'ah ah ah', 'eh eh eh',
        ]
        
        for pattern in noise_patterns:
            if pattern in text_lower:
                return True
        
        # Check for repetitive characters (e.g., "aaaa", "eeeee")
        if len(text_lower) >= 3:
            unique_chars = len(set(text_lower.replace(' ', '')))
            total_chars = len(text_lower.replace(' ', ''))
            if total_chars > 3 and unique_chars <= 2:  # Very low diversity
                return True
        
        return False
        
        return result
    
    def analyze_audio_energy(self, audio_path):
        """Analisis energi audio untuk deteksi moment loud/exciting"""
        print("Analyzing audio energy...")
        y, sr = librosa.load(audio_path, sr=None)
        
        # Hitung RMS energy
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Convert frame indices to time
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # Normalize RMS
        rms_normalized = rms / np.max(rms)
        
        return times, rms_normalized
    
    def find_excitement_keywords(self, transcription):
        """Cari kata-kata excitement dalam transkripsi"""
        excitement_moments = []
        
        for segment in transcription['segments']:
            text = segment['text'].lower()
            start_time = segment['start']
            end_time = segment['end']
            
            # Cek apakah ada keyword excitement
            found_keywords = [kw for kw in config.EXCITEMENT_KEYWORDS if kw in text]
            
            if found_keywords:
                excitement_moments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': segment['text'],
                    'keywords': found_keywords,
                    'score': len(found_keywords) * 2  # Bobot tinggi untuk keywords
                })
        
        return excitement_moments
    
    def find_volume_spikes(self, times, rms, transcription):
        """Deteksi spike volume (reaksi keras, teriakan)"""
        # Hitung mean dan std
        mean_rms = np.mean(rms)
        std_rms = np.std(rms)
        
        # Threshold untuk spike
        threshold = mean_rms + (std_rms * config.VOLUME_SPIKE_MULTIPLIER)
        
        # Temukan semua spikes
        spike_indices = np.where(rms > threshold)[0]
        
        if len(spike_indices) == 0:
            return []
        
        # Group consecutive spikes
        spike_groups = []
        current_group = [spike_indices[0]]
        
        for i in range(1, len(spike_indices)):
            if spike_indices[i] - spike_indices[i-1] <= 10:  # Within 10 frames
                current_group.append(spike_indices[i])
            else:
                if len(current_group) > 0:
                    spike_groups.append(current_group)
                current_group = [spike_indices[i]]
        
        if len(current_group) > 0:
            spike_groups.append(current_group)
        
        # Convert to time moments
        volume_moments = []
        for group in spike_groups:
            start_idx = max(0, group[0] - 5)
            end_idx = min(len(times) - 1, group[-1] + 5)
            
            start_time = times[start_idx]
            end_time = times[end_idx]
            peak_rms = np.max(rms[group])
            
            volume_moments.append({
                'start': start_time,
                'end': end_time,
                'peak_volume': float(peak_rms),
                'score': float(peak_rms * 1.5)  # Volume spike score
            })
        
        return volume_moments
    
    def detect_best_moment(self, audio_path):
        """Deteksi moment terbaik dari video berdasarkan audio dan transkripsi"""
        print("\n=== Starting Moment Detection ===")
        
        # 0. Preprocess audio untuk isolate human voice
        import os
        preprocessed_path = audio_path.replace('.wav', '_clean.wav')
        self.preprocess_audio_for_speech(audio_path, preprocessed_path)
        
        # 1. Transkripsi dengan audio yang sudah di-clean
        transcription = self.transcribe_audio(preprocessed_path)
        
        # Clean up preprocessed file
        if os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)
        
        # 2. Analisis audio energy (pakai original audio untuk detect spikes)
        times, rms = self.analyze_audio_energy(audio_path)
        
        # 3. Cari excitement keywords
        keyword_moments = self.find_excitement_keywords(transcription)
        print(f"Found {len(keyword_moments)} keyword moments")
        
        # 4. Cari volume spikes
        volume_moments = self.find_volume_spikes(times, rms, transcription)
        print(f"Found {len(volume_moments)} volume spike moments")
        
        # 5. Gabungkan dan score semua moments
        all_moments = []
        
        # Add keyword moments
        for km in keyword_moments:
            all_moments.append({
                'start': km['start'],
                'end': km['end'],
                'score': km['score'],
                'type': 'keyword',
                'details': f"Keywords: {', '.join(km['keywords'])}",
                'text': km['text']
            })
        
        # Add volume moments
        for vm in volume_moments:
            all_moments.append({
                'start': vm['start'],
                'end': vm['end'],
                'score': vm['score'],
                'type': 'volume_spike',
                'details': f"Peak volume: {vm['peak_volume']:.2f}",
                'text': ''
            })
        
        # 6. Merge overlapping moments
        if len(all_moments) == 0:
            print("No exciting moments detected!")
            return None
        
        # Sort by start time
        all_moments.sort(key=lambda x: x['start'])
        
        # Merge overlapping moments
        merged_moments = []
        current = all_moments[0].copy()
        
        for moment in all_moments[1:]:
            # Check if overlapping or close
            if moment['start'] <= current['end'] + 2:  # Within 2 seconds
                # Merge
                current['end'] = max(current['end'], moment['end'])
                current['score'] += moment['score']
                current['type'] = f"{current['type']}+{moment['type']}"
                if moment['text']:
                    current['text'] += " | " + moment['text']
                if moment['details']:
                    current['details'] += " | " + moment['details']
            else:
                merged_moments.append(current)
                current = moment.copy()
        
        merged_moments.append(current)
        
        # 7. Find best moment (highest score)
        best_moment = max(merged_moments, key=lambda x: x['score'])
        
        print(f"\n=== Best Moment Found ===")
        print(f"Time: {best_moment['start']:.2f}s - {best_moment['end']:.2f}s")
        print(f"Score: {best_moment['score']:.2f}")
        print(f"Type: {best_moment['type']}")
        print(f"Details: {best_moment['details']}")
        if best_moment['text']:
            print(f"Transcript: {best_moment['text']}")
        
        # 8. Expand with buffer dan limit ke max duration
        clip_start = max(0, best_moment['start'] - config.BUFFER_BEFORE)
        clip_end = best_moment['end'] + config.BUFFER_AFTER
        
        # Ensure max duration
        duration = clip_end - clip_start
        if duration > config.MAX_CLIP_DURATION:
            # Center around the peak moment
            middle = (best_moment['start'] + best_moment['end']) / 2
            clip_start = max(0, middle - config.MAX_CLIP_DURATION / 2)
            clip_end = clip_start + config.MAX_CLIP_DURATION
        
        # Ensure min duration
        if duration < config.MIN_CLIP_DURATION:
            needed = config.MIN_CLIP_DURATION - duration
            clip_start = max(0, clip_start - needed / 2)
            clip_end = clip_end + needed / 2
        
        return {
            'start': clip_start,
            'end': clip_end,
            'duration': clip_end - clip_start,
            'transcription': transcription,
            'score': best_moment['score'],
            'type': best_moment['type'],
            'details': best_moment['details']
        }
