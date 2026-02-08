import whisper
import librosa
import numpy as np
from scipy import signal
import config

class MomentDetector:
    def __init__(self):
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model(config.WHISPER_MODEL)
        
    def transcribe_audio(self, audio_path):
        """Transkripsi audio menggunakan Whisper dan deteksi bahasa"""
        print("Transcribing audio with Whisper...")
        result = self.whisper_model.transcribe(
            audio_path,
            language='id',  # Bahasa Indonesia
            task='transcribe',
            fp16=False
        )
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
        
        # 1. Transkripsi
        transcription = self.transcribe_audio(audio_path)
        
        # 2. Analisis audio energy
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
