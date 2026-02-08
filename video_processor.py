from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.tools.subtitles import SubtitlesClip
import os
import config

class VideoProcessor:
    
    def extract_audio(self, video_path, output_audio_path):
        """Extract audio dari video"""
        print(f"Extracting audio from {video_path}...")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_audio_path, logger=None)
        video.close()
        return output_audio_path
    
    def create_subtitle_file(self, transcription, output_path):
        """Buat file subtitle SRT dari hasil transkripsi Whisper"""
        print("Creating subtitle file...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(transcription['segments'], start=1):
                # Format waktu untuk SRT
                start_time = self.format_srt_time(segment['start'])
                end_time = self.format_srt_time(segment['end'])
                
                # Tulis ke file SRT
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text'].strip()}\n\n")
        
        return output_path
    
    def format_srt_time(self, seconds):
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def create_clip_with_subtitles(self, video_path, moment_info, output_path):
        """Buat clip dari video dengan subtitle embedded"""
        print(f"\n=== Creating Clip ===")
        print(f"Input: {video_path}")
        print(f"Output: {output_path}")
        print(f"Clip time: {moment_info['start']:.2f}s - {moment_info['end']:.2f}s")
        print(f"Duration: {moment_info['duration']:.2f}s")
        
        # Load video
        video = VideoFileClip(video_path)
        
        # Extract clip
        clip = video.subclip(moment_info['start'], moment_info['end'])
        
        # Create subtitle generator
        def generator(txt):
            # Style subtitle
            return TextClip(
                txt,
                font='Arial-Bold',
                fontsize=40,
                color='white',
                stroke_color='black',
                stroke_width=2,
                method='caption',
                size=(clip.w * 0.9, None),
                align='center'
            )
        
        # Filter segments yang ada di dalam clip
        clip_start = moment_info['start']
        clip_end = moment_info['end']
        
        subtitle_data = []
        for segment in moment_info['transcription']['segments']:
            seg_start = segment['start']
            seg_end = segment['end']
            
            # Check if segment overlaps with clip
            if seg_end >= clip_start and seg_start <= clip_end:
                # Adjust timing relative to clip start
                adjusted_start = max(0, seg_start - clip_start)
                adjusted_end = min(clip.duration, seg_end - clip_start)
                
                subtitle_data.append(
                    ((adjusted_start, adjusted_end), segment['text'].strip())
                )
        
        # Add subtitles if any
        if subtitle_data:
            print(f"Adding {len(subtitle_data)} subtitle segments...")
            subtitles = SubtitlesClip(subtitle_data, generator)
            
            # Position subtitles at bottom
            subtitles = subtitles.set_position(('center', 'bottom'))
            
            # Composite video with subtitles
            final_clip = CompositeVideoClip([clip, subtitles])
        else:
            print("No subtitles in this clip range")
            final_clip = clip
        
        # Write output
        print("Rendering final video...")
        final_clip.write_videofile(
            output_path,
            codec=config.OUTPUT_CODEC,
            audio_codec=config.AUDIO_CODEC,
            bitrate=config.VIDEO_BITRATE,
            audio_bitrate=config.AUDIO_BITRATE,
            fps=video.fps,
            preset='medium',
            threads=4,
            logger=None
        )
        
        # Cleanup
        video.close()
        clip.close()
        final_clip.close()
        if subtitle_data:
            subtitles.close()
        
        print(f"✓ Clip created successfully: {output_path}")
        
        return output_path
    
    def get_video_info(self, video_path):
        """Get video information"""
        video = VideoFileClip(video_path)
        info = {
            'duration': video.duration,
            'fps': video.fps,
            'size': video.size,
            'width': video.w,
            'height': video.h
        }
        video.close()
        return info
