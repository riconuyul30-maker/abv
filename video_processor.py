from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
    
    def get_optimal_font_size(self, video_width, video_height, is_portrait):
        """Get optimal font size based on video dimensions and orientation"""
        if is_portrait:
            # Portrait (9:16, TikTok style)
            base_size = int(video_width * 0.09)  # ~9% of width
        else:
            # Landscape (16:9, YouTube style)
            base_size = int(video_height * 0.08)  # ~8% of height
        
        # Ensure readable size
        return max(32, min(base_size, 120))
    
    def get_subtitle_position(self, video_width, video_height, is_portrait):
        """Get optimal subtitle position based on orientation"""
        if is_portrait:
            # Portrait: Center-bottom, lebih tinggi untuk safe area
            margin_bottom = int(video_height * 0.25)  # 25% dari bawah
            y_position = video_height - margin_bottom
        else:
            # Landscape: Bottom with smaller margin
            margin_bottom = int(video_height * 0.15)  # 15% dari bawah
            y_position = video_height - margin_bottom
        
        return y_position
    
    def create_subtitle_image(self, text, width, height, is_portrait, font_size):
        """Create modern subtitle image like TikTok/YouTube Shorts"""
        # Create transparent image
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Load font with optimal size
        try:
            # Try different fonts in order of preference
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
            ]
            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue
            
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Word wrapping for long text
        max_width = int(width * 0.9)  # 90% of screen width
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]
            
            if line_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Limit to 2 lines for readability
        if len(lines) > 2:
            lines = lines[:2]
            lines[1] += '...'
        
        # Calculate total text block height
        line_height = int(font_size * 1.3)
        total_text_height = len(lines) * line_height
        
        # Get Y position based on orientation
        y_start = self.get_subtitle_position(width, height, is_portrait)
        y_start -= (total_text_height // 2)  # Center vertically around position
        
        # Draw each line
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Center horizontally
            x = (width - text_width) // 2
            y = y_start + (i * line_height)
            
            # Draw background box (semi-transparent black)
            padding = int(font_size * 0.3)
            box_coords = [
                x - padding,
                y - padding,
                x + text_width + padding,
                y + text_height + padding
            ]
            draw.rounded_rectangle(
                box_coords,
                radius=int(font_size * 0.2),
                fill=(0, 0, 0, 180)  # Semi-transparent black
            )
            
            # Draw text outline (thick stroke for better readability)
            outline_width = max(3, int(font_size * 0.08))
            for adj_x in range(-outline_width, outline_width + 1):
                for adj_y in range(-outline_width, outline_width + 1):
                    if adj_x*adj_x + adj_y*adj_y <= outline_width*outline_width:
                        draw.text(
                            (x + adj_x, y + adj_y),
                            line,
                            font=font,
                            fill=(0, 0, 0, 255)
                        )
            
            # Draw main text (bright white)
            draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        
        return np.array(img)
    
    def get_word_level_timings(self, transcription, clip_start):
        """
        Extract word-level timings from Whisper transcription
        This ensures subtitle appears EXACTLY when word is spoken
        """
        word_timings = []
        
        for segment in transcription['segments']:
            seg_start = segment['start']
            seg_end = segment['end']
            text = segment['text'].strip()
            
            # Check if segment has word-level timestamps
            if 'words' in segment and segment['words']:
                # Use word-level timestamps (most accurate!)
                for word_info in segment['words']:
                    word_start = word_info.get('start', seg_start)
                    word_end = word_info.get('end', seg_end)
                    word_text = word_info.get('word', '').strip()
                    
                    if word_text:
                        # Adjust timing relative to clip start
                        adjusted_start = word_start - clip_start
                        adjusted_end = word_end - clip_start
                        
                        if adjusted_start >= 0:  # Only include words within clip
                            word_timings.append({
                                'start': adjusted_start,
                                'end': adjusted_end,
                                'text': word_text
                            })
            else:
                # No word-level data, split segment into words and estimate timing
                words = text.split()
                if words:
                    duration = seg_end - seg_start
                    time_per_word = duration / len(words)
                    
                    for i, word in enumerate(words):
                        word_start = seg_start + (i * time_per_word)
                        word_end = word_start + time_per_word
                        
                        # Adjust timing relative to clip start
                        adjusted_start = word_start - clip_start
                        adjusted_end = word_end - clip_start
                        
                        if adjusted_start >= 0:
                            word_timings.append({
                                'start': adjusted_start,
                                'end': adjusted_end,
                                'text': word
                            })
        
        return word_timings
    
    def group_words_into_chunks(self, word_timings, max_words=3, max_duration=2.5):
        """
        Group words into readable chunks (like TikTok/YouTube Shorts)
        Max 3 words per chunk, max 2.5 seconds
        """
        if not word_timings:
            return []
        
        chunks = []
        current_chunk = []
        chunk_start = None
        
        for word_info in word_timings:
            if not current_chunk:
                # Start new chunk
                current_chunk = [word_info['text']]
                chunk_start = word_info['start']
                chunk_end = word_info['end']
            else:
                # Check if we should add to current chunk or start new one
                chunk_duration = word_info['end'] - chunk_start
                
                if (len(current_chunk) < max_words and 
                    chunk_duration < max_duration):
                    # Add to current chunk
                    current_chunk.append(word_info['text'])
                    chunk_end = word_info['end']
                else:
                    # Save current chunk and start new one
                    chunks.append({
                        'start': chunk_start,
                        'end': chunk_end,
                        'text': ' '.join(current_chunk)
                    })
                    current_chunk = [word_info['text']]
                    chunk_start = word_info['start']
                    chunk_end = word_info['end']
        
        # Add last chunk
        if current_chunk:
            chunks.append({
                'start': chunk_start,
                'end': chunk_end,
                'text': ' '.join(current_chunk)
            })
        
        return chunks
    
    def create_clip_with_subtitles(self, video_path, moment_info, output_path):
        """Buat clip dari video dengan subtitle embedded - TikTok/YouTube Shorts style"""
        print(f"\n=== Creating Clip ===")
        print(f"Input: {video_path}")
        print(f"Output: {output_path}")
        print(f"Clip time: {moment_info['start']:.2f}s - {moment_info['end']:.2f}s")
        
        # Load video
        video = VideoFileClip(video_path)
        
        # Ensure clip times are within video duration
        video_duration = video.duration
        clip_start = max(0, min(moment_info['start'], video_duration - 1))
        clip_end = max(clip_start + 1, min(moment_info['end'], video_duration))
        
        # Extract clip
        clip = video.subclip(clip_start, clip_end)
        
        # Detect orientation
        is_portrait = clip.h > clip.w
        orientation = "Portrait (9:16)" if is_portrait else "Landscape (16:9)"
        print(f"Video orientation: {orientation}")
        print(f"Resolution: {clip.w}x{clip.h}")
        
        # Get optimal font size
        font_size = self.get_optimal_font_size(clip.w, clip.h, is_portrait)
        print(f"Font size: {font_size}px")
        
        # Get word-level timings from transcription
        print("Extracting word-level timings...")
        word_timings = self.get_word_level_timings(moment_info['transcription'], clip_start)
        
        # Group words into readable chunks
        subtitle_chunks = self.group_words_into_chunks(word_timings)
        print(f"Created {len(subtitle_chunks)} subtitle chunks")
        
        # Create subtitle clips
        subtitle_clips = []
        for i, chunk in enumerate(subtitle_chunks):
            # Only create subtitle if within clip duration
            if chunk['start'] < clip.duration and chunk['end'] > 0:
                # Adjust timing to be within clip bounds
                start_time = max(0, chunk['start'])
                end_time = min(clip.duration, chunk['end'])
                duration = end_time - start_time
                
                if duration > 0.1:  # Minimum 0.1s duration
                    text = chunk['text']
                    print(f"  [{start_time:.2f}s - {end_time:.2f}s] \"{text}\"")
                    
                    # Create subtitle image
                    sub_img = self.create_subtitle_image(
                        text,
                        clip.w,
                        clip.h,
                        is_portrait,
                        font_size
                    )
                    
                    # Create ImageClip with exact timing
                    sub_clip = (ImageClip(sub_img)
                               .set_start(start_time)
                               .set_duration(duration)
                               .set_position('center'))
                    
                    subtitle_clips.append(sub_clip)
        
        # Composite video with subtitles
        if subtitle_clips:
            print(f"✅ Adding {len(subtitle_clips)} subtitle segments...")
            final_clip = CompositeVideoClip([clip] + subtitle_clips)
        else:
            print("⚠️  No subtitles in this clip range")
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
        for sub_clip in subtitle_clips:
            sub_clip.close()
        
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
        for sub_clip in subtitle_clips:
            sub_clip.close()
        
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
