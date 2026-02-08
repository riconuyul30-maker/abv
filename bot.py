import os
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime
import config
from moment_detector import MomentDetector
from video_processor import VideoProcessor

class GamingClipperBot:
    def __init__(self):
        # Create directories
        os.makedirs(config.DOWNLOAD_PATH, exist_ok=True)
        os.makedirs(config.OUTPUT_PATH, exist_ok=True)
        os.makedirs(config.TEMP_PATH, exist_ok=True)
        
        # Initialize processors
        self.moment_detector = None  # Will be initialized on first use
        self.video_processor = VideoProcessor()
        
        # Initialize bot application
        self.application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
        
        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(MessageHandler(filters.VIDEO, self.handle_video))
        self.application.add_handler(MessageHandler(filters.Document.VIDEO, self.handle_video))
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk command /start"""
        welcome_message = """
🎮 *Gaming Clipper Bot* 🎮

Bot ini akan otomatis:
✅ Mendeteksi moment keren di video gaming
✅ Membuat subtitle Bahasa Indonesia
✅ Menghasilkan clip highlight (max 60 detik)

📤 *Cara Pakai:*
1. Kirim video gaming ke bot
2. Bot akan proses otomatis
3. Tunggu clip hasil + subtitle!

⚙️ *Fitur AI:*
- Deteksi kata-kata excitement (gila, wow, savage, dll)
- Deteksi reaksi keras / teriakan
- Analisis audio untuk moment intense

Kirim video sekarang untuk mulai! 🚀
        """
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk command /help"""
        help_message = """
📖 *Panduan Penggunaan*

*Format Video yang Didukung:*
- MP4, AVI, MOV, MKV
- Maksimal durasi: Apapun (tapi processing lama untuk video panjang)
- Recommended: 1-10 menit

*Tips untuk Hasil Terbaik:*
- Pastikan ada suara/voice chat
- Video dengan reaksi pemain lebih mudah dideteksi
- Kualitas audio yang baik = subtitle lebih akurat

*Output:*
- Format: MP4 (1080p)
- Durasi: 10-60 detik (tergantung moment)
- Subtitle: Embedded dalam video

Butuh bantuan? Contact admin!
        """
        await update.message.reply_text(help_message, parse_mode='Markdown')
    
    async def handle_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk video yang dikirim"""
        try:
            # Notify user
            status_msg = await update.message.reply_text(
                "📥 Video diterima! Memulai processing...\n"
                "⏳ Ini mungkin memakan waktu beberapa menit...",
                parse_mode='Markdown'
            )
            
            # Get video file
            if update.message.video:
                video_file = update.message.video
                file_name = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            else:
                video_file = update.message.document
                file_name = update.message.document.file_name
            
            # Download video
            await status_msg.edit_text("📥 Downloading video...")
            file = await context.bot.get_file(video_file.file_id)
            video_path = os.path.join(config.DOWNLOAD_PATH, file_name)
            await file.download_to_drive(video_path)
            
            print(f"\n{'='*50}")
            print(f"Processing video: {file_name}")
            print(f"File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
            
            # Get video info
            video_info = self.video_processor.get_video_info(video_path)
            print(f"Duration: {video_info['duration']:.2f}s")
            print(f"Resolution: {video_info['width']}x{video_info['height']}")
            print(f"FPS: {video_info['fps']}")
            
            # Extract audio
            await status_msg.edit_text("🎵 Extracting audio...")
            audio_path = os.path.join(config.TEMP_PATH, f"audio_{file_name}.wav")
            self.video_processor.extract_audio(video_path, audio_path)
            
            # Initialize moment detector (lazy loading for faster startup)
            if self.moment_detector is None:
                await status_msg.edit_text("🤖 Loading AI models (first time only)...")
                self.moment_detector = MomentDetector()
            
            # Detect best moment
            await status_msg.edit_text("🔍 Analyzing video for best moments...\n⏳ This may take a while...")
            moment_info = self.moment_detector.detect_best_moment(audio_path)
            
            if moment_info is None:
                await status_msg.edit_text(
                    "❌ Tidak ada moment menarik yang terdeteksi dalam video ini.\n"
                    "Coba video lain dengan reaksi atau audio yang lebih jelas!"
                )
                # Cleanup
                os.remove(video_path)
                os.remove(audio_path)
                return
            
            # Create clip
            await status_msg.edit_text(
                f"✂️ Creating clip...\n"
                f"⏱ Clip duration: {moment_info['duration']:.1f}s\n"
                f"📊 Moment score: {moment_info['score']:.1f}\n"
                f"🏷 Type: {moment_info['type']}"
            )
            
            output_filename = f"clip_{file_name}"
            output_path = os.path.join(config.OUTPUT_PATH, output_filename)
            
            self.video_processor.create_clip_with_subtitles(
                video_path,
                moment_info,
                output_path
            )
            
            # Send result
            await status_msg.edit_text("📤 Uploading clip...")
            
            clip_caption = (
                f"🎮 *Gaming Clip Generated!*\n\n"
                f"⏱ Duration: {moment_info['duration']:.1f}s\n"
                f"🎯 Type: {moment_info['type']}\n"
                f"📊 Score: {moment_info['score']:.1f}\n"
                f"💬 {moment_info['details']}\n\n"
                f"✅ Subtitle: Bahasa Indonesia"
            )
            
            with open(output_path, 'rb') as video:
                await update.message.reply_video(
                    video=video,
                    caption=clip_caption,
                    parse_mode='Markdown',
                    supports_streaming=True,
                    width=1920,
                    height=1080
                )
            
            await status_msg.edit_text("✅ Processing complete!")
            
            # Cleanup
            print(f"\nCleaning up temporary files...")
            os.remove(video_path)
            os.remove(audio_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            
            print(f"{'='*50}\n")
            
        except Exception as e:
            error_message = f"❌ Error: {str(e)}\n\nSilakan coba lagi atau kirim video lain."
            await update.message.reply_text(error_message)
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Jalankan bot"""
        print("🤖 Gaming Clipper Bot Starting...")
        print(f"Bot Token: {config.TELEGRAM_BOT_TOKEN[:20]}...")
        print(f"Chat ID: {config.TELEGRAM_CHAT_ID}")
        print("\n✅ Bot is ready! Waiting for videos...\n")
        
        # Run bot
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    bot = GamingClipperBot()
    bot.run()
