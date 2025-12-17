# tts_handler.py - FULL-DUPLEX FIXED VERSION with Cartesia AI Integration
import os
import logging
import time
import threading
import warnings
from RealtimeTTS import SystemEngine, TextToAudioStream
from cartesia_tts_engine import CartesiaTTSEngine

import os
import logging
import time
import threading
from RealtimeTTS import SystemEngine, TextToAudioStream
from cartesia_tts_engine import CartesiaTTSEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class TTSHandler:
    """TTS with <150ms barge-in using continuous STT monitoring."""
    
    def __init__(self, stt_handler=None, use_cartesia=None):
        try:
            # Check if we should use Cartesia AI
            self.use_cartesia = use_cartesia if use_cartesia is not None else (
                os.getenv('USE_CARTESIA_TTS', 'false').lower() == 'true'
            )
            
            if self.use_cartesia:
                # Initialize Cartesia AI engine
                self.cartesia_engine = CartesiaTTSEngine(
                    api_key=os.getenv('CARTESIA_API_KEY'),
                    voice_id=os.getenv('CARTESIA_VOICE_ID', 'brooke'),
                    model=os.getenv('CARTESIA_MODEL', 'sonic-english')
                )
                self.engine = None
                self.stream = None
                logger.info("🎤 TTS Handler initialized with Cartesia AI (ultra-low latency)")
            else:
                # Initialize traditional RealtimeTTS engine
                self.engine = SystemEngine()
                self.stream = TextToAudioStream(self.engine)
                self.cartesia_engine = None
                logger.info("🎤 TTS Handler initialized with RealtimeTTS (AGGRESSIVE barge-in <150ms)")
            
            # CRITICAL: Reference to main STT (must be continuously listening)
            self.main_stt = stt_handler
            if not self.main_stt:
                raise ValueError("STT handler required for barge-in detection")
            
            # State management
            self.is_playing = False
            self.is_barge_in_enabled = True
            self.barge_in_detected = False
            self.stop_event = threading.Event()
            self.state_lock = threading.Lock()
            
            # Real-time monitoring state
            self.last_seen_realtime_text = ""
            self.barge_in_sensitivity = 2  # Minimum chars to trigger (very sensitive)
            
            # Test Cartesia connection if enabled
            if self.use_cartesia and self.cartesia_engine:
                if not self.cartesia_engine.test_connection():
                    logger.warning("⚠️ Cartesia connection failed, falling back to RealtimeTTS")
                    self.use_cartesia = False
                    self.engine = SystemEngine()
                    self.stream = TextToAudioStream(self.engine)
                    self.cartesia_engine = None
                else:
                    logger.info("✅ Cartesia AI connection successful")
                    
        except Exception as e:
            logger.error(f"❌ Error initializing TTS: {e}")
            # Fallback to traditional engine if Cartesia fails
            if self.use_cartesia:
                logger.info("🔄 Falling back to RealtimeTTS")
                self.use_cartesia = False
                self.engine = SystemEngine()
                self.stream = TextToAudioStream(self.engine)
                self.cartesia_engine = None
            else:
                raise
    
    # def _monitor_barge_in(self):
    #     """
    #     CRITICAL: Aggressively monitor STT real-time transcription during playback.
    #     Triggers stop within 20-50ms of speech detection.
    #     """
    #     try:
    #         if not self.is_barge_in_enabled or not self.main_stt:
    #             return
            
    #         playback_start = time.time()
    #         tts_startup_buffer = 0.15  # Ignore first 150ms (TTS ramp-up)
    #         check_interval = 0.02  # Check every 20ms (aggressive polling)
            
    #         logger.info("👂 Barge-in monitor: ACTIVE")
            
    #         # Reset tracking
    #         self.last_seen_realtime_text = ""
    #         consecutive_detections = 0
    #         min_consecutive = 2  # Need 2 consecutive detections (40ms) to confirm
            
    #         while not self.stop_event.is_set():
    #             with self.state_lock:
    #                 if not self.is_playing:
    #                     break
                
    #             current_time = time.time()
                
    #             # Skip TTS startup buffer
    #             if current_time - playback_start < tts_startup_buffer:
    #                 time.sleep(check_interval)
    #                 continue
                
                # try:
                #     # CRITICAL: Get real-time text from continuously listening STT
                #     current_realtime = self.main_stt.get_realtime_text()
                    
                #     if not current_realtime:
                #         consecutive_detections = 0
                #         time.sleep(check_interval)
                #         continue
                    
                #     # Detect NEW speech (text changed or grew)
                #     if current_realtime != self.last_seen_realtime_text:
                #         new_text = current_realtime[len(self.last_seen_realtime_text):].strip()
                        
                #         # CRITICAL: Very sensitive detection
                #         if len(new_text) >= self.barge_in_sensitivity:
                #             consecutive_detections += 1
                            
                #             logger.debug(f"🎤 Speech fragment detected: '{new_text}' "
                #                        f"(confirmations: {consecutive_detections}/{min_consecutive})")
                            
                #             # Confirm barge-in after multiple detections
                #             if consecutive_detections >= min_consecutive:
                #                 elapsed_ms = (time.time() - playback_start) * 1000
                #                 logger.info(f"🎤 BARGE-IN CONFIRMED at {elapsed_ms:.0f}ms: '{current_realtime}'")
                                
                #                 # STOP IMMEDIATELY
                #                 with self.state_lock:
                #                     self.barge_in_detected = True
                #                     self.stop_event.set()
                                
                #                 if self.stream:
                #                     try:
                #                         self.stream.stop()
                #                         logger.info("🛑 TTS stopped in <150ms")
                #                     except Exception as e:
                #                         logger.error(f"Stop failed: {e}")
                                
                #                 break
                #         else:
                #             consecutive_detections = 0
                        
                #         self.last_seen_realtime_text = current_realtime
                    
                # except Exception as e:
                #     logger.debug(f"Monitor check error: {e}")
                
                # ...existing code...
    def _monitor_barge_in(self):
        """
        CRITICAL: Aggressively monitor STT real-time transcription during playback.
        Triggers stop within 20-50ms of speech detection.
        """
        try:
            if not self.is_barge_in_enabled or not self.main_stt:
                return
            
            playback_start = time.time()
            tts_startup_buffer = 0.15  # Ignore first 150ms (TTS ramp-up)
            check_interval = 0.02  # Check every 20ms (aggressive polling)
            
            logger.info("👂 Barge-in monitor: ACTIVE")
            
            # Reset tracking
            self.last_seen_realtime_text = ""
            consecutive_detections = 0
            min_consecutive = 2  # Need 2 consecutive detections (40ms) to confirm
            
            while not self.stop_event.is_set():
                with self.state_lock:
                    if not self.is_playing:
                        break
                
                current_time = time.time()
                
                # Skip TTS startup buffer
                if current_time - playback_start < tts_startup_buffer:
                    time.sleep(check_interval)
                    continue
                
                try:
                    # Get real-time transcription
                    realtime_text = self.main_stt.get_realtime_text()
                    
                    # Detect new speech (not just continuation)
                    if realtime_text and len(realtime_text) > self.barge_in_sensitivity:
                        if realtime_text != self.last_seen_realtime_text:
                            consecutive_detections += 1
                            
                            # Confirm with consecutive detections
                            if consecutive_detections >= min_consecutive:
                                logger.info(f"🛑 Barge-in detected: {realtime_text[:30]}...")
                                with self.state_lock:
                                    self.barge_in_detected = True
                                self.stop_event.set()
                                break
                        else:
                            consecutive_detections = max(0, consecutive_detections - 1)
                    
                    self.last_seen_realtime_text = realtime_text
                except Exception as e:
                    logger.warning(f"⚠️ Monitor check error: {e}")
                
                time.sleep(check_interval)
            
            logger.info("👂 Barge-in monitor: STOPPED")
            
        except Exception as e:
            logger.error(f"❌ Barge-in monitor crashed: {e}")
    
    def _play_with_monitoring(self, text: str):
        """Play audio while monitoring for barge-in (traditional TTS)."""
        
        def play_audio():
            try:
                with self.state_lock:
                    self.is_playing = True
                    # self.barge_in_detected = False
                self.stop_event.clear()
                
                # CRITICAL: Start monitor BEFORE audio starts
                monitor_thread = threading.Thread(target=self._monitor_barge_in, daemon=True)
                monitor_thread.start()
                
                # Brief delay to ensure monitor is active
                time.sleep(0.02)
                
                # Play audio (non-blocking thread)
                if self.stream:
                    self.stream.feed(text)
                    self.stream.play_async()
                    # try:
                    #     self.stream.play()
                    # except Exception as e:
                    #     if not self.stop_event.is_set():
                    #                 logger.error(f"❌ Playback error: {e}")
                        
            except Exception as e:
                logger.error(f"❌ Playback thread error: {e}")
            finally:
                with self.state_lock:
                    self.is_playing = False
                self.stop_event.clear()
        
        # Start playback in daemon thread
        self.playback_thread = threading.Thread(target=play_audio, daemon=True)
        self.playback_thread.start()
    
    def _play_cartesia_with_monitoring(self, text: str, emotion: str = "neutral", speed: float = 1.0):
        """Play Cartesia audio while monitoring for barge-in."""
        
        def play_cartesia():
            try:
                with self.state_lock:
                    self.is_playing = True
                    self.barge_in_detected = False
                self.stop_event.clear()
                
                # CRITICAL: Start monitor BEFORE audio starts
                monitor_thread = threading.Thread(target=self._monitor_barge_in, daemon=True)
                monitor_thread.start()
                
                # Brief delay to ensure monitor is active
                time.sleep(0.02)
                
                # Play Cartesia audio with streaming support
                if self.cartesia_engine:
                    self.cartesia_engine.play_streaming(
                        text=text,
                        enable_interrupt=self.is_barge_in_enabled,
                        emotion=emotion,
                        speed=speed
                    )
                        
            except Exception as e:
                logger.error(f"❌ Cartesia playback thread error: {e}")
            finally:
                with self.state_lock:
                    self.is_playing = False
                self.stop_event.clear()
        
        # Start Cartesia playback in daemon thread
        self.playback_thread = threading.Thread(target=play_cartesia, daemon=True)
        self.playback_thread.start()
    
    def speak(self, text: str, voice: str = "default", emotive_tags: str = "",
              enable_barge_in: bool = True, emotion: str = "neutral",
              speed: float = 1.0) -> str:
        """Speak with instant barge-in detection."""
        try:
            if self.use_cartesia and self.cartesia_engine:
                return self._speak_cartesia(text, enable_barge_in, emotion, speed)
            else:
                return self._speak_traditional(text, voice, emotive_tags, enable_barge_in)
        except Exception as e:
            logger.error(f"❌ TTS error: {e}")
            return ""
    
    def _speak_cartesia(self, text: str, enable_barge_in: bool = True,
                       emotion: str = "neutral", speed: float = 1.0) -> str:
        """Speak using Cartesia AI with streaming support."""
        try:
            if not self.cartesia_engine:
                raise ValueError("Cartesia TTS not initialized")
            
            self.is_barge_in_enabled = enable_barge_in
            logger.info(f"🗣 Cartesia Speaking: {text[:50]}... (barge-in: {enable_barge_in}, emotion: {emotion})")
            
            # CRITICAL: Clear STT real-time buffer before speaking
            if self.main_stt:
                self.main_stt.clear_realtime_text()
            
            # Start Cartesia streaming playback with monitoring
            self._play_cartesia_with_monitoring(text, emotion, speed)
            
            return "cartesia_streaming"
            
        except Exception as e:
            logger.error(f"❌ Cartesia TTS error: {e}")
            # Fallback to traditional TTS
            return self._speak_traditional(text, "default", "", enable_barge_in)
    
    def _speak_traditional(self, text: str, voice: str = "default",
                          emotive_tags: str = "", enable_barge_in: bool = True) -> str:
        """Speak using traditional RealtimeTTS."""
        try:
            if not self.engine or not self.stream:
                raise ValueError("Traditional TTS not initialized")
            
            self.is_barge_in_enabled = enable_barge_in
            logger.info(f"🗣 Traditional Speaking: {text[:50]}... (barge-in: {enable_barge_in})")
            
            if emotive_tags:
                text = f"{text} {emotive_tags}"
            
            if voice != "default":
                try:
                    self.engine.set_voice(voice)
                except Exception as e:
                    logger.warning(f"Voice '{voice}' unavailable: {e}")
            
            # CRITICAL: Clear STT real-time buffer before speaking
            if self.main_stt:
                self.main_stt.clear_realtime_text()
            
            self._play_with_monitoring(text)
            
            return "audio_playing"
            
        except Exception as e:
            logger.error(f"❌ Traditional TTS error: {e}")
            return ""
    
    def wait_for_completion(self, timeout: float = 30.0) -> bool:
        """Wait for playback to complete or be interrupted."""
        try:
            start_time = time.time()
            
            while True:
                with self.state_lock:
                    if not self.is_playing:
                        was_interrupted = self.barge_in_detected
                        if was_interrupted:
                            logger.info("✅ Playback interrupted by user")
                        return not was_interrupted
                    
                    if self.barge_in_detected:
                        return False
                
                if time.time() - start_time > timeout:
                    logger.warning("⏰ Playback timeout")
                    return False
                
                time.sleep(0.01)  # 10ms polling
                
        except Exception as e:
            logger.error(f"❌ Wait error: {e}")
            return False
    
    def is_barge_in_detected(self) -> bool:
        """Check if user interrupted playback."""
        with self.state_lock:
            return self.barge_in_detected
    
    def stop_playback(self):
        """Immediately stop TTS playback."""
        try:
            if self.use_cartesia and self.cartesia_engine:
                # Stop Cartesia playback
                self.cartesia_engine.stop_playback()
                logger.info("🛑 Cartesia TTS aborted on voice detect")
            elif self.stream:
                # Stop traditional TTS playback
                self.stream.stop()
                logger.info("🛑 Traditional TTS aborted on voice detect")
            
            with self.state_lock:
                self.barge_in_detected = True
            self.stop_event.set()
        except Exception as e:
            logger.error(f"Stop playback error: {e}")
    
    def shutdown(self):
        """Clean shutdown."""
        try:
            logger.info("🧹 Shutting down TTS...")
            
            with self.state_lock:
                self.is_playing = False
            self.stop_event.set()
            
            # Shutdown Cartesia engine if active
            if self.use_cartesia and hasattr(self, 'cartesia_engine') and self.cartesia_engine:
                try:
                    self.cartesia_engine.cleanup()
                except Exception as e:
                    logger.warning(f"Cartesia cleanup error: {e}")
                self.cartesia_engine = None
                logger.info("✅ Cartesia TTS shutdown complete")
            
            # Shutdown traditional TTS engine if active
            if not self.use_cartesia:
                if hasattr(self, 'stream') and self.stream:
                    try:
                        self.stream.stop()
                    except Exception:
                        pass
                    self.stream = None
                
                if hasattr(self, 'engine'):
                    self.engine = None
                
                logger.info("✅ Traditional TTS shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")