# tts_handler_optimized.py - Hybrid TTS Handler (Cartesia + SystemEngine Fallback)
"""
Optimized TTS Handler with:
- Cartesia AI integration (40-90ms latency)
- SystemEngine fallback for reliability
- <150ms barge-in detection via STT monitoring
- Minimal overhead, maximum performance
"""
import os
import logging
import time
import threading
import asyncio

# Try to import RealtimeTTS (optional for SystemEngine fallback)
try:
    from RealtimeTTS import SystemEngine, TextToAudioStream
    REALTIMETTS_AVAILABLE = True
except ImportError:
    REALTIMETTS_AVAILABLE = False
    SystemEngine = None
    TextToAudioStream = None
    logging.warning("⚠️ RealtimeTTS not available - SystemEngine fallback disabled")

from cartesia_tts_engine_optimized import CartesiaTTSEngine, VoiceConfig, AudioConfig, CartesiaVoices

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TTSHandler:
    """TTS Handler with Cartesia AI + Barge-in Detection (<150ms)."""
    
    def __init__(self, stt_handler=None, use_cartesia=None):
        try:
            # Validate STT handler (required for barge-in)
            self.main_stt = stt_handler
            if not self.main_stt:
                raise ValueError("STT handler required for barge-in detection")
            
            # Determine TTS engine
            self.use_cartesia = use_cartesia if use_cartesia is not None else (
                os.getenv('USE_CARTESIA_TTS', 'false').lower() == 'true'
            )
            
            # State management (thread-safe)
            self.is_playing = False
            self.is_barge_in_enabled = True
            self.barge_in_detected = False
            self.stop_event = threading.Event()
            self.state_lock = threading.Lock()
            
            # Barge-in monitoring
            self.last_seen_realtime_text = ""
            self.barge_in_sensitivity = 2  # Minimum chars to trigger
            
            # Async event loop for Cartesia (if needed)
            self.cartesia_loop = None
            self.cartesia_thread = None
            self.cartesia_future = None
            
            # Initialize TTS engines
            if self.use_cartesia:
                self._init_cartesia()
            else:
                self._init_system_engine()
                    
        except Exception as e:
            logger.error(f"❌ TTS initialization error: {e}")
            # Fallback to SystemEngine
            if self.use_cartesia:
                logger.info("🔄 Falling back to SystemEngine")
                self.use_cartesia = False
                self._init_system_engine()
            else:
                raise
    
    def _init_cartesia(self):
        """Initialize Cartesia AI with optimal config."""
        try:
            self.cartesia_engine = CartesiaTTSEngine(
                voice_config=VoiceConfig(
                    voice_id=CartesiaVoices.BROOKE,
                    model="sonic-3",  # Best quality/speed balance
                    speed=1.0
                ),
                audio_config=AudioConfig(
                    sample_rate=22050  # Cartesia optimal
                )
            )
            
            # Register barge-in callback (CRITICAL)
            self.cartesia_engine.set_barge_in_callback(self._check_barge_in_status)
            
            # Create async event loop in separate thread
            self.cartesia_loop = asyncio.new_event_loop()
            self.cartesia_thread = threading.Thread(
                target=self._run_cartesia_loop,
                daemon=True
            )
            self.cartesia_thread.start()
            
            # Initialize Cartesia client
            future = asyncio.run_coroutine_threadsafe(
                self.cartesia_engine.initialize(),
                self.cartesia_loop
            )
            future.result(timeout=10.0)  # Wait for init
            
            self.engine = None
            self.stream = None
            logger.info("✅ TTS: Cartesia AI (ultra-low latency + barge-in)")
            
        except Exception as e:
            logger.error(f"❌ Cartesia init failed: {e}")
            raise
    
    def _init_system_engine(self):
        """Initialize traditional RealtimeTTS engine."""
        if not REALTIMETTS_AVAILABLE:
            raise ImportError("RealtimeTTS not available - install with: pip install RealtimeTTS==0.3.42 elevenlabs==1.1.0")
        
        self.engine = SystemEngine()
        self.stream = TextToAudioStream(self.engine)
        self.cartesia_engine = None
        logger.info("✅ TTS: SystemEngine (barge-in <150ms)")
    
    def _run_cartesia_loop(self):
        """Run async event loop for Cartesia in separate thread."""
        asyncio.set_event_loop(self.cartesia_loop)
        self.cartesia_loop.run_forever()
    
    def _check_barge_in_status(self) -> bool:
        """
        Callback for Cartesia to check if user is speaking.
        Returns True if barge-in detected.
        """
        try:
            if not self.is_barge_in_enabled or not self.main_stt:
                return False
            
            # Get real-time STT text
            realtime_text = self.main_stt.get_realtime_text()
            
            # Detect new speech
            if realtime_text and len(realtime_text) >= self.barge_in_sensitivity:
                if realtime_text != self.last_seen_realtime_text:
                    logger.info(f"🎤 Barge-in detected: {realtime_text[:30]}...")
                    with self.state_lock:
                        self.barge_in_detected = True
                    self.last_seen_realtime_text = realtime_text
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Barge-in check error: {e}")
            return False
    
    def _monitor_barge_in(self):
        """
        Monitor STT real-time transcription during playback.
        Triggers stop within 20-50ms of speech detection.
        (Used for SystemEngine only, Cartesia uses callback)
        """
        try:
            if not self.is_barge_in_enabled or not self.main_stt:
                return
            
            playback_start = time.time()
            tts_startup_buffer = 0.15  # Ignore first 150ms
            check_interval = 0.02  # Check every 20ms
            
            logger.info("👂 Barge-in monitor: ACTIVE")
            
            self.last_seen_realtime_text = ""
            consecutive_detections = 0
            min_consecutive = 2  # Need 2 consecutive detections (40ms)
            
            while not self.stop_event.is_set():
                with self.state_lock:
                    if not self.is_playing:
                        break
                
                # Skip TTS startup buffer
                if time.time() - playback_start < tts_startup_buffer:
                    time.sleep(check_interval)
                    continue
                
                try:
                    realtime_text = self.main_stt.get_realtime_text()
                    
                    if realtime_text and len(realtime_text) > self.barge_in_sensitivity:
                        if realtime_text != self.last_seen_realtime_text:
                            consecutive_detections += 1
                            
                            if consecutive_detections >= min_consecutive:
                                logger.info(f"🛑 Barge-in: {realtime_text[:30]}...")
                                with self.state_lock:
                                    self.barge_in_detected = True
                                self.stop_event.set()
                                
                                if self.stream:
                                    try:
                                        self.stream.stop()
                                    except Exception:
                                        pass
                                break
                        else:
                            consecutive_detections = max(0, consecutive_detections - 1)
                    
                    self.last_seen_realtime_text = realtime_text
                except Exception as e:
                    logger.warning(f"Monitor error: {e}")
                
                time.sleep(check_interval)
            
            logger.info("👂 Barge-in monitor: STOPPED")
            
        except Exception as e:
            logger.error(f"❌ Monitor crashed: {e}")
    
    def _play_system_engine(self, text: str):
        """Play audio with SystemEngine + monitoring."""
        def play_audio():
            try:
                with self.state_lock:
                    self.is_playing = True
                self.stop_event.clear()
                
                # Start barge-in monitor
                monitor_thread = threading.Thread(
                    target=self._monitor_barge_in,
                    daemon=True
                )
                monitor_thread.start()
                
                time.sleep(0.02)  # Ensure monitor is active
                
                # Play audio
                if self.stream:
                    self.stream.feed(text)
                    self.stream.play_async()
                        
            except Exception as e:
                logger.error(f"❌ Playback error: {e}")
            finally:
                with self.state_lock:
                    self.is_playing = False
                self.stop_event.clear()
        
        threading.Thread(target=play_audio, daemon=True).start()
    
    def speak(self, text: str, voice: str = "default", emotive_tags: str = "",
              enable_barge_in: bool = True, emotion: str = "neutral",
              speed: float = 1.0) -> str:
        """Speak with instant barge-in detection."""
        try:
            self.is_barge_in_enabled = enable_barge_in
            
            # Clear STT buffer before speaking
            if self.main_stt:
                self.main_stt.clear_realtime_text()
                self.last_seen_realtime_text = ""
            
            if self.use_cartesia and self.cartesia_engine:
                return self._speak_cartesia(text, enable_barge_in, emotion, speed)
            else:
                return self._speak_system_engine(text, voice, emotive_tags, enable_barge_in)
                
        except Exception as e:
            logger.error(f"❌ Speak error: {e}")
            # Fallback to SystemEngine
            if self.use_cartesia:
                logger.warning("🔄 Falling back to SystemEngine")
                return self._speak_system_engine(text, voice, emotive_tags, enable_barge_in)
            return ""
    
    def _speak_cartesia(self, text: str, enable_barge_in: bool = True,
                       emotion: str = "neutral", speed: float = 1.0) -> str:
        """Speak using Cartesia with integrated barge-in."""
        try:
            if not self.cartesia_engine:
                raise ValueError("Cartesia not initialized")
            
            # Stop any ongoing playback first (critical for barge-in)
            with self.state_lock:
                was_playing = self.is_playing
            
            if was_playing:
                logger.info("🛑 Stopping previous TTS before new speech")
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.cartesia_engine.stop(),
                        self.cartesia_loop
                    ).result(timeout=0.5)
                except TimeoutError:
                    logger.debug("Stop timeout (TTS may have already finished)")
                except Exception as e:
                    if str(e):  # Only log if there's an actual error message
                        logger.warning(f"Stop error: {e}")
                
                # Wait briefly for cleanup
                time.sleep(0.1)
            
            logger.info(f"🗣 Cartesia: {text[:50]}... (barge-in: {enable_barge_in})")
            
            # Update voice config if needed
            if speed != 1.0:
                self.cartesia_engine.voice_config.speed = speed
            
            # Start async synthesis
            future = asyncio.run_coroutine_threadsafe(
                self.cartesia_engine.synthesize_and_play(
                    text=text,
                    enable_barge_in=enable_barge_in
                ),
                self.cartesia_loop
            )
            
            # Store for wait_for_completion
            with self.state_lock:
                self.is_playing = True
                self.barge_in_detected = False
                self.cartesia_future = future
            
            # Start background thread to monitor completion
            def monitor_completion():
                try:
                    future.result(timeout=30.0)
                except Exception as e:
                    logger.debug(f"Playback monitoring: {e}")
                finally:
                    with self.state_lock:
                        self.is_playing = False
                        self.cartesia_future = None
            
            threading.Thread(target=monitor_completion, daemon=True).start()
            
            return "cartesia_streaming"
            
        except Exception as e:
            logger.error(f"❌ Cartesia error: {e}")
            raise  # Let speak() handle fallback
    
    def _speak_system_engine(self, text: str, voice: str = "default",
                            emotive_tags: str = "", enable_barge_in: bool = True) -> str:
        """Speak using SystemEngine."""
        try:
            if not self.engine or not self.stream:
                raise ValueError("SystemEngine not initialized")
            
            logger.info(f"🗣 SystemEngine: {text[:50]}... (barge-in: {enable_barge_in})")
            
            if emotive_tags:
                text = f"{text} {emotive_tags}"
            
            if voice != "default":
                try:
                    self.engine.set_voice(voice)
                except Exception:
                    pass
            
            self._play_system_engine(text)
            return "audio_playing"
            
        except Exception as e:
            logger.error(f"❌ SystemEngine error: {e}")
            return ""
    
    def wait_for_completion(self, timeout: float = 30.0) -> bool:
        """
        Wait for playback to complete or be interrupted.
        For Cartesia: Returns immediately to maintain full-duplex (audio plays in background).
        For SystemEngine: Polls until completion.
        Returns True if completed, False if interrupted.
        """
        try:
            # Cartesia: DON'T BLOCK - audio plays in background thread
            # This allows STT to continue listening during TTS playback (full-duplex)
            if self.use_cartesia and hasattr(self, 'cartesia_future') and self.cartesia_future:
                # Just return immediately - the audio consumer thread handles playback
                # Barge-in detection happens via callback during playback
                return True
            
            # SystemEngine: polling (legacy behavior)
            else:
                start_time = time.time()
                while True:
                    with self.state_lock:
                        if not self.is_playing:
                            return not self.barge_in_detected
                        
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
                self.cartesia_engine.stop_playback()
                logger.info("🛑 Cartesia stopped")
            elif self.stream:
                self.stream.stop()
                logger.info("🛑 SystemEngine stopped")
            
            with self.state_lock:
                self.barge_in_detected = True
            self.stop_event.set()
            
        except Exception as e:
            logger.error(f"Stop error: {e}")
    
    def shutdown(self):
        """Clean shutdown."""
        try:
            logger.info("🧹 Shutting down TTS...")
            
            with self.state_lock:
                self.is_playing = False
            self.stop_event.set()
            
            # Shutdown Cartesia
            if self.use_cartesia and hasattr(self, 'cartesia_engine') and self.cartesia_engine:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self.cartesia_engine.cleanup(),
                        self.cartesia_loop
                    )
                    future.result(timeout=5.0)
                except Exception as e:
                    logger.warning(f"Cartesia cleanup error: {e}")
                
                # Stop event loop
                if self.cartesia_loop:
                    self.cartesia_loop.call_soon_threadsafe(self.cartesia_loop.stop)
                
                self.cartesia_engine = None
                logger.info("✅ Cartesia TTS shutdown complete")
            
            # Shutdown SystemEngine
            if not self.use_cartesia:
                if hasattr(self, 'stream') and self.stream:
                    try:
                        self.stream.stop()
                    except Exception:
                        pass
                    self.stream = None
                
                self.engine = None
                logger.info("✅ SystemEngine shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")
