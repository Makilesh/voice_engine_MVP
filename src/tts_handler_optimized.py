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

# Try to import Kokoro TTS (optional local GPU engine)
try:
    from kokoro_tts_engine import KokoroTTSEngine, KokoroVoiceConfig
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    KokoroTTSEngine = None
    KokoroVoiceConfig = None
    logging.warning("⚠️ kokoro_tts_engine not found — Kokoro TTS unavailable")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TTSHandler:
    """TTS Handler with Cartesia AI + Barge-in Detection (<150ms)."""
    
    def __init__(self, stt_handler=None, use_kokoro: bool = True, use_cartesia=None):
        try:
            # Validate STT handler (required for barge-in)
            self.main_stt = stt_handler
            if not self.main_stt:
                raise ValueError("STT handler required for barge-in detection")
            
            # Determine TTS engine
            self.use_cartesia = use_cartesia if use_cartesia is not None else (
                os.getenv('USE_CARTESIA_TTS', 'false').lower() == 'true'
            )
            self.use_kokoro = use_kokoro
            
            # State management (thread-safe)
            self.is_playing = False
            self.is_barge_in_enabled = True
            self.barge_in_detected = False
            self.stop_event = threading.Event()
            self.state_lock = threading.Lock()
            
            # Barge-in monitoring
            self.last_seen_realtime_text = ""
            # Use config for sensitivity
            from config import get_config
            self.barge_in_sensitivity = get_config().tts.barge_in_min_chars
            
            # Barge-in error tracking for graceful degradation
            self.barge_in_error_count = 0
            self.max_barge_in_errors = 3
            
            # Async event loop for Cartesia (if needed)
            self.cartesia_loop = None
            self.cartesia_thread = None
            self.cartesia_future = None
            
            # Kokoro engine state
            self.kokoro_engine = None
            self.kokoro_loop = None
            self.kokoro_thread = None
            
            # Active engine tracker
            self.active_engine = "system"
            
            # Initialize TTS engines (priority: kokoro → cartesia → system)
            initialized = False
            
            if self.use_kokoro and KOKORO_AVAILABLE:
                try:
                    self._init_kokoro()
                    self.active_engine = "kokoro"
                    initialized = True
                except Exception as e:
                    logger.warning(f"⚠️ Kokoro init failed, falling through: {e}")
            
            if not initialized and self.use_cartesia:
                try:
                    self._init_cartesia()
                    self.active_engine = "cartesia"
                    initialized = True
                except Exception as e:
                    logger.warning(f"⚠️ Cartesia init failed, falling through: {e}")
            
            if not initialized:
                self._init_system_engine()
                self.active_engine = "system"
                    
        except Exception as e:
            logger.error(f"❌ TTS initialization error: {e}")
            raise
    
    def _init_kokoro(self):
        """Initialize Kokoro-82M local GPU TTS engine."""
        self.kokoro_engine = KokoroTTSEngine(
            voice_config=KokoroVoiceConfig(voice="af_heart", speed=1.0)
        )
        self.kokoro_engine.set_barge_in_callback(self._check_barge_in_status)
        
        self.kokoro_loop = asyncio.new_event_loop()
        self.kokoro_thread = threading.Thread(
            target=self._run_kokoro_loop,
            daemon=True
        )
        self.kokoro_thread.start()
        
        # 60s timeout — first run downloads ~170 MB model
        future = asyncio.run_coroutine_threadsafe(
            self.kokoro_engine.initialize(),
            self.kokoro_loop
        )
        future.result(timeout=60.0)
        
        self.engine = None
        self.stream = None
        self.cartesia_engine = None
        logger.info("✅ TTS: Kokoro-82M local GPU (~100ms latency + barge-in)")
    
    def _run_kokoro_loop(self):
        """Run async event loop for Kokoro in separate thread."""
        asyncio.set_event_loop(self.kokoro_loop)
        self.kokoro_loop.run_forever()
    
    def _init_cartesia(self):
        """Initialize Cartesia AI with optimal config."""
        try:
            self.cartesia_engine = CartesiaTTSEngine(
                voice_config=VoiceConfig(
                    voice_id=CartesiaVoices.BROOKE,
                    model="sonic-3",  # Best quality/speed balance
                    speed=0.70
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
            
            # Detect new speech (thread-safe)
            with self.state_lock:
                if realtime_text and len(realtime_text) >= self.barge_in_sensitivity:
                    if realtime_text != self.last_seen_realtime_text:
                        logger.info(f"🎤 Barge-in detected: {realtime_text[:30]}...")
                        self.barge_in_detected = True
                        self.last_seen_realtime_text = realtime_text
                        return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Barge-in check error: {e}")
            # Track errors for graceful degradation
            self.barge_in_error_count += 1
            if self.barge_in_error_count >= self.max_barge_in_errors:
                logger.error(f"❌ Barge-in disabled due to {self.max_barge_in_errors} consecutive errors")
                self.is_barge_in_enabled = False
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
            tts_startup_buffer = 0.05  # Reduced from 0.15s to 50ms for faster response
            check_interval = 0.01  # Check every 10ms (was 20ms) for quicker detection
            
            # Safety limits to prevent infinite loops
            max_monitor_duration = 60.0  # Maximum 60 seconds
            max_iterations = 3000  # Maximum 3000 iterations
            iterations = 0
            
            logger.info("👂 Barge-in monitor: ACTIVE")
            
            self.last_seen_realtime_text = ""
            consecutive_detections = 0
            min_consecutive = 1  # Instant detection (was 2, causing 40ms delay)
            
            while not self.stop_event.is_set():
                iterations += 1
                elapsed = time.time() - playback_start
                
                # Safety checks to prevent runaway threads
                if iterations > max_iterations:
                    logger.warning(f"⚠️ Monitor exceeded max iterations ({max_iterations}), stopping")
                    break
                if elapsed > max_monitor_duration:
                    logger.warning(f"⚠️ Monitor exceeded max duration ({max_monitor_duration}s), stopping")
                    break
                
                with self.state_lock:
                    if not self.is_playing:
                        break
                
                # Skip TTS startup buffer
                if elapsed < tts_startup_buffer:
                    time.sleep(check_interval)
                    continue
                
                try:
                    realtime_text = self.main_stt.get_realtime_text()
                    
                    # Thread-safe access to last_seen_realtime_text
                    with self.state_lock:
                        if realtime_text and len(realtime_text) > self.barge_in_sensitivity:
                            if realtime_text != self.last_seen_realtime_text:
                                consecutive_detections += 1
                                
                                if consecutive_detections >= min_consecutive:
                                    logger.info(f"🛑 Barge-in: {realtime_text[:30]}...")
                                    self.barge_in_detected = True
                                    self.last_seen_realtime_text = realtime_text
                                    self.stop_event.set()
                                    
                                    if self.stream:
                                        try:
                                            self.stream.stop()
                                        except (RuntimeError, AttributeError) as e:
                                            logger.debug(f"Stream stop error during barge-in (non-critical): {e}")
                                    break
                            else:
                                consecutive_detections = max(0, consecutive_detections - 1)
                        else:
                            # Update last_seen under lock even when no match
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
            # Input validation
            if not text or not isinstance(text, str):
                logger.warning("Invalid input: empty or non-string text")
                return ""
            
            if len(text) > 5000:
                logger.warning(f"Text too long: {len(text)} chars, truncating to 5000")
                text = text[:5000]
            
            # Validate speed parameter
            if not isinstance(speed, (int, float)) or speed <= 0 or speed > 3.0:
                logger.warning(f"Invalid speed {speed}, using 1.0")
                speed = 1.0
            
            self.is_barge_in_enabled = enable_barge_in
            
            # Clear STT buffer before speaking
            if self.main_stt:
                self.main_stt.clear_realtime_text()
                self.last_seen_realtime_text = ""
            
            if self.active_engine == "kokoro" and self.kokoro_engine:
                return self._speak_async_engine(text, enable_barge_in, speed, engine=self.kokoro_engine, loop=self.kokoro_loop)
            elif self.active_engine == "cartesia" and self.cartesia_engine:
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
    
    def _speak_async_engine(self, text: str, enable_barge_in: bool = True,
                            speed: float = 1.0, engine=None, loop=None) -> str:
        """Generic async-engine speak (used for Kokoro and other async engines)."""
        try:
            if not engine or not loop:
                raise ValueError("Async engine or loop not provided")
            
            # Stop any ongoing playback
            with self.state_lock:
                was_playing = self.is_playing
            
            if was_playing:
                logger.info("🛑 Stopping previous TTS before new speech")
                try:
                    asyncio.run_coroutine_threadsafe(
                        engine.stop(), loop
                    ).result(timeout=0.5)
                except TimeoutError:
                    logger.debug("Stop timeout (TTS may have already finished)")
                except Exception as e:
                    if str(e):
                        logger.warning(f"Stop error: {e}")
                time.sleep(0.1)
            
            logger.info(f"🗣 Kokoro: {text[:50]}... (barge-in: {enable_barge_in})")
            
            self.stop_event.clear()
            
            future = asyncio.run_coroutine_threadsafe(
                engine.synthesize_and_play(
                    text=text,
                    enable_barge_in=enable_barge_in
                ),
                loop
            )
            
            with self.state_lock:
                self.is_playing = True
                self.barge_in_detected = False
            
            def monitor_completion():
                try:
                    future.result(timeout=30.0)
                except Exception as e:
                    logger.debug(f"Playback monitoring: {e}")
                finally:
                    with self.state_lock:
                        self.is_playing = False
            
            threading.Thread(target=monitor_completion, daemon=True).start()
            
            return "kokoro_streaming"
            
        except Exception as e:
            logger.error(f"❌ Async engine error: {e}")
            raise
    
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
            
            # Clear stop event from any previous call
            self.stop_event.clear()

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
                except (AttributeError, ValueError) as e:
                    logger.debug(f"Voice change failed (non-critical): {e}")
            
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
            # Async engines (Kokoro / Cartesia): DON'T BLOCK - audio plays in background
            # This allows STT to continue listening during TTS playback (full-duplex)
            if self.active_engine in ("kokoro", "cartesia"):
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
        """Immediately stop TTS playback. No-op if nothing is playing."""
        try:
            with self.state_lock:
                if not self.is_playing:
                    return  # Nothing playing - ignore spurious calls from STT callbacks

            if self.active_engine == "kokoro" and self.kokoro_engine:
                self.kokoro_engine.stop_playback()
                logger.info("🛑 Kokoro stopped")
            elif self.active_engine == "cartesia" and self.cartesia_engine:
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
            
            # Shutdown Kokoro
            if hasattr(self, 'kokoro_engine') and self.kokoro_engine:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.kokoro_engine.cleanup(),
                        self.kokoro_loop
                    ).result(timeout=5.0)
                except Exception as e:
                    logger.warning(f"Kokoro cleanup error: {e}")
                
                if self.kokoro_loop:
                    self.kokoro_loop.call_soon_threadsafe(self.kokoro_loop.stop)
                    if self.kokoro_thread and self.kokoro_thread.is_alive():
                        self.kokoro_thread.join(timeout=3.0)
                    if not self.kokoro_loop.is_closed():
                        self.kokoro_loop.close()
                
                self.kokoro_engine = None
                logger.info("✅ Kokoro TTS shutdown complete")
            
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

                    # Wait for thread to complete
                    if self.cartesia_thread and self.cartesia_thread.is_alive():
                        self.cartesia_thread.join(timeout=2.0)
                        if self.cartesia_thread.is_alive():
                            logger.warning("⚠️ Cartesia thread didn't stop within timeout")

                    # Close event loop
                    if not self.cartesia_loop.is_closed():
                        self.cartesia_loop.close()
                        logger.debug("Cartesia event loop closed")

                self.cartesia_engine = None
                logger.info("✅ Cartesia TTS shutdown complete")
            
            # Shutdown SystemEngine
            if self.active_engine == "system":
                if hasattr(self, 'stream') and self.stream:
                    try:
                        self.stream.stop()
                    except (RuntimeError, AttributeError) as e:
                        logger.debug(f"Stream stop error during shutdown (non-critical): {e}")
                    self.stream = None
                
                self.engine = None
                logger.info("✅ SystemEngine shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")
