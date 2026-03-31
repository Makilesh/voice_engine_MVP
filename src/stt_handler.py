# stt_handler.py 
from datetime import datetime
import logging
import asyncio
import warnings
import concurrent.futures
import re
import threading
import numpy as np
from RealtimeSTT import AudioToTextRecorder

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)

class STTHandler:
    """Full-duplex STT with continuous real-time transcription."""
    
    # Maximum recorder restart attempts per session (WinError 6 recovery)
    MAX_RECORDER_RESTARTS = 3
    
    # Pre-compiled regex patterns for better performance (no need to compile on every call)
    CORRECTIONS = {
        re.compile(r'\b(Shambhata|Shambla Tech|Shambla|Shamlataq|Shamlaq|Shamlata|Samba|Sharma Tech|Sham Tech|Shamlata Tech)\b', re.IGNORECASE): 'Shamla Tech',
        re.compile(r'\b(eye services?|I services?|A I services?)\b', re.IGNORECASE): 'AI services',
        re.compile(r'\b(A P I|ay pee eye|a p eye)\b', re.IGNORECASE): 'API',
        re.compile(r'\b(block ?chain)\b', re.IGNORECASE): 'blockchain',
        re.compile(r'\b(crypto ?currency|cripto)\b', re.IGNORECASE): 'cryptocurrency',
        re.compile(r'\bwanna\b', re.IGNORECASE): 'want to',
        re.compile(r'\bgonna\b', re.IGNORECASE): 'going to',
        re.compile(r'\bgotta\b', re.IGNORECASE): 'got to',
        re.compile(r'\blemme\b', re.IGNORECASE): 'let me',
    }
    
    def __init__(self, mode: str = "balanced"):
        self.recorder = None
        self.is_listening = False
        self.mode = mode
        self.transcription_count = 0
        self.avg_latency = 0.0
        
        self.model_name = self._select_model(mode)
        
        # CRITICAL: Real-time transcription state (thread-safe)
        self.realtime_text = ""
        self.realtime_lock = threading.Lock()
        
        # FIX: Link STT voice detection to TTS stop for partial barge-in (e.g., "could you..." from logs)
        self.tts_stop_callback = None
        
        # Echo suppression: discard realtime updates while TTS is active
        # (speaker audio feeds back into mic → Whisper hallucinates garbage)
        self.tts_is_active = False
        
        # CRITICAL: Completed transcription queue (non-blocking)
        self.completed_transcriptions = asyncio.Queue()
        self.last_completed_text = ""
        
        # WinError 6 recovery: track restart attempts
        self._recorder_restart_count = 0
        
        logger.info(f"🎤 STT Handler initialized (FULL-DUPLEX mode: {mode}, model: {self.model_name})")
    
    def _select_model(self, mode: str) -> str:
        models = {
            "fast": "tiny.en",
            "balanced": "small.en",
            "accurate": "base.en"
        }
        return models.get(mode, "tiny.en")
    
    def _apply_corrections(self, text: str) -> str:
        if not text:
            return text
        
        original = text
        # Use pre-compiled regex patterns
        for pattern, replacement in self.CORRECTIONS.items():
            text = pattern.sub(replacement, text)
        
        if original != text:
            logger.debug(f"🔧 Corrected: '{original}' → '{text}'")
        
        return text.strip()
    
    def _on_realtime_update(self, text: str):
        """CRITICAL: Called continuously during speech (non-blocking)."""
        # Suppress echo: while TTS is playing, the mic picks up speaker audio
        # and Whisper hallucinates garbage text. Discard it.
        if self.tts_is_active:
            logger.debug(f"🔇 Suppressing echo during TTS: '{text[:30] if text else ''}...'")
            return
        # Log that we're receiving speech - proves VAD is working
        if text and text.strip():
            logger.info(f"🎙️ Heard: '{text}'")
        with self.realtime_lock:
            self.realtime_text = text
    
    def _on_transcription_complete(self, text: str):
        """CRITICAL: Called when speech segment completes (non-blocking)."""
        corrected = self._apply_corrections(text)
        if corrected:
            # Store last completed text (synchronous)
            self.last_completed_text = corrected
            logger.info(f"✅ Completed: {corrected}")
    
    async def start_listening(self):
        """Start CONTINUOUS listening with callbacks."""
        try:
            def init_recorder():
                handler_self = self
                
                def on_realtime_update(text: str):
                    handler_self._on_realtime_update(text)
                    
                    # Only trigger barge-in stop when:
                    # 1. TTS is NOT active (would be echo, not real user speech)
                    # 2. User is actually saying something (not noise)
                    if (not handler_self.tts_is_active
                            and handler_self.tts_stop_callback
                            and text and len(text.strip()) >= 4):
                        handler_self.tts_stop_callback()
                
                def on_transcription_complete(text: str):
                    handler_self._on_transcription_complete(text)
                
                def _build_recorder(device: str, compute_type: str):
                    return AudioToTextRecorder(
                        model=self.model_name,
                        language="en",
                        device=device,
                        compute_type=compute_type,
                        
                        # CRITICAL: Enable real-time callbacks
                        enable_realtime_transcription=True,
                        on_realtime_transcription_update=on_realtime_update,
                        # NOTE: on_transcription_complete not supported by installed RealtimeSTT version
                        realtime_model_type="tiny.en",  # Tiny for real-time preview, main model for final
                        
                        # PRODUCTION OPTIMIZED - Sensitive for reliable speech detection
                        realtime_processing_pause=0.1,
                        post_speech_silence_duration=0.25,  # Reduced for faster response
                        min_length_of_recording=0.2,        # Catch even short utterances
                        min_gap_between_recordings=0.15,
                        pre_recording_buffer_duration=0.3,  # Capture speech start reliably
                        
                        # VAD settings - MORE SENSITIVE to catch speech after TTS
                        silero_sensitivity=0.45,   # Increased from 0.38 (higher = more sensitive)
                        silero_use_onnx=True,
                        webrtc_sensitivity=3,      # Increased from 2 (higher = more sensitive)
                        
                        beam_size=1,  # Fastest: near-realtime (was 3)
                        initial_prompt="Shamla Tech AI services, blockchain, cryptocurrency, DeFi, API, machine learning, automation",
                        use_microphone=True
                    )
                
                # Try CUDA first (RTX 5070 Ti), fall back to CPU if CUDA runtime not installed.
                # Use ctypes DLL check: instant, no subprocess spawned.
                # get_cuda_device_count() only uses NVML (driver API) — not a reliable compute check.
                def _cuda_runtime_available() -> bool:
                    import ctypes
                    for dll in ["cudart64_12.dll", "cudart64_120.dll", "cudart64_115.dll", "cublas64_12.dll"]:
                        try:
                            ctypes.WinDLL(dll)
                            return True
                        except OSError:
                            continue
                    return False

                if _cuda_runtime_available():
                    logger.info("🚀 Whisper: CUDA runtime found — using float16 (RTX 5070 Ti)")
                    try:
                        return _build_recorder("cuda", "float16")
                    except Exception as e:
                        logger.warning(f"⚠️ CUDA recorder failed ({e}), falling back to CPU int8")

                logger.info("ℹ️ Whisper: CUDA runtime DLLs not found — using CPU int8. "
                            "To enable GPU: pip install ctranslate2[cuda12]")
                return _build_recorder("cpu", "int8")
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(init_recorder)
                self.recorder = await asyncio.wait_for(
                    asyncio.wrap_future(future), 
                    timeout=30.0
                )
                self.is_listening = True
                logger.info(f"✅ STT listening continuously (model: {self.model_name})")
                
        except asyncio.TimeoutError:
            logger.error("❌ STT initialization timeout")
            raise TimeoutError("STT initialization timed out")
        except Exception as e:
            logger.error(f"❌ STT start error: {e}")
            raise
    
    def get_realtime_text(self) -> str:
        """Get current real-time transcription (INSTANT, non-blocking)."""
        with self.realtime_lock:
            return self.realtime_text
    
    def clear_realtime_text(self):
        """Clear real-time buffer."""
        with self.realtime_lock:
            self.realtime_text = ""
    
    def flush_recorder(self):
        """Reset STT state after TTS playback.
        NOTE: We do NOT call clear_audio_queue() — it corrupts VAD state on Windows
        and causes recorder.text() to hang. Let VAD naturally handle the transition.
        The tts_is_active flag already prevents echo from being transcribed."""
        self.clear_realtime_text()
        logger.debug("🔄 STT buffer cleared, ready for new speech")

    def get_audio_rms(self) -> float:
        """Compute RMS amplitude of the current mic audio buffer.
        Uses the recorder's internal audio_buffer (always populated, even when not recording).
        Returns 0.0 on error or if no data."""
        if not self.recorder or not hasattr(self.recorder, 'audio_buffer'):
            return 0.0
        try:
            chunks = list(self.recorder.audio_buffer)  # thread-safe deque snapshot
            if not chunks:
                return 0.0
            data = b''.join(chunks)
            # Guard: buffer must be even-length for int16 casting
            if len(data) % 2 != 0:
                return 0.0
            audio = np.frombuffer(data, dtype=np.int16)
            if len(audio) == 0:
                return 0.0
            return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        except Exception:
            return 0.0
    
    async def _restart_recorder(self):
        """Attempt to restart the recorder after WinError 6.
        Returns True if restart succeeded, False otherwise."""
        if self._recorder_restart_count >= self.MAX_RECORDER_RESTARTS:
            logger.error(f"❌ Max recorder restarts ({self.MAX_RECORDER_RESTARTS}) reached this session")
            return False
        
        self._recorder_restart_count += 1
        logger.info(f"🔄 Restarting STT recorder (attempt {self._recorder_restart_count}/{self.MAX_RECORDER_RESTARTS})")
        
        try:
            # Tear down existing recorder properly to avoid zombie processes on Windows
            if self.recorder:
                try:
                    # Stop/abort the recorder to clean up multiprocessing resources
                    if hasattr(self.recorder, 'stop'):
                        self.recorder.stop()
                    elif hasattr(self.recorder, 'abort'):
                        self.recorder.abort()
                except Exception:
                    pass
                self.recorder = None
            self.is_listening = False
            
            # Re-initialize with same parameters
            await self.start_listening()
            logger.info("✅ STT recorder restarted successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Recorder restart failed: {e}")
            return False

    async def get_transcription(self) -> str:
        """
        BLOCKING call to get next completed transcription.
        WARNING: Only use when NOT playing TTS. For full-duplex, use get_realtime_text().
        """
        try:
            if not self.recorder:
                raise ValueError("Recorder not initialized")
            
            start_time = asyncio.get_event_loop().time()
            
            # CRITICAL: Force-reset tts_is_active at the start of every transcription
            # to prevent stuck flag from exceptions in TTS wait_for_completion()
            if self.tts_is_active:
                logger.warning("⚠️ tts_is_active was True at transcription start - forcing reset")
            self.tts_is_active = False
            
            # Clear previous state and any stale echo text
            self.last_completed_text = ""
            self.clear_realtime_text()
            
            # CRITICAL: Force recorder into "listening" state to ensure VAD is active
            # This is essential after TTS playback — the recorder may be in a stale state
            try:
                self.recorder.listen()
                logger.info("🎧 STT: Recorder in listening state, VAD active")
            except Exception as e:
                logger.debug(f"recorder.listen() call: {e}")
            
            # CRITICAL: Run blocking .text() in thread pool with timeout
            loop = asyncio.get_event_loop()
            try:
                # Use config value for timeout
                from config import get_config
                timeout = get_config().stt.transcription_timeout
                text = await asyncio.wait_for(
                    loop.run_in_executor(None, self.recorder.text),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ STT timeout after {timeout}s — no speech detected (normal)")
                return ""
            except OSError as e:
                # WinError 6: handle invalidated by multiprocessing pipe on Windows.
                # Attempt to restart the recorder rather than crash the session.
                if hasattr(e, 'winerror') and e.winerror == 6:
                    logger.warning("⚠️ STT pipe handle invalid (WinError 6) — attempting recorder restart")
                    restarted = await self._restart_recorder()
                    if restarted:
                        logger.info("🎤 Recorder restarted — ready for next transcription")
                    return ""
                raise
            
            latency = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # If recorder.text() returned empty but we have a completed transcription
            # from the callback, use that as fallback
            if not text and self.last_completed_text:
                text = self.last_completed_text
                logger.debug(f"📝 Using fallback from on_transcription_complete: {text}")
            
            if text:
                text = self._apply_corrections(text)
                self.transcription_count += 1
                self.avg_latency = (self.avg_latency * (self.transcription_count - 1) + latency) / self.transcription_count
                logger.info(f"📝 [{latency:.0f}ms] {text}")
                return text
            else:
                return ""
                
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return ""
    
    async def stop_listening(self):
        """Stop listening and cleanup."""
        try:
            if self.recorder:
                self.recorder = None
            self.is_listening = False
            
            if self.transcription_count > 0:
                logger.info(f"📊 Session stats: {self.transcription_count} transcriptions, "
                          f"avg latency: {self.avg_latency:.0f}ms")
            
            logger.info("🎤 STT stopped")
        except Exception as e:
            logger.error(f"❌ Stop error: {e}")
    
    def get_performance_stats(self) -> dict:
        return {
            "model": self.model_name,
            "transcription_count": self.transcription_count,
            "avg_latency_ms": round(self.avg_latency, 1),
            "is_listening": self.is_listening
        }