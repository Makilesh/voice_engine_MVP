# stt_handler.py 
from datetime import datetime
import logging
import asyncio
import warnings
import concurrent.futures
import re
import threading
from RealtimeSTT import AudioToTextRecorder

from datetime import datetime
import logging
import asyncio
import concurrent.futures
import re
import threading
from RealtimeSTT import AudioToTextRecorder

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)

class STTHandler:
    """Full-duplex STT with continuous real-time transcription."""
    
    CORRECTIONS = {
        r'\b(Shambla Tech|Shambla|Shamlataq|Shamlaq|Shamlata|Samba|Sharma Tech)\b': 'Shamla Tech',
        r'\b(eye services?|I services?|A I services?)\b': 'AI services',
        r'\b(A P I|ay pee eye|a p eye)\b': 'API',
        r'\b(block ?chain)\b': 'blockchain',
        r'\b(crypto ?currency|cripto)\b': 'cryptocurrency',
        r'\bwanna\b': 'want to',
        r'\bgonna\b': 'going to',
        r'\bgotta\b': 'got to',
        r'\blemme\b': 'let me',
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
        
        # CRITICAL: Completed transcription queue (non-blocking)
        self.completed_transcriptions = asyncio.Queue()
        self.last_completed_text = ""
        
        logger.info(f"🎤 STT Handler initialized (FULL-DUPLEX mode: {mode}, model: {self.model_name})")
    
    def _select_model(self, mode: str) -> str:
        models = {
            "fast": "tiny",
            "balanced": "small",
            "accurate": "base"
        }
        return models.get(mode, "small")
    
    def _apply_corrections(self, text: str) -> str:
        if not text:
            return text
        
        original = text
        for pattern, replacement in self.CORRECTIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        if original != text:
            logger.debug(f"🔧 Corrected: '{original}' → '{text}'")
        
        return text.strip()
    
    def _on_realtime_update(self, text: str):
        """CRITICAL: Called continuously during speech (non-blocking)."""
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
                    
                    # FIX: Trigger TTS stop on voice detection
                    if handler_self.tts_stop_callback:
                        handler_self.tts_stop_callback()
                
                def on_transcription_complete(text: str):
                    handler_self._on_transcription_complete(text)
                
                return AudioToTextRecorder(
                    model=self.model_name,
                    language="en",
                    compute_type="int8",
                    
                    # CRITICAL: Enable real-time callbacks
                    enable_realtime_transcription=True,
                    on_realtime_transcription_update=on_realtime_update,
                    # on_transcription_complete=on_transcription_complete,
                    realtime_model_type=self.model_name,
                    
                    # OPTIMIZED timing
                    realtime_processing_pause=0.08,
                    post_speech_silence_duration=0.25,
                    min_length_of_recording=0.3,
                    min_gap_between_recordings=0.08,
                    pre_recording_buffer_duration=0.15,
                    
                    # VAD settings
                    silero_sensitivity=0.5,
                    silero_use_onnx=True,
                    webrtc_sensitivity=2,
                    
                    beam_size=3,
                    initial_prompt="Shamla Tech, AI, blockchain, cryptocurrency, API",
                    use_microphone=True
                )
            
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
    
    async def get_transcription(self) -> str:
        """
        BLOCKING call to get next completed transcription.
        WARNING: Only use when NOT playing TTS. For full-duplex, use get_realtime_text().
        """
        try:
            if not self.recorder:
                raise ValueError("Recorder not initialized")
            
            start_time = asyncio.get_event_loop().time()
            
            # Clear previous state
            self.last_completed_text = ""
            self.clear_realtime_text()
            
            # CRITICAL: Run blocking .text() in thread pool
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, self.recorder.text)
            
            latency = (asyncio.get_event_loop().time() - start_time) * 1000
            
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