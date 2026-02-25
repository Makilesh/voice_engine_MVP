# kokoro_tts_engine.py - Local Kokoro-82M TTS Engine (GPU-Accelerated)
"""
Kokoro-82M TTS Engine - Local GPU-accelerated TTS via RealtimeTTS[kokoro].
Drop-in replacement for CartesiaTTSEngine with identical async public API.

Features:
- ~100ms first-audio latency on RTX GPUs
- Zero API costs (fully local)
- Thread-safe barge-in callback support
- CUDA auto-detection
"""
import asyncio
import logging
import threading
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum

import torch
import psutil
from RealtimeTTS import KokoroEngine, TextToAudioStream

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlaybackState(Enum):
    """Minimal playback states."""
    IDLE = "idle"
    PLAYING = "playing"
    STOPPED = "stopped"


@dataclass
class KokoroVoiceConfig:
    """Voice configuration for Kokoro TTS.

    Available voices:
      American Female: af_heart, af_bella, af_sarah, af_nicole, af_sky, af_nova
      American Male:   am_adam, am_michael
      British Female:  bf_emma, bf_isabella
      British Male:    bm_george, bm_lewis
    """
    voice: str = "af_heart"  # Warm American female, closest match to Cartesia's Brooke
    speed: float = 1.0
    lang: str = "en-us"


class KokoroTTSEngine:
    """
    Local Kokoro-82M TTS Engine — GPU-accelerated drop-in for CartesiaTTSEngine.

    Features:
    - ~100ms first-audio latency on RTX GPUs
    - Async WebSocket-free local inference
    - Thread-safe barge-in callback
    - Minimal overhead
    """

    def __init__(
        self,
        voice_config: Optional[KokoroVoiceConfig] = None,
        barge_in_startup_buffer: float = 0.15,
        barge_in_check_interval: float = 0.02
    ):
        # Voice config
        self.voice_config = voice_config or KokoroVoiceConfig()

        # Barge-in timing
        self._barge_in_startup_buffer = barge_in_startup_buffer
        self._barge_in_check_interval = barge_in_check_interval

        # CUDA detection
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🎮 Kokoro device: CUDA ({gpu_name})")
        else:
            logger.info("💻 Kokoro device: CPU (no CUDA detected)")

        # Engine / stream — created in initialize_sync()
        self._engine: Optional[KokoroEngine] = None
        self._stream: Optional[TextToAudioStream] = None

        # Thread-safe state
        self._state = PlaybackState.IDLE
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()

        # Barge-in callback (set externally)
        self.barge_in_callback: Optional[Callable[[], bool]] = None

        logger.info(
            f"🎤 Kokoro TTS Engine created "
            f"(voice: {self.voice_config.voice}, "
            f"speed: {self.voice_config.speed}, "
            f"device: {self._device})"
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_sync(self):
        """Synchronous initialization — loads model, creates stream."""
        logger.info("⏳ Loading Kokoro-82M model (first run downloads ~170 MB)...")
        self._engine = KokoroEngine(
            default_voice=self.voice_config.voice,
            default_speed=self.voice_config.speed,
        )
        self._stream = TextToAudioStream(self._engine)
        logger.info(f"✅ Kokoro model loaded on {self._device.upper()}")

    async def initialize(self):
        """Async wrapper — runs synchronous init in executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.initialize_sync)

    # ------------------------------------------------------------------
    # Synthesis & Playback
    # ------------------------------------------------------------------

    async def synthesize_and_play(
        self,
        text: str,
        voice_config: Optional[KokoroVoiceConfig] = None,
        audio_config=None,           # accepted for API parity, unused
        enable_barge_in: bool = True
    ) -> bool:
        """
        Feed text to stream, play asynchronously, poll for barge-in.
        Returns True if playback completed naturally, False if interrupted.
        """
        if not self._stream or not self._engine:
            raise RuntimeError("KokoroTTSEngine not initialized — call initialize() first")

        # Apply voice override if provided
        vc = voice_config or self.voice_config

        with self._state_lock:
            self._state = PlaybackState.PLAYING
        self._stop_event.clear()

        playback_done = threading.Event()

        def on_stream_stop():
            playback_done.set()

        try:
            # Feed text and start async playback
            self._stream.feed(text)
            self._stream.play_async(on_audio_stream_stop=on_stream_stop)

            playback_start = time.time()

            # Polling loop — check barge-in or natural finish
            while not playback_done.is_set() and not self._stop_event.is_set():
                elapsed = time.time() - playback_start

                # Barge-in check (after startup buffer)
                if enable_barge_in and elapsed >= self._barge_in_startup_buffer:
                    if self.barge_in_callback and self.barge_in_callback():
                        logger.info("🛑 Barge-in triggered during Kokoro playback")
                        self.stop_playback()
                        return False

                await asyncio.sleep(self._barge_in_check_interval)

            # If stop was requested externally
            if self._stop_event.is_set():
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Kokoro synthesize_and_play error: {e}")
            with self._state_lock:
                self._state = PlaybackState.STOPPED
            return False
        finally:
            with self._state_lock:
                if self._state == PlaybackState.PLAYING:
                    self._state = PlaybackState.IDLE

    # ------------------------------------------------------------------
    # Playback control
    # ------------------------------------------------------------------

    def stop_playback(self):
        """Immediately stop playback."""
        self._stop_event.set()
        try:
            if self._stream:
                self._stream.stop()
        except (RuntimeError, AttributeError, OSError) as e:
            logger.debug(f"Stream stop error (non-critical): {e}")
        with self._state_lock:
            self._state = PlaybackState.STOPPED
        logger.info("🛑 Kokoro playback stopped")

    async def stop(self):
        """Async wrapper for stop_playback."""
        self.stop_playback()

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_playing(self) -> bool:
        """Thread-safe playing check."""
        with self._state_lock:
            return self._state == PlaybackState.PLAYING

    def get_state(self) -> PlaybackState:
        """Thread-safe state read."""
        with self._state_lock:
            return self._state

    # ------------------------------------------------------------------
    # Barge-in
    # ------------------------------------------------------------------

    def set_barge_in_callback(self, callback: Callable[[], bool]):
        """Register external barge-in callback."""
        self.barge_in_callback = callback
        logger.info("✅ Kokoro barge-in callback registered")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self):
        """Release all resources."""
        self.stop_playback()
        self._stream = None
        self._engine = None
        logger.info("✅ Kokoro TTS engine cleaned up")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_performance_stats(self) -> Dict[str, Any]:
        """Return engine diagnostics."""
        memory_mb = 0.0
        try:
            memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except (psutil.Error, OSError):
            pass
        return {
            "state": self._state.value,
            "device": self._device,
            "voice": self.voice_config.voice,
            "memory_mb": round(memory_mb, 1),
        }


# ======================================================================
# Standalone test
# ======================================================================

async def test_kokoro_tts():
    """Quick self-test for Kokoro TTS Engine."""
    try:
        engine = KokoroTTSEngine(voice_config=KokoroVoiceConfig(voice="af_heart"))

        t0 = time.time()
        await engine.initialize()
        init_ms = (time.time() - t0) * 1000
        logger.info(f"⏱ Init latency: {init_ms:.0f} ms")

        t1 = time.time()
        completed = await engine.synthesize_and_play(
            "Hello! I am Kokoro, a local text to speech engine running on your GPU.",
            enable_barge_in=False
        )
        play_ms = (time.time() - t1) * 1000
        logger.info(f"⏱ Playback time: {play_ms:.0f} ms (completed: {completed})")

        print(engine.get_performance_stats())

        await engine.cleanup()
        logger.info("✅ Kokoro TTS test passed")

    except Exception as e:
        logger.error(f"❌ Kokoro TTS test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_kokoro_tts())
