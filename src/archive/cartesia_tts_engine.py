# cartesia_tts_engine.py - Ultra-Low Latency Cartesia AI TTS Engine
"""
Cartesia AI TTS Engine with WebSocket Streaming
- 40-90ms first-byte latency (Sonic Turbo/Sonic 3)
- Full-duplex support with real-time barge-in detection
- Async-first architecture for non-blocking operation
- Thread-safe state management
- PyAudio integration for real-time playback
"""

import os
import logging
import asyncio
import threading
import time
import queue
from typing import Optional, Callable, Dict, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

import pyaudio
import numpy as np
from cartesia import AsyncCartesia

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce noise from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)


class PlaybackState(Enum):
    """TTS playback states for thread-safe management."""
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    PLAYING = "playing"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AudioConfig:
    """Audio configuration for Cartesia TTS."""
    sample_rate: int = 22050  # Cartesia optimal: 22050, 24000, 44100
    channels: int = 1
    format: int = pyaudio.paFloat32  # pcm_f32le
    encoding: str = "pcm_f32le"
    container: str = "raw"
    chunk_size: int = 1024  # PyAudio buffer size
    
    def to_cartesia_format(self) -> Dict[str, Any]:
        """Convert to Cartesia API output format."""
        return {
            "container": self.container,
            "encoding": self.encoding,
            "sample_rate": self.sample_rate
        }


@dataclass
class VoiceConfig:
    """Voice configuration for Cartesia TTS."""
    voice_id: str = "e07c00bc-4134-4eae-9ea4-1a55fb45746b"  # Default: Brooke
    model: str = "sonic-3"  # sonic-3, sonic-turbo
    language: str = "en"
    speed: float = 1.0  # 0.5 - 2.0
    emotion: Optional[str] = None  # neutral, happy, sad, angry, etc.
    
    def to_cartesia_voice(self) -> Dict[str, str]:
        """Convert to Cartesia API voice format."""
        return {
            "mode": "id",
            "id": self.voice_id
        }

class CartesiaTTSEngine:
    """
    Ultra-low latency Cartesia AI TTS Engine with WebSocket streaming.
    
    Features:
    - 40-90ms first-byte latency
    - Async WebSocket streaming
    - Real-time playback with PyAudio
    - Thread-safe barge-in support
    - Automatic reconnection
    - Performance monitoring
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_config: Optional[VoiceConfig] = None,
        audio_config: Optional[AudioConfig] = None,
        enable_monitoring: bool = True
    ):
        """
        Initialize Cartesia TTS Engine.
        
        Args:
            api_key: Cartesia API key (from env if not provided)
            voice_config: Voice configuration (Brooke by default)
            audio_config: Audio output configuration
            enable_monitoring: Enable performance monitoring
        """
        # API Configuration
        self.api_key = api_key or os.getenv('CARTESIA_API_KEY')
        if not self.api_key:
            raise ValueError("CARTESIA_API_KEY is required. Set environment variable or pass api_key.")
        
        # Voice & Audio Configuration
        self.voice_config = voice_config or VoiceConfig()
        self.audio_config = audio_config or AudioConfig()
        
        # Cartesia Client (async)
        self.client: Optional[AsyncCartesia] = None
        self.websocket = None
        
        # PyAudio for real-time playback
        self.pyaudio_instance: Optional[pyaudio.PyAudio] = None
        self.audio_stream: Optional[pyaudio.Stream] = None
        
        # Thread-safe state management
        self.state = PlaybackState.IDLE
        self.state_lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Audio streaming queue (producer: WebSocket, consumer: PyAudio)
        self.audio_queue: queue.Queue = queue.Queue(maxsize=50)
        
        # Performance monitoring
        self.enable_monitoring = enable_monitoring
        self.stats = {
            "synthesis_count": 0,
            "total_latency_ms": 0.0,
            "first_byte_latency_ms": 0.0,
            "avg_latency_ms": 0.0,
            "total_audio_duration_sec": 0.0,
            "errors": 0
        }
        
        # Barge-in callback (set by TTS handler)
        self.barge_in_callback: Optional[Callable[[], bool]] = None
        
        logger.info(
            f"🎤 Cartesia TTS Engine initialized "
            f"(voice: {self.voice_config.voice_id[:8]}..., "
            f"model: {self.voice_config.model}, "
            f"sample_rate: {self.audio_config.sample_rate}Hz)"
        )

    async def initialize(self):
        """Initialize async components (Cartesia client)."""
        try:
            if not self.client:
                self.client = AsyncCartesia(api_key=self.api_key)
                logger.info("✅ Cartesia client initialized")
            
            # Test connection
            if await self._test_connection():
                logger.info("✅ Cartesia API connection verified")
            else:
                logger.warning("⚠️ Cartesia API connection test failed")
                
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
            raise
    
    async def _test_connection(self) -> bool:
        """Test Cartesia API connection."""
        try:
            if not self.client:
                return False
            
            # Get available voices to verify API key
            voices = await self.client.voices.list()
            voice_list = [v async for v in voices]
            
            if voice_list:
                logger.debug(f"✅ Found {len(voice_list)} available voices")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False
    
    def _init_audio_output(self):
        """Initialize PyAudio for real-time playback."""
        try:
            if not self.pyaudio_instance:
                self.pyaudio_instance = pyaudio.PyAudio()
            
            if not self.audio_stream or not self.audio_stream.is_active():
                self.audio_stream = self.pyaudio_instance.open(
                    format=self.audio_config.format,
                    channels=self.audio_config.channels,
                    rate=self.audio_config.sample_rate,
                    output=True,
                    frames_per_buffer=self.audio_config.chunk_size
                )
                logger.debug("✅ PyAudio stream opened")
                
        except Exception as e:
            logger.error(f"❌ PyAudio initialization error: {e}")
            raise
    
    def _cleanup_audio_output(self):
        """Cleanup PyAudio resources."""
        try:
            if self.audio_stream:
                if self.audio_stream.is_active():
                    self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
            
            # Keep pyaudio_instance alive for reuse
            
        except Exception as e:
            logger.error(f"❌ PyAudio cleanup error: {e}")
    
    async def _stream_audio_from_websocket(
        self,
        text: str,
        voice_config: Optional[VoiceConfig] = None,
        audio_config: Optional[AudioConfig] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio chunks from Cartesia WebSocket.
        
        Args:
            text: Text to synthesize
            voice_config: Voice configuration (uses default if not provided)
            audio_config: Audio configuration (uses default if not provided)
            
        Yields:
            Audio chunks as bytes
        """
        try:
            voice_cfg = voice_config or self.voice_config
            audio_cfg = audio_config or self.audio_config
            
            # Connect to WebSocket
            if not self.websocket:
                self.websocket = await self.client.tts.websocket()
                logger.debug("✅ WebSocket connected")
            
            # Prepare synthesis request
            output_format = audio_cfg.to_cartesia_format()
            voice = voice_cfg.to_cartesia_voice()
            
            # Start streaming synthesis
            logger.debug(f"🎙 Streaming synthesis: '{text[:50]}...'")
            
            first_chunk = True
            start_time = time.time()
            
            # Get the async generator from send()
            response = await self.websocket.send(
                model_id=voice_cfg.model,
                transcript=text,
                voice=voice,
                language=voice_cfg.language,
                output_format=output_format,
                stream=True
            )
            
            # Now iterate over the async generator
            async for chunk in response:
                # Check for stop signal
                if self.stop_event.is_set():
                    logger.debug("🛑 Stop signal received during streaming")
                    break
                
                # Extract audio data
                audio_data = chunk.audio if hasattr(chunk, 'audio') else chunk.get('audio')
                
                if audio_data:
                    if first_chunk and self.enable_monitoring:
                        first_byte_latency = (time.time() - start_time) * 1000
                        self.stats["first_byte_latency_ms"] = first_byte_latency
                        logger.info(f"⚡ First byte latency: {first_byte_latency:.1f}ms")
                        first_chunk = False
                    
                    yield audio_data
            
            # Update stats
            if self.enable_monitoring:
                total_latency = (time.time() - start_time) * 1000
                self.stats["synthesis_count"] += 1
                self.stats["total_latency_ms"] += total_latency
                self.stats["avg_latency_ms"] = (
                    self.stats["total_latency_ms"] / self.stats["synthesis_count"]
                )
                logger.debug(f"✅ Synthesis completed in {total_latency:.1f}ms")
            
        except Exception as e:
            logger.error(f"❌ WebSocket streaming error: {e}")
            self.stats["errors"] += 1
            raise
    
    async def _audio_producer(
        self,
        text: str,
        voice_config: Optional[VoiceConfig] = None,
        audio_config: Optional[AudioConfig] = None
    ):
        """
        Producer: Stream audio from WebSocket to queue.
        Runs in async context.
        """
        try:
            async for audio_chunk in self._stream_audio_from_websocket(
                text, voice_config, audio_config
            ):
                if self.stop_event.is_set():
                    break
                
                # Put audio in queue for consumer
                try:
                    self.audio_queue.put(audio_chunk, timeout=1.0)
                except queue.Full:
                    logger.warning("⚠️ Audio queue full, dropping chunk")
            
            # Signal end of stream
            self.audio_queue.put(None)
            logger.debug("✅ Audio producer finished")
            
        except Exception as e:
            logger.error(f"❌ Audio producer error: {e}")
            self.audio_queue.put(None)  # Signal error/end
    
    def _audio_consumer(self):
        """
        Consumer: Play audio from queue using PyAudio.
        Runs in separate thread.
        """
        try:
            self._init_audio_output()
            
            with self.state_lock:
                self.state = PlaybackState.PLAYING
            
            logger.debug("🔊 Audio consumer started")
            
            while not self.stop_event.is_set():
                # Check for barge-in
                if self.barge_in_callback and self.barge_in_callback():
                    logger.info("🎤 Barge-in detected by consumer")
                    self.stop_event.set()
                    break
                
                try:
                    # Get audio chunk from queue (with timeout)
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    
                    if audio_chunk is None:  # End of stream signal
                        logger.debug("✅ End of stream received")
                        break
                    
                    # Play audio chunk
                    if self.audio_stream and self.audio_stream.is_active():
                        self.audio_stream.write(audio_chunk)
                    
                except queue.Empty:
                    continue  # No data yet, keep waiting
                except Exception as e:
                    logger.error(f"❌ Playback error: {e}")
                    break
            
            logger.debug("✅ Audio consumer finished")
            
        except Exception as e:
            logger.error(f"❌ Audio consumer error: {e}")
        finally:
            with self.state_lock:
                self.state = PlaybackState.STOPPED
    
    async def synthesize_and_play(
        self,
        text: str,
        voice_config: Optional[VoiceConfig] = None,
        audio_config: Optional[AudioConfig] = None,
        enable_barge_in: bool = True
    ) -> bool:
        """
        Synthesize text and play audio with real-time streaming.
        
        Args:
            text: Text to synthesize
            voice_config: Voice configuration
            audio_config: Audio configuration
            enable_barge_in: Enable barge-in detection
            
        Returns:
            True if completed successfully, False if interrupted
        """
        try:
            # Initialize if needed
            if not self.client:
                await self.initialize()
            
            # Reset state
            with self.state_lock:
                self.state = PlaybackState.STREAMING
            self.stop_event.clear()
            self.audio_queue.queue.clear()  # Clear any old data
            
            # Start audio consumer in separate thread
            consumer_thread = threading.Thread(
                target=self._audio_consumer,
                daemon=True,
                name="CartesiaTTS-Consumer"
            )
            consumer_thread.start()
            
            # Run audio producer (async)
            await self._audio_producer(text, voice_config, audio_config)
            
            # Wait for consumer to finish
            consumer_thread.join(timeout=30.0)
            
            # Check if completed or interrupted
            completed = not self.stop_event.is_set()
            
            if completed:
                logger.debug("✅ Playback completed successfully")
            else:
                logger.debug("🛑 Playback interrupted")
            
            return completed
            
        except Exception as e:
            logger.error(f"❌ Synthesis/playback error: {e}")
            self.stats["errors"] += 1
            return False
        finally:
            self._cleanup_audio_output()
            with self.state_lock:
                self.state = PlaybackState.IDLE
    
    def stop_playback(self):
        """Immediately stop audio playback."""
        try:
            logger.debug("🛑 Stopping playback...")
            with self.state_lock:
                self.state = PlaybackState.STOPPING
            
            self.stop_event.set()
            
            # Clear audio queue
            try:
                while not self.audio_queue.empty():
                    self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Stop audio stream
            if self.audio_stream and self.audio_stream.is_active():
                self.audio_stream.stop_stream()
            
            logger.info("✅ Playback stopped")
            
        except Exception as e:
            logger.error(f"❌ Stop playback error: {e}")
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        with self.state_lock:
            return self.state in [PlaybackState.STREAMING, PlaybackState.PLAYING]
    
    def get_state(self) -> PlaybackState:
        """Get current playback state."""
        with self.state_lock:
            return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.stats.copy()
    
    def set_barge_in_callback(self, callback: Callable[[], bool]):
        """
        Set barge-in detection callback.
        
        Args:
            callback: Function that returns True if barge-in detected
        """
        self.barge_in_callback = callback
        logger.debug("✅ Barge-in callback registered")
    
    async def cleanup(self):
        """Clean up all resources."""
        try:
            logger.info("🧹 Cleaning up Cartesia TTS Engine...")
            
            # Stop any active playback
            self.stop_playback()
            
            # Wait for state to settle
            await asyncio.sleep(0.1)
            
            # Close WebSocket
            if self.websocket:
                try:
                    await self.websocket.close()
                    self.websocket = None
                except Exception as e:
                    logger.error(f"WebSocket close error: {e}")
            
            # Close Cartesia client
            if self.client:
                try:
                    await self.client.close()
                    self.client = None
                except Exception as e:
                    logger.error(f"Client close error: {e}")
            
            # Cleanup PyAudio
            self._cleanup_audio_output()
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
            
            logger.info("✅ Cartesia TTS Engine cleanup complete")
            
            # Print stats if monitoring enabled
            if self.enable_monitoring and self.stats["synthesis_count"] > 0:
                logger.info(
                    f"📊 Session Stats: "
                    f"{self.stats['synthesis_count']} syntheses, "
                    f"avg latency: {self.stats['avg_latency_ms']:.1f}ms, "
                    f"first byte: {self.stats['first_byte_latency_ms']:.1f}ms, "
                    f"errors: {self.stats['errors']}"
                )
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")


# Voice ID Constants for common Cartesia voices
class CartesiaVoices:
    """Common Cartesia voice IDs."""
    BROOKE = "e07c00bc-4134-4eae-9ea4-1a55fb45746b"  # Female, American
    CLYDE = "2ee87190-8f84-4925-97da-e52547f9462c"   # Male, American
    EDDIE = "63ff761f-c1e8-414b-b969-d1833d1c870c"   # Male, American
    ALEX = "694f9389-aac1-45b6-b726-9d9369183238"    # Male, British
    EMILY = "6906f7e1-52e8-4f78-98a0-e6e4ecf4d8d5"   # Female, British


# Example usage and testing
async def test_cartesia_tts():
    """Test Cartesia TTS Engine with various scenarios."""
    try:
        print("🎙 Testing Cartesia TTS Engine...\n")
        
        # Initialize engine
        engine = CartesiaTTSEngine(
            voice_config=VoiceConfig(
                voice_id=CartesiaVoices.BROOKE,
                model="sonic-3"  # Use sonic-3 for best quality
            ),
            audio_config=AudioConfig(sample_rate=22050)
        )
        
        await engine.initialize()
        
        # Test 1: Simple synthesis
        print("Test 1: Simple synthesis")
        await engine.synthesize_and_play(
            "Hello! I'm Brooke from Shamla Tech. How can I help you today?"
        )
        
        await asyncio.sleep(1)
        
        # Test 2: Longer text
        print("\nTest 2: Longer text with natural speech")
        await engine.synthesize_and_play(
            "Shamla Tech specializes in cutting-edge AI solutions, blockchain technology, "
            "and cryptocurrency services. Our team is dedicated to delivering innovative "
            "solutions that drive real business value."
        )
        
        await asyncio.sleep(1)
        
        # Test 3: Barge-in simulation
        print("\nTest 3: Barge-in simulation")
        
        # Start long speech
        task = asyncio.create_task(
            engine.synthesize_and_play(
                "This is a long message that will be interrupted. I'm going to keep talking "
                "for a while to demonstrate the barge-in functionality. You can interrupt me "
                "at any time and I will stop speaking immediately. This is very important for "
                "natural conversation flow."
            )
        )
        
        # Simulate barge-in after 2 seconds
        await asyncio.sleep(2)
        print("🎤 Simulating barge-in...")
        engine.stop_playback()
        
        await task
        
        # Print stats
        stats = engine.get_stats()
        print(f"\n📊 Performance Stats:")
        print(f"   Syntheses: {stats['synthesis_count']}")
        print(f"   Avg Latency: {stats['avg_latency_ms']:.1f}ms")
        print(f"   First Byte: {stats['first_byte_latency_ms']:.1f}ms")
        print(f"   Errors: {stats['errors']}")
        
        # Cleanup
        await engine.cleanup()
        print("\n✅ Tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_cartesia_tts())