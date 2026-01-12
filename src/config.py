# config.py - Centralized Configuration with Validation
"""
Central configuration management for Voice MVP project.

Features:
- Environment variable validation
- Default values with fallbacks
- Type-safe configuration access
- Dynamic settings (queue sizes, timeouts, etc.)
- Memory management settings
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API keys and endpoints configuration."""
    
    # LLM APIs (Primary: OpenAI, Fallbacks: Gemini, Groq, Ollama)
    openai_api_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    gemini_api_key: str = field(default_factory=lambda: os.getenv('GEMINI_API_KEY', ''))
    groq_api_key: str = field(default_factory=lambda: os.getenv('GROQ_API_KEY', ''))
    
    # Ollama (local LLM, no API key needed)
    ollama_enabled: bool = field(default_factory=lambda: os.getenv('OLLAMA_ENABLED', 'false').lower() == 'true')
    ollama_base_url: str = field(default_factory=lambda: os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'))
    
    # TTS API
    cartesia_api_key: str = field(default_factory=lambda: os.getenv('CARTESIA_API_KEY', ''))
    
    # API endpoints
    openai_base_url: str = "https://api.openai.com/v1/chat/completions"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/models"
    groq_base_url: str = "https://api.groq.com/openai/v1/chat/completions"
    
    def validate(self) -> bool:
        """Validate required API keys and configuration values."""
        errors = []
        warnings = []
        
        # At least one LLM API key or Ollama required
        if not any([self.openai_api_key, self.gemini_api_key, self.groq_api_key, self.ollama_enabled]):
            errors.append("At least one LLM API key or Ollama required (OpenAI/Gemini/Groq/Ollama)")
        
        # Validate API key formats (basic check)
        if self.openai_api_key and not self.openai_api_key.startswith('sk-'):
            warnings.append("OpenAI API key should start with 'sk-'")
        
        if self.groq_api_key and not self.groq_api_key.startswith('gsk_'):
            warnings.append("Groq API key should start with 'gsk_'")
        
        # TTS API key required
        if not self.cartesia_api_key:
            errors.append("CARTESIA_API_KEY required for TTS")
        
        if self.cartesia_api_key and not self.cartesia_api_key.startswith('sk_car_'):
            warnings.append("Cartesia API key should start with 'sk_car_'")
        
        # Log warnings
        for warning in warnings:
            logger.warning(f"⚠️ Config validation: {warning}")
        
        if errors:
            for error in errors:
                logger.error(f"❌ Config validation: {error}")
            return False
        
        logger.info("✅ API configuration validated")
        return True
    
    def get_llm_priority(self) -> list:
        """Get LLM providers in priority order."""
        providers = []
        if self.openai_api_key:
            providers.append('openai')
        if self.gemini_api_key:
            providers.append('gemini')
        if self.groq_api_key:
            providers.append('groq')
        if self.ollama_enabled:
            providers.append('ollama')
        return providers


@dataclass
class LLMConfig:
    """LLM model configuration."""
    
    # Primary model (OpenAI)
    openai_model: str = "gpt-4o-mini"
    openai_max_tokens: int = 120
    openai_temperature: float = 0.75
    
    # Fallback model (Gemini)
    gemini_model: str = "gemini-1.5-flash"
    gemini_max_tokens: int = 120
    
    # Second fallback (Groq)
    groq_model: str = "llama-3.1-8b-instant"
    groq_max_tokens: int = 120
    
    # Third fallback (Ollama - local)
    ollama_model: str = "llama3.1:latest"  # 8B model
    ollama_max_tokens: int = 120
    
    # Timeouts
    request_timeout: float = 12.0
    connect_timeout: float = 5.0
    ollama_timeout: float = 30.0  # Local models may take longer
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0  # Initial delay (exponential backoff)
    max_retry_delay: float = 10.0
    
    # Context management
    max_history_turns: int = 6
    
    def get_model_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for specific provider."""
        configs = {
            'openai': {
                'model': self.openai_model,
                'max_tokens': self.openai_max_tokens,
                'temperature': self.openai_temperature
            },
            'gemini': {
                'model': self.gemini_model,
                'max_tokens': self.gemini_max_tokens
            },
            'groq': {
                'model': self.groq_model,
                'max_tokens': self.groq_max_tokens
            },
            'ollama': {
                'model': self.ollama_model,
                'max_tokens': self.ollama_max_tokens
            }
        }
        return configs.get(provider, {})


@dataclass
class TTSConfig:
    """TTS configuration."""
    
    # Cartesia TTS
    use_cartesia: bool = field(default_factory=lambda: os.getenv('USE_CARTESIA_TTS', 'true').lower() == 'true')
    cartesia_voice_id: str = "e07c00bc-4134-4eae-9ea4-1a55fb45746b"  # Brooke
    cartesia_model: str = "sonic-3"
    
    # Audio settings
    sample_rate: int = 22050
    channels: int = 1
    chunk_size: int = 1024
    
    # Barge-in settings
    barge_in_startup_buffer: float = 0.15  # Ignore first 150ms of playback
    barge_in_check_interval: float = 0.02  # Check every 20ms
    barge_in_min_chars: int = 2  # Minimum characters to trigger barge-in
    
    # Timeout settings
    cartesia_init_timeout: float = 10.0  # Cartesia initialization timeout
    queue_put_timeout: float = 1.0  # Audio queue put timeout
    playback_timeout: float = 30.0  # Maximum playback wait time
    websocket_timeout: float = 15.0  # WebSocket connection timeout
    
    # Fallback TTS (SystemEngine)
    fallback_enabled: bool = True
    fallback_voice: str = "default"


@dataclass
class STTConfig:
    """STT configuration."""
    
    # RealtimeSTT modes
    mode: str = "accurate"  # Options: fast, balanced, accurate (using base.en for production)
    
    # Model selection by mode (optimized for lower latency)
    models = {
        'fast': 'tiny.en',      # ~100-150ms latency, fastest
        'balanced': 'small.en',    # ~150-250ms latency, better accuracy
        'accurate': 'base.en'   # ~200-350ms latency, best accuracy
    }
    
    # VAD settings
    silero_sensitivity: float = 0.5
    webrtc_sensitivity: int = 3
        # Timeout settings
    transcription_timeout: float = 30.0  # Maximum time to wait for transcription
    def get_model(self) -> str:
        """Get model for current mode."""
        return self.models.get(self.mode, 'base.en')


@dataclass
class MemoryConfig:
    """Memory and performance configuration."""
    
    # Latency mode configuration
    latency_mode: str = "low_latency"  # Options: low_latency, balanced, stable
    
    # Audio queue sizing (dynamic) - adjusted by latency mode
    audio_queue_min_size: int = 50
    audio_queue_max_size: int = 200
    audio_queue_default_size: int = 100
    
    # Memory monitoring
    enable_memory_monitoring: bool = True
    memory_check_interval: float = 5.0  # seconds
    memory_warning_threshold_mb: float = 500.0  # MB
    memory_critical_threshold_mb: float = 1000.0  # MB
    
    # Queue management
    queue_timeout: float = 1.0  # seconds
    queue_high_water_mark: float = 0.8  # 80% full triggers warning
    
    # Performance monitoring
    enable_performance_metrics: bool = True
    latency_warning_threshold_ms: float = 500.0
    
    def get_queue_size_for_mode(self) -> int:
        """Get optimal queue size based on latency mode."""
        mode_sizes = {
            "low_latency": 25,   # ~1-1.5s buffer, minimal latency
            "balanced": 60,       # ~2-3s buffer, good balance
            "stable": 100         # ~4-5s buffer, maximum stability
        }
        size = mode_sizes.get(self.latency_mode, 60)
        logger.debug(f"Queue size for {self.latency_mode} mode: {size}")
        return size
    
    def get_dynamic_queue_size(self, current_usage_percent: float) -> int:
        """
        Calculate optimal queue size based on current usage.
        
        Args:
            current_usage_percent: Current queue usage (0.0 - 1.0)
            
        Returns:
            Optimal queue size
        """
        if current_usage_percent > 0.9:
            # High usage - increase size
            return min(self.audio_queue_max_size, int(self.audio_queue_default_size * 1.5))
        elif current_usage_percent < 0.3:
            # Low usage - decrease size
            return max(self.audio_queue_min_size, int(self.audio_queue_default_size * 0.8))
        else:
            # Normal usage
            return self.audio_queue_default_size


@dataclass
class ErrorRecoveryConfig:
    """Error handling and recovery configuration."""
    
    # Retry configuration
    enable_retry: bool = True
    max_retries: int = 3
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 10.0
    exponential_base: float = 2.0
    
    # Fallback behavior
    enable_fallback: bool = True
    fallback_timeout: float = 5.0
    
    # Error tracking
    max_consecutive_errors: int = 3
    error_cooldown_seconds: float = 30.0
    
    def calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.
        
        Args:
            attempt: Current retry attempt (0-indexed)
            
        Returns:
            Delay in seconds
        """
        delay = self.initial_retry_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_retry_delay)


@dataclass
class AppConfig:
    """Main application configuration."""
    
    api: APIConfig = field(default_factory=APIConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    error_recovery: ErrorRecoveryConfig = field(default_factory=ErrorRecoveryConfig)
    
    # Application settings
    app_name: str = "Voice MVP"
    version: str = "2.0.0"
    debug_mode: bool = field(default_factory=lambda: os.getenv('DEBUG', 'false').lower() == 'true')
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO').upper())
    
    # Conversation settings
    max_conversation_history: int = 10
    enable_personality: bool = True
    enable_sentiment_analysis: bool = True
    
    def validate(self) -> bool:
        """Validate all configuration."""
        logger.info("🔍 Validating configuration...")
        
        # Validate API keys
        if not self.api.validate():
            return False
        
        # Validate numeric ranges
        if not (0.0 <= self.llm.openai_temperature <= 2.0):
            logger.error("❌ LLM temperature must be between 0.0 and 2.0")
            return False
        
        if not (self.memory.audio_queue_min_size <= self.memory.audio_queue_default_size <= self.memory.audio_queue_max_size):
            logger.error("❌ Invalid audio queue size configuration")
            return False
        
        logger.info("✅ Configuration validation successful")
        return True
    
    def get_summary(self) -> str:
        """Get configuration summary."""
        llm_providers = self.api.get_llm_priority()
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║  {self.app_name} v{self.version} - Configuration Summary     
╠══════════════════════════════════════════════════════════════╣
║  LLM: {' → '.join(llm_providers).upper()} (with fallback)
║  TTS: {'Cartesia AI' if self.tts.use_cartesia else 'SystemEngine'}
║  STT: RealtimeSTT ({self.stt.mode} mode)
║  Memory: {self.memory.audio_queue_default_size}-element queue, monitoring {'ON' if self.memory.enable_memory_monitoring else 'OFF'}
║  Error Recovery: Retries={self.error_recovery.max_retries}, Fallback={'ON' if self.error_recovery.enable_fallback else 'OFF'}
║  Debug: {self.debug_mode}, Log Level: {self.log_level}
╚══════════════════════════════════════════════════════════════╝
"""
        return summary


# Global configuration instance
config = AppConfig()


def load_and_validate_config() -> AppConfig:
    """
    Load and validate configuration.
    
    Returns:
        Validated AppConfig instance
        
    Raises:
        ValueError: If configuration is invalid
    """
    global config
    
    if not config.validate():
        raise ValueError("Configuration validation failed. Check environment variables.")
    
    logger.info(config.get_summary())
    return config


def get_config() -> AppConfig:
    """Get current configuration instance."""
    return config


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        cfg = load_and_validate_config()
        print(cfg.get_summary())
        print(f"\nLLM Priority: {cfg.api.get_llm_priority()}")
        print(f"Dynamic queue size (50% usage): {cfg.memory.get_dynamic_queue_size(0.5)}")
        print(f"Retry delay (attempt 2): {cfg.error_recovery.calculate_retry_delay(2)}s")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
