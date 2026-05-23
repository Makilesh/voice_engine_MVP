
# INFORMATION2 - Personal Memory Notes

These are concise, personal reminders for each file so you (future me) remember what was implemented and why.

## Project origin & start (personal notes)

- Why I started: build a realistic full-duplex voice assistant for live calls/demos — continuous STT, fast TTS, and robust LLM fallback so it works in noisy or flaky network conditions.
- Initial design choices: keep audio path separate from LLM, use `deque` for bounded history, prefer local TTS (Kokoro) when GPU available, fall back to Cartesia for low-latency cloud TTS, and always have a system fallback.
- Milestones I remember:
	1. Prototype STT + simple TTS (proof of concept)
	2. Added deque-based history + simple LLM calls
	3. Implemented echo suppression and barge-in (big UX win)
	4. Optimized TTS: Cartesia streaming + Kokoro wrapper
	5. Added LLM multi-provider fallback and retry/backoff
	6. Hardened Windows recorder restart logic

## Setup & Run (quick commands)

Set required environment variables (example):

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:CARTESIA_API_KEY = "sk_car_..."
$env:USE_CARTESIA_TTS = "true"
$env:OLLAMA_ENABLED = "false"
python src/app/main.py
```

Notes:
- On Windows, `multiprocessing.freeze_support()` is used in `main.py` to avoid spawn issues.
- If using Kokoro (local GPU), ensure CUDA runtime DLLs are available (torch.cuda.is_available()).

## Troubleshooting (cheat sheet)

- No audio / mic not found: check system mic permissions and that no other app holds exclusive access. Logs: `[src/app/stt_handler.py]` initialization.
- RealtimeSTT WinError 6: recorder restart logic is in `start_listening()`; check logs for "recorder restart" messages.
- TTS echoing: verify `main_stt.set_current_tts_text(...)` calls and `tts_is_active` toggles in `tts_handler_optimized.py` + `stt_handler.py`.
- LLM timeouts/failures: check `[src/app/llm_handler.py]` logs; provider fallback order is defined in `APIConfig.get_llm_priority()`.
- High memory: `cartesia_tts_engine_optimized.py` monitors memory and will reduce queue size when thresholds are exceeded.

--


- `src/app/config.py`:
	- Central config and validation. I added LLM provider priority, dynamic queue sizing, memory thresholds, and retry/backoff helpers. Used dataclasses so validation + defaults are easy to adjust.

- `src/app/main.py`:
	- Orchestrator and full-duplex loop. `ConversationManager` uses a `deque(maxlen=...)` to keep the last N turns (cheap O(1) append + automatic pop). Linked `stt_handler.tts_stop_callback = tts_handler.stop_playback` for partial barge-in.

- `src/app/stt_handler.py`:
	- Continuous RealtimeSTT wrapper. Implemented selective echo filtering (word-overlap ratio) to avoid hearing TTS. Exposed `get_realtime_text()` for TTS barge-in checks and added WinError-6 recorder restart logic to recover on Windows.

- `src/app/llm_handler.py`:
	- Multi-provider LLM logic: priority list from config, `_call_llm_with_fallback()` tries providers with exponential backoff. Added `SentimentAnalyzer` and `ConversationalPersonality` to shape short, voice-friendly replies. Message cache to avoid rebuilding history every turn.

- `src/tts_handler_optimized.py`:
	- Hybrid TTS manager. Prefers Kokoro (local GPU) -> Cartesia (API) -> SystemEngine (fallback). Implements barge-in detection via STT callbacks and a small startup grace window; resets STT echo state on play.

- `src/app/cartesia_tts_engine_optimized.py`:
	- Cartesia TTS client: WebSocket stream → bounded `queue.Queue` → PyAudio consumer thread. Handles backpressure, dynamic queue resizing, memory monitoring, and clean shutdown. Uses sentinels (`None`/"STOP") to manage stream boundaries.

- `src/app/kokoro_tts_engine.py`:
	- Local Kokoro engine wrapper. GPU autodetect, `synthesize_and_play()` with polling for barge-in and aggressive stream/stream player force-close to ensure instantaneous stop.

- `src/utils/audio_utils.py`:
	- Small utilities (RMS, format conversion) used by STT/TTS. Keep audio math centralized.

- `src/app/tts_handler.py` (archive):
	- Legacy TTS reference kept for fallback and comparison. Don't confuse with optimized handler.

- `README.md`:
	- High-level goals and run notes. Good place to add quick start commands later.

Quick personal reminders:

- If playback echoes are weird, check `_current_tts_text` flow between STT and TTS and `tts_is_active` flag — that's usually the cause.
- On Windows, WinError 6 is still the main pain; `start_listening()` has recorder-restart logic I added; use logs to confirm restarts.
- LLM fallbacks rely on env vars — if a provider randomly fails, make sure the priority list in `APIConfig.get_llm_priority()` is correct.

TODO (future me):

- Add a small test harness to simulate barge-in timings (automate tests for echo-vs-barge-in).
- Wire `INFORMATION2.md` entries to direct links with line numbers for faster navigation.


**src/app/llm_handler.py**
- Provider priority: reads `APIConfig.get_llm_priority()` and tries each provider in order until one succeeds.
- Uses `httpx.AsyncClient` for pooled connections and async POST/stream calls; streaming handler (`_call_openai_streaming`) yields partial sentences via callback.
- Message caching: `message_cache` + `cache_history_id` to avoid rebuilding conversation messages every call (hash of tuple(conversation_history)).
- Retry/fallback: `_call_llm_with_fallback()` retries per-provider using `config.llm.max_retries` and `error_recovery.calculate_retry_delay(attempt)` for exponential backoff.
- Provider adapters: `_call_openai`, `_call_gemini`, `_call_groq`, `_call_ollama` include provider-specific payload shaping (Gemini turn merging, Ollama timeout handling).
- Response shaping: `SentimentAnalyzer` → system prompt via `_build_dynamic_system_prompt()` (LRU cached) → `_post_process_response()` adds fillers/continuer and cleans prefixes.
- Error/circuit tracking: `consecutive_errors`, `last_error_time` used to log and surface when many failures occur; `shutdown()` closes `httpx` client.


**src/tts_handler_optimized.py**
- Engine priority: Kokoro (local GPU) → Cartesia (cloud streaming) → SystemEngine fallback.
- Barge-in hybrid: Kokoro uses RMS energy + STT callback; Cartesia uses STT callback + frequent chunk checks; SystemEngine uses monitor thread.
- Echo check: `_is_echo()` computes word-overlap ratio (>=30% of STT words in current TTS text → treat as echo).
- Grace periods: `_barge_in_grace_period` computed dynamically (per-word heuristics) to avoid false positives from speaker bleed.
- Concurrency: separate asyncio loops in threads for async engines; `state_lock` + `stop_event` protect `is_playing` and barge-in signals.
- STT coordination: sets `main_stt.tts_is_active`, `set_current_tts_text()`, and clears/flushes STT after playback in all exit paths (via `_ensure_stt_active`).


**src/app/cartesia_tts_engine_optimized.py**
- Uses `AsyncCartesia` WebSocket to stream audio chunks; `_stream_audio_from_websocket()` pushes bytes into a bounded `queue.Queue`.
- Consumer thread (`_audio_consumer_loop`) reads queue and writes to PyAudio; uses sentinels `None` (end-of-stream) and "STOP" to control flow.
- Barge-in: `barge_in_callback` is checked both while streaming (before queue.put) and inside consumer loop to stop and clear queue immediately.
- Backpressure & memory: dynamic `current_queue_size` adjusted by `_adjust_queue_size()`; `_check_memory_usage()` uses `psutil` and config thresholds to shrink queue under pressure.
- Robust shutdown: `cleanup()` cancels pending async tasks, joins consumer thread, terminates PyAudio, and closes client connections.


**src/app/kokoro_tts_engine.py**
- Local GPU engine wrapper around `RealtimeTTS.KokoroEngine` + `TextToAudioStream` with `initialize_sync()` for model loading.
- `synthesize_and_play()` calls `play_async()` and polls `is_playing()` while checking `barge_in_callback()` after a brief startup buffer.
- Stop/force-close: `stop_playback()` attempts graceful stop then force-clears audio buffer and force-closes underlying PyAudio stream to prevent lingering audio.
- Device-aware: auto-detects CUDA via `torch.cuda.is_available()` and logs GPU name when present.
- Diagnostics: `get_performance_stats()` returns memory, device, and voice info for debugging.


**src/app/config.py**
- Central single-source configuration using dataclasses: `APIConfig`, `LLMConfig`, `TTSConfig`, `STTConfig`, `MemoryConfig`, `ErrorRecoveryConfig`, and `AppConfig`.
- Validation: `APIConfig.validate()` enforces presence/format of API keys (OpenAI, Groq, Cartesia) and `AppConfig.validate()` runs broader checks (temperature ranges, queue size invariants).
- LLM priority: `APIConfig.get_llm_priority()` returns providers ordered by preference (groq → ollama → gemini → openai) based on env/API presence.
- Dynamic queue sizing: `MemoryConfig.get_queue_size_for_mode()` and `get_dynamic_queue_size()` encapsulate rules used by Cartesia engine to resize audio queue under load.
- Error recovery: `ErrorRecoveryConfig.calculate_retry_delay(attempt)` implements exponential backoff with caps used by `_call_llm_with_fallback()`.
- Barge-in tuning: `TTSConfig` and `STTConfig` expose knobs (`barge_in_startup_buffer`, `barge_in_check_interval`, `barge_in_min_chars`, model choices) you tweak when adjusting responsiveness vs false positives.
- Usage: call `load_and_validate_config()` early in `main.py` to fail fast on missing keys; `get_config()` returns the global `config` singleton.

Quick notes for future debugging:
- To change provider order globally, edit `APIConfig.get_llm_priority()` or set env flags to prefer Ollama/Groq.
- Lower `memory.memory_warning_threshold_mb` if running on small VMs to force earlier queue shrinking.
- Set `DEBUG=true` to enable more verbose logs and get the configuration summary printed at startup.


**src/app/main.py**
- Orchestrator and full-duplex loop. `ConversationManager` uses a `deque(maxlen=...)` to hold recent turns and provides simple error/circuit logic (`should_abort`).
- Turn flow: `get_transcription()` (blocking) → `process_text_with_history()` → `tts_handler.speak()` → `wait_for_completion()`; supports interruption handling by checking `tts_handler.is_barge_in_detected()` and processing `stt_handler.get_realtime_text()`.
- Logging and safety: suppresses noisy warnings, filters WinError 6 logs, defers config loading to protect Windows multiprocessing spawn, and uses `multiprocessing.freeze_support()` for Windows executables.
- Startup sequence: STT initialized first (continuous listen), then LLM, then TTS (STT passed into TTS for barge-in coordination). Welcome message is spoken without barge-in enabled.
- Timing: measures per-turn timing (STT, LLM, TTS) and warns if slow; limits max_turns and gracefully shuts down services in `finally`.


**src/app/stt_handler.py**
- Continuous RealtimeSTT wrapper with CUDA-detection fallback to CPU. Uses `AudioToTextRecorder` with real-time callbacks for barge-in and completed transcription handling.
- Selective echo filtering: `_is_likely_echo()` compares word-overlap ratio vs `_current_tts_text` to ignore assistant echo while allowing different user speech to trigger barge-in.
- Two listening paths: `get_transcription()` (blocking, for normal turns) and `get_realtime_text()` (non-blocking, exposes live partial text used by TTS for immediate interruption handling).
- Robustness: WinError 6 recovery via `_restart_recorder()` with limited retries; uses threadpool to initialize recorder to avoid blocking the event loop.
- Audio metrics: `get_audio_rms()` exposes mic RMS for RMS-based barge-in in Kokoro engine; `transcription_count` and `avg_latency` tracked for session stats.
- Best-practice notes: never call recorder audio queue drains after playback (can hang on Windows); prefer `flush_recorder()` which clears realtime text only.


**src/utils/audio_utils.py**
- Small helper utilities: `save_audio_file()`, `load_audio_file()`, `normalize_audio()`, `trim_silence()`, `audio_to_mono()`.
- File I/O uses `wave` module with 16-bit PCM format; `normalize_audio()` scales to [-1,1]; `trim_silence()` uses simple absolute thresholding — useful for quick VAD trimming in tests.


**Archive check: src/app/tts_handler.py**
- File not present at expected path. The repository uses `tts_handler_optimized.py` as the active implementation; legacy `tts_handler.py` was referenced but seems removed or relocated. If you want the legacy code restored, check VCS history or backups.

---

I appended these concise analyses to help you remember logic choices, recovery behavior, and tuning knobs. Next step: I'll mark the remaining todo items complete in the task list.

