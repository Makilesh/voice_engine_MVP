# test_kokoro_local.py - Standalone smoke test for Kokoro TTS Engine
"""
Run from project root:  python src/test_kokoro_local.py
"""
import asyncio
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    from kokoro_tts_engine import KokoroTTSEngine, KokoroVoiceConfig

    results = {"passed": 0, "failed": 0}
    engine = KokoroTTSEngine(voice_config=KokoroVoiceConfig(voice="af_heart"))

    # ---- Test 1: Initialization ----
    print("\n--- Test 1: Initialize engine ---")
    try:
        t0 = time.time()
        await engine.initialize()
        elapsed_ms = (time.time() - t0) * 1000
        print(f"  ✅ PASS — initialized in {elapsed_ms:.0f} ms")
        results["passed"] += 1
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        results["failed"] += 1
        print(f"\nResults: {results['passed']} passed, {results['failed']} failed")
        return

    # ---- Test 2: Speak (no barge-in) ----
    print("\n--- Test 2: Synthesize and play ---")
    try:
        t1 = time.time()
        completed = await engine.synthesize_and_play(
            "Hello! This is a test of the Kokoro local text to speech engine.",
            enable_barge_in=False
        )
        elapsed_ms = (time.time() - t1) * 1000
        if completed:
            print(f"  ✅ PASS — playback completed in {elapsed_ms:.0f} ms")
            results["passed"] += 1
        else:
            print(f"  ❌ FAIL — playback returned False (interrupted?)")
            results["failed"] += 1
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        results["failed"] += 1

    # ---- Test 3: Performance stats ----
    print("\n--- Test 3: Performance stats ---")
    try:
        stats = engine.get_performance_stats()
        print(f"  {stats}")
        print("  ✅ PASS")
        results["passed"] += 1
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        results["failed"] += 1

    # ---- Cleanup ----
    await engine.cleanup()

    # ---- Summary ----
    total = results["passed"] + results["failed"]
    print(f"\n{'='*40}")
    if results["failed"] == 0:
        print(f"  ALL {total} TESTS PASSED ✅")
    else:
        print(f"  {results['passed']}/{total} passed, {results['failed']} FAILED ❌")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
