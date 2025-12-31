# llm_handler.py - OPTIMIZED VERSION with Multi-Provider Fallback
import os
import logging
import asyncio
import httpx
import random
import re
import time
from functools import lru_cache
from dotenv import load_dotenv
from typing import Dict, Optional
from config import get_config

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConversationalPersonality:
    """Manages personality traits and natural language variations."""
    
    FILLERS = {
        'thinking': ['Um,', 'Well,', 'Let me think,', 'Hmm,'],
        'transition': ['So,', 'Actually,', 'Right,'],
        'agreement': ['Absolutely,', 'Yeah,', 'For sure,']
    }
    
    ERROR_RESPONSES = [
        "Oops, I'm having a bit of trouble. Give me a moment?",
        "Hmm, something's not working right. Can you try again?",
        "I'm hitting a snag here. Could you rephrase that?"
    ]
    
    CONTINUERS = [
        "Anything else I can help with?",
        "What else can I do for you?",
        "Does that help?"
    ]
    
    @staticmethod
    def add_natural_pause(text: str, probability: float = 0.12) -> str:
        """Adds occasional filler words (reduced frequency)."""
        if random.random() > probability or not text:
            return text
        
        category = random.choice(list(ConversationalPersonality.FILLERS.keys()))
        filler = random.choice(ConversationalPersonality.FILLERS[category])
        
        if category in ['thinking', 'transition']:
            return f"{filler} {text}"
        else:
            sentences = text.split('. ')
            if len(sentences) > 1:
                return '. '.join(sentences[:1]) + f" {filler}. " + '. '.join(sentences[1:])
        return text
    
    @staticmethod
    def get_random_error() -> str:
        return random.choice(ConversationalPersonality.ERROR_RESPONSES)
    
    @staticmethod
    def add_continuer(text: str, probability: float = 0.25) -> str:
        """Occasionally adds follow-up question."""
        if random.random() > probability or '?' in text:
            return text
        continuer = random.choice(ConversationalPersonality.CONTINUERS)
        return f"{text} {continuer}"


class SentimentAnalyzer:
    """Simple sentiment detection for tone adjustment."""
    
    # Use sets for O(1) lookup instead of O(n) list iteration
    POSITIVE_WORDS = {'great', 'awesome', 'excellent', 'love', 'fantastic', 'happy', 'thanks', 'perfect', 'wonderful', 'amazing'}
    NEGATIVE_WORDS = {'frustrated', 'annoyed', 'upset', 'confused', 'problem', 'issue', 'broken', 'help', 'bad', 'terrible', 'wrong'}
    URGENT_WORDS = {'urgent', 'asap', 'quickly', 'immediately', 'emergency', 'now', 'fast', 'hurry'}
    
    # Pre-compile regex pattern for word boundaries
    _word_pattern = re.compile(r'\b\w+\b')
    
    @staticmethod
    def analyze(text: str) -> Dict:
        """Returns sentiment and intensity."""
        text_lower = text.lower()
        
        # Extract words once using pre-compiled regex
        words = set(SentimentAnalyzer._word_pattern.findall(text_lower))
        
        # Use set intersection for fast lookup
        positive_count = len(words & SentimentAnalyzer.POSITIVE_WORDS)
        negative_count = len(words & SentimentAnalyzer.NEGATIVE_WORDS)
        urgent_count = len(words & SentimentAnalyzer.URGENT_WORDS)
        
        if urgent_count > 0:
            sentiment = 'urgent'
        elif negative_count > positive_count:
            sentiment = 'negative'
        elif positive_count > negative_count:
            sentiment = 'positive'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'intensity': max(positive_count, negative_count, urgent_count),
            'has_question': '?' in text
        }


class LLMHandler:
    """Handles text processing with multi-provider fallback and retry logic."""
    
    def __init__(self):
        # Load configuration
        self.config = get_config()
        
        # Get LLM provider priority
        self.providers = self.config.api.get_llm_priority()
        if not self.providers:
            raise ValueError("At least one LLM API key required")
        
        self.current_provider = self.providers[0]
        logger.info(f"🤖 LLM Providers: {' → '.join(self.providers).upper()}")
        
        self.personality = ConversationalPersonality()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Async HTTP client (reusable connection pool)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                self.config.llm.request_timeout,
                connect=self.config.llm.connect_timeout
            ),
            limits=httpx.Limits(max_keepalive_connections=5)
        )

        # Message cache for conversation history (saves 20-50ms per request)
        self.message_cache = []
        self.cache_history_id = None

        self.interaction_count = 0
        self.consecutive_errors = 0
        self.last_error_time = 0
        
        logger.info(f"🤖 LLM Handler initialized (primary: {self.current_provider.upper()})")
    
    async def process_text(self, text: str) -> str:
        """Process text with sentiment-aware response (FAST mode)."""
        try:
            # Input validation
            if not text or not isinstance(text, str):
                logger.warning("Invalid input: empty or non-string text")
                return self.personality.get_random_error()
            
            if len(text) > 1000:
                logger.warning(f"Input too long: {len(text)} chars, truncating to 1000")
                text = text[:1000]
            
            sentiment = self.sentiment_analyzer.analyze(text)
            processed_text = self._preprocess_transcription(text)
            # Pass sentiment as string key for LRU cache
            system_prompt = self._build_dynamic_system_prompt(sentiment['sentiment'])
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": processed_text}
                ],
                "temperature": self._get_dynamic_temperature(sentiment),
                "max_tokens": 100,  # ⚡ REDUCED from 150 (voice should be concise)
                "top_p": 0.9,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.2
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            logger.info(f"🤖 Processing [{sentiment['sentiment']}]: {text[:50]}...")
            
            # Try with retry and fallback
            response_text = await self._call_llm_with_fallback(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": processed_text}
                ],
                max_tokens=100,
                temperature=self._get_dynamic_temperature(sentiment)
            )
            
            if not response_text:
                return self.personality.get_random_error()
            
            # Post-process
            response_text = self._post_process_response(response_text, sentiment)
            
            logger.info(f"📝 Response: {response_text[:100]}...")
            return response_text
            
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            self._track_error()
            return self.personality.get_random_error()

    async def process_text_with_history(self, text: str, conversation_history: list) -> str:
        """Process with context awareness (optimized for voice)."""
        try:
            # Input validation
            if not text or not isinstance(text, str):
                logger.warning("Invalid input: empty or non-string text")
                return self.personality.get_random_error()
            
            if len(text) > 1000:
                logger.warning(f"Input too long: {len(text)} chars, truncating to 1000")
                text = text[:1000]
            
            if not isinstance(conversation_history, list):
                logger.warning("Invalid conversation_history, using empty list")
                conversation_history = []
            
            self.interaction_count += 1
            
            sentiment = self.sentiment_analyzer.analyze(text)
            processed_text = self._preprocess_transcription(text)
            
            # Build messages (pass sentiment as string key for LRU cache)
            system_prompt = self._build_dynamic_system_prompt(sentiment['sentiment'], has_history=True)

            # Check if we can use cached message history
            current_history_id = id(conversation_history)
            if current_history_id != self.cache_history_id:
                # Rebuild message cache with length limits
                MAX_MESSAGE_LENGTH = 500
                self.message_cache = []
                for exchange in conversation_history[-6:]:
                    if exchange.startswith("System:"):
                        continue  # Skip system messages in history

                    if exchange.startswith("User:"):
                        content = exchange[5:].strip()
                        if len(content) > MAX_MESSAGE_LENGTH:
                            content = content[:MAX_MESSAGE_LENGTH] + "..."
                            logger.debug(f"Truncated long User message: {len(exchange)} -> {MAX_MESSAGE_LENGTH}")
                        self.message_cache.append({"role": "user", "content": content})
                    elif exchange.startswith("Agent:"):
                        content = exchange[6:].strip()
                        if len(content) > MAX_MESSAGE_LENGTH:
                            content = content[:MAX_MESSAGE_LENGTH] + "..."
                            logger.debug(f"Truncated long Agent message: {len(exchange)} -> {MAX_MESSAGE_LENGTH}")
                        self.message_cache.append({"role": "assistant", "content": content})

                self.cache_history_id = current_history_id
                logger.debug(f"Rebuilt message cache with {len(self.message_cache)} messages")

            # Build final message list from cache (much faster)
            messages = [{"role": "system", "content": system_prompt}] + self.message_cache + [{"role": "user", "content": processed_text}]
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": self._get_dynamic_temperature(sentiment),
                "max_tokens": 120,  # ⚡ REDUCED from 200 (faster without quality loss)
                "top_p": 0.9,
                "frequency_penalty": 0.35,  # Higher to avoid repetition
                "presence_penalty": 0.25
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            logger.info(f"🤖 Processing with history [{sentiment['sentiment']}]")
            
            # Try with retry and fallback
            response_text = await self._call_llm_with_fallback(
                messages=messages,
                max_tokens=120,
                temperature=self._get_dynamic_temperature(sentiment)
            )
            
            if not response_text:
                return self.personality.get_random_error()
            
            response_text = self._post_process_response(response_text, sentiment)
            
            return response_text
            
        except Exception as e:
            logger.error(f"❌ Error with history: {e}")
            self._track_error()
            return self.personality.get_random_error()

    def _preprocess_transcription(self, text: str) -> str:
        """Pre-process transcription."""
        if not text:
            return text
        
        processed = text
        
        # Fix company name
        shamla_pattern = re.compile(r'\b(sham[bl]a\s*tech?|sham[bl]a)\b', re.IGNORECASE)
        processed = shamla_pattern.sub('Shamla Tech', processed)
        
        # Common contractions
        contractions = {
            r'\bwanna\b': 'want to',
            r'\bgonna\b': 'going to',
            r'\bgotta\b': 'got to',
        }
        
        for pattern, replacement in contractions.items():
            processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
        
        return processed.strip()

    @lru_cache(maxsize=16)
    def _build_dynamic_system_prompt(self, sentiment_key: str, has_history: bool = False) -> str:
        """Creates adaptive system prompts with caching (OPTIMIZED for voice).
        
        Args:
            sentiment_key: String key like 'urgent', 'negative', 'positive', 'neutral'
            has_history: Whether conversation history exists
        """
        
        base = """You are Alex, a warm AI voice assistant for Shamla Tech.

VOICE GUIDELINES:
- Keep responses SHORT (1-2 sentences for simple queries, 3 max for complex)
- Use contractions (I'm, you're, that's)
- Be conversational and natural
- No lists or bullet points (this is voice!)

About Shamla Tech:
- AI solutions, blockchain, cryptocurrency services
- Cutting-edge tech team

CRITICAL: Your response will be spoken aloud. Keep it concise and natural."""
        
        # Tone modifiers (shortened)
        if sentiment_key == 'urgent':
            tone = "\nTONE: User seems urgent - be direct and efficient."
        elif sentiment_key == 'negative':
            tone = "\nTONE: User seems frustrated - be patient and reassuring."
        elif sentiment_key == 'positive':
            tone = "\nTONE: User is happy - match their energy!"
        else:
            tone = "\nTONE: Standard friendly conversation."
        
        history_note = ""
        if has_history:
            history_note = "\nNote: Reference previous conversation naturally when relevant."
        
        return base + tone + history_note
    
    def _get_dynamic_temperature(self, sentiment: Dict) -> float:
        """Adjusts creativity based on context."""
        if sentiment['sentiment'] == 'urgent':
            return 0.6  # More focused
        elif sentiment['sentiment'] == 'positive':
            return 0.85  # More playful
        else:
            return 0.75  # Balanced
    
    def _post_process_response(self, response_text: str, sentiment: Dict) -> str:
        """Enhances response with natural touches."""
        if not response_text:
            return response_text
        
        # Clean prefixes
        cleaned = self._clean_prefixes(response_text)
        
        # Add occasional filler (less likely for urgent)
        filler_prob = 0.03 if sentiment['sentiment'] == 'urgent' else 0.12
        cleaned = self.personality.add_natural_pause(cleaned, filler_prob)
        
        # Add continuer
        continuer_prob = 0.1 if sentiment['has_question'] else 0.25
        cleaned = self.personality.add_continuer(cleaned, continuer_prob)
        
        # Clean spacing and capitalization
        cleaned = " ".join(cleaned.split())
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned.strip()
    
    def _clean_prefixes(self, text: str) -> str:
        """Remove bot-like prefixes."""
        prefixes = [
            "agent:", "assistant:", "ai:", "alex:", "bot:"
        ]
        
        text_lower = text.lower()
        for prefix in prefixes:
            if text_lower.startswith(prefix):
                return text[len(prefix):].strip()
        
        return text
    
    async def _call_llm_with_fallback(
        self,
        messages: list,
        max_tokens: int,
        temperature: float
    ) -> Optional[str]:
        """Call LLM with retry and multi-provider fallback."""
        
        # Try each provider in priority order
        for provider_idx, provider in enumerate(self.providers):
            logger.debug(f"Trying provider: {provider.upper()} (attempt {provider_idx + 1}/{len(self.providers)})")
            
            # Retry with exponential backoff for current provider
            for attempt in range(self.config.llm.max_retries):
                try:
                    response_text = await self._call_provider(
                        provider=provider,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    
                    if response_text:
                        # Success - reset error tracking
                        if provider != self.current_provider:
                            logger.info(f"✅ Switched to {provider.upper()} provider")
                            self.current_provider = provider
                        self.consecutive_errors = 0
                        return response_text
                        
                except httpx.TimeoutException:
                    logger.warning(f"⏱️ {provider.upper()} timeout (attempt {attempt + 1}/{self.config.llm.max_retries})")
                except httpx.HTTPStatusError as e:
                    logger.warning(f"⚠️ {provider.upper()} HTTP {e.response.status_code} (attempt {attempt + 1})")
                except Exception as e:
                    logger.warning(f"⚠️ {provider.upper()} error: {str(e)[:50]}")
                
                # Exponential backoff before retry
                if attempt < self.config.llm.max_retries - 1:
                    delay = self.config.error_recovery.calculate_retry_delay(attempt)
                    logger.debug(f"⏳ Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
            
            # All retries failed for this provider, try next
            logger.warning(f"❌ {provider.upper()} failed after {self.config.llm.max_retries} retries")
        
        # All providers failed
        logger.error("❌ All LLM providers failed")
        self._track_error()
        return None
    
    async def _call_provider(
        self,
        provider: str,
        messages: list,
        max_tokens: int,
        temperature: float
    ) -> Optional[str]:
        """Call specific LLM provider."""
        
        if provider == 'openai':
            return await self._call_openai(messages, max_tokens, temperature)
        elif provider == 'gemini':
            return await self._call_gemini(messages, max_tokens, temperature)
        elif provider == 'groq':
            return await self._call_groq(messages, max_tokens, temperature)
        elif provider == 'ollama':
            return await self._call_ollama(messages, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def _call_openai(
        self,
        messages: list,
        max_tokens: int,
        temperature: float
    ) -> Optional[str]:
        """Call OpenAI API."""
        payload = {
            "model": self.config.llm.openai_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.2,
            "stream": False
        }
        
        headers = {
            "Authorization": f"Bearer {self.config.api.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        response = await self.client.post(
            self.config.api.openai_base_url,
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    
    async def _call_openai_streaming(
        self,
        messages: list,
        max_tokens: int,
        temperature: float,
        callback
    ) -> Optional[str]:
        """Call OpenAI API with streaming for reduced latency."""
        import json
        
        payload = {
            "model": self.config.llm.openai_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.2,
            "stream": True
        }
        
        headers = {
            "Authorization": f"Bearer {self.config.api.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        full_response = ""
        sentence_buffer = ""
        
        try:
            async with self.client.stream(
                "POST",
                self.config.api.openai_base_url,
                json=payload,
                headers=headers,
                timeout=self.config.llm.request_timeout
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    
                    if line.startswith("data: "):
                        try:
                            json_str = line[6:]
                            chunk = json.loads(json_str)
                            
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                
                                if content:
                                    full_response += content
                                    sentence_buffer += content
                                    
                                    # Check for sentence boundaries
                                    if any(punct in content for punct in ['. ', '? ', '! ', '\n']):
                                        sentences = re.split(r'([.!?]\s+)', sentence_buffer)
                                        
                                        # Send complete sentences
                                        for i in range(0, len(sentences) - 1, 2):
                                            if i + 1 < len(sentences):
                                                complete_sentence = sentences[i] + sentences[i + 1]
                                                if complete_sentence.strip():
                                                    await callback(complete_sentence.strip())
                                        
                                        # Keep incomplete sentence
                                        if len(sentences) % 2 == 1:
                                            sentence_buffer = sentences[-1]
                                        else:
                                            sentence_buffer = ""
                        
                        except json.JSONDecodeError:
                            continue
                
                # Send remaining text
                if sentence_buffer.strip():
                    await callback(sentence_buffer.strip())
            
            return full_response.strip()
            
        except Exception as e:
            logger.warning(f"OpenAI streaming error: {e}")
            return None
    
    async def _call_gemini(
        self,
        messages: list,
        max_tokens: int,
        temperature: float
    ) -> Optional[str]:
        """Call Gemini API (fallback)."""
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            if msg['role'] == 'system':
                # Gemini doesn't have system role, prepend to first user message
                contents.append({"role": "user", "parts": [{"text": msg['content']}]})
            elif msg['role'] == 'user':
                contents.append({"role": "user", "parts": [{"text": msg['content']}]})
            elif msg['role'] == 'assistant':
                contents.append({"role": "model", "parts": [{"text": msg['content']}]})
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature
            }
        }
        
        url = f"{self.config.api.gemini_base_url}/{self.config.llm.gemini_model}:generateContent?key={self.config.api.gemini_api_key}"
        
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text'].strip()
    
    async def _call_groq(
        self,
        messages: list,
        max_tokens: int,
        temperature: float
    ) -> Optional[str]:
        """Call Groq API (second fallback)."""
        payload = {
            "model": self.config.llm.groq_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        headers = {
            "Authorization": f"Bearer {self.config.api.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        response = await self.client.post(
            self.config.api.groq_base_url,
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    
    async def _call_ollama(
        self,
        messages: list,
        max_tokens: int,
        temperature: float
    ) -> Optional[str]:
        """Call Ollama API (local LLM, third fallback)."""
        payload = {
            "model": self.config.llm.ollama_model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": False
        }
        
        url = f"{self.config.api.ollama_base_url}/api/chat"
        
        # Create temporary client with longer timeout for Ollama
        async with httpx.AsyncClient(timeout=self.config.llm.ollama_timeout) as ollama_client:
            response = await ollama_client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result['message']['content'].strip()
    
    def _track_error(self):
        """Track consecutive errors for circuit breaker pattern."""
        self.consecutive_errors += 1
        self.last_error_time = time.time()
        
        if self.consecutive_errors >= self.config.error_recovery.max_consecutive_errors:
            logger.error(
                f"⚠️ {self.consecutive_errors} consecutive errors detected. "
                f"Consider checking API status."
            )
    
    async def shutdown(self):
        """Close HTTP client."""
        try:
            await self.client.aclose()
            logger.info("✅ LLM Handler shutdown")
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")


async def main():
    """Example usage."""
    llm = LLMHandler()
    response = await llm.process_text("Hello, tell me about Shamla Tech")
    print(f"Response: {response}")
    await llm.shutdown()

if __name__ == "__main__":
    asyncio.run(main())