# import os
# import logging
# import asyncio
# import requests
# import random
# import re
# from dotenv import load_dotenv
# from typing import Dict, List, Optional

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# class ConversationalPersonality:
#     """Manages personality traits and natural language variations"""
    
#     # Natural filler words for voice (use sparingly - 10-20% of responses)
#     FILLERS = {
#         'thinking': ['Um,', 'Well,', 'Let me think,', 'Hmm,', 'You know,'],
#         'transition': ['So,', 'Actually,', 'Oh,', 'I mean,', 'Right,'],
#         'clarifying': ['I mean,', 'In other words,', 'What I\'m saying is,'],
#         'agreement': ['Absolutely,', 'Totally,', 'Yeah,', 'For sure,', 'Definitely,']
#     }
    
#     # Varied error responses (more human than generic messages)
#     ERROR_RESPONSES = [
#         "Oops, I'm having a bit of a brain freeze right now. Give me a sec?",
#         "Ah man, something's not clicking on my end. Can you try that again?",
#         "You know what? I'm having trouble with that. Let me try to help differently.",
#         "Hmm, I'm hitting a snag here. Could you rephrase that for me?",
#         "Ugh, technical difficulties on my end. Mind repeating that?"
#     ]
    
#     # Conversation continuers (engage without being pushy)
#     CONTINUERS = [
#         "Anything else I can help with?",
#         "What else can I do for you?",
#         "Is there more you'd like to know?",
#         "Want to know anything else?",
#         "Does that help, or should I explain differently?"
#     ]
    
#     @staticmethod
#     def add_natural_pause(text: str, probability: float = 0.15) -> str:
#         """Adds occasional filler words for naturalness (not every response)"""
#         if random.random() > probability:
#             return text
        
#         # Pick a random category and filler
#         category = random.choice(list(ConversationalPersonality.FILLERS.keys()))
#         filler = random.choice(ConversationalPersonality.FILLERS[category])
        
#         # Insert at beginning for thinking/transition, or mid-sentence for others
#         if category in ['thinking', 'transition']:
#             return f"{filler} {text}"
#         else:
#             # Insert after first sentence for mid-conversation fillers
#             sentences = text.split('. ')
#             if len(sentences) > 1:
#                 sentences[1] = f"{filler} {sentences[1]}"
#                 return '. '.join(sentences)
        
#         return text
    
#     @staticmethod
#     def get_random_error() -> str:
#         """Returns varied error messages"""
#         return random.choice(ConversationalPersonality.ERROR_RESPONSES)
    
#     @staticmethod
#     def add_continuer(text: str, probability: float = 0.3) -> str:
#         """Occasionally adds a follow-up question to keep conversation going"""
#         if random.random() > probability or '?' in text:
#             return text  # Don't add if already asking a question
        
#         continuer = random.choice(ConversationalPersonality.CONTINUERS)
#         return f"{text} {continuer}"


# class SentimentAnalyzer:
#     """Simple sentiment detection to adjust assistant tone"""
    
#     POSITIVE_WORDS = ['great', 'awesome', 'excellent', 'love', 'fantastic', 'happy', 'excited', 'thanks', 'thank you', 'perfect', 'wonderful']
#     NEGATIVE_WORDS = ['frustrated', 'annoyed', 'upset', 'angry', 'confused', 'problem', 'issue', 'broken', 'doesn\'t work', 'help', 'worried']
#     URGENT_WORDS = ['urgent', 'asap', 'quickly', 'immediately', 'emergency', 'critical', 'now']
    
#     @staticmethod
#     def analyze(text: str) -> Dict[str, any]:
#         """Returns sentiment and intensity"""
#         text_lower = text.lower()
        
#         positive_count = sum(1 for word in SentimentAnalyzer.POSITIVE_WORDS if word in text_lower)
#         negative_count = sum(1 for word in SentimentAnalyzer.NEGATIVE_WORDS if word in text_lower)
#         urgent_count = sum(1 for word in SentimentAnalyzer.URGENT_WORDS if word in text_lower)
        
#         # Determine overall sentiment
#         if urgent_count > 0:
#             sentiment = 'urgent'
#         elif negative_count > positive_count:
#             sentiment = 'negative'
#         elif positive_count > negative_count:
#             sentiment = 'positive'
#         else:
#             sentiment = 'neutral'
        
#         intensity = max(positive_count, negative_count, urgent_count)
        
#         return {
#             'sentiment': sentiment,
#             'intensity': intensity,
#             'has_question': '?' in text
#         }


# class LLMHandler:
#     """Handles text processing with human-like conversational abilities"""
    
#     def __init__(self):
#         self.api_key = os.getenv('OPENAI_API_KEY')
#         if not self.api_key:
#             logger.error("❌ OPENAI_API_KEY not found in environment variables.")
#             raise ValueError("OpenAI API key is required for LLM functionality.")
        
#         self.base_url = "https://api.openai.com/v1/chat/completions"
#         self.personality = ConversationalPersonality()
#         self.sentiment_analyzer = SentimentAnalyzer()
        
#         # Track conversation metadata
#         self.user_preferences = {}  # Store learned user info
#         self.interaction_count = 0
        
#         logger.info("🤖 LLM Handler initialized with enhanced personality system.")
    
#     async def process_text(self, text: str) -> str:
#         """Process text with sentiment-aware, natural responses"""
#         try:
#             # Analyze sentiment to adjust tone
#             sentiment = self.sentiment_analyzer.analyze(text)
            
#             # Pre-process transcription
#             processed_text = self._preprocess_transcription(text)
            
#             # Build dynamic system prompt based on sentiment
#             system_prompt = self._build_dynamic_system_prompt(sentiment)
            
#             payload = {
#                 "model": "gpt-4o-mini",
#                 "messages": [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": processed_text}
#                 ],
#                 "temperature": self._get_dynamic_temperature(sentiment),
#                 "max_tokens": 1000,
#                 "top_p": 0.9,
#                 "frequency_penalty": 0.3,  # Higher to avoid repetition
#                 "presence_penalty": 0.2
#             }
            
#             headers = {
#                 "Authorization": f"Bearer {self.api_key}",
#                 "Content-Type": "application/json"
#             }
            
#             logger.info(f"🤖 Processing [{sentiment['sentiment']}]: {text}")
            
#             response = requests.post(self.base_url, json=payload, headers=headers)
#             response.raise_for_status()
            
#             result = response.json()
#             response_text = result['choices'][0]['message']['content'].strip()
            
#             # Post-process with natural enhancements
#             response_text = self._post_process_response(response_text, sentiment)
            
#             logger.info(f"📝 Enhanced Response: {response_text}")
#             return response_text
            
#         except requests.exceptions.RequestException as e:
#             logger.error(f"❌ API Error: {e}")
#             return self.personality.get_random_error()
#         except Exception as e:
#             logger.error(f"❌ Unexpected error: {e}")
#             return self.personality.get_random_error()
    
#     async def process_text_with_history(self, text: str, conversation_history: list) -> str:
#         """Process with context awareness and personality memory"""
#         try:
#             self.interaction_count += 1
            
#             # Analyze sentiment
#             sentiment = self.sentiment_analyzer.analyze(text)
#             processed_text = self._preprocess_transcription(text)
            
#             # Build messages with dynamic prompt
#             system_prompt = self._build_dynamic_system_prompt(sentiment, has_history=True)
#             messages = [{"role": "system", "content": system_prompt}]
            
#             # Add conversation history (last 6-8 exchanges)
#             for i, exchange in enumerate(conversation_history[-8:]):
#                 role = "user" if i % 2 == 0 else "assistant"
#                 messages.append({"role": role, "content": exchange})
            
#             messages.append({"role": "user", "content": processed_text})
            
#             payload = {
#                 "model": "gpt-4o-mini",
#                 "messages": messages,
#                 "temperature": self._get_dynamic_temperature(sentiment),
#                 "max_tokens": 800,
#                 "top_p": 0.9,
#                 "frequency_penalty": 0.3,
#                 "presence_penalty": 0.2
#             }
            
#             headers = {
#                 "Authorization": f"Bearer {self.api_key}",
#                 "Content-Type": "application/json"
#             }
            
#             logger.info(f"🤖 Processing with history [{sentiment['sentiment']}]: {text}")
            
#             response = requests.post(self.base_url, json=payload, headers=headers)
#             response.raise_for_status()
            
#             result = response.json()
#             response_text = result['choices'][0]['message']['content'].strip()
            
#             # Post-process with natural enhancements
#             response_text = self._post_process_response(response_text, sentiment)
            
#             logger.info(f"📝 Enhanced Response: {response_text}")
#             return response_text
            
#         except Exception as e:
#             logger.error(f"❌ Error processing with history: {e}")
#             return self.personality.get_random_error()
    
#     def _build_dynamic_system_prompt(self, sentiment: Dict, has_history: bool = False) -> str:
#         """Creates adaptive system prompts based on conversation context"""
        
#         base_personality = """You are Alex, a warm and genuinely helpful voice assistant for Shamla Tech.

# CRITICAL VOICE GUIDELINES:
# 🎙️ You're being heard, not read - so speak naturally:
# - Use contractions constantly (I'm, you're, that's, we'll, don't)
# - Vary your sentence structure - mix short and long sentences
# - Occasional incomplete thoughts are fine ("So that means... yeah, we can definitely help with that")
# - Ask follow-up questions when genuinely curious or when it helps clarify
# - Reference earlier conversation points naturally ("Like you mentioned before...")

# About Shamla Tech:
# - Cutting-edge AI solutions & blockchain/crypto services
# - We help businesses transform with innovative tools
# - Passionate tech team that genuinely cares about clients

# TRANSCRIPTION HANDLING:
# - Shamla Tech variations (Shambla/Shamla/etc.) → always "Shamla Tech"
# - Forgive pronunciation errors, focus on intent
# - If unclear, ask friendlyly"""
        
#         # Adjust tone based on sentiment
#         if sentiment['sentiment'] == 'urgent':
#             tone_modifier = """

# 🚨 TONE FOR THIS RESPONSE: User seems urgent/stressed
# - Be efficient and direct, but still warm
# - Skip extra pleasantries, get to the solution
# - Show you understand the urgency: "I've got you, let me help right away"
# - Offer quick action items"""
        
#         elif sentiment['sentiment'] == 'negative':
#             tone_modifier = """

# 💙 TONE FOR THIS RESPONSE: User seems frustrated/confused
# - Lead with empathy: "I hear you, that's frustrating"
# - Be patient and reassuring
# - Break things down simply
# - Offer to explain differently if needed"""
        
#         elif sentiment['sentiment'] == 'positive':
#             tone_modifier = """

# ✨ TONE FOR THIS RESPONSE: User seems happy/excited
# - Match their energy! Be enthusiastic
# - Use more exclamation points (but not overdone)
# - Celebrate their interest: "That's awesome that you're interested in..."
# - Keep the positive momentum going"""
        
#         else:  # neutral
#             tone_modifier = """

# 💬 TONE FOR THIS RESPONSE: Standard conversation
# - Friendly and approachable
# - Balance helpfulness with casualness
# - Show personality but stay professional"""
        
#         # Add context awareness if history exists
#         history_modifier = ""
#         if has_history:
#             history_modifier = """

# 📝 CONVERSATION CONTINUITY:
# - Reference what they said earlier when relevant: "Going back to what you asked about..."
# - Show you're building on previous exchanges
# - Don't repeat yourself - build on previous answers
# - If they're asking follow-ups, acknowledge you remember: "Right, so building on that..."""
        
#         return base_personality + tone_modifier + history_modifier
    
#     def _get_dynamic_temperature(self, sentiment: Dict) -> float:
#         """Adjusts creativity based on context"""
#         if sentiment['sentiment'] == 'urgent':
#             return 0.6  # More focused, less creative for urgent issues
#         elif sentiment['sentiment'] == 'positive':
#             return 0.9  # More playful for positive interactions
#         else:
#             return 0.8  # Default conversational
    
#     def _post_process_response(self, response_text: str, sentiment: Dict) -> str:
#         """Enhances response with natural language touches"""
#         if not response_text:
#             return response_text
        
#         # Clean unwanted prefixes
#         cleaned = self._clean_prefixes(response_text)
        
#         # Add occasional natural filler (less likely for urgent)
#         filler_probability = 0.05 if sentiment['sentiment'] == 'urgent' else 0.18
#         cleaned = self.personality.add_natural_pause(cleaned, filler_probability)
        
#         # Add conversation continuers (less likely if user asked a question)
#         continuer_probability = 0.15 if sentiment['has_question'] else 0.35
#         cleaned = self.personality.add_continuer(cleaned, continuer_probability)
        
#         # Clean up spacing
#         cleaned = " ".join(cleaned.split())
        
#         # Ensure proper capitalization
#         if cleaned and cleaned[0].islower():
#             cleaned = cleaned[0].upper() + cleaned[1:]
        
#         return cleaned.strip()
    
#     def _clean_prefixes(self, text: str) -> str:
#         """Remove bot-like prefixes"""
#         prefixes = [
#             "agent:", "agent: ", "assistant:", "assistant: ", 
#             "ai:", "ai: ", "alex:", "alex: ",
#             "shamla tech agent:", "bot:"
#         ]
        
#         for prefix in prefixes:
#             if text.lower().startswith(prefix):
#                 return text[len(prefix):].strip()
        
#         return text
    
#     def _preprocess_transcription(self, text: str) -> str:
#         """Pre-process transcription to handle variations"""
#         if not text:
#             return text
        
#         processed = text
        
#         # Fix company name variations (case-insensitive)
#         shamla_pattern = re.compile(r'\b(sham[bl]a\s*tech?|sham[bl]a)\b', re.IGNORECASE)
#         processed = shamla_pattern.sub('Shamla Tech', processed)
        
#         # Common spoken contractions
#         contractions = {
#             r'\bwanna\b': 'want to',
#             r'\bgonna\b': 'going to',
#             r'\bgotta\b': 'got to',
#             r'\blemme\b': 'let me',
#             r'\bgimme\b': 'give me',
#             r'\bkinda\b': 'kind of',
#             r'\bsorta\b': 'sort of',
#         }
        
#         for pattern, replacement in contractions.items():
#             processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
        
#         # Common transcription errors
#         errors = {
#             r'\bblocked?\b': 'about',
#             r'\bblacked?\b': 'about',
#         }
        
#         for pattern, replacement in errors.items():
#             processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
        
#         return processed.strip()


# async def main():
#     """Test the enhanced handler"""
#     llm_handler = LLMHandler()
    
#     # Test different sentiment scenarios
#     test_messages = [
#         "Hey, I'm really excited to learn about Shamla Tech!",
#         "I'm frustrated, this isn't working at all",
#         "Can you help me ASAP? This is urgent!",
#         "Tell me about blockchain services"
#     ]
    
#     for msg in test_messages:
#         print(f"\n🗣️  User: {msg}")
#         response = await llm_handler.process_text(msg)
#         print(f"🤖 Alex: {response}")
#         print("-" * 60)


# if __name__ == "__main__":
#     asyncio.run(main())