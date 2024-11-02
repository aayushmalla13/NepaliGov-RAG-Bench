#!/usr/bin/env python3
"""
🌐 CP11.5 Translation Layer
UI-Level EN→NE Translation with Token Preservation
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class TranslatorBackend(ABC):
    """Abstract base class for translation backends."""
    
    @abstractmethod
    def translate(self, text: str, target_lang: str = "ne") -> str:
        """Translate text to target language."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass

class MarianMTBackend(TranslatorBackend):
    """Helsinki-NLP MarianMT backend for EN→NE translation."""
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._available = None
    
    @property
    def name(self) -> str:
        return "MarianMT"
    
    def is_available(self) -> bool:
        """Check if MarianMT is available."""
        if self._available is not None:
            return self._available
            
        try:
            from transformers import MarianMTModel, MarianTokenizer
            # Try to load the model
            model_name = "Helsinki-NLP/opus-mt-en-hi"  # Use Hindi as proxy for Nepali
            self._tokenizer = MarianTokenizer.from_pretrained(model_name)
            self._model = MarianMTModel.from_pretrained(model_name)
            self._available = True
            logger.info(f"✅ {self.name} backend loaded successfully")
        except Exception as e:
            logger.warning(f"❌ {self.name} backend unavailable: {e}")
            self._available = False
            
        return self._available
    
    def translate(self, text: str, target_lang: str = "ne") -> str:
        """Translate text using MarianMT."""
        if not self.is_available():
            raise RuntimeError(f"{self.name} backend not available")
        
        try:
            # Tokenize and translate
            inputs = self._tokenizer(text, return_tensors="pt", padding=True)
            translated = self._model.generate(**inputs)
            result = self._tokenizer.decode(translated[0], skip_special_tokens=True)
            return result
        except Exception as e:
            logger.error(f"Translation failed with {self.name}: {e}")
            raise

class FallbackTranslatorBackend(TranslatorBackend):
    """Fallback translator using simple romanization rules."""
    
    # Enhanced English to Nepali romanization mapping with proper nouns preservation
    ROMANIZATION_MAP = {
        # Core government terms
        "health": "स्वास्थ्य",
        "government": "सरकार", 
        "constitution": "संविधान",
        "rights": "अधिकार",
        "fundamental": "मौलिक",
        "nepal": "नेपाल",
        "nepali": "नेपाली",
        "service": "सेवा",
        "services": "सेवाहरू",
        "policy": "नीति",
        "policies": "नीतिहरू",
        "law": "कानून",
        "act": "ऐन",
        "ministry": "मन्त्रालय",
        "department": "विभाग",
        "according": "अनुसार",
        "article": "धारा",
        "section": "खण्ड",
        "chapter": "अध्याय",
        "part": "भाग",
        "page": "पृष्ठ",
        "document": "कागजात",
        "report": "प्रतिवेदन",
        
        # Question words
        "what": "के",
        "how": "कसरी",
        "when": "कहिले",
        "where": "कहाँ",
        "why": "किन",
        "which": "कुन",
        "who": "को",
        
        # Enhanced vocabulary
        "budget": "बजेट",
        "allocation": "बाँडफाँड",
        "resources": "स्रोतहरू",
        "funding": "कोष",
        "primary": "प्राथमिक",
        "care": "हेरचाह",
        "system": "प्रणाली",
        "public": "सार्वजनिक",
        "private": "निजी",
        "citizen": "नागरिक",
        "citizens": "नागरिकहरू",
        "committee": "समिति",
        "parliament": "संसद",
        "assembly": "सभा",
        "court": "अदालत",
        "justice": "न्याय",
        "education": "शिक्षा",
        "hospital": "अस्पताल",
        "clinic": "क्लिनिक",
        "doctor": "डाक्टर",
        "nurse": "नर्स",
        "medicine": "औषधि",
        "treatment": "उपचार",
        "emergency": "आपतकाल",
        "disaster": "प्रकोप",
        "management": "व्यवस्थापन",
        "development": "विकास",
        "program": "कार्यक्रम",
        "project": "परियोजना",
        "plan": "योजना",
        "strategy": "रणनीति",
        "implementation": "कार्यान्वयन",
        "monitoring": "अनुगमन",
        "evaluation": "मूल्याङ्कन"
    }
    
    # Proper nouns that should NOT be translated
    PROPER_NOUNS = {
        "WHO", "World Health Organization", "United Nations", "UN",
        "COVID-19", "SARS-CoV-2", "HIV", "AIDS", "TB", "Tuberculosis",
        "Kathmandu", "Pokhara", "Chitwan", "Dharan", "Biratnagar",
        "Ministry of Health and Population", "MOHP", "DoHS", 
        "Department of Health Services", "HEOC", "PHC", "SHP",
        "Basic Health Services", "BHS", "Strategic Plan", "STP"
    }
    
    @property
    def name(self) -> str:
        return "FallbackTranslator"
    
    def is_available(self) -> bool:
        return True
    
    def translate(self, text: str, target_lang: str = "ne") -> str:
        """Enhanced fallback translation with proper noun preservation."""
        if target_lang == "ne":
            # EN→NE translation
            return self._translate_en_to_ne(text)
        elif target_lang == "en":
            # NE→EN translation  
            return self._translate_ne_to_en(text)
        else:
            return text
    
    def _translate_en_to_ne(self, text: str) -> str:
        """Translate English to Nepali."""
        words = text.split()
        translated_words = []
        
        for word in words:
            # Check if it's a proper noun first (preserve exactly)
            if word in self.PROPER_NOUNS or word.upper() in self.PROPER_NOUNS:
                translated_words.append(word)
                continue
            
            # Clean punctuation for lookup
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            if clean_word in self.ROMANIZATION_MAP:
                # Preserve original punctuation
                translated = self.ROMANIZATION_MAP[clean_word]
                # Add back punctuation
                if word != clean_word:
                    punct = re.findall(r'[^\w]', word)
                    if punct:
                        translated += ''.join(punct)
                translated_words.append(translated)
            else:
                translated_words.append(word)
        
        return ' '.join(translated_words)
    
    def _translate_ne_to_en(self, text: str) -> str:
        """High-quality Nepali to English translation."""
        # Comprehensive word mapping
        word_map = {
            'स्वास्थ्य': 'health', 'सेवा': 'service', 'सेवाहरू': 'services',
            'सरकार': 'government', 'सरकारी': 'governmental', 'सरकारले': 'government',
            'मन्त्रालय': 'ministry', 'विभाग': 'department', 'संस्था': 'institution',
            'संविधान': 'constitution', 'संविधानले': 'constitution', 'संविधानको': 'constitutional',
            'अधिकार': 'rights', 'अधिकारहरू': 'rights', 'मौलिक': 'fundamental',
            'नेपाल': 'Nepal', 'नेपाली': 'Nepali', 'नेपालमा': 'in Nepal',
            'छ': 'is', 'छन्': 'are', 'हुन्छ': 'happens', 'गर्छ': 'does', 'गर्छन्': 'do',
            'भएको': 'been', 'गरेको': 'done', 'दिएको': 'given', 'लिएको': 'taken',
            'को': 'of', 'का': 'of', 'की': 'of', 'ले': 'by', 'लाई': 'to', 'बाट': 'from',
            'मा': 'in', 'र': 'and', 'तर': 'but', 'वा': 'or', 'पनि': 'also',
            'अनुसार': 'according to', 'धारा': 'article', 'ऐन': 'act', 'कानून': 'law',
            'नीति': 'policy', 'योजना': 'plan', 'कार्यक्रम': 'program', 'प्रणाली': 'system',
            'व्यवस्था': 'system', 'विकास': 'development', 'सुधार': 'improvement',
            'जनता': 'people', 'नागरिक': 'citizen', 'समुदाय': 'community',
            'क्षेत्र': 'sector', 'उपलब्ध': 'available', 'प्रदान': 'provide',
            'कस्तो': 'what kind of', 'कति': 'how much', 'कहाँ': 'where',
            'कसरी': 'how', 'किन': 'why', 'के': 'what', 'कुन': 'which'
        }
        
        words = text.split()
        translated = []
        
        for word in words:
            clean = word.strip('।,.:;!?()[]{}"\'')
            if clean in word_map:
                translated.append(word_map[clean])
            elif clean.lower() in word_map:
                translated.append(word_map[clean.lower()])
            else:
                # Check for partial matches
                found = False
                for nepali, english in word_map.items():
                    if len(nepali) > 3 and nepali in clean:
                        translated.append(english)
                        found = True
                        break
                if not found:
                    # Keep original word instead of "term"
                    translated.append(clean)
        
        return ' '.join(translated)
class TokenPreservingTranslator:
    """Main translator with token preservation capabilities."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("cache/translations")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()
        
        # Initialize backends in priority order
        self.backends = [
            MarianMTBackend(),
            FallbackTranslatorBackend()
        ]
        
        # Find first available backend
        self.active_backend = None
        for backend in self.backends:
            if backend.is_available():
                self.active_backend = backend
                logger.info(f"✅ Using translation backend: {backend.name}")
                break
        
        if not self.active_backend:
            logger.error("❌ No translation backends available")
    
    def _load_cache(self) -> Dict[str, str]:
        """Load translation cache."""
        cache_file = self.cache_dir / "translations.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load translation cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save translation cache."""
        cache_file = self.cache_dir / "translations.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save translation cache: {e}")
    
    def _get_cache_key(self, text: str, target_lang: str) -> str:
        """Generate cache key for text."""
        content = f"{text}|{target_lang}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def translate_preserving_tokens(
        self, 
        text: str, 
        target_lang: str = "ne",
        preserve_regex: str = r"\[\[.*?\]\]|\[[A-Z]{2}\]|\d[\d\.\,:%/-]*|`[^`]*`",
        return_alignment: bool = False
    ) -> Union[str, Tuple[str, List[Dict[str, str]]]]:
        """
        Translate text while preserving special tokens.
        
        Args:
            text: Text to translate
            target_lang: Target language code
            preserve_regex: Regex pattern for tokens to preserve
            return_alignment: If True, return sentence-level alignment info
        
        Returns:
            Translated text with preserved tokens, optionally with alignment info
        """
        if not self.active_backend:
            logger.warning("No translation backend available, returning original text")
            if return_alignment:
                return text, []
            return text
        
        # Check cache first
        cache_key = self._get_cache_key(f"{text}|alignment={return_alignment}", target_lang)
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if return_alignment and isinstance(cached_result, tuple):
                return cached_result
            elif not return_alignment and isinstance(cached_result, str):
                return cached_result
        
        # Find and mask tokens to preserve
        tokens_to_preserve = []
        token_pattern = re.compile(preserve_regex)
        
        def mask_token(match):
            token = match.group(0)
            token_id = f"__TOKEN_{len(tokens_to_preserve)}__"
            tokens_to_preserve.append((token_id, token))
            return token_id
        
        # Mask tokens
        masked_text = token_pattern.sub(mask_token, text)
        
        try:
            # For sentence alignment, split into sentences first
            if return_alignment:
                sentences = re.split(r'(?<=[.!?])\s+', masked_text)
                translated_sentences = []
                alignment_info = []
                
                for sentence in sentences:
                    if sentence.strip():
                        translated_sentence = self.active_backend.translate(sentence, target_lang)
                        
                        # Restore tokens in this sentence
                        original_sentence = sentence
                        for token_id, original_token in tokens_to_preserve:
                            original_sentence = original_sentence.replace(token_id, original_token)
                            translated_sentence = translated_sentence.replace(token_id, original_token)
                        
                        translated_sentences.append(translated_sentence)
                        alignment_info.append({
                            'original': original_sentence,
                            'translated': translated_sentence,
                            'confidence': 0.8  # Default confidence for fallback
                        })
                
                translated_text = ' '.join(translated_sentences)
                result = (translated_text, alignment_info)
            else:
                # Translate masked text
                translated_text = self.active_backend.translate(masked_text, target_lang)
                
                # Restore tokens
                for token_id, original_token in tokens_to_preserve:
                    translated_text = translated_text.replace(token_id, original_token)
                
                result = translated_text
            
            # Cache result
            self.cache[cache_key] = result
            self._save_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            if return_alignment:
                return text, []
            return text  # Return original on failure
    
    def batch_translate(
        self, 
        texts: List[str], 
        target_lang: str = "ne",
        max_chunk_size: int = 1000
    ) -> List[str]:
        """
        Batch translate multiple texts with chunking.
        
        Args:
            texts: List of texts to translate
            target_lang: Target language code
            max_chunk_size: Maximum characters per chunk
        
        Returns:
            List of translated texts
        """
        results = []
        
        for text in texts:
            # For now, translate individually
            # TODO: Implement proper batching for supported backends
            translated = self.translate_preserving_tokens(text, target_lang)
            results.append(translated)
        
        return results

# Transliteration support for Nepali input
class NepaliTransliterator:
    """Roman to Devanagari transliteration for Nepali input."""
    
    # Basic romanization to Devanagari mapping
    TRANSLITERATION_MAP = {
        # Vowels
        'a': 'अ', 'aa': 'आ', 'i': 'इ', 'ii': 'ई', 'u': 'उ', 'uu': 'ऊ',
        'e': 'ए', 'ai': 'ऐ', 'o': 'ओ', 'au': 'औ',
        
        # Consonants
        'ka': 'क', 'kha': 'ख', 'ga': 'ग', 'gha': 'घ', 'nga': 'ङ',
        'cha': 'च', 'chha': 'छ', 'ja': 'ज', 'jha': 'झ', 'nya': 'ञ',
        'ta': 'त', 'tha': 'थ', 'da': 'द', 'dha': 'ध', 'na': 'न',
        'pa': 'प', 'pha': 'फ', 'ba': 'ब', 'bha': 'भ', 'ma': 'म',
        'ya': 'य', 'ra': 'र', 'la': 'ल', 'wa': 'व', 'va': 'व',
        'sha': 'श', 'shha': 'ष', 'sa': 'स', 'ha': 'ह',
        
        # Special combinations
        'gya': 'ज्ञ', 'ksha': 'क्ष', 'tra': 'त्र',
        
        # Common words and phrases
        'nepal': 'नेपाल', 'nepali': 'नेपाली', 
        'sarkar': 'सरकार', 'swasthya': 'स्वास्थ्य',
        'samvidhan': 'संविधान', 'adhikar': 'अधिकार',
        'maulik': 'मौलिक', 'seva': 'सेवा',
        'niti': 'नीति', 'kanun': 'कानून',
        'mantralaya': 'मन्त्रालय', 'vibhag': 'विभाग',
        'hospital': 'अस्पताल', 'clinic': 'क्लिनिक',
        'doctor': 'डाक्टर', 'daktar': 'डाक्टर',
        'nurse': 'नर्स', 'aushadhi': 'औषधि',
        'upchar': 'उपचार', 'ilaj': 'इलाज',
        'biramari': 'बिरामी', 'rog': 'रोग',
        'swastha': 'स्वस्थ', 'sudhar': 'सुधार',
        'vikas': 'विकास', 'yojana': 'योजना',
        'karyakram': 'कार्यक्रम', 'parisad': 'परिषद्',
        'samiti': 'समिति', 'sabha': 'सभा',
        'adalat': 'अदालत', 'nyaya': 'न्याय',
        'siksha': 'शिक्षा', 'vidyalaya': 'विद्यालय',
        'budget': 'बजेट', 'kosh': 'कोष',
        'prabandhan': 'प्रबन्धन', 'vyavasthapan': 'व्यवस्थापन',
        
        # Question words and common phrases
        'ke': 'के', 'kasari': 'कसरी', 'kaha': 'कहाँ',
        'kahile': 'कहिले', 'kun': 'कुन', 'kina': 'किन',
        'ko': 'को', 'ka': 'का', 'ki': 'की',
        'ma': 'मा', 'lai': 'लाई', 'bata': 'बाट',
        'dekhi': 'देखि', 'samma': 'सम्म',
        'anusar': 'अनुसार', 'dwara': 'द्वारा'
    }
    
    def transliterate(self, roman_text: str) -> str:
        """
        Convert romanized Nepali to Devanagari.
        
        Args:
            roman_text: Romanized Nepali text
            
        Returns:
            Devanagari text
        """
        if not roman_text:
            return roman_text
        
        # Check if already in Devanagari
        if any('\u0900' <= char <= '\u097F' for char in roman_text):
            return roman_text  # Already in Devanagari, return as-is
        
        words = roman_text.lower().split()
        transliterated_words = []
        
        for word in words:
            # Remove punctuation for lookup
            clean_word = re.sub(r'[^\w]', '', word)
            
            if clean_word in self.TRANSLITERATION_MAP:
                transliterated = self.TRANSLITERATION_MAP[clean_word]
                # Add back punctuation
                if word != clean_word:
                    punct = re.findall(r'[^\w]', word)
                    if punct:
                        transliterated += ''.join(punct)
                transliterated_words.append(transliterated)
            else:
                # Try character-by-character transliteration for unknown words
                transliterated = self._char_by_char_transliterate(clean_word)
                # Add back punctuation
                if word != clean_word:
                    punct = re.findall(r'[^\w]', word)
                    if punct:
                        transliterated += ''.join(punct)
                transliterated_words.append(transliterated)
        
        return ' '.join(transliterated_words)
    
    def _char_by_char_transliterate(self, word: str) -> str:
        """Fallback character-by-character transliteration."""
        # This is a simplified version - in practice, you'd want more sophisticated
        # phonetic mapping and context-aware rules
        result = ""
        i = 0
        while i < len(word):
            # Try 3-char combinations first, then 2-char, then 1-char
            found = False
            for length in [3, 2, 1]:
                if i + length <= len(word):
                    substr = word[i:i+length]
                    if substr in self.TRANSLITERATION_MAP:
                        result += self.TRANSLITERATION_MAP[substr]
                        i += length
                        found = True
                        break
            
            if not found:
                result += word[i]  # Keep original character
                i += 1
        
        return result

# Global instances
_translator = None
_transliterator = None

def get_translator() -> TokenPreservingTranslator:
    """Get global translator instance."""
    global _translator
    if _translator is None:
        _translator = TokenPreservingTranslator()
    return _translator

def get_transliterator() -> NepaliTransliterator:
    """Get global transliterator instance."""
    global _transliterator
    if _transliterator is None:
        _transliterator = NepaliTransliterator()
    return _transliterator
