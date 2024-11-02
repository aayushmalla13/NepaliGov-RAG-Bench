#!/usr/bin/env python3
"""
Font utilities for CP11.5 - Devanagari font injection and Unicode handling
"""

import base64
import os
import unicodedata
from pathlib import Path

def get_font_base64(font_path: str) -> str:
    """Convert TTF font to base64 for CSS embedding."""
    try:
        with open(font_path, 'rb') as f:
            font_data = f.read()
        return base64.b64encode(font_data).decode('utf-8')
    except FileNotFoundError:
        print(f"⚠️ Font not found: {font_path}")
        return ""

def inject_devanagari_fonts() -> str:
    """Generate CSS with embedded Devanagari fonts."""
    
    # Get font paths
    base_dir = Path(__file__).parent.parent
    regular_path = base_dir / "assets" / "fonts" / "NotoSansDevanagari-Regular.ttf"
    bold_path = base_dir / "assets" / "fonts" / "NotoSansDevanagari-Bold.ttf"
    
    # Get base64 encoded fonts
    regular_b64 = get_font_base64(str(regular_path))
    bold_b64 = get_font_base64(str(bold_path))
    
    css = """
    <style>
    @font-face {
        font-family: 'Noto Sans Devanagari';
        src: url(data:font/truetype;charset=utf-8;base64,""" + regular_b64 + """) format('truetype');
        font-weight: normal;
        font-style: normal;
        font-display: swap;
    }
    
    @font-face {
        font-family: 'Noto Sans Devanagari';
        src: url(data:font/truetype;charset=utf-8;base64,""" + bold_b64 + """) format('truetype');
        font-weight: bold;
        font-style: normal;
        font-display: swap;
    }
    
    /* Apply to all text inputs and Nepali content */
    .stTextInput input, .stTextArea textarea, .nepali-text, 
    .devanagari-input, [lang="ne"] {
        font-family: 'Noto Sans Devanagari', 'Noto Sans', Arial, sans-serif !important;
        font-feature-settings: 'kern' 1, 'liga' 1;
        text-rendering: optimizeLegibility;
    }
    
    /* Ensure proper line height for Devanagari */
    .nepali-text, [lang="ne"] {
        line-height: 1.6 !important;
        letter-spacing: 0.02em;
    }
    
    /* Input method hints */
    .devanagari-input {
        ime-mode: active;
        composition-mode: active;
    }
    </style>
    """
    
    return css

def normalize_unicode(text: str) -> str:
    """Normalize Unicode text to NFC form for consistent rendering."""
    if not text:
        return text
    return unicodedata.normalize('NFC', text)

def is_devanagari(text: str) -> bool:
    """Check if text contains Devanagari characters."""
    if not text:
        return False
    
    # Devanagari Unicode range: U+0900–U+097F
    devanagari_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        return False
    
    return (devanagari_chars / total_chars) > 0.3

def validate_font_rendering(test_text: str = "नेपाल स्वास्थ्य") -> dict:
    """Validate that Devanagari text renders correctly."""
    normalized = normalize_unicode(test_text)
    
    return {
        'original': test_text,
        'normalized': normalized,
        'is_devanagari': is_devanagari(normalized),
        'length_original': len(test_text),
        'length_normalized': len(normalized),
        'normalization_changed': test_text != normalized,
        'sha256_original': __import__('hashlib').sha256(test_text.encode()).hexdigest()[:16],
        'sha256_normalized': __import__('hashlib').sha256(normalized.encode()).hexdigest()[:16]
    }

if __name__ == "__main__":
    # Test font utilities
    print("🔤 Testing Devanagari font utilities...")
    
    # Test validation
    test_cases = [
        "नेपाल",
        "स्वास्थ्य सेवा", 
        "Hello World",
        "नेपाल Government",
        ""
    ]
    
    for test in test_cases:
        result = validate_font_rendering(test)
        print(f"Text: '{test}' -> Devanagari: {result['is_devanagari']}, Normalized: {result['normalization_changed']}")
    
    # Test CSS generation
    css = inject_devanagari_fonts()
    print(f"✅ Generated CSS: {len(css)} characters")
    print("🎯 Font utilities ready for CP11.5")
