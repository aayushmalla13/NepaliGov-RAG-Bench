#!/usr/bin/env python3
"""
Quick fix for translation quality - replace the problematic method
"""

import sys
import os

# Read the current translation.py file
with open('api/translation.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Create a much better translation method
new_method = '''    def _translate_ne_to_en(self, text: str) -> str:
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
            clean = word.strip('।,.:;!?()[]{}"\\'')
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
        
        return ' '.join(translated)'''

# Find the old method and replace it
start_marker = "    def _translate_ne_to_en(self, text: str) -> str:"
end_marker = "        return \" \".join(translated_words)"

start_idx = content.find(start_marker)
if start_idx == -1:
    start_idx = content.find("    def _translate_ne_to_en(self, text: str) -> str:")

if start_idx != -1:
    # Find the end of the method
    end_idx = content.find(end_marker, start_idx)
    if end_idx != -1:
        end_idx = content.find('\n', end_idx) + 1
        
        # Replace the method
        new_content = content[:start_idx] + new_method + content[end_idx:]
        
        # Write back
        with open('api/translation.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ Translation method replaced successfully!")
        print("🔄 Please restart the app to see the changes")
    else:
        print("❌ Could not find method end")
else:
    print("❌ Could not find translation method to replace")
