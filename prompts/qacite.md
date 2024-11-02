# Q-A-Cite Generation Prompt for NepaliGov-RAG-Bench

## Task
Generate question-answer pairs with precise citations from authoritative Nepal government documents. Each answer must be grounded in the provided text chunk with exact character spans and bounding boxes.

## Guidelines

### Question Generation
- Generate questions in the SAME LANGUAGE as the source chunk
- For Nepali chunks: Ask questions in Nepali using appropriate question words (के, कहाँ, कसरी, किन, कति, etc.)
- For English chunks: Ask questions in English using standard question words (what, where, how, why, when, etc.)
- Questions should be specific and answerable from the chunk content
- Avoid overly broad or generic questions

### Answer Generation
- Provide concise, direct answers (1-3 sentences maximum)
- Answer must be EXACTLY extractable from the source chunk text
- For table chunks: Quote minimal cell values or specific data points
- Maintain factual accuracy and avoid interpretation beyond the text

### Citation Requirements
- Every answer must include exact character spans from the source chunk
- Character spans must be continuous substrings that support the answer
- Multiple spans allowed if answer draws from different parts of the chunk
- Spans should be minimal but complete (include full words/phrases)

### Language-Specific Examples

**Nepali Example:**
```
Chunk: "नेपालको संविधानले मौलिक अधिकारको ग्यारेन्टी दिन्छ। यसमा जीवनको अधिकार, स्वतन्त्रताको अधिकार र समानताको अधिकार समावेश छ।"

Question: "नेपालको संविधानले कुन अधिकारहरूको ग्यारेन्टी दिन्छ?"
Answer: "नेपालको संविधानले मौलिक अधिकारको ग्यारेन्टी दिन्छ जसमा जीवनको अधिकार, स्वतन्त्रताको अधिकार र समानताको अधिकार समावेश छ।"
Spans: ["मौलिक अधिकारको ग्यारेन्टी", "जीवनको अधिकार, स्वतन्त्रताको अधिकार र समानताको अधिकार"]
```

**English Example:**
```
Chunk: "The Health Emergency Operations Center (HEOC) coordinates emergency response activities. It serves as the central hub for information management and resource coordination during health crises."

Question: "What is the primary function of the Health Emergency Operations Center?"
Answer: "The HEOC coordinates emergency response activities and serves as the central hub for information management and resource coordination during health crises."
Spans: ["coordinates emergency response activities", "central hub for information management and resource coordination during health crises"]
```

**Table Example:**
```
Chunk Type: table
Chunk: "[TABLE: 3 rows × 2 cols]\nDistrict\tCases\nKathmandu\t150\nPokhara\t75"

Question: "How many cases were reported in Kathmandu?"
Answer: "150 cases were reported in Kathmandu."
Spans: ["Kathmandu\t150"]
```

### Quality Standards
- Questions must be natural and realistic (something a user might actually ask)
- Answers must be factually grounded and verifiable from the chunk
- Character spans must exactly match substrings in the original text
- Avoid hallucination or information not present in the chunk
- Maintain appropriate formality level for government documents

### Output Format
Each generated Q-A pair should include:
- `question`: The generated question
- `answer_exact`: The precise answer text
- `char_spans`: List of [start, end] character positions in the chunk
- `language`: Language code ('ne' or 'en')
- `chunk_type`: Type of source chunk ('text' or 'table')



