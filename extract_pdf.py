import fitz
import sys

if len(sys.argv) < 3:
    print("Usage: python extract_pdf.py <input> <output>")
    sys.exit(1)

doc = fitz.open(sys.argv[1])
text = ""
for page in doc:
    text += page.get_text()

with open(sys.argv[2], 'w', encoding='utf-8') as f:
    f.write(text)
