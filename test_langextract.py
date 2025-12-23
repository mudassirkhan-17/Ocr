"""
Simple text extraction from document
Just extract what's in the document - no modifications
"""

import os
import json

# Read the text file
text_file = "encovaop/baltic_fil1.txt"
if not os.path.exists(text_file):
    print(f"✗ File not found: {text_file}")
    exit(1)

print(f"Reading file: {text_file}")
with open(text_file, "r", encoding="utf-8") as f:
    text_content = f.read()

print(f"File size: {len(text_content):,} characters")
print(f"Total lines: {len(text_content.splitlines()):,}\n")

# Extract the document content as-is
print("=" * 70)
print("DOCUMENT CONTENT (as-is):")
print("=" * 70)
print(text_content)

# Save to output file
output_file = "baltic_extracted_content.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(text_content)

print(f"\n✓ Content saved to: {output_file}")
