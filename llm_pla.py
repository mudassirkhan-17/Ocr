"""
LLM-Based Certificate Field Extraction
Extracts key fields from ACORD insurance certificates using GPT-4.1-mini
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


class CertificateExtractor:
    """Extract fields from insurance certificates using LLM"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the extractor
        
        Args:
            model: OpenAI model to use (default: gpt-4o-mini)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def parse_triple_extraction(self, combo_text: str) -> tuple[str, str, str]:
        """
        Parse combo file to extract pdfplumber, PyMuPDF, and Tesseract sections separately
        
        Args:
            combo_text: Combined extraction text with all three methods
            
        Returns:
            Tuple of (pdfplumber_text, pymupdf_text, tesseract_text)
        """
        pdfplumber_text = ""
        pymupdf_text = ""
        tesseract_text = ""
        
        # Split by the extraction method markers (new format from cert_extract_pla.py)
        if "--- PDFPLUMBER (Table-aware) ---" in combo_text:
            parts = combo_text.split("--- PDFPLUMBER (Table-aware) ---")
            if len(parts) > 1:
                pdfplumber_section = parts[1]
                
                # Extract pdfplumber text (everything until PyMuPDF section)
                if "--- PYMUPDF (Text layer) ---" in pdfplumber_section:
                    pdfplumber_text = pdfplumber_section.split("--- PYMUPDF (Text layer) ---")[0].strip()
                    remaining = pdfplumber_section.split("--- PYMUPDF (Text layer) ---")[1]
                    
                    # Extract PyMuPDF text (everything until Tesseract section)
                    if "--- TESSERACT (OCR) ---" in remaining:
                        pymupdf_text = remaining.split("--- TESSERACT (OCR) ---")[0].strip()
                        tesseract_text = remaining.split("--- TESSERACT (OCR) ---")[1].strip()
                    else:
                        pymupdf_text = remaining.strip()
                else:
                    pdfplumber_text = pdfplumber_section.strip()
        
        # Fallback: try old format (Tesseract + PyMuPDF only)
        if not pdfplumber_text and not pymupdf_text and not tesseract_text:
            if "--- TESSERACT (Buffer=1) ---" in combo_text:
                parts = combo_text.split("--- TESSERACT (Buffer=1) ---")
                if len(parts) > 1:
                    tesseract_section = parts[1]
                    if "--- PYMUPDF (Buffer=0) ---" in tesseract_section:
                        tesseract_text = tesseract_section.split("--- PYMUPDF (Buffer=0) ---")[0].strip()
                        pymupdf_text = tesseract_section.split("--- PYMUPDF (Buffer=0) ---")[1].strip()
                    else:
                        tesseract_text = tesseract_section.strip()
        
        # If parsing failed, return the whole text as single source
        if not pdfplumber_text and not pymupdf_text and not tesseract_text:
            pdfplumber_text = combo_text
        
        return pdfplumber_text, pymupdf_text, tesseract_text
    
    def create_extraction_prompt(self, pdfplumber_text: str, pymupdf_text: str = None, tesseract_text: str = None) -> str:
        """
        Create the extraction prompt for the LLM with triple extraction validation
        
        Args:
            pdfplumber_text: Extraction text from pdfplumber (table-aware, primary source)
            pymupdf_text: Extraction text from PyMuPDF (text layer, optional)
            tesseract_text: OCR text from Tesseract (OCR, optional)
            
        Returns:
            Formatted prompt string
        """
        # TODO: Build prompt step by step
        prompt = ""
        
        return prompt
    
    def _extract_property_section_page(self, text: str) -> Optional[str]:
        """
        Extract PAGE 1 (header info) + PROPERTY SECTION page from a multi-page ACORD 140 document.
        This reduces the context size and helps the LLM focus on the coverages.
        
        Args:
            text: Full document text
            
        Returns:
            PAGE 1 + PROPERTY SECTION page content, or None if not an ACORD 140
        """
        import re
        
        # Check if this is an ACORD 140 with PROPERTY SECTION
        if "PROPERTY SECTION" not in text:
            return None
        
        # Split by PAGE markers
        page_pattern = r'(={60,}\nPAGE \d+\n={60,})'
        parts = re.split(page_pattern, text)
        
        # Reconstruct pages: combine markers with their content
        pages_dict = {}
        current_marker = None
        for part in parts:
            if re.match(r'={60,}\nPAGE \d+\n={60,}', part):
                current_marker = part
                page_match = re.search(r'PAGE (\d+)', part)
                if page_match:
                    page_num = int(page_match.group(1))
            elif current_marker:
                pages_dict[page_num] = current_marker + part
                current_marker = None
        
        # Build filtered content: PAGE 1 (header) + PROPERTY SECTION page
        filtered_parts = []
        
        # Always include PAGE 1 for header info (insured name, dates, addresses)
        if 1 in pages_dict:
            filtered_parts.append("=== PAGE 1 - HEADER INFO ===\n" + pages_dict[1])
        
        # Find and include the PROPERTY SECTION page (usually PAGE 6)
        for page_num, content in pages_dict.items():
            if page_num == 1:
                continue  # Already added
            if "PROPERTY SECTION" in content:
                # Check for actual coverage data
                if "Building" in content or "Business Income" in content or "Equipment Breakdown" in content:
                    filtered_parts.append(f"\n\n=== PAGE {page_num} - PROPERTY SECTION (COVERAGES) ===\n" + content)
                    break
        
        if len(filtered_parts) > 1:  # Have both header and coverages
            return "\n".join(filtered_parts)
        
        return None
    
    def extract_fields(self, ocr_text: str, use_dual_validation: bool = True) -> Dict[str, Optional[str]]:
        """
        Extract fields from certificate text using LLM
        
        Args:
            ocr_text: The OCR extracted text (may be combo file with dual OCR)
            use_dual_validation: If True, parse and validate both OCR sources
            
        Returns:
            Dictionary with extracted fields
        """
        # Try to parse triple extraction if available
        pdfplumber_text, pymupdf_text, tesseract_text = "", "", ""
        
        if use_dual_validation:
            pdfplumber_text, pymupdf_text, tesseract_text = self.parse_triple_extraction(ocr_text)
            sources = []
            if pdfplumber_text:
                sources.append("pdfplumber")
            if pymupdf_text:
                sources.append("PyMuPDF")
            if tesseract_text:
                sources.append("Tesseract")
            
            if len(sources) > 1:
                print(f"✅ Detected multiple extraction sources ({', '.join(sources)}) - using cross-validation")
            elif pdfplumber_text:
                print("ℹ️  Single extraction source detected (pdfplumber)")
            else:
                pdfplumber_text = ocr_text
                print("ℹ️  Single extraction source detected")
        else:
            pdfplumber_text = ocr_text
        
        # For ACORD 140 forms, filter to just the PROPERTY SECTION page to avoid LLM confusion
        property_section_text = self._extract_property_section_page(pdfplumber_text)
        if property_section_text:
            print(f"✅ Found PROPERTY SECTION - filtering to relevant page ({len(property_section_text)} chars)")
            # Replace the full text with just the property section for coverage extraction
            pdfplumber_text = property_section_text
            # Also filter PyMuPDF if available
            if pymupdf_text:
                pymupdf_property = self._extract_property_section_page(pymupdf_text)
                if pymupdf_property:
                    pymupdf_text = pymupdf_property
        
        # Create prompt
        prompt = self.create_extraction_prompt(
            pdfplumber_text, 
            pymupdf_text if pymupdf_text else None,
            tesseract_text if tesseract_text else None
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert insurance document analyzer. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,  # Deterministic output
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result_text = response.choices[0].message.content.strip()
            extracted_data = json.loads(result_text)
            
            return extracted_data
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse LLM response as JSON: {e}")
            print(f"Response was: {result_text}")
            return {
                "policy_number": None,
                "effective_date": None,
                "expiration_date": None,
                "insured_name": None,
                "mailing_address": None,
                "location_address": None,
                "error": "JSON parsing failed"
            }
        except Exception as e:
            print(f"❌ Error calling LLM API: {e}")
            return {
                "policy_number": None,
                "effective_date": None,
                "expiration_date": None,
                "insured_name": None,
                "mailing_address": None,
                "location_address": None,
                "error": str(e)
            }
    
    def extract_from_file(self, file_path: Path) -> Dict[str, Optional[str]]:
        """
        Extract fields from a certificate text file
        
        Args:
            file_path: Path to the OCR text file
            
        Returns:
            Dictionary with extracted fields
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read the OCR text
        with open(file_path, 'r', encoding='utf-8') as f:
            ocr_text = f.read()
        
        # Extract fields
        return self.extract_fields(ocr_text)


def main():
    """Main function to extract fields from certificate"""
    
    print("\n" + "="*80)
    print("CERTIFICATE FIELD EXTRACTION (LLM-Based)")
    print("="*80)
    print()
    
    # Get input file
    if len(sys.argv) < 2:
        print("⚠️  No input provided, using default: james_pl")
        base_name = "naiya_pla"
    else:
        base_name = sys.argv[1]
    
    # Carrier directory (change this to switch between nationwideop, encovaop, etc.)
    carrier_dir = "nonstandardop"
    
    # Look for the combo file (best extraction)
    input_file = Path(f"{carrier_dir}/{base_name}_combo.txt")
    
    if not input_file.exists():
        # Try alternatives
        alternatives = [
            Path(f"{carrier_dir}/{base_name}1.txt"),  # pdfplumber
            Path(f"{carrier_dir}/{base_name}2.txt"),  # PyMuPDF
            Path(f"{carrier_dir}/{base_name}3.txt"),  # Tesseract
        ]
        for alt in alternatives:
            if alt.exists():
                input_file = alt
                break
    
    if not input_file.exists():
        print(f"❌ No OCR file found for: {base_name}")
        print("   Please run cert_extract_pl.py or cert_extract_gl.py first")
        return
    
    print(f"📄 Input file: {input_file}")
    print(f"   Size: {input_file.stat().st_size:,} bytes")
    
    # Check if it's a combo file (triple extraction)
    is_combo = "_combo.txt" in str(input_file)
    if is_combo:
        print(f"   Type: Triple extraction (pdfplumber + PyMuPDF + Tesseract)")
    else:
        print(f"   Type: Single extraction")
    print()
    
    # Initialize extractor
    try:
        extractor = CertificateExtractor()
        print(f"✅ LLM initialized: {extractor.model}\n")
    except ValueError as e:
        print(f"❌ {e}")
        print("   Please add OPENAI_API_KEY to your .env file")
        return
    
    # Extract fields
    print("🔍 Extracting fields with LLM cross-validation...\n")
    result = extractor.extract_from_file(input_file)
    
    # Display results
    print("\n" + "="*80)
    print("EXTRACTED FIELDS")
    print("="*80)
    print()
    print(json.dumps(result, indent=2))
    print()
    
    # Save results
    output_file = Path(f"{carrier_dir}/{base_name}_extracted_real.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()

