"""
LLM-Based Certificate Field Extraction
Extracts key fields from ACORD GL certificates (ACORD 25) using GPT-4o-mini
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
    
    def parse_dual_ocr(self, combo_text: str) -> tuple[str, str]:
        """
        Parse combo file to extract Tesseract and PyMuPDF sections separately
        
        Args:
            combo_text: Combined OCR text with both methods
            
        Returns:
            Tuple of (tesseract_text, pymupdf_text)
        """
        tesseract_text = ""
        pymupdf_text = ""
        
        # Split by the buffer markers
        if "--- TESSERACT (Buffer=1) ---" in combo_text:
            parts = combo_text.split("--- TESSERACT (Buffer=1) ---")
            if len(parts) > 1:
                tesseract_section = parts[1]
                
                # Extract Tesseract text (everything until PyMuPDF section)
                if "--- PYMUPDF (Buffer=0) ---" in tesseract_section:
                    tesseract_text = tesseract_section.split("--- PYMUPDF (Buffer=0) ---")[0].strip()
                    pymupdf_text = tesseract_section.split("--- PYMUPDF (Buffer=0) ---")[1].strip()
                else:
                    tesseract_text = tesseract_section.strip()
        
        # If parsing failed, return the whole text as single source
        if not tesseract_text and not pymupdf_text:
            tesseract_text = combo_text
        
        return tesseract_text, pymupdf_text
    
    def create_extraction_prompt(self, tesseract_text: str, pymupdf_text: str = None) -> str:
        """
        Create the extraction prompt for the LLM with dual OCR validation
        
        Args:
            tesseract_text: OCR text from Tesseract method
            pymupdf_text: OCR text from PyMuPDF method (optional)
            
        Returns:
            Formatted prompt string
        """
        if pymupdf_text:
            # Dual OCR mode - cross-validation (ACORD 25 - GL)
            prompt = """You are an expert in ACORD 25 (Certificate of Liability Insurance) extraction.

You are given TWO OCR sources for the SAME document. Cross-validate them and choose the most accurate values.

==================================================
CRITICAL: DATE CLARIFICATION
==================================================
- The top "DATE (MM/DD/YYYY)" is the CERTIFICATE ISSUE DATE. Do NOT use it as policy dates.
- Policy Effective/Expiration dates are in the coverage table columns ("POLICY EFF" / "POLICY EXP") per coverage line.

==================================================
FIELDS TO EXTRACT (ACORD 25)
==================================================
Return ONLY a valid JSON object with:

1) Top-level (best overall / primary policy info):
- policy_number: string or null (prefer CGL policy number if multiple exist)
- effective_date: MM/DD/YYYY or null (prefer CGL effective date)
- expiration_date: MM/DD/YYYY or null (prefer CGL expiration date)
- insured_name: string or null
- mailing_address: string or null (combine lines with commas)
- location_address: string or null (often found in "DESCRIPTION OF OPERATIONS / LOCATIONS / VEHICLES"; may be absent)

2) Certificate holder (bottom section labeled "CERTIFICATE HOLDER"):
- If certificate holder is missing or is a generic placeholder like "TO WHOM IT MAY CONCERN", OMIT certificate holder fields entirely (no nulls, no empty strings, do not add the keys).
- If exactly 1 certificate holder:
  - certificate_holder_name: string
  - certificate_holder_address: string
- If 2+ certificate holders (rare; schedule/attachments):
  - certificate_holders: [{"name": "...", "address": "..."}, ...]

3) GL coverages (ACORD 25 coverages table):
- Extract ONLY coverages that are present (do not invent).
- Return as a nested object under key "coverages".
- Each coverage object MUST include policy_number/policy_eff/policy_exp for that line if present.

Coverage keys to use (only when present):

A) commercial_general_liability:
{
  "policy_number": "...",
  "policy_eff": "MM/DD/YYYY or null",
  "policy_exp": "MM/DD/YYYY or null",
  "claims_made": true/false/null,
  "occur": true/false/null,
  "general_aggregate_applies_per": "POLICY|PROJECT|LOC|null",
  "limits": {
    "each_occurrence": "string or null",
    "damage_to_rented_premises": "string or null",
    "med_exp": "string or null",
    "personal_adv_injury": "string or null",
    "general_aggregate": "string or null",
    "products_comp_op_agg": "string or null"
  },
  "deductible_or_retention": "string or null"
}

B) automobile_liability:
{
  "policy_number": "...",
  "policy_eff": "MM/DD/YYYY or null",
  "policy_exp": "MM/DD/YYYY or null",
  "any_auto": true/false/null,
  "owned_autos_only": true/false/null,
  "hired_autos_only": true/false/null,
  "scheduled_autos": true/false/null,
  "non_owned_autos_only": true/false/null,
  "limits": {
    "combined_single_limit": "string or null",
    "bodily_injury_per_person": "string or null",
    "bodily_injury_per_accident": "string or null",
    "property_damage": "string or null"
  },
  "deductible_or_retention": "string or null"
}

C) umbrella_liability (or excess_liability):
{
  "policy_number": "...",
  "policy_eff": "MM/DD/YYYY or null",
  "policy_exp": "MM/DD/YYYY or null",
  "claims_made": true/false/null,
  "occur": true/false/null,
  "limits": {
    "each_occurrence": "string or null",
    "aggregate": "string or null"
  },
  "deductible_or_retention": "string or null"
}

D) workers_compensation:
{
  "policy_number": "...",
  "policy_eff": "MM/DD/YYYY or null",
  "policy_exp": "MM/DD/YYYY or null",
  "wc_statutory": true/false/null,
  "employers_liability": {
    "each_accident": "string or null",
    "disease_each_employee": "string or null",
    "disease_policy_limit": "string or null"
  }
}

E) employment_practices_liability (if present as separate line):
{
  "policy_number": "...",
  "policy_eff": "MM/DD/YYYY or null",
  "policy_exp": "MM/DD/YYYY or null",
  "limits": {
    "each_limit": "string or null",
    "aggregate_limit": "string or null"
  }
}

F) liquor_liability:
{
  "policy_number": "...",
  "policy_eff": "MM/DD/YYYY or null",
  "policy_exp": "MM/DD/YYYY or null",
  "limits": {
    "each_limit": "string or null",
    "aggregate_limit": "string or null"
  }
}

G) garagekeepers_liability:
{
  "policy_number": "...",
  "policy_eff": "MM/DD/YYYY or null",
  "policy_exp": "MM/DD/YYYY or null",
  "limits": {
    "limit": "string or null"
  },
  "comprehensive_deductible": "string or null",
  "collision_deductible": "string or null",
  "notes": "string or null"
}

4) validation_notes: string (brief notes about OCR conflicts or assumptions)

==================================================
OCR SOURCE 1 (Tesseract)
==================================================
""" + tesseract_text + """

==================================================
OCR SOURCE 2 (PyMuPDF)
==================================================
""" + pymupdf_text + """

Return ONLY the JSON object now."""
        else:
            # Single OCR mode (ACORD 25 - GL)
            prompt = """You are an expert in ACORD 25 (Certificate of Liability Insurance) extraction.

CRITICAL: The top "DATE (MM/DD/YYYY)" is the certificate issue date. Do NOT use it for policy effective/expiration.
Policy dates are in the coverage table columns ("POLICY EFF" / "POLICY EXP") per coverage line.

Return ONLY a valid JSON object with:
- policy_number, effective_date, expiration_date, insured_name, mailing_address, location_address
- certificate holder fields (omit entirely if "TO WHOM IT MAY CONCERN")
- coverages (nested objects for any present coverages: CGL/Auto/Umbrella/WC/Liquor/EPL/Garagekeepers)
- validation_notes

Follow the same JSON shapes described in the dual-OCR instructions.

Certificate OCR Text:
---
""" + tesseract_text + """
---

Return ONLY the JSON object now."""
        
        return prompt
    
    def extract_fields(self, ocr_text: str, use_dual_validation: bool = True) -> Dict[str, Optional[str]]:
        """
        Extract fields from certificate text using LLM
        
        Args:
            ocr_text: The OCR extracted text (may be combo file with dual OCR)
            use_dual_validation: If True, parse and validate both OCR sources
            
        Returns:
            Dictionary with extracted fields
        """
        # Try to parse dual OCR if available
        tesseract_text, pymupdf_text = "", ""
        
        if use_dual_validation:
            tesseract_text, pymupdf_text = self.parse_dual_ocr(ocr_text)
            if tesseract_text and pymupdf_text:
                print("✅ Detected dual OCR sources - using cross-validation")
            else:
                tesseract_text = ocr_text
                print("ℹ️  Single OCR source detected")
        else:
            tesseract_text = ocr_text
        
        # Create prompt
        prompt = self.create_extraction_prompt(tesseract_text, pymupdf_text if pymupdf_text else None)
        
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
        print("⚠️  No input provided, using default: aaniya_gl")
        base_name = "westside_gl"
    else:
        base_name = sys.argv[1]
    
    # Carrier directory (change this to switch between nationwideop, encovaop, etc.)
    carrier_dir = "nationwideop"
    
    # Look for the combo file (best extraction)
    # NOTE: for GL we typically use base names like "aaniya_gl" so the file becomes "aaniya_gl_combo.txt"
    input_file = Path(f"{carrier_dir}/{base_name}_combo.txt")
    
    if not input_file.exists():
        # Try alternatives
        alternatives = [
            Path(f"{carrier_dir}/{base_name}2.txt"),  # PyMuPDF
            Path(f"{carrier_dir}/{base_name}1.txt"),  # Tesseract
        ]
        for alt in alternatives:
            if alt.exists():
                input_file = alt
                break
    
    if not input_file.exists():
        print(f"❌ No OCR file found for: {base_name}")
        print("   Please run cert_extract_gl.py first")
        return
    
    print(f"📄 Input file: {input_file}")
    print(f"   Size: {input_file.stat().st_size:,} bytes")
    
    # Check if it's a combo file (dual OCR)
    is_combo = "_combo.txt" in str(input_file)
    if is_combo:
        print(f"   Type: Dual OCR (Tesseract + PyMuPDF)")
    else:
        print(f"   Type: Single OCR")
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

