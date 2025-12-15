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
            # Dual OCR mode - cross-validation
            prompt = """You are an expert in business insurance and ACORD (Association for Cooperative Operations Research and Development) forms. ACORD forms have a STANDARDIZED, FIXED FORMAT that never changes.

**IMPORTANT: You have TWO independent OCR extractions of the SAME document below.**

**ACORD Form Standard Structure (ALWAYS THE SAME):**
ACORD forms follow a fixed layout structure:
- **Top Section**: Agency information (name, address, phone, email)
- **Header Row**: Contains INSURED name, POLICY NUMBER, EFFECTIVE DATE, EXPIRATION DATE
- **Middle Section**: Property/Liability information, Coverage details
- **Bottom Section**: Additional interests, Authorized representative

**Standard Field Locations (NEVER CHANGE):**
1. **INSURED** - Always in the header row, labeled "INSURED", contains the insured party name(s)
2. **MAILING ADDRESS** - Always immediately below INSURED name (street, city, state, zip on separate lines)
3. **POLICY NUMBER** - Always in the header row, after INSURED name, labeled "POLICY NUMBER"
4. **EFFECTIVE DATE** - Always in the header row, next to EXPIRATION DATE, labeled "EFFECTIVE DATE"
5. **EXPIRATION DATE** - Always in the header row, next to EFFECTIVE DATE, labeled "EXPIRATION DATE" (may be OCR'd as "EXPRATION DATE")
6. **LOCATION ADDRESS** - Always in "PROPERTY INFORMATION" section, under "LOCATION/DESCRIPTION" label (street, city, state, zip on separate lines)
7. **ADDITIONAL INTEREST** - Always in the bottom section, labeled "ADDITIONAL INTEREST" (may also be "MORTGAGEE", "LOSS PAYEE", or "ADDITIONAL INSURED")
   - Can have 0, 1, 2, or more entries
   - Each entry has name and address on separate lines
   - **CRITICAL:** Generic placeholders like "TO WHOM IT MAY CONCERN" are NOT valid additional interests (count as 0)

**IMPORTANT DATE CLARIFICATION:**
- The **certificate issue date** appears at the very top of the form (e.g., "DATE (MM/DD/YYYY) 7/14/2025") - DO NOT use this
- The **EFFECTIVE DATE** is the policy start date - look for it in the header row AFTER the policy number (e.g., "06/15/2025")
- The **EXPIRATION DATE** comes immediately after the effective date (e.g., "06/15/2026")
- DO NOT confuse the certificate issue date with the policy effective date

Your task is to:
1. **Compare both OCR outputs** to identify discrepancies
2. **Cross-validate** the information between both sources
3. **Choose the most accurate value** when there are conflicts
4. **Handle OCR errors** intelligently using context from both sources
5. **Fill gaps** where one OCR captured data the other missed
6. **Use the standard ACORD form structure** - fields are always in the same relative positions

**Fields to Extract:**
1. **Policy Number** - Found in the header row, after INSURED name, labeled "POLICY NUMBER"
2. **Effective Date** - Found in the header row, labeled "EFFECTIVE DATE", format: MM/DD/YYYY
3. **Expiration Date** - Found in the header row, labeled "EXPIRATION DATE" (or "EXPRATION DATE" if OCR error), format: MM/DD/YYYY
4. **Insured Name** - Found in the header row, labeled "INSURED"
   - This is the name of the person or entity covered by the policy
   - Examples: "MURFF, JAMES P; KENDALL, WILLIAM E", "SEAN SKA PROPERTIES LLC", "AANIYA GAS AND FOOD INC DBA QUICK SHOP"
   - May contain multiple names separated by semicolons
   - Extract exactly as shown (name only, not address)
5. **Mailing Address** - Found immediately below INSURED name in the header section
   - This is the mailing address where the insured receives mail
   - Full address format: "Street, City, State ZIP"
   - Example: "1812 MORRIS LANDERS DR NE, ATLANTA, GA 30345-4104"
   - Combine all address lines into a single string with commas
6. **Location Address** - Found in "PROPERTY INFORMATION" or "LOCATION/DESCRIPTION" section
   - This is the physical location of the insured property
   - Full address format: "Street, City, State ZIP"
   - Example: "4940 LAVISTA RD, TUCKER, GA 30084-4403"
   - Combine all address lines into a single string with commas
   - May be the same as mailing address or different
7. **Additional Interest(s)** - Found in "ADDITIONAL INTEREST" section at the bottom of the form
   - Parties with an insurable interest (mortgagee, lender, property owner, etc.)
   - **IMPORTANT - CONDITIONAL FORMATTING BASED ON COUNT:**
     - **If 0 additional interests found** → Do NOT include any additional interest fields (no null, no empty string, DO NOT ADD THE KEYS AT ALL)
     - **If EXACTLY 1 additional interest found** → Use flat structure:
       - "additional_interest_name": "name only"
       - "additional_interest_address": "full address"
     - **If 2 OR MORE additional interests found** → Use array structure:
       - "additional_interests": [{"name": "...", "address": "..."}, {...}]
   - **CRITICAL:** "TO WHOM IT MAY CONCERN" is NOT a valid additional interest → count as 0 → OMIT FIELDS ENTIRELY
   - Extract all legitimate additional interests (can be 0, 1, 2, 3, or more)
8. **Coverages** - Found in "COVERAGE INFORMATION" section, under "COVERAGE / PERILS / FORMS" and "AMOUNT OF INSURANCE" columns
   - This is a table with coverage items and their insurance amounts
   - **IMPORTANT - DYNAMIC EXTRACTION:**
     - Extract ONLY the coverages that are present in the certificate
     - Do NOT include coverages that are not listed
     - Each certificate may have different coverages (variable, not fixed)
   - **Format as a JSON object** with coverage name as key and amount as value
   - Common coverage examples: "Building", "Pumps", "Canopy", "Business Personal Property", "Business Income", "Equipment Breakdown", "Employee Dishonesty", "Money and Securities", "Outdoor Signs", "Wind and Hail", "Windstorm or Hail"
   - Amount values can be:
     - Numeric (e.g., "377,743", "100,000", "10,000")
     - "Included" (coverage is included without specific limit)
     - "Actual Loss Sustained" (for business income)
   - Extract the amount exactly as shown (including commas in numbers)

**Validation Rules:**
- If both sources agree → high confidence, use that value
- If sources disagree → use the one that makes more logical sense
- If one source has formatting issues → prefer the cleaner formatted value
- Dates must be in MM/DD/YYYY format
- Policy numbers should be numeric (may have periods or hyphens)

**Output Format:**
Return ONLY a valid JSON object with CONDITIONAL structure for additional interests:

**If 0 additional interests (e.g., "TO WHOM IT MAY CONCERN" or no entries):**
{
    "policy_number": "string or null",
    "effective_date": "MM/DD/YYYY or null",
    "expiration_date": "MM/DD/YYYY or null",
    "insured_name": "string or null",
    "mailing_address": "string or null",
    "location_address": "string or null",
    "coverages": {
        "Building": "377,743",
        "Pumps": "100,000",
        "Business Personal Property": "53,000"
    },
    "validation_notes": "..."
}
NOTE: NO additional_interest fields at all - the keys should NOT exist in the JSON
NOTE: coverages object should ONLY contain items that are present in the certificate

**If EXACTLY 1 additional interest:**
{
    "policy_number": "string or null",
    "effective_date": "MM/DD/YYYY or null",
    "expiration_date": "MM/DD/YYYY or null",
    "insured_name": "string or null",
    "mailing_address": "string or null",
    "location_address": "string or null",
    "coverages": {
        "Building": "344,000",
        "Business Personal Property": "55,900"
    },
    "additional_interest_name": "string",
    "additional_interest_address": "string",
    "validation_notes": "..."
}

**If 2 OR MORE additional interests:**
{
    "policy_number": "string or null",
    "effective_date": "MM/DD/YYYY or null",
    "expiration_date": "MM/DD/YYYY or null",
    "insured_name": "string or null",
    "mailing_address": "string or null",
    "location_address": "string or null",
    "coverages": {
        "Building": "949,400",
        "Business Income": "Actual Loss Sustained",
        "Equipment Breakdown": "Included"
    },
    "additional_interests": [
        {"name": "string", "address": "string"},
        {"name": "string", "address": "string"}
    ],
    "validation_notes": "..."
}

**OCR Source 1 (Tesseract):**
---
""" + tesseract_text + """
---

**OCR Source 2 (PyMuPDF):**
---
""" + pymupdf_text + """
---

Now analyze both sources, cross-validate, and return the JSON object:"""
        else:
            # Single OCR mode
            prompt = """You are an expert in business insurance and ACORD (Association for Cooperative Operations Research and Development) forms. ACORD forms have a STANDARDIZED, FIXED FORMAT that never changes.

**ACORD Form Standard Structure (ALWAYS THE SAME):**
- **Top Section**: Agency information (name, address, phone, email)
- **Header Row**: Contains INSURED name, POLICY NUMBER, EFFECTIVE DATE, EXPIRATION DATE
- **Middle Section**: Property/Liability information, Coverage details
- **Bottom Section**: Additional interests, Authorized representative

**Standard Field Locations (NEVER CHANGE):**
1. **INSURED** - Always in the header row, labeled "INSURED", contains the insured party name(s)
2. **MAILING ADDRESS** - Always immediately below INSURED name (street, city, state, zip on separate lines)
3. **POLICY NUMBER** - Always in the header row, after INSURED name, labeled "POLICY NUMBER"
4. **EFFECTIVE DATE** - Always in the header row, next to EXPIRATION DATE, labeled "EFFECTIVE DATE"
5. **EXPIRATION DATE** - Always in the header row, next to EFFECTIVE DATE, labeled "EXPIRATION DATE" (may be OCR'd as "EXPRATION DATE")
6. **LOCATION ADDRESS** - Always in "PROPERTY INFORMATION" section, under "LOCATION/DESCRIPTION" label (street, city, state, zip on separate lines)
7. **ADDITIONAL INTEREST** - Always in the bottom section, labeled "ADDITIONAL INTEREST" (may also be "MORTGAGEE", "LOSS PAYEE", or "ADDITIONAL INSURED")
   - Can have 0, 1, 2, or more entries
   - Each entry has name and address on separate lines
   - **CRITICAL:** Generic placeholders like "TO WHOM IT MAY CONCERN" are NOT valid additional interests (count as 0)
8. **COVERAGES** - Always in "COVERAGE INFORMATION" section, table with coverage items and amounts
   - Extract ONLY the coverages that are present in the certificate
   - Do NOT include coverages that are not listed

**IMPORTANT DATE CLARIFICATION:**
- The **certificate issue date** appears at the very top of the form (e.g., "DATE (MM/DD/YYYY) 7/14/2025") - DO NOT use this
- The **EFFECTIVE DATE** is the policy start date - look for it in the header row AFTER the policy number (e.g., "06/15/2025")
- The **EXPIRATION DATE** comes immediately after the effective date (e.g., "06/15/2026")
- DO NOT confuse the certificate issue date with the policy effective date

**Your Task:**
Extract the following fields using the standard ACORD form structure:
1. **Policy Number** - Found in header row, after INSURED name, labeled "POLICY NUMBER"
2. **Effective Date** - Found in header row, labeled "EFFECTIVE DATE", format: MM/DD/YYYY
3. **Expiration Date** - Found in header row, labeled "EXPIRATION DATE" (or "EXPRATION DATE" if OCR error), format: MM/DD/YYYY
4. **Insured Name** - Found in header row, labeled "INSURED"
   - This is the name of the person or entity covered by the policy
   - Examples: "MURFF, JAMES P; KENDALL, WILLIAM E", "SEAN SKA PROPERTIES LLC"
   - May contain multiple names separated by semicolons
   - Extract exactly as shown (name only, not address)
5. **Mailing Address** - Found immediately below INSURED name in the header section
   - Full address format: "Street, City, State ZIP"
   - Example: "1812 MORRIS LANDERS DR NE, ATLANTA, GA 30345-4104"
   - Combine all address lines into a single string with commas
6. **Location Address** - Found in "PROPERTY INFORMATION" or "LOCATION/DESCRIPTION" section
   - Full address format: "Street, City, State ZIP"
   - Example: "4940 LAVISTA RD, TUCKER, GA 30084-4403"
   - Combine all address lines into a single string with commas
7. **Additional Interest(s)** - Found in "ADDITIONAL INTEREST" section at the bottom of the form
   - Parties with an insurable interest (mortgagee, lender, property owner, etc.)
   - **IMPORTANT - CONDITIONAL FORMATTING BASED ON COUNT:**
     - **If 0 additional interests found** → Do NOT include any additional interest fields (no null, no empty string, DO NOT ADD THE KEYS AT ALL)
     - **If EXACTLY 1 additional interest found** → Use flat structure:
       - "additional_interest_name": "name only"
       - "additional_interest_address": "full address"
     - **If 2 OR MORE additional interests found** → Use array structure:
       - "additional_interests": [{"name": "...", "address": "..."}, {...}]
   - **CRITICAL:** "TO WHOM IT MAY CONCERN" is NOT a valid additional interest → count as 0 → OMIT FIELDS ENTIRELY
8. **Coverages** - Found in "COVERAGE INFORMATION" table
   - Extract ONLY coverages that are present
   - Format as JSON object: {"Coverage Name": "Amount"}
   - Amount values: numeric with commas (e.g., "377,743"), "Included", or "Actual Loss Sustained"

**Output Format:**
Return ONLY a valid JSON object with CONDITIONAL structure for additional interests:

**If 0 additional interests (e.g., "TO WHOM IT MAY CONCERN" or no entries):**
{
    "policy_number": "string or null",
    "effective_date": "MM/DD/YYYY or null",
    "expiration_date": "MM/DD/YYYY or null",
    "insured_name": "string or null",
    "mailing_address": "string or null",
    "location_address": "string or null",
    "coverages": {
        "Building": "344,000",
        "Business Personal Property": "55,900"
    }
}
NOTE: NO additional_interest fields at all - the keys should NOT exist in the JSON

**If EXACTLY 1 additional interest:**
{
    "policy_number": "string or null",
    "effective_date": "MM/DD/YYYY or null",
    "expiration_date": "MM/DD/YYYY or null",
    "insured_name": "string or null",
    "mailing_address": "string or null",
    "location_address": "string or null",
    "coverages": {
        "Building": "344,000",
        "Business Income": "Actual Loss Sustained"
    },
    "additional_interest_name": "string",
    "additional_interest_address": "string"
}

**If 2 OR MORE additional interests:**
{
    "policy_number": "string or null",
    "effective_date": "MM/DD/YYYY or null",
    "expiration_date": "MM/DD/YYYY or null",
    "insured_name": "string or null",
    "mailing_address": "string or null",
    "location_address": "string or null",
    "coverages": {
        "Building": "949,400",
        "Outdoor Signs": "2,500"
    },
    "additional_interests": [
        {"name": "string", "address": "string"},
        {"name": "string", "address": "string"}
    ]
}

**Important Guidelines:**
- Use MM/DD/YYYY format for dates
- Return ONLY the JSON object, no additional text or explanation
- If a field is not found or unclear, use null
- Be precise and extract exactly what appears on the certificate
- Handle OCR errors intelligently based on context

**Certificate OCR Text:**
---
""" + tesseract_text + """
---

Return the JSON object now:"""
        
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
        print("⚠️  No input provided, using default: james_pl")
        base_name = "stay_pl"
    else:
        base_name = sys.argv[1]
    
    # Look for the combo file (best extraction)
    input_file = Path(f"nationwideop/{base_name}_combo.txt")
    
    if not input_file.exists():
        # Try alternatives
        alternatives = [
            Path(f"nationwideop/{base_name}2.txt"),  # PyMuPDF
            Path(f"nationwideop/{base_name}1.txt"),  # Tesseract
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
    output_file = Path(f"nationwideop/{base_name}_extracted_real.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()

