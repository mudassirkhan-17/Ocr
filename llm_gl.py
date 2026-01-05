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
    
    def parse_dual_extraction(self, combo_text: str) -> tuple[str, str]:
        """
        Parse combo file to extract pdfplumber and PyMuPDF sections separately
        
        Args:
            combo_text: Combined extraction text with both methods
            
        Returns:
            Tuple of (pdfplumber_text, pymupdf_text)
        """
        pdfplumber_text = ""
        pymupdf_text = ""
        
        # Split by the extraction method markers
        if "--- PDFPLUMBER (Table-aware) ---" in combo_text:
            parts = combo_text.split("--- PDFPLUMBER (Table-aware) ---")
            if len(parts) > 1:
                pdfplumber_section = parts[1]
                
                # Extract pdfplumber text (everything until PyMuPDF section)
                if "--- PYMUPDF (Text layer) ---" in pdfplumber_section:
                    pdfplumber_text = pdfplumber_section.split("--- PYMUPDF (Text layer) ---")[0].strip()
                    pymupdf_text = pdfplumber_section.split("--- PYMUPDF (Text layer) ---")[1].strip()
                else:
                    pdfplumber_text = pdfplumber_section.strip()
        
        # If parsing failed, return the whole text as single source
        if not pdfplumber_text and not pymupdf_text:
            pdfplumber_text = combo_text
        
        return pdfplumber_text, pymupdf_text
    
    def create_extraction_prompt(self, pdfplumber_text: str, pymupdf_text: str = None) -> str:
        """
        Create the extraction prompt for the LLM with dual extraction validation
        
        Args:
            pdfplumber_text: Extraction text from pdfplumber (table-aware)
            pymupdf_text: Extraction text from PyMuPDF (text layer, optional)
            
        Returns:
            Formatted prompt string
        """
        if pymupdf_text:
            # Dual extraction mode - cross-validation (ACORD 25 - GL)
            prompt = """You are an expert in ACORD 25 (Certificate of Liability Insurance) extraction.

You are given TWO extraction sources for the SAME document:
1. **PDFPLUMBER (Table-aware)**: Preserves table structure - USE THIS AS PRIMARY SOURCE for coverage data
2. **PYMUPDF (Text layer)**: Raw text extraction - use as cross-validation/fallback

**PRIORITY**: For coverage extraction, prioritize pdfplumber's TABLE sections (especially TABLE 2 which contains the coverage table).
Cross-validate with PyMuPDF text when needed, but trust pdfplumber's structured table data first.

**CRITICAL - HIDDEN/TEMPLATE TEXT WARNING**:
- Both pdfplumber and PyMuPDF may extract hidden/template text that is NOT visible on the actual certificate
- If a coverage (especially Workers Compensation) appears in both sources but the row structure is unclear or doesn't match the clear format of other coverages (CGL, Umbrella, EPL), it may be template text → OMIT IT
- Workers Compensation is particularly prone to this - if the row doesn't have the same clear structure as CGL/Umbrella/EPL, it's likely template text → DO NOT INCLUDE

==================================================
CRITICAL: DATE CLARIFICATION
==================================================
- The top "DATE (MM/DD/YYYY)" is the CERTIFICATE ISSUE DATE. Do NOT use it as policy dates.
- Policy Effective/Expiration dates are in the coverage table columns ("POLICY EFF" / "POLICY EXP") per coverage line.

==================================================
CRITICAL: ADDRESS EXTRACTION RULES
==================================================
ACORD 25 certificates can have ONE or TWO addresses:

**IF TWO ADDRESSES EXIST:**
1. **Mailing Address** = Address in INSURED section (header, directly below insured name)
   - This is where mail is sent to the insured
   - Location: Top section, under "INSURED" label, immediately after insured name
   - Format: Combine all address lines with commas (e.g., "37 E MAIN ST N, HAMPTON, GA 30228-5501")
   - **CRITICAL**: This is the INSURED's address, NOT the PRODUCER address
2. **Location Address** = Address in "DESCRIPTION OF OPERATIONS / LOCATIONS / VEHICLES" section
   - This is the physical location where business operations occur
   - Location: Middle section, in the description box labeled "DESCRIPTION OF OPERATIONS / LOCATIONS / VEHICLES"
   - Format: Combine all address lines with commas

**IF ONLY ONE ADDRESS EXISTS:**
- Use it as **location_address** (physical location)
- Set **mailing_address = null** (do not use the single address for mailing)

**CRITICAL - DO NOT CONFUSE ADDRESSES:**
- **PRODUCER address** = Top left section labeled "PRODUCER" (agency/broker address like "McKinney & Co., P.O Box 7, Tucker, GA") → **IGNORE THIS - DO NOT EXTRACT**
- **INSURED address** = Section labeled "INSURED" (insured party address) → **THIS IS WHAT YOU EXTRACT**
- **Location address** = "DESCRIPTION OF OPERATIONS" section (if present and different from INSURED address)

**CRITICAL REMINDER - SINGLE ADDRESS RULE**:
- If you see an address ONLY in the INSURED section and NO address in the "DESCRIPTION OF OPERATIONS" section:
  → location_address = address from INSURED section
  → mailing_address = null
- DO NOT put the single address in mailing_address - it must go to location_address
- The single address represents the physical location, not a mailing address

**EXAMPLES:**
- Certificate shows "37 E MAIN ST N, HAMPTON, GA 30228-5501" in INSURED section only (no address in Description section)
  → location_address = "37 E MAIN ST N, HAMPTON, GA 30228-5501"
  → mailing_address = null
  **CORRECT**: Single address goes to location_address

- Certificate shows address in INSURED section AND different address in Description section
  → mailing_address = address from INSURED section
  → location_address = address from Description section
  **CORRECT**: Two addresses - INSURED = mailing, Description = location

**FIRST PRIORITY: Check insured name first, then determine address locations based on above rules.**

==================================================
FIELDS TO EXTRACT (ACORD 25)
==================================================
Return ONLY a valid JSON object with:

1) Top-level (best overall / primary policy info):
- policy_number: string or null (prefer CGL policy number if multiple exist)
- effective_date: MM/DD/YYYY or null (prefer CGL effective date)
- expiration_date: MM/DD/YYYY or null (prefer CGL expiration date)
- insured_name: string or null (extract from INSURED section header)
- mailing_address: string or null (combine lines with commas)
  **LOCATION**: INSURED section (header), directly below insured name
  **CRITICAL**: This is the INSURED's mailing address, NOT the PRODUCER address (ignore PRODUCER section)
  **RULE**: Only use if TWO addresses exist. If only ONE address exists, set to null
  **VERIFY**: Check "DESCRIPTION OF OPERATIONS" section - if it has an address, then use INSURED address as mailing_address. If no address in Description section → set mailing_address = null
  
- location_address: string or null (combine lines with commas)
  **LOCATION**: "DESCRIPTION OF OPERATIONS / LOCATIONS / VEHICLES" section (middle of certificate)
  **RULE**: Physical location where business operations occur.
  **CRITICAL**: If only ONE address exists (only in INSURED section, no address in Description section), use it here as location_address and set mailing_address = null
  **VERIFY**: If Description section has no address → use the INSURED address as location_address

2) Certificate holder (bottom section labeled "CERTIFICATE HOLDER"):
- If certificate holder is missing or is a generic placeholder like "TO WHOM IT MAY CONCERN", OMIT certificate holder fields entirely (no nulls, no empty strings, do not add the keys).
- If exactly 1 certificate holder:
  - certificate_holder_name: string
  - certificate_holder_address: string
- If 2+ certificate holders (rare; schedule/attachments):
  - certificate_holders: [{"name": "...", "address": "..."}, ...]

3) GL coverages (ACORD 25 coverages table):
- Extract ALL coverages that are present WITH DATA (do not invent, but do not skip valid ones).
- Return as a nested object under key "coverages".
- Each coverage object MUST include policy_number/policy_eff/policy_exp for that line if present.

**⛔⛔⛔ CRITICAL ANTI-HALLUCINATION RULE FOR COVERAGES ⛔⛔⛔**
- ONLY include coverages that have a ROW in the coverages table WITH ACTUAL DATA (policy numbers, limits, dates, etc.)
- If a coverage row is BLANK or has NO limits/policy numbers/dates, DO NOT include it in the output
- If a coverage is NOT listed in the table at all, DO NOT include it
- **WHEN IN DOUBT, OMIT THE COVERAGE** - it's better to miss something than to invent it

- **PRIMARY RULE - POLICY NUMBER REQUIRED**: 
  - **A coverage MUST have a policy_number to be included** - if there's no policy number in the POLICY NUMBER column, DO NOT include that coverage at all
  - Policy number is the PRIMARY indicator of whether a coverage is actually present - if it's missing, the coverage is not active
  - Limits alone are NOT sufficient - if there's no policy number, the row is likely incomplete/template/hidden text → OMIT IT
  - **VERIFY**: Before including any coverage, ask: "Does this row have a REAL, VISIBLE policy number in the POLICY NUMBER column?" If NO → OMIT IT

- **SPECIAL RULE FOR WORKERS COMPENSATION - EXTRA STRICT**: 
  - Workers Comp is OFTEN template text/hidden text that appears in both pdfplumber and PyMuPDF but is NOT actually on the certificate
  - Workers Comp MUST have a CLEAR, VISIBLE policy number in the POLICY NUMBER column of the same row (just like CGL and Umbrella have)
  - Workers Comp MUST have the SAME clear row structure as CGL/Umbrella/EPL - all fields properly aligned in the table
  - If Workers Comp row shows ONLY "PER STATUTE" checked with NO clear policy number visible in the table structure → DO NOT include workers_compensation
  - If the policy number appears to be from a different row, hidden text, or template text → DO NOT include workers_compensation
  - **VERIFY**: Look at the table structure - does Workers Comp have the same clear row format as CGL/Umbrella/EPL? 
    - Compare: Does CGL have "A" in INSR LTR, clear policy number, dates, limits? YES → Include CGL
    - Compare: Does Workers Comp have the SAME clear structure? If NO or UNCLEAR → OMIT IT
  - **WHEN IN DOUBT ABOUT WORKERS COMP → OMIT IT** - it's better to miss it than to hallucinate it

- Examples of what NOT to include:
  - If "AUTOMOBILE LIABILITY" row exists but all fields are blank → DO NOT include automobile_liability
  - If "UMBRELLA LIAB" row exists but all fields are blank → DO NOT include umbrella_liability
  - If "LIQUOR LIABILITY" is NOT in the table OR has no policy number → DO NOT include liquor_liability
  - If "WORKERS COMPENSATION" row exists but has no policy number → DO NOT include workers_compensation

**⚠️⚠️⚠️ CRITICAL - EXTRACT ALL COVERAGES WITH DATA ⚠️⚠️⚠️**
- **DO NOT SKIP ANY COVERAGE** that has a policy number (policy number is required, limits are secondary)
- **Check for ALL possible coverage types** in the table:
  ✓ Commercial General Liability (CGL)
  ✓ Automobile Liability
  ✓ Umbrella Liability / Excess Liability
  ✓ Workers Compensation
  ✓ Employment Practices Liability (EPL)
  ✓ Liquor Liability
  ✓ Garagekeepers Liability
  ✓ Any other coverage types present
- **If a coverage has a policy number OR limits, you MUST include it** - do not skip it
- **Example**: If you see "UMBRELLA LIAB" with policy "20 SBA AV6JXA" and limits "$2,000,000", you MUST include umbrella_liability in the output
- **Before returning, verify**: Did I check for ALL coverage types? Did I extract every coverage that has data?

Coverage keys to use (only when present WITH DATA):

A) commercial_general_liability:
{
  "policy_number": "...",
  "policy_eff": "MM/DD/YYYY or null",
  "policy_exp": "MM/DD/YYYY or null",
  "additional_insured": true/false/null,  # Extract from "ADDL INSD" column - "Y" = true, blank/unchecked = false
  "claims_made": true/false/null,
  "occur": true/false/null,
  **CRITICAL**: In the table, if "OCCUR" appears checked/marked, set "occur": true and "claims_made": false.
  If "CLAIMS-MADE" appears checked/marked, set "claims_made": true and "occur": false.
  Read the table row carefully - the checked checkbox indicates which one is true.
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
  "additional_insured": true/false/null,  # Extract from "ADDL INSD" column - "Y" = true, blank/unchecked = false
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
  "additional_insured": true/false/null,  # Extract from "ADDL INSD" column - "Y" = true, blank/unchecked = false
  "claims_made": true/false/null,
  "occur": true/false/null,
  **CRITICAL**: In the table, if "OCCUR" appears checked/marked, set "occur": true and "claims_made": false.
  If "CLAIMS-MADE" appears checked/marked, set "claims_made": true and "occur": false.
  Read the table row carefully - the checked checkbox indicates which one is true.
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
  "additional_insured": true/false/null,  # Extract from "ADDL INSD" column - "Y" = true, blank/unchecked = false
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
  "additional_insured": true/false/null,  # Extract from "ADDL INSD" column - "Y" = true, blank/unchecked = false
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
  "additional_insured": true/false/null,  # Extract from "ADDL INSD" column - "Y" = true, blank/unchecked = false
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
  "additional_insured": true/false/null,  # Extract from "ADDL INSD" column - "Y" = true, blank/unchecked = false
  "limits": {
    "limit": "string or null"
  },
  "comprehensive_deductible": "string or null",
  "collision_deductible": "string or null",
  "notes": "string or null"
}

4) Additional Insured parties (if mentioned in "DESCRIPTION OF OPERATIONS / LOCATIONS / VEHICLES" section):
- Extract parties explicitly listed as "Additional Insured" in the Description section
   - **IMPORTANT - CONDITIONAL FORMATTING BASED ON COUNT:**
  - **If 0 additional insured parties found** → Do NOT include any additional insured fields (no null, no empty string, DO NOT ADD THE KEYS AT ALL)
  - **If EXACTLY 1 additional insured party found** → Use flat structure:
    - "additional_insured_name": "name only"
    - "additional_insured_address": "full address" (if available, otherwise null)
  - **If 2 OR MORE additional insured parties found** → Use array structure:
    - "additional_insureds": [{"name": "...", "address": "..."}, {...}]
- **NOTE**: The "ADDL INSD" column in the coverages table (Y/blank) is DIFFERENT - that indicates whether each coverage has additional insured provisions, and goes in each coverage object as "additional_insured": true/false
- **NOTE**: If Description section says "Certificate Holder is listed as an Additional Insured" but doesn't provide separate name/address, extract from Certificate Holder section
- **CRITICAL**: Generic placeholders like "TO WHOM IT MAY CONCERN" are NOT valid additional insured parties → count as 0 → OMIT FIELDS ENTIRELY

5) validation_notes: string (brief notes about OCR conflicts or assumptions)

**Output Format Examples:**

**If 0 additional insured parties:**
{
  "policy_number": "...",
  "effective_date": "...",
  "expiration_date": "...",
  "insured_name": "...",
  "mailing_address": "...",
  "location_address": "...",
  "certificate_holder_name": "...",
  "certificate_holder_address": "...",
    "coverages": {
    "commercial_general_liability": {
      "policy_number": "...",
      "additional_insured": true,
      ...
    }
    },
    "validation_notes": "..."
}
NOTE: NO additional_insured_name or additional_insureds fields at all

**If EXACTLY 1 additional insured party:**
{
  "policy_number": "...",
  "effective_date": "...",
  "expiration_date": "...",
  "insured_name": "...",
  "mailing_address": "...",
  "location_address": "...",
  "certificate_holder_name": "...",
  "certificate_holder_address": "...",
  "coverages": {...},
  "additional_insured_name": "PINNACLE BANK, A TENNESSEE BANK",
  "additional_insured_address": "PO BOX 702726, DALLAS, TX 75370",
    "validation_notes": "..."
}

**If 2 OR MORE additional insured parties:**
{
  "policy_number": "...",
  "effective_date": "...",
  "expiration_date": "...",
  "insured_name": "...",
  "mailing_address": "...",
  "location_address": "...",
  "certificate_holder_name": "...",
  "certificate_holder_address": "...",
  "coverages": {...},
  "additional_insureds": [
    {"name": "...", "address": "..."},
    {"name": "...", "address": "..."}
    ],
    "validation_notes": "..."
}

==================================================
EXTRACTION SOURCE 1: PDFPLUMBER (Table-aware) - PRIMARY SOURCE
==================================================
**PRIORITY**: Use the TABLE sections (especially TABLE 2) for coverage data extraction.
The tables preserve structure and are more reliable than raw text.

""" + pdfplumber_text + """

==================================================
EXTRACTION SOURCE 2: PYMUPDF (Text layer) - CROSS-VALIDATION
==================================================
**USE AS**: Fallback/cross-reference when pdfplumber data is unclear or missing.

""" + pymupdf_text + """

Return ONLY the JSON object now."""
        else:
            # Single extraction mode (ACORD 25 - GL)
            prompt = """You are an expert in ACORD 25 (Certificate of Liability Insurance) extraction.

**NOTE**: If you see TABLE sections in the text, prioritize those for coverage extraction as they preserve structure.

==================================================
CRITICAL: DATE CLARIFICATION
==================================================
- The top "DATE (MM/DD/YYYY)" is the CERTIFICATE ISSUE DATE. Do NOT use it as policy dates.
- Policy Effective/Expiration dates are in the coverage table columns ("POLICY EFF" / "POLICY EXP") per coverage line.

==================================================
CRITICAL: ADDRESS EXTRACTION RULES
==================================================
ACORD 25 certificates can have ONE or TWO addresses:

**IF TWO ADDRESSES EXIST:**
1. **Mailing Address** = Address in INSURED section (header, directly below insured name)
   - This is where mail is sent to the insured
   - Location: Top section, under "INSURED" label, immediately after insured name
   - Format: Combine all address lines with commas (e.g., "37 E MAIN ST N, HAMPTON, GA 30228-5501")
   - **CRITICAL**: This is the INSURED's address, NOT the PRODUCER address
2. **Location Address** = Address in "DESCRIPTION OF OPERATIONS / LOCATIONS / VEHICLES" section
   - This is the physical location where business operations occur
   - Location: Middle section, in the description box labeled "DESCRIPTION OF OPERATIONS / LOCATIONS / VEHICLES"
   - Format: Combine all address lines with commas

**IF ONLY ONE ADDRESS EXISTS:**
- Use it as **location_address** (physical location)
- Set **mailing_address = null** (do not use the single address for mailing)

**CRITICAL - DO NOT CONFUSE ADDRESSES:**
- **PRODUCER address** = Top left section labeled "PRODUCER" (agency/broker address like "McKinney & Co., P.O Box 7, Tucker, GA") → **IGNORE THIS - DO NOT EXTRACT**
- **INSURED address** = Section labeled "INSURED" (insured party address) → **THIS IS WHAT YOU EXTRACT**
- **Location address** = "DESCRIPTION OF OPERATIONS" section (if present and different from INSURED address)

**CRITICAL REMINDER - SINGLE ADDRESS RULE**:
- If you see an address ONLY in the INSURED section and NO address in the "DESCRIPTION OF OPERATIONS" section:
  → location_address = address from INSURED section
  → mailing_address = null
- DO NOT put the single address in mailing_address - it must go to location_address
- The single address represents the physical location, not a mailing address

**EXAMPLES:**
- Certificate shows "37 E MAIN ST N, HAMPTON, GA 30228-5501" in INSURED section only (no address in Description section)
  → location_address = "37 E MAIN ST N, HAMPTON, GA 30228-5501"
  → mailing_address = null
  **CORRECT**: Single address goes to location_address

- Certificate shows address in INSURED section AND different address in Description section
  → mailing_address = address from INSURED section
  → location_address = address from Description section
  **CORRECT**: Two addresses - INSURED = mailing, Description = location

**FIRST PRIORITY: Check insured name first, then determine address locations based on above rules.**

==================================================
FIELDS TO EXTRACT (ACORD 25)
==================================================
Return ONLY a valid JSON object with:
- policy_number, effective_date, expiration_date, insured_name, mailing_address, location_address
  **NOTE**: Follow address extraction rules above - if only one address exists, use it as location_address and set mailing_address = null
  **CRITICAL**: DO NOT put a single address in mailing_address - it must go to location_address
  **CRITICAL**: mailing_address is from INSURED section, NOT PRODUCER section
- certificate holder fields (omit entirely if "TO WHOM IT MAY CONCERN")
- coverages (nested objects for any present coverages WITH DATA - do not invent coverages, but do not skip valid ones)
  **CRITICAL**: Only include coverages that have policy numbers OR limits in the table - if a coverage row is blank, omit it
  **CRITICAL**: Extract ALL coverages with data - check for CGL, Auto, Umbrella, WC, EPL, Liquor, Garagekeepers, etc.
  **CRITICAL**: Each coverage object MUST include "additional_insured": true/false/null from the "ADDL INSD" column
  **VERIFY**: Before returning, check that you extracted every coverage that has a policy number or limits
- additional insured parties (conditional formatting - see dual-extraction instructions for details)
  **IMPORTANT**: If 0 additional insured parties → omit fields entirely
  **IMPORTANT**: If exactly 1 → use "additional_insured_name" and "additional_insured_address"
  **IMPORTANT**: If 2+ → use "additional_insureds" array
- validation_notes

Follow the same JSON shapes described in the dual-extraction instructions.

Certificate Extraction Text:
---
""" + pdfplumber_text + """
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
        # Try to parse dual extraction if available
        pdfplumber_text, pymupdf_text = "", ""
        
        if use_dual_validation:
            pdfplumber_text, pymupdf_text = self.parse_dual_extraction(ocr_text)
            if pdfplumber_text and pymupdf_text:
                print("✅ Detected dual extraction sources (pdfplumber + PyMuPDF) - using cross-validation")
            else:
                pdfplumber_text = ocr_text
                print("ℹ️  Single extraction source detected")
        else:
            pdfplumber_text = ocr_text
        
        # Create prompt
        prompt = self.create_extraction_prompt(pdfplumber_text, pymupdf_text if pymupdf_text else None)
        
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
        print("⚠️  No input provided, using default: wilkes_gl")
        base_name = "naiya_gl"
    else:
        base_name = sys.argv[1]
    
    # Carrier directory (change this to switch between nationwideop, encovaop, etc.)
    carrier_dir = "nonstandardop"
    
    # Look for the combo file (best extraction)
    # NOTE: for GL we typically use base names like "aaniya_gl" so the file becomes "aaniya_gl_combo.txt"
    input_file = Path(f"{carrier_dir}/{base_name}_combo.txt")
    
    if not input_file.exists():
        # Try alternatives
        alternatives = [
            Path(f"{carrier_dir}/{base_name}1.txt"),  # pdfplumber
            Path(f"{carrier_dir}/{base_name}2.txt"),  # PyMuPDF
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
    
    # Check if it's a combo file (dual extraction)
    is_combo = "_combo.txt" in str(input_file)
    if is_combo:
        print(f"   Type: Dual extraction (pdfplumber + PyMuPDF)")
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

