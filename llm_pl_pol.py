"""
LLM-Based Policy Validation
Validates certificate fields against policy document using GPT-4o-mini
Handles dual OCR sources, endorsements, and full policy context
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


class PolicyValidator:
    """Validate certificate information against policy document using LLM"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the validator
        
        Args:
            model: OpenAI model to use (default: gpt-4o-mini)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def create_validation_prompt(self, cert_data: Dict, policy_text: str) -> str:
        """
        Create comprehensive validation prompt with full context
        
        Args:
            cert_data: Certificate data (ground truth)
            policy_text: Full policy document with dual OCR sources
            
        Returns:
            Formatted prompt string
        """
        
        # Extract basic fields for validation
        fields_to_validate = {
            "policy_number": cert_data.get("policy_number"),
            "effective_date": cert_data.get("effective_date"),
            "expiration_date": cert_data.get("expiration_date"),
            "insured_name": cert_data.get("insured_name"),
            "mailing_address": cert_data.get("mailing_address"),
            "location_address": cert_data.get("location_address")
        }
        
        prompt = f"""You are an expert Policy Quality Control Specialist performing validation between a certificate and the actual policy document.

==================================================
CRITICAL INSTRUCTIONS - READ CAREFULLY
==================================================

**YOUR ROLE:**
You are performing the same validation process as a human QC specialist. This requires:
1. **FULL CONTEXT AWARENESS** - Read and understand the ENTIRE policy document structure
2. **CROSS-REFERENCING** - Check declarations page AND endorsements for modifications
3. **DUAL OCR VALIDATION** - Use the clearer of two OCR sources per page
4. **DETAIL ORIENTATION** - Consider surrounding context, dates, locations, and modifications

**DUAL OCR STRUCTURE:**
This policy document contains TWO independent OCR extraction sources for EACH page:
- **TESSERACT (Buffer=1)** - First OCR extraction
- **PYMUPDF (Buffer=0)** - Second OCR extraction

For each page:
- If one source says "[Page not found]" or is garbled → use the other source
- If both are readable → use whichever is clearer/more complete
- Cross-reference between sources to verify accuracy
- In your evidence field, ALWAYS specify which OCR source you used

**POLICY DOCUMENT STRUCTURE:**
This is a COMPLETE policy document containing:
1. **Common Declarations** (usually pages 1-5)
   - Policy number, effective/expiration dates
   - Named insured, mailing address
   - Premium information
   
2. **Property/Coverage Declarations** (usually pages 5-15)
   - Location addresses (premises)
   - Building limits, deductibles
   - Coverage details
   
3. **Liability Declarations** (if applicable)

4. **Forms and Endorsements** (remaining pages)
   - **CRITICAL:** Endorsements may MODIFY values from declarations
   - Check for amendments, corrections, or changes
   - Verify effective dates of endorsements

**CONTEXT REQUIREMENTS:**
When validating each field, you MUST:
1. **Read the full declarations page** to understand the primary values
2. **Note surrounding context**:
   - For policy number: Check it appears consistently
   - For dates: Verify they're policy dates, not certificate issue dates
   - For insured name: Check for "dba" or multiple names
   - For mailing address: This is the insured's address (not property location)
   - For location address: Check "Premises Address" or "Location" sections
3. **Scan all endorsements** to check for modifications
4. **Cross-reference** multiple occurrences to ensure consistency

**VALIDATION RULES:**

**1. POLICY NUMBER:**
- Must match EXACTLY (case-insensitive, but format must be the same)
- Usually appears on Common Declarations page
- Format examples: "ACP BP013230657249", "H0054PR002844-00"
- Check multiple pages to ensure consistency

**2. EFFECTIVE DATE:**
- Format: MM/DD/YYYY
- This is the policy START date (not certificate issue date)
- Must appear on Common Declarations page labeled "Effective" or "Policy Period"
- Watch for: "Policy Period: From MM/DD/YYYY To MM/DD/YYYY"

**3. EXPIRATION DATE:**
- Format: MM/DD/YYYY
- This is the policy END date
- Usually appears immediately after effective date
- Typical policy period: 1 year

**4. INSURED NAME (NAMED INSURED):**
- Check "Named Insured" field on Common Declarations
- May include "dba" (doing business as)
- Examples: "INDIAN TRAIL COMMONS CONDO ASSOCIATION", "StaycaVaca LLC dba BOOK A MEMORY"
- Handle variations: Case differences acceptable, but core name must match
- "JOHN SMITH" = "John Smith" → MATCH (case difference only)
- "JOHN SMITH" vs "JOHN SMITH LLC" → MISMATCH (entity difference)

**5. MAILING ADDRESS:**
- This is the INSURED's address (where mail is sent)
- Found under "Mailing Address" on Common Declarations
- NOT the property/location address
- Format: Street, City, State ZIP
- Handle abbreviations: "Rd" = "Road", "ST" = "Street" → MATCH with note
- May not always be present in policy (some policies only have location address)

**6. LOCATION ADDRESS:**
- This is the PROPERTY/PREMISES address being insured
- Found in "Description of Premises" or "Premises Address" section
- May be labeled "Premises: 001 / Building: 001" with address below
- If multiple locations exist, match the PRIMARY or MAIN location
- If certificate specifies a specific address, find that exact location in the schedule

**MATCHING TOLERANCE:**
- **Exact Match Required:** Policy number
- **Format Flexible:** Dates ("09/26/2025" = "9/26/2025")
- **Abbreviations OK:** Addresses ("Rd" = "Road", "NW" = "Northwest")
- **Case Insensitive:** Names and addresses
- **Whitespace Flexible:** Extra spaces ignored

==================================================
CERTIFICATE DATA (GROUND TRUTH)
==================================================

The following information was extracted from the certificate. This is what you need to VALIDATE against the policy:

{json.dumps(fields_to_validate, indent=2)}

==================================================
POLICY DOCUMENT (FULL TEXT WITH DUAL OCR)
==================================================

{policy_text}

==================================================
VALIDATION TASK
==================================================

For EACH field in the certificate data above, perform the following validation:

**STEP 1: LOCATE IN POLICY**
- Search the ENTIRE policy document (all pages, both OCR sources)
- Find the corresponding field in declarations
- Note the page number and OCR source

**STEP 2: CHECK FOR MODIFICATIONS**
- Scan endorsements section
- Look for amendments to this field
- Check effective dates of any changes

**STEP 3: DETERMINE FINAL VALUE**
- What is the value in declarations?
- Are there any endorsements that modify it?
- What is the FINAL, EFFECTIVE value?

**STEP 4: COMPARE WITH CERTIFICATE**
- Does the policy value MATCH the certificate value?
- Apply tolerance rules (dates, abbreviations, case)
- Determine status: MATCH, MISMATCH, or NOT_FOUND

**STEP 5: PROVIDE EVIDENCE**
- Quote the exact text from policy (declarations)
- Quote any modifying endorsements (if applicable)
- Specify page number and OCR source used
- Explain your reasoning

==================================================
OUTPUT FORMAT
==================================================

Return ONLY a valid JSON object with this EXACT structure:

{{
  "validation_results": {{
    "policy_number": {{
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "certificate_value": "value from certificate or null",
      "policy_value": "value found in policy or null",
      "evidence": "Exact quote from policy (OCR_SOURCE, Page X)",
      "notes": "Explanation of reasoning, any tolerance applied, or issues found"
    }},
    "effective_date": {{
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "certificate_value": "MM/DD/YYYY or null",
      "policy_value": "MM/DD/YYYY or null",
      "evidence": "Exact quote from policy (OCR_SOURCE, Page X)",
      "notes": "Explanation..."
    }},
    "expiration_date": {{
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "certificate_value": "MM/DD/YYYY or null",
      "policy_value": "MM/DD/YYYY or null",
      "evidence": "Exact quote from policy (OCR_SOURCE, Page X)",
      "notes": "Explanation..."
    }},
    "insured_name": {{
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "certificate_value": "string or null",
      "policy_value": "string or null",
      "evidence": "Exact quote from policy (OCR_SOURCE, Page X)",
      "notes": "Explanation..."
    }},
    "mailing_address": {{
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "certificate_value": "string or null",
      "policy_value": "string or null",
      "evidence": "Exact quote from policy (OCR_SOURCE, Page X)",
      "notes": "Explanation..."
    }},
    "location_address": {{
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "certificate_value": "string or null",
      "policy_value": "string or null",
      "evidence": "Exact quote from policy (OCR_SOURCE, Page X)",
      "notes": "Explanation..."
    }}
  }},
  "summary": {{
    "total_fields": 6,
    "matched": 0,
    "mismatched": 0,
    "not_found": 0
  }},
  "qc_notes": "Overall observations about the validation process, document quality, or concerns"
}}

**STATUS DEFINITIONS:**
- **MATCH**: Policy value matches certificate value (considering tolerance rules)
- **MISMATCH**: Policy value differs from certificate value
- **NOT_FOUND**: Field not found in policy document

**EVIDENCE FORMAT EXAMPLES:**
- "Policy Number: ACP BP013230657249 (TESSERACT, Page 31)"
- "Policy Period: From 01-23-2025 To 01-23-2026 (PYMUPDF, Page 31)"
- "Named Insured: INDIAN TRAIL COMMONS CONDO ASSOCIATION (TESSERACT, Page 31)"
- "Mailing Address: 1040 INDIAN TRAIL LILBURN RD NW BUILDING A-B-C LILBURN, GA 30047-6831 (TESSERACT, Page 31)"

**NOTES FORMAT EXAMPLES:**
- "Exact match found on declarations page"
- "Minor formatting difference: certificate shows '09/26/2025', policy shows '9/26/2025' - considered MATCH"
- "Address abbreviation: certificate shows 'Rd', policy shows 'Road' - considered MATCH"
- "Case difference only: certificate shows uppercase, policy shows mixed case - considered MATCH"
- "Found in Premises 001/Building 001 section as primary location"
- "Multiple locations in policy, matched certificate address to Premises 004"
- "No mailing address explicitly labeled; used premises address as proxy"

Now analyze the policy document and return the validation JSON:"""
        
        return prompt
    
    def validate_policy(self, cert_json_path: str, policy_combo_path: str) -> Dict:
        """
        Validate certificate against policy document
        
        Args:
            cert_json_path: Path to certificate JSON file
            policy_combo_path: Path to policy combo text file
            
        Returns:
            Validation results dictionary
        """
        # Load certificate data
        print(f"\n{'='*70}")
        print("POLICY VALIDATION - Certificate vs Policy Document")
        print(f"{'='*70}\n")
        
        print(f"[1/5] Loading certificate: {cert_json_path}")
        with open(cert_json_path, 'r', encoding='utf-8') as f:
            cert_data = json.load(f)
        
        # Basic validation fields count
        fields_count = sum([
            1 for field in ['policy_number', 'effective_date', 'expiration_date', 
                           'insured_name', 'mailing_address', 'location_address']
            if cert_data.get(field)
        ])
        print(f"      Certificate fields to validate: {fields_count}")
        
        # Load policy document
        print(f"\n[2/5] Loading policy: {policy_combo_path}")
        with open(policy_combo_path, 'r', encoding='utf-8') as f:
            policy_text = f.read()
        
        policy_size_kb = len(policy_text) / 1024
        policy_lines = len(policy_text.split('\n'))
        print(f"      Policy size: {policy_size_kb:.1f} KB ({policy_lines:,} lines)")
        
        # Create validation prompt
        print(f"\n[3/5] Creating validation prompt...")
        prompt = self.create_validation_prompt(cert_data, policy_text)
        prompt_size_kb = len(prompt) / 1024
        print(f"      Prompt size: {prompt_size_kb:.1f} KB")
        
        # Call LLM
        print(f"\n[4/5] Calling LLM for validation (model: {self.model})...")
        print(f"      Please wait, analyzing full policy document...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Policy Quality Control Specialist. You validate certificate information against policy documents with high accuracy, considering all endorsements and modifications."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent, factual responses
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result_text = response.choices[0].message.content
            validation_result = json.loads(result_text)
            
            # Add metadata
            validation_result['metadata'] = {
                'model': self.model,
                'certificate_file': cert_json_path,
                'policy_file': policy_combo_path,
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            print(f"      ✓ Validation complete")
            print(f"      Tokens used: {response.usage.total_tokens:,} (prompt: {response.usage.prompt_tokens:,}, completion: {response.usage.completion_tokens:,})")
            
            return validation_result
            
        except Exception as e:
            print(f"      ✗ Error during LLM call: {e}")
            raise
    
    def save_validation_results(self, results: Dict, output_path: str):
        """Save validation results to JSON file and display on CLI"""
        print(f"\n[5/5] Saving results to: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print detailed results
        print(f"\n{'='*70}")
        print("DETAILED VALIDATION RESULTS")
        print(f"{'='*70}\n")
        
        validation_results = results.get('validation_results', {})
        
        for field_name, field_data in validation_results.items():
            status = field_data.get('status', 'UNKNOWN')
            cert_value = field_data.get('certificate_value', 'N/A')
            policy_value = field_data.get('policy_value', 'N/A')
            evidence = field_data.get('evidence', 'N/A')
            notes = field_data.get('notes', 'N/A')
            
            # Status icon
            if status == 'MATCH':
                icon = '✓'
                color = ''
            elif status == 'MISMATCH':
                icon = '✗'
                color = ''
            else:
                icon = '?'
                color = ''
            
            # Display field name
            print(f"{icon} {field_name.upper().replace('_', ' ')}")
            print(f"  Status: {status}")
            print(f"  Certificate: {cert_value}")
            print(f"  Policy:      {policy_value}")
            
            # Truncate long evidence for display
            if len(evidence) > 100:
                evidence_display = evidence[:97] + "..."
            else:
                evidence_display = evidence
            print(f"  Evidence: {evidence_display}")
            
            # Truncate long notes for display
            if len(notes) > 100:
                notes_display = notes[:97] + "..."
            else:
                notes_display = notes
            print(f"  Notes: {notes_display}")
            print()
        
        # Print summary
        summary = results.get('summary', {})
        print(f"{'='*70}")
        print("VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total fields validated:  {summary.get('total_fields', 0)}")
        print(f"  ✓ Matched:             {summary.get('matched', 0)}")
        print(f"  ✗ Mismatched:          {summary.get('mismatched', 0)}")
        print(f"  ? Not Found:           {summary.get('not_found', 0)}")
        
        if 'qc_notes' in results:
            print(f"\nQC Notes: {results['qc_notes']}")
        
        print(f"{'='*70}\n")


def main():
    """Main execution function"""
    # ========== EDIT THESE VALUES ==========
    cert_prefix = "stay"              # Certificate name prefix (e.g., "stay", "indian", "evergreen")
    carrier_dir = "nationwideop"      # Carrier directory
    # =======================================
    
    # Construct paths
    cert_json_path = os.path.join(carrier_dir, f"{cert_prefix}_pl_extracted_real.json")
    policy_combo_path = os.path.join(carrier_dir, f"{cert_prefix}_pol_combo.txt")
    output_path = os.path.join(carrier_dir, f"{cert_prefix}_validation.json")
    
    # Check if files exist
    if not os.path.exists(cert_json_path):
        print(f"Error: Certificate JSON not found: {cert_json_path}")
        sys.exit(1)
    
    if not os.path.exists(policy_combo_path):
        print(f"Error: Policy combo text not found: {policy_combo_path}")
        sys.exit(1)
    
    # Run validation
    validator = PolicyValidator()
    
    try:
        results = validator.validate_policy(cert_json_path, policy_combo_path)
        validator.save_validation_results(results, output_path)
        
        print("✓ Validation completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

