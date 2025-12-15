"""
LLM-Based Coverage Validation - Building Coverage Only
Validates Building coverage values from certificate against policy document
Handles multiple buildings using location address context
"""

import os
import json
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


class BuildingCoverageValidator:
    """Validate Building coverage from certificate against policy"""
    
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
    
    def extract_building_coverages(self, cert_data: Dict) -> List[Dict]:
        """
        Extract all Building-related coverages from certificate
        
        Args:
            cert_data: Certificate JSON data
            
        Returns:
            List of dicts with building name and value
        """
        coverages = cert_data.get("coverages", {})
        buildings = []
        
        for coverage_name, coverage_value in coverages.items():
            # Match any coverage with "Building" in the name
            if "building" in coverage_name.lower():
                buildings.append({
                    "name": coverage_name,
                    "value": coverage_value
                })
        
        return buildings
    
    def create_validation_prompt(self, cert_data: Dict, buildings: List[Dict], policy_text: str) -> str:
        """
        Create validation prompt for Building coverages
        
        Args:
            cert_data: Certificate data with location context
            buildings: List of building coverages to validate
            policy_text: Full policy document text
            
        Returns:
            Formatted prompt string
        """
        
        # Extract context from certificate
        location_address = cert_data.get("location_address", "Not specified")
        insured_name = cert_data.get("insured_name", "Not specified")
        policy_number = cert_data.get("policy_number", "Not specified")
        
        prompt = f"""You are an expert Property Insurance QC Specialist validating Building coverage limits.

==================================================
CRITICAL INSTRUCTIONS
==================================================

**YOUR TASK:**
Validate BUILDING coverage limits from the certificate against the policy document.

**CONTEXT FROM CERTIFICATE:**
- Insured Name: {insured_name}
- Policy Number: {policy_number}
- Location Address: {location_address}

**BUILDING COVERAGES TO VALIDATE:**
{json.dumps(buildings, indent=2)}

==================================================
POLICY DOCUMENT (DUAL OCR SOURCES)
==================================================

This policy document contains TWO OCR extraction sources per page:
- **TESSERACT (Buffer=1)** - First OCR source
- **PYMUPDF (Buffer=0)** - Second OCR source

Use whichever source is clearer. ALWAYS cite which OCR source you used.

{policy_text}

==================================================
VALIDATION PROCESS
==================================================

For EACH Building coverage in the certificate:

**STEP 1: UNDERSTAND THE CERTIFICATE STRUCTURE**
- Is this a single building or multiple buildings?
- What is the location address? (Use this to match the right building in policy)
- What are the building names/numbers?

**STEP 2: SEARCH POLICY FOR BUILDING LIMITS**
- Look in the DECLARATIONS section (usually pages 1-10)
- Find the section labeled "Building" or "Coverages Provided"
- Match to the correct building using:
  * Location address
  * Premises number
  * Building number
  * Description

**STEP 3: CHECK FOR ENDORSEMENTS**
- Scan the ENTIRE policy for endorsements that modify building limits
- Look for forms like:
  * "BUILDING COVERAGE ENDORSEMENT"
  * "LIMIT OF INSURANCE - BUILDING"
  * Any amendment or correction forms
- Check effective dates of endorsements

**STEP 4: DETERMINE FINAL VALUE**
- What is the base limit in declarations?
- Are there any endorsements that increase/decrease the limit?
- What is the FINAL, EFFECTIVE limit for the building?

**STEP 5: COMPARE VALUES**
- Does the policy limit match the certificate limit?
- Handle dollar formatting differences: "$1,320,000" = "1,320,000" = "$1.32M"
- Consider:
  * Exact match = MATCH
  * Different value = MISMATCH
  * Not found in policy = NOT_FOUND

**IMPORTANT - MULTIPLE BUILDINGS:**
If the certificate has multiple buildings (e.g., "Building", "Building 2", "Building 01", "Building 02"):
- Match EACH certificate building to the corresponding policy building
- Use premises numbers, building numbers, or location descriptions
- Validate each one separately

**IMPORTANT - LOCATION MATCHING:**
The location address in the certificate tells you WHICH building to look for:
- If policy has multiple premises, find the one matching the certificate location
- Focus on that specific building's limit

==================================================
OUTPUT FORMAT
==================================================

Return ONLY a valid JSON object with this structure:

{{
  "building_validations": [
    {{
      "cert_building_name": "Name from certificate (e.g., 'Building', 'Building 01')",
      "cert_building_value": "Value from certificate",
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "policy_building_name": "How it appears in policy (e.g., 'Building - Premises 001')",
      "policy_building_value": "Final effective limit in policy",
      "policy_location": "Location/premises description from policy",
      "evidence_declarations": "Quote from declarations page (OCR_SOURCE, Page X)",
      "evidence_endorsements": "Quote from any modifying endorsements (OCR_SOURCE, Page X) or null",
      "notes": "Explanation: How did you match this? Any modifications applied? Why MATCH/MISMATCH/NOT_FOUND?"
    }}
  ],
  "summary": {{
    "total_buildings": 0,
    "matched": 0,
    "mismatched": 0,
    "not_found": 0
  }},
  "qc_notes": "Overall observations about the validation"
}}

**STATUS DEFINITIONS:**
- **MATCH**: Policy building limit matches certificate value (exact dollar amount)
- **MISMATCH**: Policy building limit differs from certificate value
- **NOT_FOUND**: Building coverage not found in policy document

**EVIDENCE FORMAT:**
Always include page number and OCR source, e.g.:
- "Building: $1,320,000 Special Coverage (TESSERACT, Page 4)"
- "Limit of Insurance - Building 2: $80,000 (PYMUPDF, Page 27)"

Return ONLY the JSON object. No other text.
"""
        
        return prompt
    
    def validate_buildings(self, cert_json_path: str, policy_combo_path: str, output_path: str):
        """
        Main validation workflow
        
        Args:
            cert_json_path: Path to certificate JSON file
            policy_combo_path: Path to policy combo text file
            output_path: Path for output JSON file
        """
        
        print(f"\n{'='*70}")
        print("BUILDING COVERAGE VALIDATION")
        print(f"{'='*70}\n")
        
        # Load certificate
        print(f"[1/5] Loading certificate: {cert_json_path}")
        with open(cert_json_path, 'r', encoding='utf-8') as f:
            cert_data = json.load(f)
        
        # Extract building coverages
        buildings = self.extract_building_coverages(cert_data)
        
        if not buildings:
            print("      ❌ No Building coverages found in certificate!")
            print("      Certificate may be GL policy or missing coverage data.")
            return
        
        print(f"      Found {len(buildings)} Building coverage(s):")
        for b in buildings:
            print(f"        - {b['name']}: {b['value']}")
        
        # Load policy
        print(f"\n[2/5] Loading policy: {policy_combo_path}")
        with open(policy_combo_path, 'r', encoding='utf-8') as f:
            policy_text = f.read()
        
        policy_size_kb = len(policy_text) / 1024
        print(f"      Policy size: {policy_size_kb:.1f} KB")
        
        # Create prompt
        print(f"\n[3/5] Creating validation prompt...")
        prompt = self.create_validation_prompt(cert_data, buildings, policy_text)
        prompt_size_kb = len(prompt) / 1024
        print(f"      Prompt size: {prompt_size_kb:.1f} KB")
        
        # Call LLM
        print(f"\n[4/5] Calling LLM for validation (model: {self.model})...")
        print(f"      Analyzing policy for Building coverage limits...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert Property Insurance QC Specialist. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            results = json.loads(result_text)
            
            # Add metadata
            results["metadata"] = {
                "model": self.model,
                "certificate_file": cert_json_path,
                "policy_file": policy_combo_path,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            print(f"      ✓ LLM validation complete")
            print(f"      Tokens used: {response.usage.total_tokens:,} (prompt: {response.usage.prompt_tokens:,}, completion: {response.usage.completion_tokens:,})")
            
        except Exception as e:
            print(f"      ❌ Error calling LLM: {str(e)}")
            raise
        
        # Save results
        self.save_validation_results(results, output_path)
        
        # Display results
        self.display_results(results)
        
        print(f"\n✓ Validation completed successfully!")
    
    def save_validation_results(self, results: Dict, output_path: str):
        """Save validation results to JSON file"""
        print(f"\n[5/5] Saving results to: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"      ✓ Results saved")
    
    def display_results(self, results: Dict):
        """Display validation results on console"""
        print(f"\n{'='*70}")
        print("BUILDING VALIDATION RESULTS")
        print(f"{'='*70}\n")
        
        validations = results.get('building_validations', [])
        
        for validation in validations:
            status = validation.get('status', 'UNKNOWN')
            cert_name = validation.get('cert_building_name', 'N/A')
            cert_value = validation.get('cert_building_value', 'N/A')
            policy_name = validation.get('policy_building_name', 'N/A')
            policy_value = validation.get('policy_building_value', 'N/A')
            policy_location = validation.get('policy_location', 'N/A')
            evidence_decl = validation.get('evidence_declarations', 'N/A')
            evidence_end = validation.get('evidence_endorsements', None)
            notes = validation.get('notes', 'N/A')
            
            # Status icon
            if status == 'MATCH':
                icon = '✓'
            elif status == 'MISMATCH':
                icon = '✗'
            else:
                icon = '?'
            
            print(f"{icon} {cert_name}")
            print(f"  Status: {status}")
            print(f"  Certificate Value: {cert_value}")
            print(f"  Policy Value: {policy_value}")
            print(f"  Policy Building: {policy_name}")
            print(f"  Policy Location: {policy_location}")
            
            # Truncate evidence if too long (handle None)
            if evidence_decl and len(evidence_decl) > 100:
                evidence_decl = evidence_decl[:97] + "..."
            print(f"  Evidence (Declarations): {evidence_decl if evidence_decl else 'N/A'}")
            
            if evidence_end:
                if len(evidence_end) > 100:
                    evidence_end = evidence_end[:97] + "..."
                print(f"  Evidence (Endorsements): {evidence_end}")
            
            # Truncate notes if too long (handle None)
            if notes and len(notes) > 150:
                notes = notes[:147] + "..."
            print(f"  Notes: {notes if notes else 'N/A'}")
            print()
        
        # Print summary
        summary = results.get('summary', {})
        print(f"{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Total Buildings:  {summary.get('total_buildings', 0)}")
        print(f"  ✓ Matched:      {summary.get('matched', 0)}")
        print(f"  ✗ Mismatched:   {summary.get('mismatched', 0)}")
        print(f"  ? Not Found:    {summary.get('not_found', 0)}")
        
        if 'qc_notes' in results:
            qc_notes = results['qc_notes']
            if len(qc_notes) > 200:
                qc_notes = qc_notes[:197] + "..."
            print(f"\nQC Notes: {qc_notes}")
        
        print(f"{'='*70}\n")


def main():
    """Main execution function"""
    # ========== EDIT THESE VALUES ==========
    cert_prefix = "drive"              # Change to: james, indian, etc.
    carrier_dir = "travelerop"      # Change to: hartfordop, encovaop, etc.
    # =======================================
    
    # Construct paths
    cert_json_path = os.path.join(carrier_dir, f"{cert_prefix}_pl_extracted_real.json")
    policy_combo_path = os.path.join(carrier_dir, f"{cert_prefix}_pol_combo.txt")
    output_path = os.path.join(carrier_dir, f"{cert_prefix}_building_validation.json")
    
    # Check if files exist
    if not os.path.exists(cert_json_path):
        print(f"Error: Certificate JSON not found: {cert_json_path}")
        exit(1)
    
    if not os.path.exists(policy_combo_path):
        print(f"Error: Policy combo text not found: {policy_combo_path}")
        exit(1)
    
    # Create validator and run
    try:
        validator = BuildingCoverageValidator()
        validator.validate_buildings(cert_json_path, policy_combo_path, output_path)
    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

