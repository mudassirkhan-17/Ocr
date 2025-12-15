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
    """Validate Building + BPP + Money & Securities + Equipment Breakdown coverages from certificate against policy (single LLM call)"""
    
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

    def extract_bpp_coverages(self, cert_data: Dict) -> List[Dict]:
        """
        Extract Business Personal Property (BPP) coverages from certificate.
        Targets the main BPP limit (not off-premises/in-transit extensions).
        """
        coverages = cert_data.get("coverages", {}) or {}
        bpps = []

        for coverage_name, coverage_value in coverages.items():
            name = (coverage_name or "").strip()
            n = name.lower()

            is_bpp = (
                "business personal property" in n
                or n == "bpp"
                or n.startswith("bpp ")
                or n.endswith(" bpp")
            )

            is_extension = any(
                kw in n
                for kw in [
                    "off premises",
                    "off-premises",
                    "away from premises",
                    "in transit",
                    "transit",
                    "portable storage",
                    "temporarily",
                    "newly acquired",
                    "newly constructed",
                    "coverage extension",
                    "extension",
                ]
            )

            if is_bpp and not is_extension:
                bpps.append({"name": name, "value": coverage_value})

        return bpps

    def extract_money_securities_coverages(self, cert_data: Dict) -> List[Dict]:
        """
        Extract Money & Securities coverages from certificate.

        Notes:
        - Usually a dollar limit (e.g., "10,000"), sometimes "Included"
        - Sometimes the policy has Inside/Outside split even if certificate shows one number
        """
        coverages = cert_data.get("coverages", {}) or {}
        ms_items: List[Dict] = []

        for coverage_name, coverage_value in coverages.items():
            name = (coverage_name or "").strip()
            n = name.lower()

            is_ms = (
                ("money" in n and "secur" in n)  # securities / security
                or "money & securities" in n
                or "money and securities" in n
            )

            # Avoid confusing with unrelated lines like "counterfeit money" if it exists
            is_excluded = any(
                kw in n
                for kw in [
                    "counterfeit",
                    "money orders",
                    "forgery",
                    "alteration",
                    "funds transfer",
                    "computer fraud",
                ]
            )

            if is_ms and not is_excluded:
                ms_items.append({"name": name, "value": coverage_value})

        return ms_items

    def extract_equipment_breakdown_coverages(self, cert_data: Dict) -> List[Dict]:
        """
        Extract Equipment Breakdown coverages from certificate.

        Notes:
        - Often "Included" / "Yes"
        - Sometimes a dollar limit
        - Avoid picking up deductibles or other non-limit fields
        """
        coverages = cert_data.get("coverages", {}) or {}
        eb_items: List[Dict] = []

        for coverage_name, coverage_value in coverages.items():
            name = (coverage_name or "").strip()
            n = name.lower()

            is_eb = (
                "equipment breakdown" in n
                or ("equip" in n and "breakdown" in n)
                or "boiler and machinery" in n
                or "boiler & machinery" in n
            )

            # Exclude non-limit fields that sometimes appear near EB
            is_excluded = any(
                kw in n
                for kw in [
                    "deductible",
                    "ded.",
                    "coinsurance",
                    "waiting period",
                    "waiting",
                    "service interruption",
                ]
            )

            if is_eb and not is_excluded:
                eb_items.append({"name": name, "value": coverage_value})

        return eb_items
    
    def create_validation_prompt(self, cert_data: Dict, buildings: List[Dict], bpp_items: List[Dict], ms_items: List[Dict], eb_items: List[Dict], policy_text: str) -> str:
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
        
        all_coverages = cert_data.get("coverages", {}) or {}

        prompt = f"""You are an expert Property Insurance QC Specialist validating coverage limits.

==================================================
CRITICAL INSTRUCTIONS
==================================================

**YOUR TASK:**
Validate BUILDING, Business Personal Property (BPP), Money & Securities, and Equipment Breakdown coverages from the certificate against the policy document.

**CONTEXT FROM CERTIFICATE:**
- Insured Name: {insured_name}
- Policy Number: {policy_number}
- Location Address: {location_address}

**ALL CERTIFICATE COVERAGES (for context):**
{json.dumps(all_coverages, indent=2)}

**BUILDING COVERAGES TO VALIDATE:**
{json.dumps(buildings, indent=2)}

**BPP COVERAGES TO VALIDATE:**
{json.dumps(bpp_items, indent=2)}

**MONEY & SECURITIES COVERAGES TO VALIDATE:**
{json.dumps(ms_items, indent=2)}

**EQUIPMENT BREAKDOWN COVERAGES TO VALIDATE:**
{json.dumps(eb_items, indent=2)}

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
MONEY & SECURITIES VALIDATION RULES (STRICT)
==================================================

For EACH Money & Securities item:
- Prefer declarations/optional coverages sections where "Money and Securities" is listed with a limit.
- If the policy shows an Inside/Outside split:
  - If certificate shows a single number (e.g., "10,000"), treat as MATCH if the key split limit(s) equal that value (commonly $10,000 inside and $10,000 outside).
  - Record the split in the output.
- Do NOT confuse with: Forgery/Alteration, Money Orders/Counterfeit Money, Computer Fraud/Funds Transfer, or other crime/cyber sublimits.
- Formatting differences are not mismatches: "10,000" == "$10,000" == "$ 10,000"
- If certificate says "Included", treat as MATCH only if policy indicates it is covered/included (or shows a limit as part of the form).

==================================================
EQUIPMENT BREAKDOWN VALIDATION RULES (STRICT)
==================================================

For EACH Equipment Breakdown item:
- The certificate value may be "Included" / "Yes" / "Provided" instead of a dollar amount.
- MATCH rules:
  - If certificate is "Included"/"Yes": MATCH if policy indicates Equipment Breakdown is included/covered OR provides a limit as part of the Equipment Breakdown coverage.
  - If certificate is a dollar limit: MATCH only if the policy's Equipment Breakdown limit matches (ignore formatting like $ and commas).
- Do NOT confuse Equipment Breakdown coverage with:
  - Equipment Breakdown deductible
  - Service Interruption sublimit
  - Other mechanical breakdown wording that is not a coverage grant/limit
- Evidence must include page number and OCR source.

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
  "bpp_validations": [
    {{
      "cert_bpp_name": "Name from certificate (e.g., 'Business Personal Property')",
      "cert_bpp_value": "Value from certificate",
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "policy_bpp_name": "How it appears in policy",
      "policy_bpp_value": "Final effective limit in policy",
      "policy_location": "Location/premises/building description from policy",
      "policy_premises_building": "Premises/Building identifier if available (e.g., 'Premises 001 / Building 002')",
      "evidence_declarations": "Quote from declarations page (OCR_SOURCE, Page X)",
      "evidence_endorsements": "Quote from any modifying endorsement (OCR_SOURCE, Page X) or null",
      "notes": "How you matched location/premises and why MATCH/MISMATCH/NOT_FOUND (avoid matching sublimits/extensions)."
    }}
  ],
  "money_securities_validations": [
    {{
      "cert_ms_name": "Name from certificate (e.g., 'Money & Securities')",
      "cert_ms_value": "Value from certificate (e.g., '10,000' or 'Included')",
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "policy_ms_name": "How it appears in policy",
      "policy_ms_value": "Primary limit in policy (if a single limit)",
      "policy_ms_split": "If split exists, capture like 'Inside $X; Outside $Y' otherwise null",
      "policy_location": "Location/premises/building description from policy (or null if policy-wide)",
      "evidence_declarations": "Quote from declarations/optional coverages (OCR_SOURCE, Page X)",
      "evidence_endorsements": "Quote from modifying endorsement (OCR_SOURCE, Page X) or null",
      "notes": "Explain how you matched and why MATCH/MISMATCH/NOT_FOUND."
    }}
  ],
  "equipment_breakdown_validations": [
    {{
      "cert_eb_name": "Name from certificate (e.g., 'Equipment Breakdown')",
      "cert_eb_value": "Value from certificate (e.g., 'Included' or '100,000')",
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "policy_eb_name": "How it appears in policy",
      "policy_eb_value": "Policy value (Included/Yes or a dollar limit) or null",
      "policy_location": "Location/premises/building description from policy (or null if policy-wide)",
      "evidence_declarations": "Quote from declarations/coverage schedule (OCR_SOURCE, Page X)",
      "evidence_endorsements": "Quote from any modifying endorsement (OCR_SOURCE, Page X) or null",
      "notes": "Explain how you matched and why MATCH/MISMATCH/NOT_FOUND."
    }}
  ],
  "summary": {{
    "total_buildings": 0,
    "matched": 0,
    "mismatched": 0,
    "not_found": 0,
    "total_bpp_items": 0,
    "bpp_matched": 0,
    "bpp_mismatched": 0,
    "bpp_not_found": 0,
    "total_ms_items": 0,
    "ms_matched": 0,
    "ms_mismatched": 0,
    "ms_not_found": 0,
    "total_eb_items": 0,
    "eb_matched": 0,
    "eb_mismatched": 0,
    "eb_not_found": 0
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
        
        # Extract building + BPP + Money & Securities + Equipment Breakdown coverages (single LLM call)
        buildings = self.extract_building_coverages(cert_data)
        bpp_items = self.extract_bpp_coverages(cert_data)
        ms_items = self.extract_money_securities_coverages(cert_data)
        eb_items = self.extract_equipment_breakdown_coverages(cert_data)
        
        if not buildings and not bpp_items and not ms_items and not eb_items:
            print("      ❌ No Building, BPP, Money & Securities, or Equipment Breakdown coverages found in certificate!")
            print("      Certificate may be GL policy or missing coverage data.")
            return
        
        if buildings:
            print(f"      Found {len(buildings)} Building coverage(s):")
            for b in buildings:
                print(f"        - {b['name']}: {b['value']}")
        if bpp_items:
            print(f"      Found {len(bpp_items)} BPP coverage(s):")
            for b in bpp_items:
                print(f"        - {b['name']}: {b['value']}")
        if ms_items:
            print(f"      Found {len(ms_items)} Money & Securities coverage(s):")
            for m in ms_items:
                print(f"        - {m['name']}: {m['value']}")
        if eb_items:
            print(f"      Found {len(eb_items)} Equipment Breakdown coverage(s):")
            for e in eb_items:
                print(f"        - {e['name']}: {e['value']}")
        
        # Load policy
        print(f"\n[2/5] Loading policy: {policy_combo_path}")
        with open(policy_combo_path, 'r', encoding='utf-8') as f:
            policy_text = f.read()
        
        policy_size_kb = len(policy_text) / 1024
        print(f"      Policy size: {policy_size_kb:.1f} KB")
        
        # Create prompt
        print(f"\n[3/5] Creating validation prompt...")
        prompt = self.create_validation_prompt(cert_data, buildings, bpp_items, ms_items, eb_items, policy_text)
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
        print("COVERAGE VALIDATION RESULTS (BUILDING + BPP + MONEY & SECURITIES + EQUIPMENT BREAKDOWN)")
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

        # Display BPP validations (if present)
        bpp_validations = results.get('bpp_validations', [])
        if bpp_validations:
            print(f"{'='*70}")
            print("BPP VALIDATION RESULTS")
            print(f"{'='*70}\n")

            for v in bpp_validations:
                status = v.get('status', 'UNKNOWN')
                cert_name = v.get('cert_bpp_name', 'N/A')
                cert_value = v.get('cert_bpp_value', 'N/A')
                policy_name = v.get('policy_bpp_name', 'N/A')
                policy_value = v.get('policy_bpp_value', 'N/A')
                policy_location = v.get('policy_location', 'N/A')
                policy_pb = v.get('policy_premises_building', 'N/A')
                evidence_decl = v.get('evidence_declarations', 'N/A')
                evidence_end = v.get('evidence_endorsements', None)
                notes = v.get('notes', 'N/A')

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
                print(f"  Policy Label: {policy_name}")
                print(f"  Policy Location: {policy_location}")
                print(f"  Policy Prem/Building: {policy_pb}")

                if evidence_decl and len(evidence_decl) > 100:
                    evidence_decl = evidence_decl[:97] + "..."
                print(f"  Evidence (Declarations): {evidence_decl if evidence_decl else 'N/A'}")

                if evidence_end:
                    if len(evidence_end) > 100:
                        evidence_end = evidence_end[:97] + "..."
                    print(f"  Evidence (Endorsements): {evidence_end}")

                if notes and len(notes) > 150:
                    notes = notes[:147] + "..."
                print(f"  Notes: {notes if notes else 'N/A'}")
                print()

        # Display Money & Securities validations (if present)
        ms_validations = results.get('money_securities_validations', [])
        if ms_validations:
            print(f"{'='*70}")
            print("MONEY & SECURITIES VALIDATION RESULTS")
            print(f"{'='*70}\n")

            for v in ms_validations:
                status = v.get('status', 'UNKNOWN')
                cert_name = v.get('cert_ms_name', 'N/A')
                cert_value = v.get('cert_ms_value', 'N/A')
                policy_name = v.get('policy_ms_name', 'N/A')
                policy_value = v.get('policy_ms_value', 'N/A')
                policy_split = v.get('policy_ms_split', None)
                policy_location = v.get('policy_location', 'N/A')
                evidence_decl = v.get('evidence_declarations', 'N/A')
                evidence_end = v.get('evidence_endorsements', None)
                notes = v.get('notes', 'N/A')

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
                if policy_split:
                    print(f"  Policy Split: {policy_split}")
                print(f"  Policy Label: {policy_name}")
                print(f"  Policy Location: {policy_location}")

                if evidence_decl and len(evidence_decl) > 100:
                    evidence_decl = evidence_decl[:97] + "..."
                print(f"  Evidence (Declarations): {evidence_decl if evidence_decl else 'N/A'}")

                if evidence_end:
                    if len(evidence_end) > 100:
                        evidence_end = evidence_end[:97] + "..."
                    print(f"  Evidence (Endorsements): {evidence_end}")

                if notes and len(notes) > 150:
                    notes = notes[:147] + "..."
                print(f"  Notes: {notes if notes else 'N/A'}")
                print()

        # Display Equipment Breakdown validations (if present)
        eb_validations = results.get('equipment_breakdown_validations', [])
        if eb_validations:
            print(f"{'='*70}")
            print("EQUIPMENT BREAKDOWN VALIDATION RESULTS")
            print(f"{'='*70}\n")

            for v in eb_validations:
                status = v.get('status', 'UNKNOWN')
                cert_name = v.get('cert_eb_name', 'N/A')
                cert_value = v.get('cert_eb_value', 'N/A')
                policy_name = v.get('policy_eb_name', 'N/A')
                policy_value = v.get('policy_eb_value', 'N/A')
                policy_location = v.get('policy_location', 'N/A')
                evidence_decl = v.get('evidence_declarations', 'N/A')
                evidence_end = v.get('evidence_endorsements', None)
                notes = v.get('notes', 'N/A')

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
                print(f"  Policy Label: {policy_name}")
                print(f"  Policy Location: {policy_location}")

                if evidence_decl and len(evidence_decl) > 100:
                    evidence_decl = evidence_decl[:97] + "..."
                print(f"  Evidence (Declarations): {evidence_decl if evidence_decl else 'N/A'}")

                if evidence_end:
                    if len(evidence_end) > 100:
                        evidence_end = evidence_end[:97] + "..."
                    print(f"  Evidence (Endorsements): {evidence_end}")

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

        if 'total_bpp_items' in summary:
            print(f"\nTotal BPP Items:  {summary.get('total_bpp_items', 0)}")
            print(f"  ✓ Matched:      {summary.get('bpp_matched', 0)}")
            print(f"  ✗ Mismatched:   {summary.get('bpp_mismatched', 0)}")
            print(f"  ? Not Found:    {summary.get('bpp_not_found', 0)}")

        if 'total_ms_items' in summary:
            print(f"\nTotal Money & Securities Items:  {summary.get('total_ms_items', 0)}")
            print(f"  ✓ Matched:      {summary.get('ms_matched', 0)}")
            print(f"  ✗ Mismatched:   {summary.get('ms_mismatched', 0)}")
            print(f"  ? Not Found:    {summary.get('ms_not_found', 0)}")

        if 'total_eb_items' in summary:
            print(f"\nTotal Equipment Breakdown Items:  {summary.get('total_eb_items', 0)}")
            print(f"  ✓ Matched:      {summary.get('eb_matched', 0)}")
            print(f"  ✗ Mismatched:   {summary.get('eb_mismatched', 0)}")
            print(f"  ? Not Found:    {summary.get('eb_not_found', 0)}")
        
        if 'qc_notes' in results:
            qc_notes = results['qc_notes']
            if len(qc_notes) > 200:
                qc_notes = qc_notes[:197] + "..."
            print(f"\nQC Notes: {qc_notes}")
        
        print(f"{'='*70}\n")


def main():
    """Main execution function"""
    # ========== EDIT THESE VALUES ==========
    cert_prefix = "salem"              # Change to: james, indian, etc.
    carrier_dir = "encovaop"      # Change to: hartfordop, encovaop, etc.
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

