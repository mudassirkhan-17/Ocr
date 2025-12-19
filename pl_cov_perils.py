"""
LLM-Based Coverage Validation - Perils (Theft + Wind/Hail)
Validates Theft and Wind/Hail coverages from certificate against policy document
Uses causes of loss (Basic/Broad/Special) to determine coverage inclusion
"""

import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


class PerilsCoverageValidator:
    """Validate Theft and Wind/Hail perils from certificate against policy."""
    
    def __init__(self, model: str = "gpt-4.1-mini"):
        """
        Initialize the validator
        
        Args:
            model: OpenAI model to use (default: gpt-4.1-mini)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def extract_theft_coverages(self, cert_data: Dict) -> List[Dict]:
        """
        Extract Theft coverage from certificate.
        Avoids keys like "Theft Deductible" which are not the coverage itself.
        """
        coverages = cert_data.get("coverages", {}) or {}
        items: List[Dict] = []

        for coverage_name, coverage_value in coverages.items():
            name = (coverage_name or "").strip()
            n = name.lower()

            if "theft" not in n:
                continue

            # Exclude deductible-only rows/keys
            if "deductible" in n or "ded." in n:
                continue

            # Keep only the core theft coverage entry
            items.append({"name": name, "value": coverage_value})

        return items

    def extract_wind_hail_coverages(self, cert_data: Dict) -> List[Dict]:
        """
        Extract Wind/Hail (Windstorm & Hail) coverage from certificate.
        Excludes deductible-only rows/keys (handled as notes during validation).
        """
        coverages = cert_data.get("coverages", {}) or {}
        items: List[Dict] = []

        for coverage_name, coverage_value in coverages.items():
            name = (coverage_name or "").strip()
            n = name.lower()

            is_wind_hail = (
                "wind" in n and "hail" in n
            ) or ("windstorm" in n and "hail" in n) or ("windstorm" in n)

            if not is_wind_hail:
                continue

            # Exclude deductible rows
            if "deductible" in n or "ded." in n:
                continue

            items.append({"name": name, "value": coverage_value})

        return items

    def _norm_name(self, s: Optional[str]) -> str:
        """Normalize coverage names for loose matching between requested items and LLM output."""
        if not s:
            return ""
        s = s.lower()
        # keep alphanumerics only to be robust to '&' vs 'and', punctuation, spacing
        return "".join(ch for ch in s if ch.isalnum())

    def _filter_validations_to_requested(
        self,
        validations: List[Dict],
        requested_items: List[Dict],
        cert_name_field: str,
    ) -> List[Dict]:
        """
        Keep only validation entries that correspond to requested certificate items.
        If no requested items exist, returns an empty list.
        """
        if not requested_items:
            return []

        requested_norms = [self._norm_name((it or {}).get("name")) for it in requested_items]
        requested_norms = [x for x in requested_norms if x]
        if not requested_norms:
            return []

        filtered: List[Dict] = []
        for v in validations or []:
            cert_name = self._norm_name((v or {}).get(cert_name_field))
            if not cert_name:
                continue

            # Loose containment match either direction
            if any(r in cert_name or cert_name in r for r in requested_norms):
                filtered.append(v)

        # If the LLM used unexpected labeling and nothing matched, fall back to
        # taking the first N validations to avoid dropping real results.
        if not filtered:
            return list((validations or [])[: len(requested_items)])

        # Prevent unexpected inflation: cap at number of requested items.
        if len(filtered) > len(requested_items):
            filtered = filtered[: len(requested_items)]

        return filtered

    def _recompute_summary_counts(self, results: Dict) -> None:
        """Recompute summary counts from the validation arrays to avoid hallucinated totals."""
        def _count(arr: List[Dict]) -> Dict[str, int]:
            c = {"total": 0, "match": 0, "mismatch": 0, "not_found": 0}
            for v in arr or []:
                c["total"] += 1
                s = (v.get("status") or "").upper()
                if s == "MATCH":
                    c["match"] += 1
                elif s == "MISMATCH":
                    c["mismatch"] += 1
                elif s == "NOT_FOUND":
                    c["not_found"] += 1
            return c

        summary = results.get("summary", {}) or {}

        theft = _count(results.get("theft_validations", []))
        summary.update(
            {
                "total_theft_items": theft["total"],
                "theft_matched": theft["match"],
                "theft_mismatched": theft["mismatch"],
                "theft_not_found": theft["not_found"],
            }
        )

        wh = _count(results.get("wind_hail_validations", []))
        summary.update(
            {
                "total_wind_hail_items": wh["total"],
                "wind_hail_matched": wh["match"],
                "wind_hail_mismatched": wh["mismatch"],
                "wind_hail_not_found": wh["not_found"],
            }
        )

        results["summary"] = summary
    
    def create_validation_prompt(self, cert_data: Dict, theft_items: List[Dict], wind_hail_items: List[Dict], policy_text: str) -> str:
        """
        Create validation prompt for Theft and Wind/Hail perils
        
        Args:
            cert_data: Certificate data with location context
            theft_items: List of theft coverages to validate
            wind_hail_items: List of wind/hail coverages to validate
            policy_text: Full policy document text
            
        Returns:
            Formatted prompt string
        """
        
        # Extract context from certificate
        location_address = cert_data.get("location_address", "Not specified")
        insured_name = cert_data.get("insured_name", "Not specified")
        policy_number = cert_data.get("policy_number", "Not specified")
        
        all_coverages = cert_data.get("coverages", {}) or {}

        prompt = f"""You are an expert Property Insurance QC Specialist validating peril coverage (Theft and Wind/Hail).

==================================================
⛔⛔⛔ ANTI-HALLUCINATION RULES (READ FIRST) ⛔⛔⛔
==================================================

**IF YOU CANNOT FIND SOMETHING, RETURN null OR "NOT_FOUND" - DO NOT HALLUCINATE**

1. **WHEN SEARCHING FOR A COVERAGE:**
   - Search thoroughly through all pages
   - If you CANNOT find it after careful search, return status="NOT_FOUND"
   - Return null for all evidence fields
   - Return null for all policy_* fields (policy_value, policy_name, etc.)
   - NEVER invent text or values that don't exist

2. **WHEN YOU FIND A COVERAGE BUT CAN'T FIND ITS VALUE:**
   - Return status="NOT_FOUND"
   - Return null for the value fields
   - Explain in notes: "Coverage name found but specific value/limit not located in policy"

3. **WHEN YOU PARTIALLY FIND SOMETHING:**
   - Example: You find "Theft" mentioned but can't determine if it's included
   - Return status="NOT_FOUND" with explanation in notes
   - Better to return NOT_FOUND than to guess

4. **CRITICAL - NEVER GUESS:**
   - Do NOT estimate values
   - Do NOT use "typical" amounts
   - Do NOT use numbers from other sections
   - If unclear, return null and NOT_FOUND

5. **EVIDENCE REQUIREMENTS:**
   - Evidence MUST be verbatim text from the policy
   - Evidence MUST contain BOTH the coverage name AND the value/inclusion status
   - If you can't find evidence with both, return null
   - ONLY cite page numbers that are clearly marked in the policy

**When in doubt, return NOT_FOUND with null values. This is better than hallucinating.**

==================================================
CRITICAL INSTRUCTIONS
==================================================

**YOUR TASK:**
Validate Theft and Wind/Hail (Windstorm & Hail) coverages from the certificate against the policy document.

**CONTEXT FROM CERTIFICATE:**
- Insured Name: {insured_name}
- Policy Number: {policy_number}
- Location Address: {location_address}

**ALL CERTIFICATE COVERAGES (for context):**
{json.dumps(all_coverages, indent=2)}

**THEFT COVERAGES TO VALIDATE:**
{json.dumps(theft_items, indent=2)}

**WIND / HAIL COVERAGES TO VALIDATE:**
{json.dumps(wind_hail_items, indent=2)}

==================================================
POLICY DOCUMENT (DUAL OCR SOURCES)
==================================================

This policy document contains TWO OCR extraction sources per page:
- **TESSERACT (Buffer=1)** - First OCR source
- **PYMUPDF (Buffer=0)** - Second OCR source

Use whichever source is clearer. ALWAYS cite which OCR source you used.

{policy_text}

==================================================
THEFT + WIND/HAIL VALIDATION RULES (ENCOVA-SAFE)
==================================================

These two are often NOT written as simple standalone lines. You MUST validate using the policy's per-premises/building "Causes of Loss" (Basic/Broad/Special) and any exclusions/endorsements.

**IMPORTANT: HOW TO READ ENCOVA DECLARATIONS TABLES**
- Many policies show a table with columns BASIC / BROAD / SPECIAL and an X/mark indicating which applies for a given premises/building.
- You MUST quote the line(s) that show which causes of loss applies (OCR source + page).

**THEFT (Peril inclusion)**
- Theft is typically INCLUDED only when Causes of Loss is **SPECIAL** (unless excluded by endorsement).
- If certificate says Theft = "Included":
  - MATCH only if the policy shows **SPECIAL** for the relevant premises/building AND you do NOT find a Theft exclusion endorsement.
  - If policy shows BASIC or BROAD for that premises/building, Theft is generally NOT included -> MISMATCH (unless a separate theft coverage endorsement explicitly adds it).
- If certificate gives a Theft dollar limit (rare):
  - MATCH only if the policy shows a theft sublimit/coverage limit specifically for Theft (not employee dishonesty/crime).
- Do NOT confuse Theft with:
  - "Theft Deductible" rows
  - Employee Dishonesty / Employee Theft (crime/fidelity)

**WIND/HAIL (Windstorm & Hail)**
- Windstorm/Hail can appear as "Wind and Hail", "Wind & Hail", "Windstorm & Hail" and may be shown only as a deductible/percentage.
- If certificate says Wind/Hail = "Included":
  - MATCH if the policy includes wind/hail as a covered peril for the relevant premises/building (often via BASIC/BROAD/SPECIAL) and there is no wind/hail exclusion endorsement.
  - If the policy only shows a Wind/Hail deductible/percentage, that still supports "Included" (capture it).
- If certificate gives a Wind/Hail limit (rare):
  - MATCH only if a specific wind/hail limit/sublimit is found and matches.
- Evidence must cite causes-of-loss selection AND any deductible/exclusion language if present.

==================================================
OUTPUT FORMAT (WITH NULL VALUES FOR NOT FOUND)
==================================================

⚠️ **IMPORTANT:** Use null for any field you cannot verify in the policy:
- If NOT_FOUND: all policy_* fields = null, all evidence_* fields = null
- Do NOT leave empty strings "" - use null instead
- Do NOT invent page numbers

Return ONLY a valid JSON object with this structure:

{{
  "theft_validations": [
    {{
      "cert_theft_name": "Name from certificate (e.g., 'Theft')",
      "cert_theft_value": "Value from certificate (e.g., 'Included' or a limit)",
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "policy_theft_value": "Included/Not Included/Limit or null",
      "policy_causes_of_loss": "Basic | Broad | Special | Unknown",
      "policy_location": "Premises/building description used",
      "evidence_causes_of_loss": "Quote showing Basic/Broad/Special selection (OCR_SOURCE, Page X)",
      "evidence_exclusions": "Quote from theft exclusion/endorsement if found, else null",
      "notes": "Explain why theft is included or not (must reference causes-of-loss)."
    }}
  ],
  "wind_hail_validations": [
    {{
      "cert_wind_hail_name": "Name from certificate (e.g., 'Wind and Hail')",
      "cert_wind_hail_value": "Value from certificate (e.g., 'Included' or a limit)",
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "policy_wind_hail_value": "Included/Excluded/Limit or null",
      "policy_causes_of_loss": "Basic | Broad | Special | Unknown",
      "policy_wind_hail_deductible": "If present (e.g., '1%') else null",
      "policy_location": "Premises/building description used",
      "evidence_causes_of_loss": "Quote showing Basic/Broad/Special selection (OCR_SOURCE, Page X)",
      "evidence_deductible_or_endorsement": "Quote showing wind/hail deductible or exclusion/endorsement if present, else null",
      "notes": "Explain why wind/hail is included/excluded and how you matched synonyms."
    }}
  ],
  "summary": {{
    "total_theft_items": 0,
    "theft_matched": 0,
    "theft_mismatched": 0,
    "theft_not_found": 0,
    "total_wind_hail_items": 0,
    "wind_hail_matched": 0,
    "wind_hail_mismatched": 0,
    "wind_hail_not_found": 0
  }},
  "qc_notes": "Overall observations about the validation"
}}

**STATUS DEFINITIONS:**
- **MATCH**: Policy coverage status EXACTLY matches certificate value (with all evidence fields populated)
- **MISMATCH**: Policy coverage status CONFIRMED but DIFFERS from certificate value (with evidence fields populated)
- **NOT_FOUND**: Coverage not found in policy OR evidence cannot be verified (set all policy_* and evidence_* fields to null)

⚠️ **CRITICAL CHECKLIST BEFORE RETURNING:**
For each validation result, verify:
✓ If NOT_FOUND: Are ALL policy_* fields set to null?
✓ If NOT_FOUND: Are ALL evidence_* fields set to null?
✓ If MATCH/MISMATCH: Does evidence contain coverage name + causes of loss + page number?
✓ If MATCH/MISMATCH: Is the page number actually visible in the policy text?
✓ Did I search thoroughly before declaring NOT_FOUND?
✓ Did I avoid inventing values, page numbers, or evidence?

**EVIDENCE FORMAT:**
Always include page number and OCR source, e.g.:
- "Causes of Loss - SPECIAL (TESSERACT, Page 4)"
- "Wind/Hail Deductible: 1% (PYMUPDF, Page 27)"

Return ONLY the JSON object. No other text.
"""
        
        return prompt
    
    def validate_perils(self, cert_json_path: str, policy_combo_path: str, output_path: str):
        """
        Main validation workflow
        
        Args:
            cert_json_path: Path to certificate JSON file
            policy_combo_path: Path to policy combo text file
            output_path: Path for output JSON file
        """
        
        print(f"\n{'='*70}")
        print("PERILS COVERAGE VALIDATION (THEFT + WIND/HAIL)")
        print(f"{'='*70}\n")
        
        # Load certificate
        print(f"[1/5] Loading certificate: {cert_json_path}")
        with open(cert_json_path, 'r', encoding='utf-8') as f:
            cert_data = json.load(f)
        
        # Extract coverages to validate
        theft_items = self.extract_theft_coverages(cert_data)
        wind_hail_items = self.extract_wind_hail_coverages(cert_data)
        
        if not theft_items and not wind_hail_items:
            print("      ❌ No Theft or Wind/Hail coverages found in certificate!")
            return
        
        if theft_items:
            print(f"      Found {len(theft_items)} Theft coverage(s):")
            for t in theft_items:
                print(f"        - {t['name']}: {t['value']}")
        if wind_hail_items:
            print(f"      Found {len(wind_hail_items)} Wind/Hail coverage(s):")
            for w in wind_hail_items:
                print(f"        - {w['name']}: {w['value']}")
        
        # Load policy
        print(f"\n[2/5] Loading policy: {policy_combo_path}")
        with open(policy_combo_path, 'r', encoding='utf-8') as f:
            policy_text = f.read()
        
        policy_size_kb = len(policy_text) / 1024
        print(f"      Policy size: {policy_size_kb:.1f} KB")
        
        # Create prompt
        print(f"\n[3/5] Creating validation prompt...")
        prompt = self.create_validation_prompt(
            cert_data,
            theft_items,
            wind_hail_items,
            policy_text,
        )
        prompt_size_kb = len(prompt) / 1024
        print(f"      Prompt size: {prompt_size_kb:.1f} KB")
        
        # Call LLM
        print(f"\n[4/5] Calling LLM for validation (model: {self.model})...")
        print(f"      Analyzing policy for Theft and Wind/Hail coverage...")
        
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

            # Guardrail: keep only validations for items actually present in the certificate extraction.
            results["theft_validations"] = self._filter_validations_to_requested(
                results.get("theft_validations", []),
                theft_items,
                "cert_theft_name",
            )
            results["wind_hail_validations"] = self._filter_validations_to_requested(
                results.get("wind_hail_validations", []),
                wind_hail_items,
                "cert_wind_hail_name",
            )

            self._recompute_summary_counts(results)
            
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
        print("PERILS COVERAGE VALIDATION RESULTS (THEFT + WIND/HAIL)")
        print(f"{'='*70}\n")
        
        # Display Theft validations (if present)
        theft_validations = results.get('theft_validations', [])
        if theft_validations:
            print(f"{'='*70}")
            print("THEFT VALIDATION RESULTS")
            print(f"{'='*70}\n")

            for v in theft_validations:
                status = v.get('status', 'UNKNOWN')
                cert_name = v.get('cert_theft_name', 'N/A')
                cert_value = v.get('cert_theft_value', 'N/A')
                policy_value = v.get('policy_theft_value', 'N/A')
                causes = v.get('policy_causes_of_loss', 'Unknown')
                policy_location = v.get('policy_location', 'N/A')
                evidence_col = v.get('evidence_causes_of_loss', 'N/A')
                evidence_excl = v.get('evidence_exclusions', None)
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
                print(f"  Causes of Loss: {causes}")
                print(f"  Policy Location: {policy_location}")

                if evidence_col and len(evidence_col) > 140:
                    evidence_col = evidence_col[:137] + "..."
                print(f"  Evidence (Causes of Loss): {evidence_col if evidence_col else 'N/A'}")

                if evidence_excl:
                    if len(evidence_excl) > 140:
                        evidence_excl = evidence_excl[:137] + "..."
                    print(f"  Evidence (Exclusions/Endorsements): {evidence_excl}")

                if notes and len(notes) > 170:
                    notes = notes[:167] + "..."
                print(f"  Notes: {notes if notes else 'N/A'}")
                print()

        # Display Wind/Hail validations (if present)
        wh_validations = results.get('wind_hail_validations', [])
        if wh_validations:
            print(f"{'='*70}")
            print("WIND / HAIL VALIDATION RESULTS")
            print(f"{'='*70}\n")

            for v in wh_validations:
                status = v.get('status', 'UNKNOWN')
                cert_name = v.get('cert_wind_hail_name', 'N/A')
                cert_value = v.get('cert_wind_hail_value', 'N/A')
                policy_value = v.get('policy_wind_hail_value', 'N/A')
                causes = v.get('policy_causes_of_loss', 'Unknown')
                deductible = v.get('policy_wind_hail_deductible', None)
                policy_location = v.get('policy_location', 'N/A')
                evidence_col = v.get('evidence_causes_of_loss', 'N/A')
                evidence_other = v.get('evidence_deductible_or_endorsement', None)
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
                print(f"  Causes of Loss: {causes}")
                if deductible:
                    print(f"  Wind/Hail Deductible: {deductible}")
                print(f"  Policy Location: {policy_location}")

                if evidence_col and len(evidence_col) > 140:
                    evidence_col = evidence_col[:137] + "..."
                print(f"  Evidence (Causes of Loss): {evidence_col if evidence_col else 'N/A'}")

                if evidence_other:
                    if len(evidence_other) > 140:
                        evidence_other = evidence_other[:137] + "..."
                    print(f"  Evidence (Deductible/Endorsement): {evidence_other}")

                if notes and len(notes) > 170:
                    notes = notes[:167] + "..."
                print(f"  Notes: {notes if notes else 'N/A'}")
                print()
        
        # Print summary
        summary = results.get('summary', {})
        print(f"{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        
        if 'total_theft_items' in summary:
            print(f"\nTotal Theft Items:  {summary.get('total_theft_items', 0)}")
            print(f"  ✓ Matched:      {summary.get('theft_matched', 0)}")
            print(f"  ✗ Mismatched:   {summary.get('theft_mismatched', 0)}")
            print(f"  ? Not Found:    {summary.get('theft_not_found', 0)}")

        if 'total_wind_hail_items' in summary:
            print(f"\nTotal Wind/Hail Items:  {summary.get('total_wind_hail_items', 0)}")
            print(f"  ✓ Matched:      {summary.get('wind_hail_matched', 0)}")
            print(f"  ✗ Mismatched:   {summary.get('wind_hail_mismatched', 0)}")
            print(f"  ? Not Found:    {summary.get('wind_hail_not_found', 0)}")
        
        if 'qc_notes' in results:
            qc_notes = results['qc_notes']
            if len(qc_notes) > 200:
                qc_notes = qc_notes[:197] + "..."
            print(f"\nQC Notes: {qc_notes}")
        
        print(f"{'='*70}\n")


def main():
    """Main execution function"""
    # ========== EDIT THESE VALUES ==========
    cert_prefix = "westside"              # Change to: james, indian, etc.
    carrier_dir = "nationwideop"      # Change to: hartfordop, encovaop, etc.
    # =======================================
    
    # Construct paths
    cert_json_path = os.path.join(carrier_dir, f"{cert_prefix}_pl_extracted_real.json")
    policy_combo_path = os.path.join(carrier_dir, f"{cert_prefix}_pol_combo.txt")
    output_path = os.path.join(carrier_dir, f"{cert_prefix}_perils_validation.json")
    
    # Check if files exist
    if not os.path.exists(cert_json_path):
        print(f"Error: Certificate JSON not found: {cert_json_path}")
        exit(1)
    
    if not os.path.exists(policy_combo_path):
        print(f"Error: Policy combo text not found: {policy_combo_path}")
        exit(1)
    
    # Create validator and run
    try:
        validator = PerilsCoverageValidator()
        validator.validate_perils(cert_json_path, policy_combo_path, output_path)
    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
