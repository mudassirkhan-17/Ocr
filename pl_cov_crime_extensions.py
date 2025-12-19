"""
LLM-Based Coverage Validation - Crime & Extensions (Money & Securities + Employee Dishonesty)
Validates Money & Securities and Employee Dishonesty coverages from certificate against policy document
"""

import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


class CrimeExtensionsCoverageValidator:
    """Validate Money & Securities and Employee Dishonesty coverages from certificate against policy."""
    
    def __init__(self, model: str = "gpt-4.1-mini"):
        """
        Initialize the validator
        
        Args:
            model: OpenAI model to use (default: gpt-4.1-nano for cost efficiency)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
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

    def extract_employee_dishonesty_coverages(self, cert_data: Dict) -> List[Dict]:
        """
        Extract Employee Dishonesty coverages from certificate.

        Notes:
        - Usually a dollar limit (e.g., 10,000 / 25,000 / 50,000)
        - Sometimes "Included" / "Yes"
        - Avoid confusing with other crime/cyber coverages when possible
        """
        coverages = cert_data.get("coverages", {}) or {}
        ed_items: List[Dict] = []

        for coverage_name, coverage_value in coverages.items():
            name = (coverage_name or "").strip()
            n = name.lower()

            is_ed = (
                "employee dishonesty" in n
                or ("employee" in n and "dishon" in n)
                or (n == "dishonesty")
            )

            # Exclude non-coverage details if they appear as keys
            is_excluded = any(
                kw in n
                for kw in [
                    "deductible",
                    "ded.",
                    "coinsurance",
                    "waiting",
                    "waiting period",
                    "description",
                ]
            )

            if is_ed and not is_excluded:
                ed_items.append({"name": name, "value": coverage_value})

        return ed_items

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

        ms = _count(results.get("money_securities_validations", []))
        summary.update(
            {
                "total_ms_items": ms["total"],
                "ms_matched": ms["match"],
                "ms_mismatched": ms["mismatch"],
                "ms_not_found": ms["not_found"],
            }
        )

        ed = _count(results.get("employee_dishonesty_validations", []))
        summary.update(
            {
                "total_ed_items": ed["total"],
                "ed_matched": ed["match"],
                "ed_mismatched": ed["mismatch"],
                "ed_not_found": ed["not_found"],
            }
        )

        results["summary"] = summary
    
    def create_validation_prompt(self, cert_data: Dict, ms_items: List[Dict], ed_items: List[Dict], policy_text: str) -> str:
        """
        Create validation prompt for Money & Securities and Employee Dishonesty coverages
        
        Args:
            cert_data: Certificate data with location context
            ms_items: List of Money & Securities coverages to validate
            ed_items: List of Employee Dishonesty coverages to validate
            policy_text: Full policy document text
            
        Returns:
            Formatted prompt string
        """
        
        # Extract context from certificate
        location_address = cert_data.get("location_address", "Not specified")
        insured_name = cert_data.get("insured_name", "Not specified")
        policy_number = cert_data.get("policy_number", "Not specified")
        
        all_coverages = cert_data.get("coverages", {}) or {}

        prompt = f"""You are an expert Property Insurance QC Specialist validating crime and extension coverages (Money & Securities and Employee Dishonesty).

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
   - Example: You find "Money & Securities" mentioned but can't find the actual limit
   - Return status="NOT_FOUND" with explanation in notes
   - Better to return NOT_FOUND than to guess a number

4. **CRITICAL - NEVER GUESS:**
   - Do NOT estimate values
   - Do NOT use "typical" amounts
   - Do NOT use numbers from other sections
   - If unclear, return null and NOT_FOUND

5. **EVIDENCE REQUIREMENTS:**
   - Evidence MUST be verbatim text from the policy
   - Evidence MUST contain BOTH the coverage name AND the value
   - If you can't find evidence with both, return null
   - ONLY cite page numbers that are clearly marked in the policy

**When in doubt, return NOT_FOUND with null values. This is better than hallucinating.**

==================================================
CRITICAL INSTRUCTIONS
==================================================

**YOUR TASK:**
Validate Money & Securities and Employee Dishonesty coverages from the certificate against the policy document.

**CONTEXT FROM CERTIFICATE:**
- Insured Name: {insured_name}
- Policy Number: {policy_number}
- Location Address: {location_address}

**ALL CERTIFICATE COVERAGES (for context):**
{json.dumps(all_coverages, indent=2)}

**MONEY & SECURITIES COVERAGES TO VALIDATE:**
{json.dumps(ms_items, indent=2)}

**EMPLOYEE DISHONESTY COVERAGES TO VALIDATE:**
{json.dumps(ed_items, indent=2)}

==================================================
POLICY DOCUMENT (DUAL OCR SOURCES)
==================================================

This policy document contains TWO OCR extraction sources per page:
- **TESSERACT (Buffer=1)** - First OCR source
- **PYMUPDF (Buffer=0)** - Second OCR source

Use whichever source is clearer. ALWAYS cite which OCR source you used.

{policy_text}

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
- Evidence must include page number and OCR source.

==================================================
EMPLOYEE DISHONESTY VALIDATION RULES (STRICT)
==================================================

For EACH Employee Dishonesty item:
- Certificate value may be "Included" / "Yes" or a dollar limit.
- Prefer evidence from declarations, schedules of coverages/optional coverages, or the specific crime endorsement/form where the limit is shown.
- MATCH rules:
  - If certificate is "Included"/"Yes": MATCH if policy indicates Employee Dishonesty is covered/included OR shows a limit for Employee Dishonesty.
  - If certificate is a dollar limit: MATCH only if the policy's Employee Dishonesty limit matches (ignore $/commas/spacing).
- Do NOT confuse with:
  - Forgery/Alteration
  - Money Orders & Counterfeit Money
  - Computer Fraud / Funds Transfer Fraud
  - Other crime/cyber coverages that are not Employee Dishonesty
- Evidence must include OCR source + page number.

==================================================
OUTPUT FORMAT (WITH NULL VALUES FOR NOT FOUND)
==================================================

⚠️ **IMPORTANT:** Use null for any field you cannot verify in the policy:
- If NOT_FOUND: all policy_* fields = null, all evidence_* fields = null
- Do NOT leave empty strings "" - use null instead
- Do NOT invent page numbers

Return ONLY a valid JSON object with this structure:

{{
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
  "employee_dishonesty_validations": [
    {{
      "cert_ed_name": "Name from certificate (e.g., 'Employee Dishonesty')",
      "cert_ed_value": "Value from certificate (e.g., 'Included' or '25,000')",
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "policy_ed_name": "How it appears in policy",
      "policy_ed_value": "Policy value (Included/Yes or a dollar limit) or null",
      "policy_location": "Location/premises/building description from policy (or null if policy-wide)",
      "evidence_declarations": "Quote from declarations/schedule (OCR_SOURCE, Page X)",
      "evidence_endorsements": "Quote from crime/endorsement form if applicable (OCR_SOURCE, Page X) or null",
      "notes": "Explain how you matched and why MATCH/MISMATCH/NOT_FOUND."
    }}
  ],
  "summary": {{
    "total_ms_items": 0,
    "ms_matched": 0,
    "ms_mismatched": 0,
    "ms_not_found": 0,
    "total_ed_items": 0,
    "ed_matched": 0,
    "ed_mismatched": 0,
    "ed_not_found": 0
  }},
  "qc_notes": "Overall observations about the validation"
}}

**STATUS DEFINITIONS:**
- **MATCH**: Policy limit EXACTLY matches certificate value (with all evidence fields populated)
- **MISMATCH**: Policy limit CONFIRMED but DIFFERS from certificate value (with evidence fields populated)
- **NOT_FOUND**: Coverage not found in policy OR evidence cannot be verified (set all policy_* and evidence_* fields to null)

⚠️ **CRITICAL CHECKLIST BEFORE RETURNING:**
For each validation result, verify:
✓ If NOT_FOUND: Are ALL policy_* fields set to null?
✓ If NOT_FOUND: Are ALL evidence_* fields set to null?
✓ If MATCH/MISMATCH: Does evidence contain coverage name + value + page number?
✓ If MATCH/MISMATCH: Is the page number actually visible in the policy text?
✓ Did I search thoroughly before declaring NOT_FOUND?
✓ Did I avoid inventing values, page numbers, or evidence?

**EVIDENCE FORMAT:**
Always include page number and OCR source, e.g.:
- "Money and Securities: $10,000 (TESSERACT, Page 4)"
- "Employee Dishonesty Limit: $25,000 (PYMUPDF, Page 27)"

Return ONLY the JSON object. No other text.
"""
        
        return prompt
    
    def validate_crime_extensions(self, cert_json_path: str, policy_combo_path: str, output_path: str):
        """
        Main validation workflow
        
        Args:
            cert_json_path: Path to certificate JSON file
            policy_combo_path: Path to policy combo text file
            output_path: Path for output JSON file
        """
        
        print(f"\n{'='*70}")
        print("CRIME & EXTENSIONS COVERAGE VALIDATION (MONEY & SECURITIES + EMPLOYEE DISHONESTY)")
        print(f"{'='*70}\n")
        
        # Load certificate
        print(f"[1/5] Loading certificate: {cert_json_path}")
        with open(cert_json_path, 'r', encoding='utf-8') as f:
            cert_data = json.load(f)
        
        # Extract coverages to validate
        ms_items = self.extract_money_securities_coverages(cert_data)
        ed_items = self.extract_employee_dishonesty_coverages(cert_data)
        
        if not ms_items and not ed_items:
            print("      ❌ No Money & Securities or Employee Dishonesty coverages found in certificate!")
            return
        
        if ms_items:
            print(f"      Found {len(ms_items)} Money & Securities coverage(s):")
            for m in ms_items:
                print(f"        - {m['name']}: {m['value']}")
        if ed_items:
            print(f"      Found {len(ed_items)} Employee Dishonesty coverage(s):")
            for e in ed_items:
                print(f"        - {e['name']}: {e['value']}")
        
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
            ms_items,
            ed_items,
            policy_text,
        )
        prompt_size_kb = len(prompt) / 1024
        print(f"      Prompt size: {prompt_size_kb:.1f} KB")
        
        # Call LLM
        print(f"\n[4/5] Calling LLM for validation (model: {self.model})...")
        print(f"      Analyzing policy for Money & Securities and Employee Dishonesty coverage...")
        
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
            results["money_securities_validations"] = self._filter_validations_to_requested(
                results.get("money_securities_validations", []),
                ms_items,
                "cert_ms_name",
            )
            results["employee_dishonesty_validations"] = self._filter_validations_to_requested(
                results.get("employee_dishonesty_validations", []),
                ed_items,
                "cert_ed_name",
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
        print("CRIME & EXTENSIONS COVERAGE VALIDATION RESULTS (MONEY & SECURITIES + EMPLOYEE DISHONESTY)")
        print(f"{'='*70}\n")
        
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

        # Display Employee Dishonesty validations (if present)
        ed_validations = results.get('employee_dishonesty_validations', [])
        if ed_validations:
            print(f"{'='*70}")
            print("EMPLOYEE DISHONESTY VALIDATION RESULTS")
            print(f"{'='*70}\n")

            for v in ed_validations:
                status = v.get('status', 'UNKNOWN')
                cert_name = v.get('cert_ed_name', 'N/A')
                cert_value = v.get('cert_ed_value', 'N/A')
                policy_name = v.get('policy_ed_name', 'N/A')
                policy_value = v.get('policy_ed_value', 'N/A')
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
        
        if 'total_ms_items' in summary:
            print(f"\nTotal Money & Securities Items:  {summary.get('total_ms_items', 0)}")
            print(f"  ✓ Matched:      {summary.get('ms_matched', 0)}")
            print(f"  ✗ Mismatched:   {summary.get('ms_mismatched', 0)}")
            print(f"  ? Not Found:    {summary.get('ms_not_found', 0)}")

        if 'total_ed_items' in summary:
            print(f"\nTotal Employee Dishonesty Items:  {summary.get('total_ed_items', 0)}")
            print(f"  ✓ Matched:      {summary.get('ed_matched', 0)}")
            print(f"  ✗ Mismatched:   {summary.get('ed_mismatched', 0)}")
            print(f"  ? Not Found:    {summary.get('ed_not_found', 0)}")
        
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
    output_path = os.path.join(carrier_dir, f"{cert_prefix}_crime_extensions_validation.json")
    
    # Check if files exist
    if not os.path.exists(cert_json_path):
        print(f"Error: Certificate JSON not found: {cert_json_path}")
        exit(1)
    
    if not os.path.exists(policy_combo_path):
        print(f"Error: Policy combo text not found: {policy_combo_path}")
        exit(1)
    
    # Create validator and run
    try:
        validator = CrimeExtensionsCoverageValidator()
        validator.validate_crime_extensions(cert_json_path, policy_combo_path, output_path)
    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
