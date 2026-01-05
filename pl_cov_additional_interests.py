"""
LLM-Based Coverage Validation - Additional Interests
Validates Additional Interests (Mortgagee, Loss Payee, Additional Insured) from certificate against policy document
Uses filtered pages to focus on relevant sections
"""

import os
import json
import re
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Regex to find pages with "additional insure*" or "additional interests" (mortgagee, loss payee, etc.)
_ADDITIONAL_INSURE_RE = re.compile(r"additional\s+(insure\w*|interests?)", re.IGNORECASE)


def _split_policy_combo_into_pages(policy_text: str) -> List[Dict]:
    """Split policy text into pages based on PAGE markers"""
    text = policy_text.replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(r"^={40,}\nPAGE\s+(\d+)\n={40,}\n", re.MULTILINE)
    matches = list(pattern.finditer(text))
    
    if not matches:
        return [{"page_number": 0, "text": text}]
    
    pages = []
    for i, m in enumerate(matches):
        page_num = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip("\n")
        pages.append({"page_number": page_num, "text": chunk})
    
    return pages


def _filter_pages_with_additional_insure(pages: List[Dict]) -> List[int]:
    """Find all pages containing 'additional insure*' or 'additional interests' or related terms (mortgagee, loss payee, etc.)"""
    out = []
    # Also search for mortgagee, loss payee, lienholder keywords
    additional_keywords = re.compile(
        r"(additional\s+(insure\w*|interests?)|mortgagee|loss\s+payee|lienholder)", 
        re.IGNORECASE
    )
    for p in pages:
        if additional_keywords.search(p["text"]):
            out.append(p["page_number"])
    return out


def _expand_neighbors(page_nums: List[int], radius: int) -> List[int]:
    """Add neighboring pages"""
    if radius <= 0:
        return sorted(set(page_nums))
    s = set(page_nums)
    for n in list(s):
        for i in range(1, radius + 1):
            s.add(n - i)
            s.add(n + i)
    return sorted(x for x in s if x >= 0)


def _build_filtered_policy_text(pages: List[Dict], keep_page_nums: List[int], max_pages: int) -> str:
    """Build filtered text from selected pages"""
    keep_set = set(keep_page_nums)
    kept = [p for p in pages if p["page_number"] in keep_set]
    kept.sort(key=lambda p: p["page_number"])
    
    if max_pages and len(kept) > max_pages:
        kept = kept[:max_pages]
    
    return "\n\n".join(p["text"] for p in kept)


class AdditionalInterestsCoverageValidator:
    """Validate Additional Interests from certificate against policy."""
    
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
    
    def extract_additional_interests(self, cert_data: Dict) -> List[Dict]:
        """
        Extract additional interests from certificate (handle all formats)
        
        Returns:
            List of additional interests with name and address
        """
        interests = []
        
        # Handle array format (2+ interests)
        if "additional_interests" in cert_data and isinstance(cert_data.get("additional_interests"), list):
            interests = cert_data["additional_interests"]
        # Handle flat format (1 interest)
        elif "additional_interest_name" in cert_data and cert_data.get("additional_interest_name"):
            interests = [{
                "name": cert_data.get("additional_interest_name"),
                "address": cert_data.get("additional_interest_address", "")
            }]
        # If neither exists, interests remains empty (0 interests)
        
        return interests
    
    def _is_name_variation(self, name1: str, name2: str) -> bool:
        """
        Check if two names are likely the same entity with OCR variation
        
        Args:
            name1: First name (uppercase)
            name2: Second name (uppercase)
            
        Returns:
            True if names appear to be same entity with OCR error
        """
        # Remove common suffixes for comparison
        suffixes = [' LLC', ' INC', ' CORP', ' L.L.C.', ' I.N.C.', ' CORP.']
        n1 = name1.upper()
        n2 = name2.upper()
        for suffix in suffixes:
            n1 = n1.replace(suffix, '')
            n2 = n2.replace(suffix, '')
        
        # Check if very similar (1-2 character difference)
        if len(n1) == len(n2):
            diff = sum(c1 != c2 for c1, c2 in zip(n1, n2))
            if diff <= 2 and len(n1) > 5:  # Allow 1-2 char differences for names > 5 chars
                return True
        
        # Check for common OCR errors: G vs H, O vs 0, I vs 1, etc.
        if len(n1) > 8 and len(n2) > 8:
            common_errors = [('G', 'H'), ('H', 'G'), ('O', '0'), ('0', 'O'), ('I', '1'), ('1', 'I')]
            n1_variants = [n1]
            n2_variants = [n2]
            for old, new in common_errors:
                n1_variants.append(n1.replace(old, new))
                n2_variants.append(n2.replace(old, new))
            
            for v1 in n1_variants:
                for v2 in n2_variants:
                    if v1 == v2:
                        return True
        
        return False
    
    def _recompute_summary_counts(self, results: Dict) -> None:
        """Recompute summary counts from validation arrays"""
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
        
        counts = _count(results.get("additional_interests_validations", []))
        summary = results.get("summary", {}) or {}
        summary.update({
            "total_additional_interests": counts["total"],
            "additional_interests_matched": counts["match"],
            "additional_interests_mismatched": counts["mismatch"],
            "additional_interests_not_found": counts["not_found"],
        })
        results["summary"] = summary
    
    def create_validation_prompt(self, cert_data: Dict, cert_interests: List[Dict], filtered_policy_text: str) -> str:
        """
        Create validation prompt for Additional Interests
        
        Args:
            cert_data: Certificate data with context
            cert_interests: List of additional interests from certificate
            filtered_policy_text: Filtered policy text focusing on relevant sections
            
        Returns:
            Formatted prompt string
        """
        
        insured_name = cert_data.get("insured_name", "Not specified")
        policy_number = cert_data.get("policy_number", "Not specified")
        location_address = cert_data.get("location_address", "Not specified")
        
        prompt = f"""You are an expert Property Insurance QC Specialist validating Additional Interests (Mortgagee, Loss Payee, Additional Insured).

==================================================
🔴 CRITICAL EVIDENCE EXTRACTION RULES 🔴
==================================================

**YOU MUST FOLLOW THIS EXACT PROCESS FOR EACH ADDITIONAL INTEREST:**

**STEP 1: SEARCH**
- Scan the policy text below for the entity name
- Look in: Additional Insured schedules, Mortgagee clauses, Loss Payee sections, Endorsements
- Check both TESSERACT and PYMUPDF sources (one may be clearer than the other)
- Search thoroughly - the entity may appear in multiple locations

**STEP 2: LOCATE THE PAGE NUMBER**
- Find the PAGE marker in the text (e.g., "PAGE 143" or "PAGE 67")
- The page number MUST come from these PAGE markers - DO NOT INVENT OR GUESS
- If you see the entity but cannot find a clear PAGE marker nearby, search up/down in the text for the closest marker

**STEP 3: COPY EXACT TEXT (DO NOT PARAPHRASE)**
- Copy the text WORD-FOR-WORD from the policy - no changes, no summaries
- Include at least 15-30 words for proper context
- Include the label/header if present (e.g., "Additional Insured:", "Mortgagee:")
- DO NOT change, summarize, rephrase, or "clean up" ANY words

**STEP 4: VERIFY YOUR WORK BEFORE SUBMITTING**
Ask yourself these questions:
✓ Did I copy this text EXACTLY as written? (not paraphrased, not summarized)
✓ Is the page number from an actual PAGE marker I can see in the text below?
✓ Does the entity name appear in my copied text?
✓ Did I include enough context (15-30 words minimum)?
✓ Would someone be able to Ctrl+F and find this exact text on this page?

⛔ IF YOU CANNOT FIND THE ENTITY AFTER THOROUGH SEARCH:
- Return status: "NOT_FOUND"
- Set evidence: null
- Set policy_interest_name: null
- Set policy_interest_address: null
- Set policy_interest_type: null
- Set match_type: null
- DO NOT INVENT, GUESS, OR PARAPHRASE

==================================================
EVIDENCE EXAMPLES - STUDY THESE CAREFULLY
==================================================

Certificate Entity: "DGR HOLDING LLC"

✅ EXAMPLE 1 - EXCELLENT (Exact copy with full context):
"Name Of Person(s) Or Organization(s) (Additional Insured): DGR HOLDING LLC, 123 MAIN STREET, ATLANTA, GA 30303. Coverage applies per the terms of this endorsement. (PYMUPDF, Page 143)"

✅ EXAMPLE 2 - EXCELLENT (From schedule with context):
"SCHEDULE OF ADDITIONAL INSUREDS: 1. DGR HOLDING LLC - 123 MAIN ST - ATLANTA GA 30303 - BLANKET ADDITIONAL INSURED 2. ABC COMPANY (TESSERACT, Page 67)"

✅ EXAMPLE 3 - EXCELLENT (Mortgagee clause with surrounding text):
"Loss Payable Clause: Loss, if any, under this policy shall be payable to DGR HOLDING LLC, 123 Main Street, Atlanta, GA 30303 as mortgagee as interest may appear. (PYMUPDF, Page 89)"

✅ EXAMPLE 4 - GOOD (Endorsement with entity name):
"This endorsement modifies insurance provided under COMMERCIAL GENERAL LIABILITY COVERAGE FORM. Additional Insured: DGR HOLDING LLC (TESSERACT, Page 145)"

❌ EXAMPLE 1 - BAD (Paraphrased, not exact):
"The policy lists DGR Holding LLC as an additional insured on page 143"
PROBLEM: Paraphrased instead of word-for-word copy

❌ EXAMPLE 2 - BAD (Missing page number):
"Name Of Person(s) Or Organization(s) (Additional Insured): DGR HOLDING LLC"
PROBLEM: No page number provided

❌ EXAMPLE 3 - BAD (Invented page number):
"DGR HOLDING LLC appears as additional insured (Page 143)"
PROBLEM: Not a direct quote, page number may be invented, no actual text copied

❌ EXAMPLE 4 - BAD (Insufficient context):
"DGR HOLDING LLC (Page 143)"
PROBLEM: Too short, no context, can't verify what this refers to

❌ EXAMPLE 5 - BAD (Summarized):
"Additional insured entity is listed in the schedule on page 67"
PROBLEM: Summary instead of exact quote

==================================================
CONTEXT FROM CERTIFICATE
==================================================

- Insured Name: {insured_name}
- Policy Number: {policy_number}
- Location Address: {location_address}

==================================================
CERTIFICATE ENTITIES TO FIND IN POLICY
==================================================

{json.dumps(cert_interests, indent=2) if cert_interests else "[] (No additional interests on certificate)"}

For EACH entity above, you must:
1. Search for it in the policy text below
2. Copy the exact text where it appears (15-30 words minimum)
3. Note the page number from the PAGE marker
4. Note the OCR source (TESSERACT or PYMUPDF)

==================================================
EVIDENCE EXTRACTION CHECKLIST
==================================================

Before submitting each validation, verify:

**IF YOU FOUND THE ENTITY (STATUS = MATCH or MISMATCH):**
□ I copied the text WORD-FOR-WORD (no paraphrasing)
□ My copied text includes at least 15-30 words of context
□ The entity name appears in my copied text
□ I got the page number from a PAGE marker in the text
□ I noted the OCR source (TESSERACT or PYMUPDF)
□ I included the section label (e.g., "Additional Insured:", "Mortgagee:")
□ Someone could find this exact text by searching the policy

**IF YOU DID NOT FIND THE ENTITY (STATUS = NOT_FOUND):**
□ I searched thoroughly through all the filtered text
□ I checked both TESSERACT and PYMUPDF sections
□ I set all policy_* fields to null
□ I set evidence to null
□ I set match_type to null
□ I did NOT invent or guess any information

==================================================
POLICY TEXT (FILTERED - ADDITIONAL INTERESTS SECTIONS)
==================================================

**⚠️ IMPORTANT: This is a FILTERED extract containing ONLY pages mentioning additional interests.**

The text below contains TWO OCR extraction sources per page:
- **--- TESSERACT (Buffer=1) ---** sections
- **--- PYMUPDF (Buffer=0) ---** sections

Use whichever source is clearer for your evidence extraction.

**SEARCH THIS FILTERED TEXT CAREFULLY:**

{filtered_policy_text}

==================================================
SEARCH PATTERNS TO LOOK FOR
==================================================

Common patterns where additional interests appear:

1. **Endorsement schedules (MOST COMMON):**
   "Name Of Person(s) Or Organization(s) (Additional Insured):" [entity name on next lines]
   "Additional Insured:" [entity name]
   
2. **Standalone schedules:**
   "SCHEDULE OF ADDITIONAL INSUREDS"
   "MORTGAGEE HOLDERS"
   "LOSS PAYEE SCHEDULE"
   
3. **Mortgagee/Loss Payee clauses:**
   "Mortgagee:" [entity name and address]
   "Loss Payable to:" [entity name and address]
   
4. **Endorsement forms:**
   Look for "CG 20" forms or endorsements with additional insured language

==================================================
NAME MATCHING RULES
==================================================

- **EXACT MATCH**: Names identical (case-insensitive) → status: "MATCH", match_type: "EXACT"
- **NAME VARIATION**: Names similar but NOT identical → status: "MISMATCH", match_type: "NAME_VARIATION"
  Examples: "GOLDING" vs "HOLDING", "SMITH" vs "SMITHE" (OCR errors)
  Even if likely same entity, names don't match exactly = MISMATCH
- **NOT FOUND**: Entity not in policy after thorough search → status: "NOT_FOUND", all fields null

**CRITICAL:** "TO WHOM IT MAY CONCERN" is NOT a valid additional interest

==================================================
OUTPUT FORMAT - FOLLOW EXACTLY
==================================================

Return ONLY a valid JSON object with this structure:

{{
  "additional_interests_validations": [
    {{
      "cert_interest_name": "Entity name from certificate",
      "cert_interest_address": "Address from certificate or null",
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "policy_interest_name": "Exact name from policy or null if NOT_FOUND",
      "policy_interest_address": "Address from policy or null if NOT_FOUND",
      "policy_interest_type": "MORTGAGEE | LOSS_PAYEE | ADDITIONAL_INSURED | LIENHOLDER | OTHER | null if NOT_FOUND",
      "match_type": "EXACT | NAME_VARIATION | null if NOT_FOUND",
      "evidence": "EXACT word-for-word quote (15-30 words minimum) from policy with (OCR_SOURCE, Page X) or null if NOT_FOUND",
      "notes": "Explanation of match/mismatch/not found decision"
    }}
  ],
  "summary": {{
    "total_additional_interests": 0,
    "additional_interests_matched": 0,
    "additional_interests_mismatched": 0,
    "additional_interests_not_found": 0
  }},
  "qc_notes": "Overall observations"
}}

==================================================
FINAL VERIFICATION BEFORE SUBMITTING
==================================================

For EACH validation, double-check:

**IF STATUS = "MATCH" OR "MISMATCH":**
✓ evidence contains EXACT copied text (not paraphrased)?
✓ evidence includes 15-30 words minimum?
✓ evidence includes page number from PAGE marker?
✓ evidence includes OCR source (TESSERACT or PYMUPDF)?
✓ policy_interest_name is filled in?
✓ I can find this exact text in the policy above if I search for it?

**IF STATUS = "NOT_FOUND":**
✓ evidence is null (not empty string)?
✓ policy_interest_name is null?
✓ policy_interest_address is null?
✓ policy_interest_type is null?
✓ match_type is null?
✓ I searched thoroughly before declaring NOT_FOUND?

**Remember: It's better to return NOT_FOUND than to hallucinate or guess.**

Return ONLY the JSON object now. No additional text or explanations.
"""
        
        return prompt
    
    def validate_additional_interests(self, cert_json_path: str, policy_combo_path: str, output_path: str):
        """
        Main validation workflow
        
        Args:
            cert_json_path: Path to certificate JSON file
            policy_combo_path: Path to policy combo text file
            output_path: Path for output JSON file
        """
        
        print(f"\n{'='*70}")
        print("ADDITIONAL INTERESTS COVERAGE VALIDATION")
        print(f"{'='*70}\n")
        
        # Load certificate
        print(f"[1/5] Loading certificate: {cert_json_path}")
        with open(cert_json_path, 'r', encoding='utf-8') as f:
            cert_data = json.load(f)
        
        # Extract additional interests to validate
        cert_interests = self.extract_additional_interests(cert_data)
        
        if not cert_interests:
            print("      ⚠️  No additional interests found in certificate!")
            print("      (This is valid - some certificates have no additional interests)")
            # Still create output with empty validations
            results = {
                "additional_interests_validations": [],
                "summary": {
                    "total_additional_interests": 0,
                    "additional_interests_matched": 0,
                    "additional_interests_mismatched": 0,
                    "additional_interests_not_found": 0
                },
                "qc_notes": "No additional interests on certificate to validate."
            }
            self.save_validation_results(results, output_path)
            return
        
        print(f"      Found {len(cert_interests)} additional interest(s) on certificate:")
        for i, interest in enumerate(cert_interests, 1):
            print(f"        {i}. {interest.get('name', 'N/A')}")
            if interest.get('address'):
                print(f"           Address: {interest.get('address')}")
        
        # Load policy
        print(f"\n[2/5] Loading policy: {policy_combo_path}")
        with open(policy_combo_path, 'r', encoding='utf-8') as f:
            policy_text = f.read()
        
        policy_size_kb = len(policy_text) / 1024
        print(f"      Policy size: {policy_size_kb:.1f} KB")
        
        # Filter policy pages for additional interests
        print(f"\n[3/5] Filtering policy pages for additional interests...")
        pages = _split_policy_combo_into_pages(policy_text)
        print(f"      Total pages: {len(pages)}")
        
        # Find all pages containing "additional insure*", "additional interests", "mortgagee", "loss payee", etc.
        ai_pages = _filter_pages_with_additional_insure(pages)
        print(f"      Pages with additional interests/mortgagee/loss payee keywords: {len(ai_pages)} pages")
        
        # Expand to include neighboring pages for context
        expanded = _expand_neighbors(ai_pages, radius=1)
        filtered_text = _build_filtered_policy_text(pages, expanded, max_pages=None)
        
        filtered_size_kb = len(filtered_text) / 1024
        reduction = ((len(pages) - len(expanded)) / len(pages) * 100) if pages else 0
        print(f"      After neighbor expansion: {len(expanded)} pages")
        print(f"      Filtered: {len(pages)} → {len(expanded)} pages ({reduction:.1f}% reduction)")
        print(f"      Filtered text size: {filtered_size_kb:.1f} KB")
        
        # Create prompt
        print(f"\n[4/5] Creating validation prompt...")
        prompt = self.create_validation_prompt(cert_data, cert_interests, filtered_text)
        prompt_size_kb = len(prompt) / 1024
        print(f"      Prompt size: {prompt_size_kb:.1f} KB")
        
        # Call LLM
        print(f"\n[5/5] Calling LLM for validation (model: {self.model})...")
        print(f"      Analyzing filtered policy sections for additional interests...")
        
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
                "pages_total": len(pages),
                "pages_with_additional_insure": ai_pages,
                "pages_expanded": expanded,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Post-process: detect name variations and mark as MISMATCH (even if likely OCR error)
            for validation in results.get("additional_interests_validations", []):
                cert_name = (validation.get("cert_interest_name") or "").upper()
                policy_name = (validation.get("policy_interest_name") or "").upper()
                
                if cert_name and policy_name and cert_name != policy_name:
                    # Check if it's a name variation (likely OCR error)
                    if self._is_name_variation(cert_name, policy_name):
                        # Mark as MISMATCH (names don't match exactly, even if likely same entity)
                        validation["status"] = "MISMATCH"
                        validation["match_type"] = "NAME_VARIATION"
                        if not validation.get("notes"):
                            validation["notes"] = f"Name mismatch detected (likely OCR error): Certificate has '{validation.get('cert_interest_name')}' but policy has '{validation.get('policy_interest_name')}'. Names don't match exactly."
            
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
        print(f"\n[6/6] Saving results to: {output_path}")
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"      ✓ Results saved")
    
    def display_results(self, results: Dict):
        """Display validation results on console"""
        print(f"\n{'='*70}")
        print("ADDITIONAL INTERESTS VALIDATION RESULTS")
        print(f"{'='*70}\n")
        
        validations = results.get('additional_interests_validations', [])
        if not validations:
            print("No additional interests to validate.")
            return
        
        for v in validations:
            status = v.get('status', 'UNKNOWN')
            cert_name = v.get('cert_interest_name', 'N/A')
            cert_address = v.get('cert_interest_address', 'N/A')
            policy_name = v.get('policy_interest_name', 'N/A')
            policy_address = v.get('policy_interest_address', 'N/A')
            policy_type = v.get('policy_interest_type', 'N/A')
            match_type = v.get('match_type', 'N/A')
            evidence = v.get('evidence', 'N/A')
            notes = v.get('notes', 'N/A')
            
            if status == 'MATCH':
                icon = '✓'
            elif status == 'MISMATCH':
                icon = '✗'
            else:
                icon = '?'
            
            print(f"{icon} {cert_name}")
            print(f"  Status: {status}")
            if match_type and match_type != 'N/A':
                print(f"  Match Type: {match_type}")
            print(f"  Certificate: {cert_name}")
            if cert_address and cert_address != 'N/A':
                print(f"    Address: {cert_address}")
            print(f"  Policy: {policy_name if policy_name != 'N/A' else 'Not found'}")
            if policy_address and policy_address != 'N/A':
                print(f"    Address: {policy_address}")
            if policy_type and policy_type != 'N/A':
                print(f"    Type: {policy_type}")
            
            if evidence and evidence != 'N/A':
                if len(evidence) > 140:
                    evidence = evidence[:137] + "..."
                print(f"  Evidence: {evidence}")
            
            if notes and notes != 'N/A':
                if len(notes) > 170:
                    notes = notes[:167] + "..."
                print(f"  Notes: {notes}")
            print()
        
        # Print summary
        summary = results.get('summary', {})
        print(f"{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Total Additional Interests:  {summary.get('total_additional_interests', 0)}")
        print(f"  ✓ Matched:      {summary.get('additional_interests_matched', 0)}")
        print(f"  ✗ Mismatched:   {summary.get('additional_interests_mismatched', 0)}")
        print(f"  ? Not Found:    {summary.get('additional_interests_not_found', 0)}")
        
        if 'qc_notes' in results:
            qc_notes = results['qc_notes']
            if len(qc_notes) > 200:
                qc_notes = qc_notes[:197] + "..."
            print(f"\nQC Notes: {qc_notes}")
        
        print(f"{'='*70}\n")


def main():
    """Main execution function"""
    # ========== EDIT THESE VALUES ==========
    cert_prefix = "arrr"              # Change to: westside, james, etc.
    carrier_dir = "usgnonop"            # Change to: nationwideop, hartfordop, etc.
    # =======================================
    
    # Construct paths
    cert_json_path = os.path.join(carrier_dir, f"{cert_prefix}_pl_extracted_real.json")
    policy_combo_path = os.path.join(carrier_dir, f"{cert_prefix}_pol_combo.txt")
    output_path = os.path.join(carrier_dir, f"{cert_prefix}_additional_interests_validation.json")
    
    # Check if files exist
    if not os.path.exists(cert_json_path):
        print(f"Error: Certificate JSON not found: {cert_json_path}")
        exit(1)
    
    if not os.path.exists(policy_combo_path):
        print(f"Error: Policy combo text not found: {policy_combo_path}")
        exit(1)
    
    # Create validator and run
    try:
        validator = AdditionalInterestsCoverageValidator()
        validator.validate_additional_interests(cert_json_path, policy_combo_path, output_path)
    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

