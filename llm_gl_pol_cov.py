"""
LLM-Based GL (ACORD 25) Coverage Validation

Validates selected GL LIMITS from a GL certificate extraction JSON against the policy document OCR
combo text using a SINGLE LLM call.

Current scope (limits only):
- Each Occurrence
- Damage to Rented Premises (Ea occurrence)
- Med Exp (Any one person)  (can be Excluded / $0 / blank)
- Personal & Advertising Injury
- General Aggregate
- Products - Comp/Op Agg

Also supports (to avoid label collisions like "Each Occurrence" / "Aggregate" appearing multiple times):
- Umbrella/Excess: Each Occurrence, Aggregate
- Employment Practices Liability: Each Limit, Aggregate Limit
"""

import json
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class GLLimitsValidator:
    """Validate GL certificate limit fields against policy text (single LLM call)."""

    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def extract_cgl_limits(self, cert_data: Dict) -> List[Dict]:
        """
        Extract relevant CGL limits from GL certificate extraction JSON.
        Expected structure (from llm_gl.py):
          cert_data["coverages"]["commercial_general_liability"]["limits"][...]
        """
        coverages = cert_data.get("coverages", {}) or {}
        cgl = coverages.get("commercial_general_liability", {}) or {}
        limits = cgl.get("limits", {}) or {}

        def _clean(v: Optional[str]) -> Optional[str]:
            if v is None:
                return None
            s = str(v).strip()
            if not s:
                return None
            # OCR sometimes returns "$" or "0" placeholders
            if s in {"$", "$0.00", "$ 0.00"}:
                return "$0"
            return s

        items: List[Dict] = []

        mapping = [
            ("each_occurrence", "Each Occurrence"),
            ("damage_to_rented_premises", "Damage to Rented Premises (Ea occurrence)"),
            ("med_exp", "Med Exp (Any one person)"),
            ("personal_adv_injury", "Personal & Adv Injury"),
            ("general_aggregate", "General Aggregate"),
            ("products_comp_op_agg", "Products - Comp/Op Agg"),
        ]

        for key, label in mapping:
            v = _clean(limits.get(key))
            # keep even "$0" (excluded) if present; if truly missing/blank, skip to avoid inventing
            if v is not None:
                items.append(
                    {
                        "coverage_section": "commercial_general_liability",
                        "limit_key": key,
                        "limit_label": label,
                        "value": v,
                    }
                )

        return items

    def extract_umbrella_limits(self, cert_data: Dict) -> List[Dict]:
        """
        Extract Umbrella/Excess limits from certificate extraction JSON.
        Expected structure (from llm_gl.py):
          cert_data["coverages"]["umbrella_liability"]["limits"][...]
        """
        coverages = cert_data.get("coverages", {}) or {}
        umb = coverages.get("umbrella_liability", {}) or coverages.get("excess_liability", {}) or {}
        limits = umb.get("limits", {}) or {}

        def _clean(v: Optional[str]) -> Optional[str]:
            if v is None:
                return None
            s = str(v).strip()
            if not s:
                return None
            if s in {"$", "$0.00", "$ 0.00"}:
                return "$0"
            return s

        mapping = [
            ("each_occurrence", "Umbrella Each Occurrence"),
            ("aggregate", "Umbrella Aggregate"),
        ]

        items: List[Dict] = []
        for key, label in mapping:
            v = _clean(limits.get(key))
            if v is not None:
                items.append(
                    {
                        "coverage_section": "umbrella_liability",
                        "limit_key": key,
                        "limit_label": label,
                        "value": v,
                    }
                )
        return items

    def extract_epl_limits(self, cert_data: Dict) -> List[Dict]:
        """
        Extract Employment Practices Liability limits (Each Limit / Aggregate Limit) from certificate.
        Expected structure (from llm_gl.py):
          cert_data["coverages"]["employment_practices_liability"]["limits"][...]
        """
        coverages = cert_data.get("coverages", {}) or {}
        epl = coverages.get("employment_practices_liability", {}) or {}
        limits = epl.get("limits", {}) or {}

        def _clean(v: Optional[str]) -> Optional[str]:
            if v is None:
                return None
            s = str(v).strip()
            if not s:
                return None
            if s in {"$", "$0.00", "$ 0.00"}:
                return "$0"
            return s

        mapping = [
            ("each_limit", "EPL Each Limit"),
            ("aggregate_limit", "EPL Aggregate Limit"),
        ]

        items: List[Dict] = []
        for key, label in mapping:
            v = _clean(limits.get(key))
            if v is not None:
                items.append(
                    {
                        "coverage_section": "employment_practices_liability",
                        "limit_key": key,
                        "limit_label": label,
                        "value": v,
                    }
                )
        return items

    def extract_liquor_limits(self, cert_data: Dict) -> List[Dict]:
        """
        Extract Liquor Liability limits (Each Limit / Aggregate Limit) from certificate.
        Expected structure (from llm_gl.py):
          cert_data["coverages"]["liquor_liability"]["limits"][...]
        """
        coverages = cert_data.get("coverages", {}) or {}
        liquor = coverages.get("liquor_liability", {}) or {}
        limits = liquor.get("limits", {}) or {}

        def _clean(v: Optional[str]) -> Optional[str]:
            if v is None:
                return None
            s = str(v).strip()
            if not s:
                return None
            if s in {"$", "$0.00", "$ 0.00"}:
                return "$0"
            return s

        mapping = [
            ("each_limit", "Liquor Liability Each Limit"),
            ("aggregate_limit", "Liquor Liability Aggregate Limit"),
        ]

        items: List[Dict] = []
        for key, label in mapping:
            v = _clean(limits.get(key))
            if v is not None:
                items.append(
                    {
                        "coverage_section": "liquor_liability",
                        "limit_key": key,
                        "limit_label": label,
                        "value": v,
                    }
                )
        return items

    def _norm_name(self, s: Optional[str]) -> str:
        if not s:
            return ""
        s = s.lower()
        return "".join(ch for ch in s if ch.isalnum())

    def _filter_validations_to_requested(
        self, validations: List[Dict], requested_items: List[Dict], key_field: str
    ) -> List[Dict]:
        if not requested_items:
            return []
        requested_norms = [self._norm_name((it or {}).get(key_field)) for it in requested_items]
        requested_norms = [x for x in requested_norms if x]
        if not requested_norms:
            return []

        filtered: List[Dict] = []
        for v in validations or []:
            k = self._norm_name((v or {}).get("cert_limit_key"))
            if not k:
                continue
            if any(r in k or k in r for r in requested_norms):
                filtered.append(v)

        if not filtered:
            return list((validations or [])[: len(requested_items)])
        if len(filtered) > len(requested_items):
            filtered = filtered[: len(requested_items)]
        return filtered

    def _recompute_summary_counts(self, results: Dict) -> None:
        def _count(arr: List[Dict]) -> Dict[str, int]:
            t = 0
            m = 0
            mm = 0
            nf = 0
            for v in arr or []:
                t += 1
                s = (v.get("status") or "").upper()
                if s == "MATCH":
                    m += 1
                elif s == "MISMATCH":
                    mm += 1
                elif s == "NOT_FOUND":
                    nf += 1
            return {"total": t, "matched": m, "mismatched": mm, "not_found": nf}

        cgl = _count(results.get("cgl_limit_validations", []))
        umb = _count(results.get("umbrella_limit_validations", []))
        epl = _count(results.get("epl_limit_validations", []))
        liquor = _count(results.get("liquor_limit_validations", []))

        results["summary"] = {
            "total_limits": cgl["total"] + umb["total"] + epl["total"] + liquor["total"],
            "matched": cgl["matched"] + umb["matched"] + epl["matched"] + liquor["matched"],
            "mismatched": cgl["mismatched"] + umb["mismatched"] + epl["mismatched"] + liquor["mismatched"],
            "not_found": cgl["not_found"] + umb["not_found"] + epl["not_found"] + liquor["not_found"],
            "total_cgl_limits": cgl["total"],
            "total_umbrella_limits": umb["total"],
            "total_epl_limits": epl["total"],
            "total_liquor_limits": liquor["total"],
        }

    def create_validation_prompt(
        self,
        cert_data: Dict,
        cgl_items: List[Dict],
        umbrella_items: List[Dict],
        epl_items: List[Dict],
        liquor_items: List[Dict],
        policy_text: str,
    ) -> str:
        insured_name = cert_data.get("insured_name", "Not specified")
        mailing_address = cert_data.get("mailing_address", "Not specified")
        location_address = cert_data.get("location_address", None)

        coverages = cert_data.get("coverages", {}) or {}
        cgl = coverages.get("commercial_general_liability", {}) or {}
        umb = coverages.get("umbrella_liability", {}) or coverages.get("excess_liability", {}) or {}
        epl = coverages.get("employment_practices_liability", {}) or {}
        liquor = coverages.get("liquor_liability", {}) or {}

        prompt = f"""You are an expert Commercial General Liability (CGL) QC Specialist.

Return ONLY valid JSON.

==================================================
TASK
==================================================
Validate the following GL LIMITS from the GL certificate against the policy document:

1) Commercial General Liability (CGL) limits:
- Each Occurrence
- Damage to Rented Premises (Ea occurrence)
- Med Exp (Any one person) (may be Excluded / $0)
- Personal & Advertising Injury
- General Aggregate
- Products - Comp/Op Agg

2) Umbrella/Excess limits (if present on the certificate):
- Each Occurrence
- Aggregate

3) Employment Practices Liability limits (if present on the certificate):
- Each Limit
- Aggregate Limit

4) Liquor Liability limits (if present on the certificate):
- Each Limit
- Aggregate Limit

IMPORTANT:
- Validate LIMITS only. Ignore deductibles except as context.
- The same labels may appear in multiple sections. You MUST match each requested item within its correct coverage section:
  - CGL "Each Occurrence" is NOT Umbrella "Each Occurrence".
  - CGL "General Aggregate" is NOT Umbrella "Aggregate".
  - EPL "Each Limit/Aggregate Limit" is NOT CGL limits.
  - Liquor Liability "Each Limit/Aggregate Limit" is NOT EPL limits (they are separate coverages).
- "Med Exp" may be shown as "$0", "0", "Excluded", or blank on the certificate/policy. Treat "$0"/"0"/"Excluded" as equivalent.
- Formatting differences are not mismatches: "1,000,000" == "$1,000,000" == "$ 1,000,000".

==================================================
CERTIFICATE CONTEXT
==================================================
Insured Name: {insured_name}
Mailing Address: {mailing_address}
Location Address (if shown): {location_address if location_address else "Not shown"}

Commercial General Liability (from certificate extraction): 
{json.dumps(cgl, indent=2)}

CGL LIMITS TO VALIDATE (ONLY THESE):
{json.dumps(cgl_items, indent=2)}

Umbrella/Excess (from certificate extraction):
{json.dumps(umb, indent=2)}

UMBRELLA LIMITS TO VALIDATE (ONLY THESE):
{json.dumps(umbrella_items, indent=2)}

Employment Practices Liability (from certificate extraction):
{json.dumps(epl, indent=2)}

EPL LIMITS TO VALIDATE (ONLY THESE):
{json.dumps(epl_items, indent=2)}

Liquor Liability (from certificate extraction):
{json.dumps(liquor, indent=2)}

LIQUOR LIABILITY LIMITS TO VALIDATE (ONLY THESE):
{json.dumps(liquor_items, indent=2)}

==================================================
POLICY DOCUMENT (DUAL OCR)
==================================================
This policy combo text includes page separators and OCR source markers:
- TESSERACT (Buffer=1)
- PYMUPDF (Buffer=0)

Use whichever is clearer. ALWAYS cite the OCR source + page number in evidence fields.

{policy_text}

==================================================
OUTPUT FORMAT
==================================================
Return ONLY this JSON object:

{{
  "cgl_limit_validations": [
    {{
      "cert_limit_key": "each_occurrence | damage_to_rented_premises | med_exp | personal_adv_injury | general_aggregate | products_comp_op_agg",
      "cert_limit_label": "Label from the request",
      "cert_value": "Value from certificate (e.g., '$1,000,000' or '$0' or 'Excluded')",
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "policy_value": "Value from policy (or 'Excluded' / '$0' if shown) or null",
      "policy_location_context": "Premises/location context if relevant, else null",
      "evidence_declarations": "Quote showing the limit (OCR_SOURCE, Page X) or null",
      "evidence_endorsements": "Quote from endorsement changing the limit (OCR_SOURCE, Page X) or null",
      "notes": "Explain how you found it and why MATCH/MISMATCH/NOT_FOUND."
    }}
  ],
  "umbrella_limit_validations": [
    {{
      "cert_limit_key": "each_occurrence | aggregate",
      "cert_limit_label": "Label from the request (e.g., 'Umbrella Each Occurrence')",
      "cert_value": "Value from certificate",
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "policy_value": "Value from policy or null",
      "evidence_declarations": "Quote showing the limit (OCR_SOURCE, Page X) or null",
      "evidence_endorsements": "Quote from endorsement changing the limit (OCR_SOURCE, Page X) or null",
      "notes": "Explain why MATCH/MISMATCH/NOT_FOUND and confirm it is Umbrella/Excess (not CGL)."
    }}
  ],
  "epl_limit_validations": [
    {{
      "cert_limit_key": "each_limit | aggregate_limit",
      "cert_limit_label": "Label from the request (e.g., 'EPL Each Limit')",
      "cert_value": "Value from certificate",
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "policy_value": "Value from policy or null",
      "evidence_declarations": "Quote showing the limit (OCR_SOURCE, Page X) or null",
      "evidence_endorsements": "Quote from endorsement changing the limit (OCR_SOURCE, Page X) or null",
      "notes": "Explain why MATCH/MISMATCH/NOT_FOUND and confirm it is Employment Practices Liability (not CGL)."
    }}
  ],
  "liquor_limit_validations": [
    {{
      "cert_limit_key": "each_limit | aggregate_limit",
      "cert_limit_label": "Label from the request (e.g., 'Liquor Liability Each Limit')",
      "cert_value": "Value from certificate",
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "policy_value": "Value from policy or null",
      "evidence_declarations": "Quote showing the limit (OCR_SOURCE, Page X) or null",
      "evidence_endorsements": "Quote from endorsement changing the limit (OCR_SOURCE, Page X) or null",
      "notes": "Explain why MATCH/MISMATCH/NOT_FOUND and confirm it is Liquor Liability (not EPL or CGL)."
    }}
  ],
  "summary": {{
    "total_limits": 0,
    "matched": 0,
    "mismatched": 0,
    "not_found": 0,
    "total_cgl_limits": 0,
    "total_umbrella_limits": 0,
    "total_epl_limits": 0,
    "total_liquor_limits": 0
  }},
  "qc_notes": "Overall observations (optional)"
}}
"""
        return prompt

    def validate_limits(self, cert_json_path: str, policy_combo_path: str, output_path: str) -> None:
        print("\n" + "=" * 70)
        print("GL LIMIT VALIDATION (CGL + UMBRELLA + EPL + LIQUOR)")
        print("=" * 70 + "\n")

        print(f"[1/5] Loading certificate JSON: {cert_json_path}")
        with open(cert_json_path, "r", encoding="utf-8") as f:
            cert_data = json.load(f)

        cgl_items = self.extract_cgl_limits(cert_data)
        umbrella_items = self.extract_umbrella_limits(cert_data)
        epl_items = self.extract_epl_limits(cert_data)
        liquor_items = self.extract_liquor_limits(cert_data)

        if not cgl_items and not umbrella_items and not epl_items and not liquor_items:
            print("      ❌ No supported GL limit items found in certificate extraction JSON.")
            print("      Ensure llm_gl.py extracted limits under coverages.commercial_general_liability / umbrella_liability / employment_practices_liability / liquor_liability.")
            return

        if cgl_items:
            print(f"      Found {len(cgl_items)} CGL limit item(s):")
            for it in cgl_items:
                print(f"        - {it['limit_label']}: {it['value']}")
        if umbrella_items:
            print(f"      Found {len(umbrella_items)} Umbrella limit item(s):")
            for it in umbrella_items:
                print(f"        - {it['limit_label']}: {it['value']}")
        if epl_items:
            print(f"      Found {len(epl_items)} EPL limit item(s):")
            for it in epl_items:
                print(f"        - {it['limit_label']}: {it['value']}")
        if liquor_items:
            print(f"      Found {len(liquor_items)} Liquor Liability limit item(s):")
            for it in liquor_items:
                print(f"        - {it['limit_label']}: {it['value']}")

        print(f"\n[2/5] Loading policy combo text: {policy_combo_path}")
        with open(policy_combo_path, "r", encoding="utf-8") as f:
            policy_text = f.read()
        print(f"      Policy size: {len(policy_text) / 1024:.1f} KB")

        print("\n[3/5] Creating validation prompt...")
        prompt = self.create_validation_prompt(cert_data, cgl_items, umbrella_items, epl_items, liquor_items, policy_text)
        print(f"      Prompt size: {len(prompt) / 1024:.1f} KB")

        print(f"\n[4/5] Calling LLM for validation (model: {self.model})...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert GL insurance QC specialist. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        result_text = response.choices[0].message.content
        results = json.loads(result_text)

        # Guardrail: keep only validations we requested from the certificate
        results["cgl_limit_validations"] = self._filter_validations_to_requested(
            results.get("cgl_limit_validations", []),
            cgl_items,
            "limit_key",
        )
        results["umbrella_limit_validations"] = self._filter_validations_to_requested(
            results.get("umbrella_limit_validations", []),
            umbrella_items,
            "limit_key",
        )
        results["epl_limit_validations"] = self._filter_validations_to_requested(
            results.get("epl_limit_validations", []),
            epl_items,
            "limit_key",
        )
        results["liquor_limit_validations"] = self._filter_validations_to_requested(
            results.get("liquor_limit_validations", []),
            liquor_items,
            "limit_key",
        )
        self._recompute_summary_counts(results)

        results["metadata"] = {
            "model": self.model,
            "certificate_file": cert_json_path,
            "policy_file": policy_combo_path,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        print(f"      ✓ LLM validation complete")
        print(
            f"      Tokens used: {response.usage.total_tokens:,} "
            f"(prompt: {response.usage.prompt_tokens:,}, completion: {response.usage.completion_tokens:,})"
        )

        print(f"\n[5/5] Saving results to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("      ✓ Results saved\n")

        self.display_results(results)
        print("✓ Validation completed successfully!")

    def display_results(self, results: Dict) -> None:
        def _print_section(title: str, arr: List[Dict]) -> None:
            if not arr:
                return
            print("=" * 70)
            print(title)
            print("=" * 70 + "\n")
            for v in arr:
                status = v.get("status", "UNKNOWN")
                label = v.get("cert_limit_label", v.get("cert_limit_key", "N/A"))
                cert_value = v.get("cert_value", "N/A")
                policy_value = v.get("policy_value", "N/A")
                evidence_decl = v.get("evidence_declarations", None)
                evidence_end = v.get("evidence_endorsements", None)
                notes = v.get("notes", "")

                if status == "MATCH":
                    icon = "✓"
                elif status == "MISMATCH":
                    icon = "✗"
                else:
                    icon = "?"

                print(f"{icon} {label}")
                print(f"  Status: {status}")
                print(f"  Certificate Value: {cert_value}")
                print(f"  Policy Value: {policy_value}")
                if evidence_decl:
                    e = evidence_decl
                    if len(e) > 140:
                        e = e[:137] + "..."
                    print(f"  Evidence (Declarations): {e}")
                if evidence_end:
                    e = evidence_end
                    if len(e) > 140:
                        e = e[:137] + "..."
                    print(f"  Evidence (Endorsements): {e}")
                if notes:
                    n = notes
                    if len(n) > 180:
                        n = n[:177] + "..."
                    print(f"  Notes: {n}")
                print()

        _print_section("CGL LIMIT VALIDATION RESULTS", results.get("cgl_limit_validations", []) or [])
        _print_section("UMBRELLA LIMIT VALIDATION RESULTS", results.get("umbrella_limit_validations", []) or [])
        _print_section("EPL LIMIT VALIDATION RESULTS", results.get("epl_limit_validations", []) or [])
        _print_section("LIQUOR LIABILITY LIMIT VALIDATION RESULTS", results.get("liquor_limit_validations", []) or [])

        summary = results.get("summary", {}) or {}
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total Limits:  {summary.get('total_limits', 0)}")
        print(f"  ✓ Matched:      {summary.get('matched', 0)}")
        print(f"  ✗ Mismatched:   {summary.get('mismatched', 0)}")
        print(f"  ? Not Found:    {summary.get('not_found', 0)}")
        if "total_cgl_limits" in summary:
            print(f"\nTotal CGL Limits:      {summary.get('total_cgl_limits', 0)}")
        if "total_umbrella_limits" in summary:
            print(f"Total Umbrella Limits: {summary.get('total_umbrella_limits', 0)}")
        if "total_epl_limits" in summary:
            print(f"Total EPL Limits:      {summary.get('total_epl_limits', 0)}")
        if "total_liquor_limits" in summary:
            print(f"Total Liquor Limits:   {summary.get('total_liquor_limits', 0)}")

        qc_notes = results.get("qc_notes", None)
        if qc_notes:
            if len(qc_notes) > 220:
                qc_notes = qc_notes[:217] + "..."
            print(f"\nQC Notes: {qc_notes}")

        print("=" * 70 + "\n")


def main() -> None:
    # ========== EDIT THESE VALUES ==========
    carrier_dir = "nationwideop"        # encovaop, hartfordop, nationwideop, travelerop, ...
    cert_prefix = "westside_gl"       # e.g. aaniya_gl, ambama_gl, evergreen_gl
    policy_prefix = "westside"        # base name used for policy combo, e.g. aaniya -> aaniya_pol_combo.txt
    # =======================================

    cert_json_path = os.path.join(carrier_dir, f"{cert_prefix}_extracted_real.json")
    policy_combo_path = os.path.join(carrier_dir, f"{policy_prefix}_pol_combo.txt")
    output_path = os.path.join(carrier_dir, f"{policy_prefix}_gl_limits_validation.json")

    if not os.path.exists(cert_json_path):
        print(f"Error: Certificate JSON not found: {cert_json_path}")
        return
    if not os.path.exists(policy_combo_path):
        print(f"Error: Policy combo text not found: {policy_combo_path}")
        print("Hint: run policy_extract.py + policy_filter.py + combine_extractions.py to produce *_pol_combo.txt")
        return

    validator = GLLimitsValidator()
    validator.validate_limits(cert_json_path, policy_combo_path, output_path)


if __name__ == "__main__":
    main()


