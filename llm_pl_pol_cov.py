"""
LLM-Based Coverage Validation - Building Coverage Only
Validates Building coverage values from certificate against policy document
Handles multiple buildings using location address context
"""

import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


class BuildingCoverageValidator:
    """Validate Property coverages from certificate against policy (single LLM call)."""
    
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
            n = (coverage_name or "").lower()
            # Avoid double-counting special combined labels (handled in Pumps/Canopy validation)
            is_building_with_pumps_canopy = (
                "building" in n and "pump" in n and "canopy" in n
            )
            if "building" in n and not is_building_with_pumps_canopy:
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

    def extract_outdoor_signs_coverages(self, cert_data: Dict) -> List[Dict]:
        """
        Extract Outdoor Signs coverages from certificate.

        Notes:
        - Can be "Included" / "Yes" or a dollar limit
        - Wording varies: "Outdoor Signs", "Signs", "Outdoor sign(s)"
        - Avoid confusing with "signs you must display" type policy language by requiring "sign" to appear as a coverage name from cert.
        """
        coverages = cert_data.get("coverages", {}) or {}
        os_items: List[Dict] = []

        for coverage_name, coverage_value in coverages.items():
            name = (coverage_name or "").strip()
            n = name.lower()

            is_outdoor_signs = (
                "outdoor sign" in n
                or "outdoor signs" in n
                or (n == "signs")
                or (n.startswith("signs ") or n.endswith(" signs"))
            )

            # Exclude non-coverage details if they appear as keys
            is_excluded = any(
                kw in n
                for kw in [
                    "deductible",
                    "ded.",
                    "coinsurance",
                    "waiting period",
                    "waiting",
                    "description",
                ]
            )

            if is_outdoor_signs and not is_excluded:
                os_items.append({"name": name, "value": coverage_value})

        return os_items

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

    def extract_pumps_canopy_coverages(self, cert_data: Dict) -> List[Dict]:
        """
        Extract Pumps / Canopy related coverages from certificate, supporting:
        - Separate: "Pumps", "Canopy"
        - Combined: "Pumps & Canopy" (or "Pumps and Canopy")
        - Combined with Building: "Building with Pumps & Canopy"
        """
        coverages = cert_data.get("coverages", {}) or {}

        # Track presence to apply precedence rules (prefer combined label if present)
        combined_building_key = None
        combined_pc_key = None
        pumps_key = None
        canopy_key = None

        for coverage_name in coverages.keys():
            name = (coverage_name or "").strip()
            n = name.lower()

            if "building" in n and "pump" in n and "canopy" in n:
                combined_building_key = coverage_name
                continue

            # Combined pumps+canopy label
            if ("pump" in n and "canopy" in n) and ("building" not in n):
                # e.g. "Pumps & Canopy"
                combined_pc_key = coverage_name
                continue

            # Separate
            if n in ("pumps", "pump") or n.startswith("pumps ") or n.endswith(" pumps"):
                pumps_key = coverage_name
                continue

            if n in ("canopy", "canopies") or n.startswith("canopy ") or n.endswith(" canopy") or n.endswith(" canopies"):
                canopy_key = coverage_name
                continue

        # Precedence:
        # 1) If "Building with Pumps & Canopy" exists, validate only that combined item (avoid double-counting components)
        # 2) Else if "Pumps & Canopy" exists, validate that combined item (components may still exist, but we avoid duplicates)
        # 3) Else validate separate Pumps/Canopy if present
        items: List[Dict] = []

        if combined_building_key:
            items.append({"name": combined_building_key, "value": coverages.get(combined_building_key)})
            return items

        if combined_pc_key:
            items.append({"name": combined_pc_key, "value": coverages.get(combined_pc_key)})
            return items

        if pumps_key:
            items.append({"name": pumps_key, "value": coverages.get(pumps_key)})
        if canopy_key:
            items.append({"name": canopy_key, "value": coverages.get(canopy_key)})

        return items

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

    def extract_business_income_coverages(self, cert_data: Dict) -> List[Dict]:
        """
        Extract Business Income coverages from certificate.

        Notes:
        - May appear once or repeated for multiple locations/buildings.
        - Value can be a dollar limit (e.g., "150,000") or "Actual Loss Sustained".
        - Avoid deductible rows/keys.
        """
        coverages = cert_data.get("coverages", {}) or {}
        items: List[Dict] = []

        for coverage_name, coverage_value in coverages.items():
            name = (coverage_name or "").strip()
            n = name.lower()

            if "business income" not in n:
                continue

            # Exclude deductibles or waiting period-only lines if they appear as keys
            if "deductible" in n or "ded." in n or "waiting" in n:
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

        b = _count(results.get("building_validations", []))
        summary.update(
            {
                "total_buildings": b["total"],
                "matched": b["match"],
                "mismatched": b["mismatch"],
                "not_found": b["not_found"],
            }
        )

        bpp = _count(results.get("bpp_validations", []))
        summary.update(
            {
                "total_bpp_items": bpp["total"],
                "bpp_matched": bpp["match"],
                "bpp_mismatched": bpp["mismatch"],
                "bpp_not_found": bpp["not_found"],
            }
        )

        bi = _count(results.get("business_income_validations", []))
        summary.update(
            {
                "total_bi_items": bi["total"],
                "bi_matched": bi["match"],
                "bi_mismatched": bi["mismatch"],
                "bi_not_found": bi["not_found"],
            }
        )

        ms = _count(results.get("money_securities_validations", []))
        summary.update(
            {
                "total_ms_items": ms["total"],
                "ms_matched": ms["match"],
                "ms_mismatched": ms["mismatch"],
                "ms_not_found": ms["not_found"],
            }
        )

        eb = _count(results.get("equipment_breakdown_validations", []))
        summary.update(
            {
                "total_eb_items": eb["total"],
                "eb_matched": eb["match"],
                "eb_mismatched": eb["mismatch"],
                "eb_not_found": eb["not_found"],
            }
        )

        os = _count(results.get("outdoor_signs_validations", []))
        summary.update(
            {
                "total_os_items": os["total"],
                "os_matched": os["match"],
                "os_mismatched": os["mismatch"],
                "os_not_found": os["not_found"],
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

        pc = _count(results.get("pumps_canopy_validations", []))
        summary.update(
            {
                "total_pc_items": pc["total"],
                "pc_matched": pc["match"],
                "pc_mismatched": pc["mismatch"],
                "pc_not_found": pc["not_found"],
            }
        )

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
    
    def create_validation_prompt(self, cert_data: Dict, buildings: List[Dict], bpp_items: List[Dict], bi_items: List[Dict], ms_items: List[Dict], eb_items: List[Dict], os_items: List[Dict], ed_items: List[Dict], pc_items: List[Dict], theft_items: List[Dict], wind_hail_items: List[Dict], policy_text: str) -> str:
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
Validate BUILDING, Business Personal Property (BPP), Business Income, Money & Securities, Equipment Breakdown, Outdoor Signs, Employee Dishonesty, Pumps/Canopy, Theft, and Wind/Hail (Windstorm & Hail) from the certificate against the policy document.

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

**BUSINESS INCOME COVERAGES TO VALIDATE:**
{json.dumps(bi_items, indent=2)}

**MONEY & SECURITIES COVERAGES TO VALIDATE:**
{json.dumps(ms_items, indent=2)}

**EQUIPMENT BREAKDOWN COVERAGES TO VALIDATE:**
{json.dumps(eb_items, indent=2)}

**OUTDOOR SIGNS COVERAGES TO VALIDATE:**
{json.dumps(os_items, indent=2)}

**EMPLOYEE DISHONESTY COVERAGES TO VALIDATE:**
{json.dumps(ed_items, indent=2)}

**PUMPS / CANOPY COVERAGES TO VALIDATE:**
{json.dumps(pc_items, indent=2)}

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
BUSINESS INCOME VALIDATION RULES (STRICT)
==================================================

For EACH Business Income item (STRICT LOCATION MATCHING):
- Business Income may be listed per location/building in the certificate (e.g., repeated for Location 01 and Location 02). You MUST match the correct premises/building context in the policy.
- If certificate value is "Actual Loss Sustained" (or similar like "A.L.S."):
  - MATCH if the policy indicates Business Income is Actual Loss Sustained (or no stated dollar limit and clearly ALS form applies) for that location/building.
  - MISMATCH if the policy clearly shows a specific dollar limit and it conflicts with ALS representation.
- If certificate value is a dollar limit:
  - MATCH if the policy's Business Income limit matches for that location/building (ignore $/commas/spacing).
- If the policy lists Business Income as part of a combined "Business Income and Extra Expense" or similar, capture the effective BI representation and explain in notes.
- Waiting period/deductible supports inclusion but is NOT the BI limit; capture it as context.
- Do NOT confuse Business Income with:
  - Extra Expense only
  - Rental Value (unless the certificate explicitly says it)
  - Waiting period / deductible entries (these are not the limit)

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
OUTDOOR SIGNS VALIDATION RULES (STRICT)
==================================================

For EACH Outdoor Signs item:
- Certificate value may be "Included" / "Yes" or a dollar limit (e.g., 10,000 / 25,000 / 50,000).
- MATCH rules:
  - If certificate is "Included"/"Yes": MATCH if policy indicates Outdoor Signs are covered/included OR shows a limit for Outdoor Signs (it can still be "included" but expressed as a limit).
  - If certificate is a dollar limit: MATCH only if the policy’s Outdoor Signs limit matches (ignore $/commas/spacing).
- Do NOT confuse Outdoor Signs with:
  - Premises/operations signage text, posting requirements, general "sign" mentions
  - Other property coverages that mention signs as part of wording
- Evidence must cite declarations/coverage schedule or the specific coverage form/endorsement and include OCR source + page.

==================================================
EMPLOYEE DISHONESTY VALIDATION RULES (STRICT)
==================================================

For EACH Employee Dishonesty item:
- Certificate value may be "Included" / "Yes" or a dollar limit.
- Prefer evidence from declarations, schedules of coverages/optional coverages, or the specific crime endorsement/form where the limit is shown.
- MATCH rules:
  - If certificate is "Included"/"Yes": MATCH if policy indicates Employee Dishonesty is covered/included OR shows a limit for Employee Dishonesty.
  - If certificate is a dollar limit: MATCH only if the policy’s Employee Dishonesty limit matches (ignore $/commas/spacing).
- Do NOT confuse with:
  - Forgery/Alteration
  - Money Orders & Counterfeit Money
  - Computer Fraud / Funds Transfer Fraud
  - Other crime/cyber coverages that are not Employee Dishonesty
- Evidence must include OCR source + page number.

==================================================
PUMPS / CANOPY VALIDATION RULES (STRICT)
==================================================

This coverage family can appear in multiple equivalent representations across certificates/policies:
- Separate: "Pumps" and "Canopy"
- Combined: "Pumps & Canopy"
- Combined with Building: "Building with Pumps & Canopy"

For EACH Pumps/Canopy item:
- If certificate item is "Pumps & Canopy":
  - MATCH if policy shows "Pumps & Canopy" with same limit OR policy shows separate "Pumps" and "Canopy" whose SUM equals the certificate limit.
- If certificate item is "Building with Pumps & Canopy":
  - MATCH if policy shows the same combined label with same limit OR policy shows:
    - Building + Pumps + Canopy (sum), OR
    - Building + (Pumps & Canopy) (sum)
- If certificate items are separate ("Pumps" and/or "Canopy"):
  - MATCH if policy shows that same separate item with same limit.
  - If policy only shows a combined "Pumps & Canopy" limit, you may MATCH the separate items if the combined limit equals the SUM of the separate certificate limits; note clearly that the policy is combined.
- Do NOT confuse pumps/canopy with other property items.
- Evidence must include OCR source + page number for each component used in a sum match.

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
  "business_income_validations": [
    {{
      "cert_bi_name": "Name from certificate (e.g., 'Business Income', 'Business Income - Location 01')",
      "cert_bi_value": "Value from certificate (e.g., '150,000' or 'Actual Loss Sustained')",
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "policy_bi_name": "How it appears in policy (e.g., 'Business Income', 'Business Income and Extra Expense')",
      "policy_bi_value": "Policy value (ALS/Included or a dollar limit) or null",
      "policy_bi_waiting_period": "If present (e.g., '72 hours') else null",
      "policy_location": "Location/premises/building description from policy (or null if policy-wide)",
      "evidence_declarations": "Quote from declarations/coverage schedule (OCR_SOURCE, Page X)",
      "evidence_endorsements": "Quote from any modifying endorsement (OCR_SOURCE, Page X) or null",
      "notes": "Explain how you matched location/building and why MATCH/MISMATCH/NOT_FOUND."
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
  "outdoor_signs_validations": [
    {{
      "cert_os_name": "Name from certificate (e.g., 'Outdoor Signs')",
      "cert_os_value": "Value from certificate (e.g., 'Included' or '25,000')",
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "policy_os_name": "How it appears in policy",
      "policy_os_value": "Policy value (Included/Yes or a dollar limit) or null",
      "policy_location": "Location/premises/building description from policy (or null if policy-wide)",
      "evidence_declarations": "Quote from declarations/coverage schedule (OCR_SOURCE, Page X)",
      "evidence_endorsements": "Quote from any modifying endorsement (OCR_SOURCE, Page X) or null",
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
  "pumps_canopy_validations": [
    {{
      "cert_pc_name": "Name from certificate (e.g., 'Pumps', 'Canopy', 'Pumps & Canopy', 'Building with Pumps & Canopy')",
      "cert_pc_value": "Value from certificate",
      "status": "MATCH | MISMATCH | NOT_FOUND",
      "policy_pc_name": "How it appears in policy",
      "policy_pc_value": "Policy value (single limit) or null",
      "policy_pc_components": "If matched by sum, list components and values like 'Building $X; Pumps $Y; Canopy $Z' or null",
      "policy_location": "Location/premises/building description from policy (or null if policy-wide)",
      "evidence_declarations": "Quote(s) from declarations/schedules (OCR_SOURCE, Page X) - include all components if sum",
      "evidence_endorsements": "Quote from endorsements if applicable (OCR_SOURCE, Page X) or null",
      "notes": "Explain match method (direct vs sum) and why MATCH/MISMATCH/NOT_FOUND."
    }}
  ],
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
    "total_buildings": 0,
    "matched": 0,
    "mismatched": 0,
    "not_found": 0,
    "total_bpp_items": 0,
    "bpp_matched": 0,
    "bpp_mismatched": 0,
    "bpp_not_found": 0,
    "total_bi_items": 0,
    "bi_matched": 0,
    "bi_mismatched": 0,
    "bi_not_found": 0,
    "total_ms_items": 0,
    "ms_matched": 0,
    "ms_mismatched": 0,
    "ms_not_found": 0,
    "total_eb_items": 0,
    "eb_matched": 0,
    "eb_mismatched": 0,
    "eb_not_found": 0,
    "total_os_items": 0,
    "os_matched": 0,
    "os_mismatched": 0,
    "os_not_found": 0,
    "total_ed_items": 0,
    "ed_matched": 0,
    "ed_mismatched": 0,
    "ed_not_found": 0,
    "total_pc_items": 0,
    "pc_matched": 0,
    "pc_mismatched": 0,
    "pc_not_found": 0,
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
        
        # Extract coverages to validate (single LLM call)
        buildings = self.extract_building_coverages(cert_data)
        bpp_items = self.extract_bpp_coverages(cert_data)
        bi_items = self.extract_business_income_coverages(cert_data)
        ms_items = self.extract_money_securities_coverages(cert_data)
        eb_items = self.extract_equipment_breakdown_coverages(cert_data)
        os_items = self.extract_outdoor_signs_coverages(cert_data)
        ed_items = self.extract_employee_dishonesty_coverages(cert_data)
        pc_items = self.extract_pumps_canopy_coverages(cert_data)
        theft_items = self.extract_theft_coverages(cert_data)
        wind_hail_items = self.extract_wind_hail_coverages(cert_data)
        
        if (
            not buildings
            and not bpp_items
            and not bi_items
            and not ms_items
            and not eb_items
            and not os_items
            and not ed_items
            and not pc_items
            and not theft_items
            and not wind_hail_items
        ):
            print("      ❌ No supported coverages found in certificate!")
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
        if bi_items:
            print(f"      Found {len(bi_items)} Business Income coverage(s):")
            for b in bi_items:
                print(f"        - {b['name']}: {b['value']}")
        if ms_items:
            print(f"      Found {len(ms_items)} Money & Securities coverage(s):")
            for m in ms_items:
                print(f"        - {m['name']}: {m['value']}")
        if eb_items:
            print(f"      Found {len(eb_items)} Equipment Breakdown coverage(s):")
            for e in eb_items:
                print(f"        - {e['name']}: {e['value']}")
        if os_items:
            print(f"      Found {len(os_items)} Outdoor Signs coverage(s):")
            for o in os_items:
                print(f"        - {o['name']}: {o['value']}")
        if ed_items:
            print(f"      Found {len(ed_items)} Employee Dishonesty coverage(s):")
            for e in ed_items:
                print(f"        - {e['name']}: {e['value']}")
        if pc_items:
            print(f"      Found {len(pc_items)} Pumps/Canopy coverage(s):")
            for p in pc_items:
                print(f"        - {p['name']}: {p['value']}")
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
            buildings,
            bpp_items,
            bi_items,
            ms_items,
            eb_items,
            os_items,
            ed_items,
            pc_items,
            theft_items,
            wind_hail_items,
            policy_text,
        )
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

            # Guardrail: keep only validations for items actually present in the certificate extraction.
            # This prevents the model from "helpfully" validating extra coverages found in the policy.
            results["building_validations"] = self._filter_validations_to_requested(
                results.get("building_validations", []),
                buildings,
                "cert_building_name",
            )
            results["bpp_validations"] = self._filter_validations_to_requested(
                results.get("bpp_validations", []),
                bpp_items,
                "cert_bpp_name",
            )
            results["business_income_validations"] = self._filter_validations_to_requested(
                results.get("business_income_validations", []),
                bi_items,
                "cert_bi_name",
            )
            results["money_securities_validations"] = self._filter_validations_to_requested(
                results.get("money_securities_validations", []),
                ms_items,
                "cert_ms_name",
            )
            results["equipment_breakdown_validations"] = self._filter_validations_to_requested(
                results.get("equipment_breakdown_validations", []),
                eb_items,
                "cert_eb_name",
            )
            results["outdoor_signs_validations"] = self._filter_validations_to_requested(
                results.get("outdoor_signs_validations", []),
                os_items,
                "cert_os_name",
            )
            results["employee_dishonesty_validations"] = self._filter_validations_to_requested(
                results.get("employee_dishonesty_validations", []),
                ed_items,
                "cert_ed_name",
            )
            results["pumps_canopy_validations"] = self._filter_validations_to_requested(
                results.get("pumps_canopy_validations", []),
                pc_items,
                "cert_pc_name",
            )
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
        print("COVERAGE VALIDATION RESULTS (BUILDING + BPP + BUSINESS INCOME + MONEY & SECURITIES + EQUIPMENT BREAKDOWN + OUTDOOR SIGNS + EMPLOYEE DISHONESTY + PUMPS/CANOPY + THEFT + WIND/HAIL)")
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

        # Display Business Income validations (if present)
        bi_validations = results.get('business_income_validations', [])
        if bi_validations:
            print(f"{'='*70}")
            print("BUSINESS INCOME VALIDATION RESULTS")
            print(f"{'='*70}\n")

            for v in bi_validations:
                status = v.get('status', 'UNKNOWN')
                cert_name = v.get('cert_bi_name', 'N/A')
                cert_value = v.get('cert_bi_value', 'N/A')
                policy_name = v.get('policy_bi_name', 'N/A')
                policy_value = v.get('policy_bi_value', 'N/A')
                waiting = v.get('policy_bi_waiting_period', None)
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
                if waiting:
                    print(f"  Waiting Period: {waiting}")
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

        # Display Outdoor Signs validations (if present)
        os_validations = results.get('outdoor_signs_validations', [])
        if os_validations:
            print(f"{'='*70}")
            print("OUTDOOR SIGNS VALIDATION RESULTS")
            print(f"{'='*70}\n")

            for v in os_validations:
                status = v.get('status', 'UNKNOWN')
                cert_name = v.get('cert_os_name', 'N/A')
                cert_value = v.get('cert_os_value', 'N/A')
                policy_name = v.get('policy_os_name', 'N/A')
                policy_value = v.get('policy_os_value', 'N/A')
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

        # Display Pumps/Canopy validations (if present)
        pc_validations = results.get('pumps_canopy_validations', [])
        if pc_validations:
            print(f"{'='*70}")
            print("PUMPS / CANOPY VALIDATION RESULTS")
            print(f"{'='*70}\n")

            for v in pc_validations:
                status = v.get('status', 'UNKNOWN')
                cert_name = v.get('cert_pc_name', 'N/A')
                cert_value = v.get('cert_pc_value', 'N/A')
                policy_name = v.get('policy_pc_name', 'N/A')
                policy_value = v.get('policy_pc_value', 'N/A')
                policy_components = v.get('policy_pc_components', None)
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
                if policy_components:
                    print(f"  Policy Components: {policy_components}")
                print(f"  Policy Label: {policy_name}")
                print(f"  Policy Location: {policy_location}")

                if evidence_decl and len(evidence_decl) > 120:
                    evidence_decl = evidence_decl[:117] + "..."
                print(f"  Evidence (Declarations): {evidence_decl if evidence_decl else 'N/A'}")

                if evidence_end:
                    if len(evidence_end) > 120:
                        evidence_end = evidence_end[:117] + "..."
                    print(f"  Evidence (Endorsements): {evidence_end}")

                if notes and len(notes) > 170:
                    notes = notes[:167] + "..."
                print(f"  Notes: {notes if notes else 'N/A'}")
                print()

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
        print(f"Total Buildings:  {summary.get('total_buildings', 0)}")
        print(f"  ✓ Matched:      {summary.get('matched', 0)}")
        print(f"  ✗ Mismatched:   {summary.get('mismatched', 0)}")
        print(f"  ? Not Found:    {summary.get('not_found', 0)}")

        if 'total_bpp_items' in summary:
            print(f"\nTotal BPP Items:  {summary.get('total_bpp_items', 0)}")
            print(f"  ✓ Matched:      {summary.get('bpp_matched', 0)}")
            print(f"  ✗ Mismatched:   {summary.get('bpp_mismatched', 0)}")
            print(f"  ? Not Found:    {summary.get('bpp_not_found', 0)}")

        if 'total_bi_items' in summary:
            print(f"\nTotal Business Income Items:  {summary.get('total_bi_items', 0)}")
            print(f"  ✓ Matched:      {summary.get('bi_matched', 0)}")
            print(f"  ✗ Mismatched:   {summary.get('bi_mismatched', 0)}")
            print(f"  ? Not Found:    {summary.get('bi_not_found', 0)}")

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

        if 'total_os_items' in summary:
            print(f"\nTotal Outdoor Signs Items:  {summary.get('total_os_items', 0)}")
            print(f"  ✓ Matched:      {summary.get('os_matched', 0)}")
            print(f"  ✗ Mismatched:   {summary.get('os_mismatched', 0)}")
            print(f"  ? Not Found:    {summary.get('os_not_found', 0)}")

        if 'total_ed_items' in summary:
            print(f"\nTotal Employee Dishonesty Items:  {summary.get('total_ed_items', 0)}")
            print(f"  ✓ Matched:      {summary.get('ed_matched', 0)}")
            print(f"  ✗ Mismatched:   {summary.get('ed_mismatched', 0)}")
            print(f"  ? Not Found:    {summary.get('ed_not_found', 0)}")

        if 'total_pc_items' in summary:
            print(f"\nTotal Pumps/Canopy Items:  {summary.get('total_pc_items', 0)}")
            print(f"  ✓ Matched:      {summary.get('pc_matched', 0)}")
            print(f"  ✗ Mismatched:   {summary.get('pc_mismatched', 0)}")
            print(f"  ? Not Found:    {summary.get('pc_not_found', 0)}")

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
    cert_prefix = "stay"              # Change to: james, indian, etc.
    carrier_dir = "nationwideop"      # Change to: hartfordop, encovaop, etc.
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

