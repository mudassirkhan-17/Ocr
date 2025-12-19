"""
Policy Additional Interests Extraction (LLM + Parallel Page Filtering)

Goal:
- Extract policy-side additional interests like Mortgagee / Loss Payee / Lienholder / Additional Insured
  from a full policy combo OCR text file (*_pol_combo.txt).

Key idea:
- Split policy into pages
- Run two cheap filters in parallel:
  1) Dollar/number-heavy pages
  2) Keyword pages (mortgagee/loss payee/etc.)
- Union those pages (optionally add neighbor pages) and send only that subset to the LLM.
"""

import os
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


KEYWORDS = [
    # common
    "additional interest",
    "additional interests",
    "additional insured",
    "additional insureds",
    "mortgagee",
    "mortgage holder",
    "mortgage holders",
    "mortgagees",
    "loss payee",
    "loss payable",
    "lienholder",
    "lien holder",
    "secured party",
    "secured parties",
    "payee",
    # sometimes appears as column headers / schedule labels
    "mortgage holder name",
    "mortgagee address",
    "mortgagee city",
    "mortgagee city state zipcode",
]


_DOLLAR_RE = re.compile(r"\$\s*\d")
_NUMBER_WITH_COMMAS_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")
_BIG_NUMBER_RE = re.compile(r"\b\d{5,}\b")  # catches big numeric blocks sometimes tied to schedules


@dataclass(frozen=True)
class PolicyPage:
    page_number: int
    text: str


def _split_policy_combo_into_pages(policy_text: str) -> List[PolicyPage]:
    """
    Split a *_pol_combo.txt into pages based on:
      ================================================================================
      PAGE X
      ================================================================================
    """
    # Normalize newlines
    text = policy_text.replace("\r\n", "\n").replace("\r", "\n")

    # Find "PAGE <num>" separators
    # We keep the "PAGE X" marker inside each chunk so the LLM can cite it.
    pattern = re.compile(r"^={40,}\nPAGE\s+(\d+)\n={40,}\n", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        # fallback: treat entire doc as a single page
        return [PolicyPage(page_number=0, text=text)]

    pages: List[PolicyPage] = []
    for i, m in enumerate(matches):
        page_num = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip("\n")
        pages.append(PolicyPage(page_number=page_num, text=chunk))
    return pages


def _filter_pages_with_dollars(pages: List[PolicyPage]) -> List[int]:
    out: List[int] = []
    for p in pages:
        t = p.text
        # Fast heuristics: any dollar sign + digit, number-with-commas, or large number blocks
        if _DOLLAR_RE.search(t) or _NUMBER_WITH_COMMAS_RE.search(t) or _BIG_NUMBER_RE.search(t):
            out.append(p.page_number)
    return out


def _filter_pages_with_keywords(pages: List[PolicyPage], keywords: List[str]) -> List[int]:
    out: List[int] = []
    for p in pages:
        lower = p.text.lower()
        if any(k in lower for k in keywords):
            out.append(p.page_number)
    return out


def _expand_neighbors(page_nums: List[int], radius: int) -> List[int]:
    if radius <= 0:
        return sorted(set(page_nums))
    s = set(page_nums)
    for n in list(s):
        for i in range(1, radius + 1):
            s.add(n - i)
            s.add(n + i)
    return sorted(x for x in s if x >= 0)


def _build_filtered_policy_text(pages: List[PolicyPage], keep_page_nums: List[int], max_pages: int) -> str:
    keep_set = set(keep_page_nums)
    kept = [p for p in pages if p.page_number in keep_set]
    kept.sort(key=lambda p: p.page_number)

    if max_pages and len(kept) > max_pages:
        kept = kept[:max_pages]

    return "\n\n".join(p.text for p in kept)


class PolicyAdditionalInterestsExtractor:
    def __init__(self, model: str = "gpt-4.1-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def create_prompt(self, cert_context: Optional[Dict], filtered_policy_text: str) -> str:
        insured = (cert_context or {}).get("insured_name") or "Unknown"
        policy_number = (cert_context or {}).get("policy_number") or "Unknown"
        location = (cert_context or {}).get("location_address") or "Unknown"

        return f"""You are an expert Commercial Property policy QC analyst.

Your task: Extract policy-side ADDITIONAL INTERESTS from the policy text below.
These can include: Mortgagee / Mortgage Holder, Loss Payee, Lienholder/Secured Party, Additional Insured.

==================================================
⛔⛔⛔ ANTI-HALLUCINATION RULES (READ FIRST) ⛔⛔⛔
==================================================
- If an entry is not explicitly present in the provided policy text, DO NOT invent it.
- If you cannot find an additional interest schedule/table or endorsement, return an empty list.
- Evidence must be VERBATIM text from the policy and must include a page number shown in the text.
- If you cannot quote evidence with the page number visible, set evidence=null and do NOT create the entry.

==================================================
CONTEXT (optional, from certificate extraction)
==================================================
- Insured: {insured}
- Policy number (from cert): {policy_number}
- Location (from cert): {location}

==================================================
POLICY TEXT (FILTERED PAGES)
==================================================
{filtered_policy_text}

==================================================
WHAT COUNTS AS AN "ADDITIONAL INTEREST" HERE
==================================================
We only want third parties with an insurable/financial interest shown on the POLICY:
- Mortgagee / Mortgage Holder / Mortgagee Clause schedules
- Loss Payee / Loss Payable schedules
- Lienholder / Secured Party schedules
- Additional Insured endorsements ONLY if they list a specific entity name

DO NOT include:
- Named insured
- Producer/agency
- Company addresses
- Generic placeholders like "TO WHOM IT MAY CONCERN"

==================================================
OUTPUT FORMAT (STRICT JSON)
==================================================
Return ONLY a valid JSON object:
{{
  "additional_interests": [
    {{
      "type": "MORTGAGEE|LOSS_PAYEE|LIENHOLDER|SECURED_PARTY|ADDITIONAL_INSURED|OTHER",
      "name": "string",
      "address": "string or null",
      "city": "string or null",
      "state": "string or null",
      "zip": "string or null",
      "premises_building": "e.g. 'Prem 1 / Bldg 1' or null",
      "evidence": "verbatim quote (OCR_SOURCE, Page X) or null",
      "notes": "brief notes, e.g. OCR issues or mapping"
    }}
  ],
  "summary": {{
    "total": 0
  }},
  "qc_notes": "string"
}}

Rules:
- If none found: additional_interests must be [] and summary.total must be 0.
- For every entry, evidence must include OCR source (TESSERACT or PYMUPDF) and Page X exactly as shown.
- Prefer extracting from schedules/tables with headers like MORTGAGE HOLDERS / LOSS PAYEE.
"""

    def extract_from_policy_combo(
        self,
        policy_combo_path: str,
        cert_json_path: Optional[str],
        output_path: str,
        neighbor_radius: int = 1,
        max_pages: int = 25,
        dry_run: bool = False,
    ) -> Dict:
        with open(policy_combo_path, "r", encoding="utf-8") as f:
            policy_text = f.read()

        cert_context = None
        if cert_json_path and os.path.exists(cert_json_path):
            with open(cert_json_path, "r", encoding="utf-8") as f:
                cert_context = json.load(f)

        pages = _split_policy_combo_into_pages(policy_text)

        # Parallel filters (cheap)
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_dollar = ex.submit(_filter_pages_with_dollars, pages)
            f_kw = ex.submit(_filter_pages_with_keywords, pages, KEYWORDS)
            dollar_pages = f_dollar.result()
            keyword_pages = f_kw.result()

        combined = sorted(set(dollar_pages) | set(keyword_pages))
        expanded = _expand_neighbors(combined, neighbor_radius)
        filtered_text = _build_filtered_policy_text(pages, expanded, max_pages=max_pages)

        result: Dict = {
            "metadata": {
                "model": self.model,
                "policy_file": policy_combo_path,
                "cert_file": cert_json_path,
                "pages_total": len(pages),
                "pages_dollar": dollar_pages,
                "pages_keyword": keyword_pages,
                "pages_combined": combined,
                "pages_expanded": expanded,
                "max_pages": max_pages,
                "neighbor_radius": neighbor_radius,
                "dry_run": dry_run,
            }
        }

        if dry_run:
            # Useful for debugging which pages we would send.
            result["filtered_policy_text_preview"] = filtered_text[:3000]
            result["additional_interests"] = []
            result["summary"] = {"total": 0}
            result["qc_notes"] = "DRY_RUN: No LLM call made."
            self._save(output_path, result)
            return result

        prompt = self.create_prompt(cert_context, filtered_text)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        data = json.loads(response.choices[0].message.content)
        data["metadata"] = {
            **result["metadata"],
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        # Ensure summary.total is consistent
        interests = data.get("additional_interests") or []
        if "summary" not in data or not isinstance(data.get("summary"), dict):
            data["summary"] = {}
        data["summary"]["total"] = len(interests)

        self._save(output_path, data)
        return data

    def _save(self, output_path: str, data: Dict) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    # ========== EDIT THESE VALUES ==========
    carrier_dir = "nationwideop"  # encovaop / hartfordop / travelerop / nationwideop
    base_name = "westside"
    # ======================================

    policy_combo_path = os.path.join(carrier_dir, f"{base_name}_pol_combo.txt")
    cert_json_path = os.path.join(carrier_dir, f"{base_name}_pl_extracted_real.json")
    output_path = os.path.join(carrier_dir, f"{base_name}_policy_additional_interests.json")

    if not os.path.exists(policy_combo_path):
        raise SystemExit(f"Policy combo not found: {policy_combo_path}")

    extractor = PolicyAdditionalInterestsExtractor(model="gpt-4.1-mini")
    extractor.extract_from_policy_combo(
        policy_combo_path=policy_combo_path,
        cert_json_path=cert_json_path if os.path.exists(cert_json_path) else None,
        output_path=output_path,
        neighbor_radius=1,
        max_pages=25,
        dry_run=False,
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()


