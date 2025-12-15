"""
Traveler Certificate + Policy Extraction (Property + GL) with QC

- Uses GPT-4.1 Mini by default
- Accepts:
  - Traveler combined policy text (Tesseract + PyMuPDF)
  - Optional ACORD 27 (Property) certificate text
  - Optional ACORD 25 (GL) certificate text
- Produces structured JSON and a deterministic QC mismatch report
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI library not installed. Install with: pip install openai")

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")

# Load environment variables from .env file
if DOTENV_AVAILABLE:
    load_dotenv()


def _read_text_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8", errors="replace")


def _normalize_money(value: Any) -> Optional[str]:
    """
    Normalize money-like values into one of:
    - digits-only string (e.g. "181472")
    - "Included"
    - None
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(int(value))
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v:
        return None
    if v.lower() == "included":
        return "Included"
    # keep percentages as-is (e.g. "1%") for deductibles
    if v.endswith("%"):
        return v
    # "Inside $10,000 / Outside $10,000" should stay as-is
    if "inside" in v.lower() or "outside" in v.lower():
        return v
    # strip $, commas, spaces
    digits = "".join(ch for ch in v if ch.isdigit())
    return digits or None


def _compare_values(a: Any, b: Any) -> bool:
    return _normalize_money(a) == _normalize_money(b)


def _qc_compare(certificate: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic QC: compares certificate vs policy on key Traveler fields.
    Produces a mismatches list with field paths and normalized values.
    """
    mismatches: List[Dict[str, Any]] = []

    # Helper to safely get nested keys
    def get(d: Dict[str, Any], path: List[str]) -> Any:
        cur: Any = d
        for k in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        return cur

    checks: List[Tuple[str, List[str], List[str]]] = [
        ("property.policy_number",
         ["property", "policy_number"],
         ["property", "policy_number"]),
        ("property.effective_date",
         ["property", "effective_date"],
         ["property", "policy_period", "effective_date"]),
        ("property.expiration_date",
         ["property", "expiration_date"],
         ["property", "policy_period", "expiration_date"]),
        ("gl.policy_number",
         ["general_liability", "policy_number"],
         ["general_liability", "policy_number"]),
        ("gl.effective_date",
         ["general_liability", "effective_date"],
         ["general_liability", "policy_period", "effective_date"]),
        ("gl.expiration_date",
         ["general_liability", "expiration_date"],
         ["general_liability", "policy_period", "expiration_date"]),
        # GL limits
        ("gl.limits.each_occurrence",
         ["general_liability", "limits", "each_occurrence"],
         ["general_liability", "limits", "each_occurrence"]),
        ("gl.limits.general_aggregate",
         ["general_liability", "limits", "general_aggregate"],
         ["general_liability", "limits", "general_aggregate"]),
        ("gl.limits.products_completed_operations_aggregate",
         ["general_liability", "limits", "products_completed_operations_aggregate"],
         ["general_liability", "limits", "products_completed_operations_aggregate"]),
        ("gl.limits.personal_advertising_injury",
         ["general_liability", "limits", "personal_advertising_injury"],
         ["general_liability", "limits", "personal_advertising_injury"]),
        ("gl.limits.damage_to_rented_premises",
         ["general_liability", "limits", "damage_to_rented_premises"],
         ["general_liability", "limits", "damage_to_rented_premises"]),
        ("gl.limits.medical_expense",
         ["general_liability", "limits", "medical_expense"],
         ["general_liability", "limits", "medical_expense"]),
        # Property location fields
        ("property.locations[0].business_personal_property",
         ["property", "locations", "0", "business_personal_property"],
         ["property", "locations", "0", "business_personal_property"]),
        ("property.locations[0].building",
         ["property", "locations", "0", "building"],
         ["property", "locations", "0", "building"]),
        ("property.locations[0].business_income",
         ["property", "locations", "0", "business_income"],
         ["property", "locations", "0", "business_income"]),
        ("property.locations[0].deductible",
         ["property", "locations", "0", "deductible"],
         ["property", "locations", "0", "deductible"]),
        # Policy-level property signals
        ("policy.property.outdoor_signs_limit",
         ["property", "locations", "0", "outdoor_signs"],  # certificate (first location)
         ["property", "outdoor_signs_limit"]),
        ("policy.property.windstorm_or_hail",
         ["property", "locations", "0", "windstorm_or_hail"],
         ["property", "windstorm_or_hail"]),
        ("policy.property.theft_sublimit",
         ["property", "locations", "0", "theft_sublimit"],
         ["property", "theft_sublimit"]),
    ]

    # For certificate locations, allow empty list
    cert_prop = certificate.get("property", {}) if isinstance(certificate, dict) else {}
    cert_locations = cert_prop.get("locations")
    if not isinstance(cert_locations, list) or len(cert_locations) == 0:
        # remove checks that depend on location 0
        checks = [c for c in checks if "locations" not in c[1]]

    for field, cert_path, pol_path in checks:
        # Handle "0" index for location lists
        def resolve_path(root: Dict[str, Any], path: List[str]) -> Any:
            cur: Any = root
            for k in path:
                if k.isdigit():
                    if not isinstance(cur, list):
                        return None
                    idx = int(k)
                    if idx >= len(cur):
                        return None
                    cur = cur[idx]
                else:
                    if not isinstance(cur, dict):
                        return None
                    cur = cur.get(k)
            return cur

        cert_val = resolve_path(certificate, cert_path)
        pol_val = resolve_path(policy, pol_path)

        # Skip if both are None (nothing to compare)
        if cert_val is None and pol_val is None:
            continue
        
        # Flag mismatch if:
        # 1. Both present but different values
        # 2. Certificate has value but policy is None (certificate claims coverage not in policy)
        # 3. Policy has value but certificate is None (policy has coverage not on certificate)
        if cert_val is None or pol_val is None:
            # One is None, the other isn't - this is a mismatch
            mismatches.append({
                "field": field,
                "certificate": _normalize_money(cert_val) if cert_val is not None else None,
                "policy": _normalize_money(pol_val) if pol_val is not None else None,
            })
        elif not _compare_values(cert_val, pol_val):
            # Both present but different values
            mismatches.append({
                "field": field,
                "certificate": _normalize_money(cert_val),
                "policy": _normalize_money(pol_val),
            })

    status = "pass" if not mismatches else "needs_review"
    return {"status": status, "mismatches": mismatches}


def _extract_first_class_amount(policy_text: str, class_no: int) -> Optional[str]:
    """
    Best-effort regex extraction for Traveler declarations:
    Finds first "$ <amount>" appearing near "Class 1"/"Class 2".
    """
    import re
    # Allow OCR line breaks between tokens
    pattern = re.compile(
        rf"Class\s*{class_no}[\s\S]{{0,200}}?\$\s*([0-9,]+)",
        re.IGNORECASE
    )
    m = pattern.search(policy_text)
    if not m:
        return None
    return _normalize_money(m.group(1))


def _policy_has_explicit_building_limit(policy_text: str) -> bool:
    """
    True if policy text shows a distinct Building coverage line with a dollar limit.
    (Avoids treating "All Personal Property" as Building.)
    """
    import re
    # IMPORTANT: exclude "Building and Personal Property ..." form name
    patterns = [
        # Table style: "1 1 Building $ 350,000"
        r"(?mi)^\s*\d+\s+\d+\s+Building(?!\s+and\s+Personal)\b[\s\S]{0,80}?\$\s*[0-9,]+",
        # Label style: "Building\n$ 983,892"
        r"(?mi)^\s*Building(?!\s+and\s+Personal)\b\s*(?:\n|\s)+\$\s*[0-9,]+",
        # Single-line style: "Building $ 425,000"
        r"(?mi)^\s*Building(?!\s+and\s+Personal)\b[\t ]+\$?\s*[0-9,]+\s*$",
    ]
    return any(re.search(p, policy_text) for p in patterns)


def _postprocess_extraction(extracted: Dict[str, Any], policy_text: str, certs_provided: bool) -> Dict[str, Any]:
    """
    Patch common, predictable OCR/LLM mistakes with deterministic rules.
    """
    if not isinstance(extracted, dict):
        return extracted

    # If no certificates were provided, mark certificate checkbox-style fields as null (not false)
    if not certs_provided and isinstance(extracted.get("certificate"), dict):
        # keep as-is; prompt already says [NOT PROVIDED] => should be nulls
        pass

    policy = extracted.get("policy")
    if not isinstance(policy, dict):
        return extracted

    prop = policy.get("property")
    if not isinstance(prop, dict):
        return extracted

    locs = prop.get("locations")
    if isinstance(locs, list):
        # Extract fallback class amounts from policy text (first match)
        class1_amt = _extract_first_class_amount(policy_text, 1)
        class2_amt = _extract_first_class_amount(policy_text, 2)
        has_building_limit = _policy_has_explicit_building_limit(policy_text)

        for loc in locs:
            if not isinstance(loc, dict):
                continue

            # If "building" is a construction type (non-numeric), move it to construction
            bld = loc.get("building")
            bld_norm = _normalize_money(bld)
            if isinstance(bld, str) and bld_norm is None and bld.strip():
                # Likely "Frame"/"Non-Combustible"/etc
                if loc.get("construction") in (None, ""):
                    loc["construction"] = bld.strip()
                loc["building"] = None

            # Pumps/canopy should be numeric (Class 1/2). If model returned Included/null, fix from regex.
            pumps = loc.get("pumps")
            if _normalize_money(pumps) in (None, "Included") and class1_amt:
                loc["pumps"] = class1_amt

            canopy = loc.get("canopy")
            if _normalize_money(canopy) in (None, "Included") and class2_amt:
                loc["canopy"] = class2_amt

            # If there is no explicit Building $ limit in the policy, do not populate building
            # with the All Personal Property/BPP amount.
            if not has_building_limit:
                bpp = loc.get("business_personal_property")
                bld = loc.get("building")
                if _normalize_money(bpp) and _normalize_money(bld) and _normalize_money(bpp) == _normalize_money(bld):
                    loc["building"] = None

    return extracted


class EncovaExtractor:
    """Extract Traveler certificate + policy data using an LLM."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1-mini"):
        """
        Initialize the extractor
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name (default: gpt-4.1-nano)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
    def load_prompt(self, prompt_file: str) -> str:
        return _read_text_file(prompt_file)

    def extract(self, *, prompt: str, policy_text: str, property_cert_text: Optional[str], gl_cert_text: Optional[str]) -> Dict[str, Any]:
        print("=" * 80)
        print("ENCOVA EXTRACTION (PROPERTY + GL) + QC")
        print("=" * 80)
        print(f"Model: {self.model}")
        print(f"Policy chars: {len(policy_text):,} (~{len(policy_text)//4:,} tokens est.)")
        if property_cert_text:
            print(f"Property cert chars: {len(property_cert_text):,}")
        if gl_cert_text:
            print(f"GL cert chars: {len(gl_cert_text):,}")
        print()

        json_instruction = (
            "\n\nIMPORTANT: Return ONLY valid JSON. No markdown. No code fences. No commentary."
        )
        cert_block = []
        if property_cert_text:
            cert_block.append("# PROPERTY CERTIFICATE (ACORD 27)\n" + property_cert_text)
        else:
            cert_block.append("# PROPERTY CERTIFICATE (ACORD 27)\n[NOT PROVIDED]")
        if gl_cert_text:
            cert_block.append("# GL CERTIFICATE (ACORD 25)\n" + gl_cert_text)
        else:
            cert_block.append("# GL CERTIFICATE (ACORD 25)\n[NOT PROVIDED]")

        full_prompt = (
            f"{prompt}{json_instruction}\n\n"
            f"# INPUTS\n\n"
            f"{cert_block[0]}\n\n"
            f"{cert_block[1]}\n\n"
            f"# POLICY TEXT (ENCOVA)\n\n{policy_text}\n"
        )
        
        print("🔄 Sending request to model...")
        print("   This may take a moment for large documents...")
        print()
        
        try:
            # Chat Completions API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert insurance analyst. Extract Encova certificate + policy data and return ONLY valid JSON."
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            # Extract response
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            try:
                result = json.loads(response_text)
                print("✅ Extraction successful!")
                print()
                return result
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Response is not valid JSON")
                print(f"   Attempting to extract JSON from response...")
                print()
                
                # Try to extract JSON from markdown code blocks
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                
                try:
                    result = json.loads(response_text)
                    print("✅ Successfully extracted JSON from response")
                    print()
                    return result
                except json.JSONDecodeError:
                    print(f"❌ Failed to parse JSON response")
                    print(f"   Error: {e}")
                    print(f"   Response preview: {response_text[:500]}...")
                return {"error": "Failed to parse JSON response", "raw_response": response_text}
        
        except Exception as e:
            print(f"❌ API call failed: {e}")
            print()
            return {"error": str(e)}
    
    def save_results(self, results: Dict, output_file: str):
        """Save extraction results to JSON file"""
        output_path = Path(output_file)
        
        # Add metadata
        output_data = results
        output_data.setdefault("timestamp", datetime.now().isoformat())
        output_data.setdefault("model", self.model)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_path.absolute()}")
        print(f"   File size: {output_path.stat().st_size:,} bytes")
        print()
        
        return str(output_path)
    
    def print_summary(self, results: Dict):
        """Print a compact summary"""
        if "error" in results:
            print("❌ Extraction failed - see error details above")
            return
        qc = results.get("qc", {})
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"QC status: {qc.get('status', 'unknown')}")
        if qc.get("mismatches"):
            print(f"Mismatches: {len(qc['mismatches'])}")
            for m in qc["mismatches"][:10]:
                print(f" - {m['field']}: cert={m['certificate']} policy={m['policy']}")
        print()


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract Traveler Property + GL from policy + optional certificates, then QC compare",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple: just provide base name (auto-detects all files)
  python llm4nano.py shelby
  
  # Policy-only (certificate fields will be null)
  python llm4nano.py --policy encovaop/shelby_combined.txt

  # With certificate OCR text
  python llm4nano.py --policy encovaop/shelby_combined.txt --property-cert encovaop/shelby_policy2.txt --gl-cert encovaop/shelby_gl_cert.txt

  # Custom prompt/model
  python llm4nano.py shelby --model gpt-4.1-mini
  
  # Use environment variable for API key
  export OPENAI_API_KEY=your_key_here
  python llm4nano.py shelby
        """
    )
    
    parser.add_argument(
        "base_name",
        nargs="?",
        type=str,
        default=None,
        help="Base name (e.g., 'shelby') - auto-detects policy and certificate files"
    )
    
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to Encova combined policy extraction file (overrides auto-detection)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="encova_extraction_prompt.txt",
        help="Path to Encova extraction prompt file"
    )

    parser.add_argument(
        "--property-cert",
        type=str,
        default=None,
        help="Optional: path to ACORD 27 certificate OCR text"
    )

    parser.add_argument(
        "--gl-cert",
        type=str,
        default=None,
        help="Optional: path to ACORD 25 certificate OCR text"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="hartford_extraction_results.json",
        help="Output JSON file path"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="Model name (default: gpt-4.1-mini)"
    )
    
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Don't print extraction summary"
    )
    
    args = parser.parse_args()
    
    # Check if OpenAI is available
    if not OPENAI_AVAILABLE:
        print("❌ OpenAI library not installed")
        print("   Install with: pip install openai")
        return 1
    
    # Auto-detect files if base_name is provided
    output_dir = Path("hartfordop")
    if args.base_name:
        base_name = args.base_name
        print(f"🔍 Auto-detecting files for: {base_name}")
        print()
        
        # Auto-detect policy file (try filtered combo first, then combined)
        if not args.policy:
            policy_candidates = [
                output_dir / f"{base_name}_pol_combo.txt",
                output_dir / f"{base_name}_combined.txt",
            ]
            for candidate in policy_candidates:
                if candidate.exists():
                    args.policy = str(candidate)
                    break
            if not args.policy:
                print(f"⚠️  Warning: No policy file found for {base_name}")
                print(f"   Tried: {[str(c) for c in policy_candidates]}")
        
        # Auto-detect property certificate
        if not args.property_cert:
            prop_cert = output_dir / f"{base_name}_pl_combo.txt"
            if prop_cert.exists():
                args.property_cert = str(prop_cert)
        
        # Auto-detect GL certificate
        if not args.gl_cert:
            gl_cert = output_dir / f"{base_name}_gl_combo.txt"
            if gl_cert.exists():
                args.gl_cert = str(gl_cert)
        
        # Auto-generate output filename
        if args.output == "hartford_extraction_results.json":  # default value
            args.output = str(output_dir / f"{base_name}_extraction_llm.json")
    
    # Fallback to default if no base_name and no explicit policy
    if not args.policy:
        args.policy = "hartfordop/khwaish_combined.txt"
        print(f"⚠️  No base name or policy provided, using default: {args.policy}")
        print()
    
    try:
        # Initialize extractor
        extractor = EncovaExtractor(api_key=args.api_key, model=args.model)
        
        # Load prompt and policy document
        print("📄 Loading files...")
        prompt = extractor.load_prompt(args.prompt)
        policy_content = _read_text_file(args.policy)
        property_cert_text = _read_text_file(args.property_cert) if args.property_cert else None
        gl_cert_text = _read_text_file(args.gl_cert) if args.gl_cert else None
        print(f"   Prompt: {args.prompt}")
        print(f"   Policy: {args.policy}")
        if args.property_cert:
            print(f"   Property cert: {args.property_cert}")
        if args.gl_cert:
            print(f"   GL cert: {args.gl_cert}")
        print()
        
        # Extract information
        extracted = extractor.extract(
            prompt=prompt,
            policy_text=policy_content,
            property_cert_text=property_cert_text,
            gl_cert_text=gl_cert_text,
        )

        certs_provided = bool(property_cert_text or gl_cert_text)
        if isinstance(extracted, dict):
            extracted = _postprocess_extraction(extracted, policy_content, certs_provided)

        # Attach deterministic QC if the model returned the expected shape
        results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "inputs": {
                "policy_file": args.policy,
                "prompt_file": args.prompt,
                "property_cert_file": args.property_cert,
                "gl_cert_file": args.gl_cert,
            },
            "extraction": extracted,
        }

        if not certs_provided:
            results["qc"] = {"status": "no_certificate", "mismatches": []}
        elif isinstance(extracted, dict) and "certificate" in extracted and "policy" in extracted:
            results["qc"] = _qc_compare(extracted.get("certificate", {}), extracted.get("policy", {}))
        else:
            results["qc"] = {"status": "unknown", "mismatches": []}
        
        # Auto-generate output filename from policy filename if using default
        output_file = args.output
        if output_file == "hartford_extraction_results.json":  # default value
            policy_path = Path(args.policy)
            policy_name = policy_path.stem.replace("_combined", "").replace("_pol_combo", "")
            output_file = policy_path.parent / f"{policy_name}_extraction_llm.json"
            output_file = str(output_file)
        
        # Save results
        output_path = extractor.save_results(results, output_file)
        
        # Print summary
        if not args.no_summary:
            extractor.print_summary(results)
        
        print("✅ Extraction complete!")
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return 1
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

