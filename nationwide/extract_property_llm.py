"""
Property Coverage Extraction using GPT-5 Nano
Extracts commercial property coverage limits from combined OCR extraction files
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional
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


class PropertyExtractor:
    """Extract property coverage information using GPT-5 Nano"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-nano"):
        """
        Initialize the extractor
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name (default: gpt-5-nano)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
    def load_prompt(self, prompt_file: str = "property_extraction_prompt.txt") -> str:
        """Load the extraction prompt from file"""
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_policy_document(self, policy_file: str) -> str:
        """Load the combined policy extraction file"""
        policy_path = Path(policy_file)
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_file}")
        
        with open(policy_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_property_info(self, policy_content: str, prompt: str) -> Dict:
        """
        Extract property information using GPT-5 Nano
        
        Args:
            policy_content: The combined policy document content
            prompt: The extraction prompt
            
        Returns:
            Dictionary with extracted property information
        """
        print("="*80)
        print("PROPERTY COVERAGE EXTRACTION - GPT-5 Nano")
        print("="*80)
        print(f"Model: {self.model}")
        print(f"Policy document: {len(policy_content):,} characters (~{len(policy_content)//4:,} tokens)")
        print()
        
        # Combine prompt and document
        # Add explicit JSON instruction since we can't use response_format parameter
        json_instruction = "\n\nIMPORTANT: Return ONLY valid JSON. Do not include any markdown formatting, code blocks, or explanatory text. Return the JSON object directly."
        full_prompt = f"{prompt}{json_instruction}\n\n# Policy Document\n\n{policy_content}"
        
        print("🔄 Sending request to GPT-5 Nano...")
        print("   This may take a moment for large documents...")
        print()
        
        try:
            # Use GPT-5 Nano Responses API format
            response = self.client.responses.create(
                model=self.model,
                input=full_prompt,
                reasoning={
                    "effort": "low"
                },
                text={
                    "verbosity": "low"
                }
            )
            
            # Extract response
            response_text = response.output_text.strip()
            
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
    
    def save_results(self, results: Dict, output_file: str = "property_extraction_results.json"):
        """Save extraction results to JSON file"""
        output_path = Path(output_file)
        
        # Add metadata
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "extraction": results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_path.absolute()}")
        print(f"   File size: {output_path.stat().st_size:,} bytes")
        print()
        
        return str(output_path)
    
    def print_summary(self, results: Dict):
        """Print a summary of extracted information"""
        if "error" in results:
            print("❌ Extraction failed - see error details above")
            return
        
        print("="*80)
        print("EXTRACTION SUMMARY")
        print("="*80)
        print()
        
        if "property" in results:
            property_data = results["property"]
            print("Property Coverage Limits:")
            print("-" * 80)
            
            for field, value in property_data.items():
                field_display = field.replace("_", " ").title()
                if value is None:
                    print(f"  {field_display:30} : Not found")
                else:
                    print(f"  {field_display:30} : {value}")
            
            print()
        
        if "extraction_metadata" in results:
            metadata = results["extraction_metadata"]
            pages = metadata.get("pages_found", {})
            snippets = metadata.get("text_snippets", {})
            
            if pages:
                print("📄 Page References & Source Text:")
                print("-" * 80)
                for field, page_num in pages.items():
                    field_display = field.replace("_", " ").title()
                    snippet = snippets.get(field, "")
                    if snippet:
                        # Truncate snippet if too long
                        if len(snippet) > 70:
                            snippet = snippet[:67] + "..."
                        print(f"  {field_display:30} : Page {page_num}")
                        print(f"    └─ \"{snippet}\"")
                    else:
                        print(f"  {field_display:30} : Page {page_num}")
                print()
        
        print("="*80)
        print()


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract Property Coverage Information using GPT-5 Nano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses default files)
  python extract_property_llm.py
  
  # Custom files
  python extract_property_llm.py --policy encovaop/aaniya_combined.txt --prompt property_extraction_prompt.txt
  
  # Custom output file
  python extract_property_llm.py --output results/property_results.json
  
  # Use environment variable for API key
  export OPENAI_API_KEY=your_key_here
  python extract_property_llm.py
        """
    )
    
    parser.add_argument(
        "--policy",
        type=str,
        default="nationwideop/sean_combined.txt",
        help="Path to combined policy extraction file (default: nationwideop/sean_combined.txt)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="property_extraction_prompt.txt",
        help="Path to extraction prompt file (default: property_extraction_prompt.txt)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="property_extraction_results.json",
        help="Output JSON file path (default: property_extraction_results.json)"
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
        default="gpt-5-nano",
        help="Model name (default: gpt-5-nano)"
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
    
    try:
        # Initialize extractor
        extractor = PropertyExtractor(api_key=args.api_key, model=args.model)
        
        # Load prompt and policy document
        print("📄 Loading files...")
        prompt = extractor.load_prompt(args.prompt)
        policy_content = extractor.load_policy_document(args.policy)
        print(f"   Prompt: {args.prompt}")
        print(f"   Policy: {args.policy}")
        print()
        
        # Extract information
        results = extractor.extract_property_info(policy_content, prompt)
        
        # Save results
        output_path = extractor.save_results(results, args.output)
        
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

