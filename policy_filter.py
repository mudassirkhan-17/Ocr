"""
Unified Policy Filter - Page Reduction
Takes both policy extraction files ({base_name}_pol1.txt and {base_name}_pol2.txt),
filters them to only pages with dollar amounts, and saves filtered outputs

Usage:
    python policy_filter.py aaniya
    python policy_filter.py aaniya_policy

Outputs:
    - {carrier_dir}/{base_name}_fil1.txt (filtered Tesseract extraction)
    - {carrier_dir}/{base_name}_fil2.txt (filtered PyMuPDF extraction)
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class PolicyPageExtractor:
    """Extract pages from policy OCR text with filtering"""
    
    # Keywords for additional interest/insured filtering
    ADDITIONAL_INTEREST_KEYWORDS = [
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
    ]
    
    def __init__(self, policy_text: str, filename: str):
        """Initialize with policy text and filename"""
        self.policy_text = policy_text
        self.filename = filename
        self.page_boundaries = self._calculate_page_boundaries()
    
    def _calculate_page_boundaries(self) -> Dict[int, Tuple[int, int]]:
        """Calculate character positions for each page"""
        boundaries = {}
        
        # Try multiple patterns for PAGE markers
        page_patterns = [
            r'={50,}\s*\nPAGE\s+(\d+)\s*\n={50,}',
            r'={50,}\s*\nPAGE\s+(\d+)\s*\n',
            r'\nPAGE\s+(\d+)\s*\n',
            r'={50,}\s*\nPage\s+(\d+)\s*\n={50,}',
        ]
        
        matches = []
        for pattern in page_patterns:
            found = list(re.finditer(pattern, self.policy_text, re.MULTILINE | re.IGNORECASE))
            if found:
                matches = found
                break
        
        if not matches:
            fallback_pattern = r'PAGE\s+(\d+)'
            fallback_matches = list(re.finditer(fallback_pattern, self.policy_text, re.IGNORECASE))
            if fallback_matches:
                matches = fallback_matches
            else:
                boundaries[1] = (0, len(self.policy_text))
                return boundaries
        
        # Process each page marker
        for i, match in enumerate(matches):
            try:
                page_num = int(match.group(1))
            except (ValueError, IndexError):
                continue
            
            page_start = match.end()
            
            if i < len(matches) - 1:
                next_match = matches[i + 1]
                page_end = next_match.start()
            else:
                page_end = len(self.policy_text)
            
            if page_num not in boundaries:
                boundaries[page_num] = (page_start, page_end)
            else:
                existing_start, existing_end = boundaries[page_num]
                if page_end > existing_end:
                    boundaries[page_num] = (existing_start, page_end)
        
        # Fix overlaps
        if boundaries:
            sorted_pages = sorted(boundaries.items(), key=lambda x: x[1][0])
            for i in range(len(sorted_pages) - 1):
                current_num, (current_start, current_end) = sorted_pages[i]
                next_num, (next_start, next_end) = sorted_pages[i + 1]
                
                if current_end > next_start:
                    boundaries[current_num] = (current_start, next_start)
        
        return boundaries
    
    def find_pages_with_dollar_amounts(self) -> List[int]:
        """Find pages that contain dollar amounts >= $200"""
        pages_with_dollars = set()
        min_amount = 200
        
        skip_patterns = ['EXAMPLE', 'CALCULATION', 'HOW TO', 'SAMPLE', 'ILLUSTRATION']
        
        for page_num, (page_start, page_end) in self.page_boundaries.items():
            page_text = self.policy_text[page_start:page_end]
            page_text_upper = page_text.upper()
            
            if any(skip in page_text_upper for skip in skip_patterns):
                continue
            
            dollar_matches = re.finditer(r'\$\s*([0-9,]+)', page_text)
            
            for match in dollar_matches:
                try:
                    amount_str = match.group(1).replace(',', '')
                    amount = int(amount_str)
                    
                    if amount >= min_amount:
                        pages_with_dollars.add(page_num)
                        break
                except (ValueError, AttributeError):
                    continue
        
        return sorted(pages_with_dollars)
    
    def find_pages_with_keywords(self) -> List[int]:
        """Find pages that contain additional interest/insured keywords"""
        pages_with_keywords = set()
        
        for page_num, (page_start, page_end) in self.page_boundaries.items():
            page_text = self.policy_text[page_start:page_end]
            page_text_lower = page_text.lower()
            
            # Check if any keyword appears in this page
            if any(keyword in page_text_lower for keyword in self.ADDITIONAL_INTEREST_KEYWORDS):
                pages_with_keywords.add(page_num)
        
        return sorted(pages_with_keywords)
    
    def merge_page_ranges(self, pages: List[int], buffer: int = 1) -> List[Tuple[int, int]]:
        """Add buffer pages and merge overlapping ranges"""
        if not pages:
            return []
        
        all_pages = sorted(self.page_boundaries.keys())
        min_page = all_pages[0] if all_pages else 1
        max_page = all_pages[-1] if all_pages else 1
        
        ranges = []
        for page in pages:
            start = max(min_page, page - buffer)
            end = min(max_page, page + buffer)
            ranges.append((start, end))
        
        ranges.sort(key=lambda x: x[0])
        
        merged = []
        for start, end in ranges:
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        return merged
    
    def extract_filtered_pages(self) -> str:
        """Extract only pages with dollar amounts OR additional interest keywords (filtered content)"""
        # Find pages with dollar amounts (existing logic - unchanged)
        dollar_pages = self.find_pages_with_dollar_amounts()
        
        # Find pages with keywords (new)
        keyword_pages = self.find_pages_with_keywords()
        
        # Combine both: union of dollar pages and keyword pages
        combined_pages = sorted(set(dollar_pages) | set(keyword_pages))
        
        if not combined_pages:
            print(f"  ⚠️  No pages with $ amounts or additional interest keywords found, returning empty")
            return ""
        
        # Merge ranges with buffer
        merged_ranges = self.merge_page_ranges(combined_pages, buffer=1)
        
        # Extract pages from each range
        filtered_text = ""
        for start_page, end_page in merged_ranges:
            if start_page in self.page_boundaries and end_page in self.page_boundaries:
                start_char = self.page_boundaries[start_page][0]
                end_char = self.page_boundaries[end_page][1]
                
                # Add page markers for clarity
                filtered_text += f"\n{'='*80}\n"
                filtered_text += f"FILTERED PAGES {start_page}-{end_page}\n"
                filtered_text += f"{'='*80}\n\n"
                filtered_text += self.policy_text[start_char:end_char]
                filtered_text += "\n"
        
        return filtered_text


def extract_base_name(input_path: str) -> str:
    """Extract base name from input (removes .pdf, paths, etc.)"""
    path = Path(input_path)
    base_name = path.stem
    
    # Strip "_policy" suffix if present
    if base_name.endswith("_policy"):
        base_name = base_name[:-7]
    
    return base_name


def filter_policy_file(input_file: Path, output_file: Path) -> bool:
    """Filter a single policy file"""
    if not input_file.exists():
        print(f"  ❌ File not found: {input_file}")
        return False
    
    print(f"  📄 Processing: {input_file.name}")
    
    # Load file
    with open(input_file, 'r', encoding='utf-8') as f:
        policy_text = f.read()
    
    # Process
    extractor = PolicyPageExtractor(policy_text, str(input_file))
    
    total_pages = len(extractor.page_boundaries)
    print(f"     Total pages: {total_pages}")
    
    # Find and extract filtered pages
    filtered_text = extractor.extract_filtered_pages()
    
    if not filtered_text:
        print(f"  ⚠️  No filtered content extracted")
        return False
    
    # Save filtered output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(filtered_text)
    
    filtered_chars = len(filtered_text)
    # Calculate reduction based on combined pages (dollar + keyword)
    dollar_pages = extractor.find_pages_with_dollar_amounts()
    keyword_pages = extractor.find_pages_with_keywords()
    combined_pages = sorted(set(dollar_pages) | set(keyword_pages))
    reduction = ((total_pages - len(combined_pages)) / total_pages * 100) if total_pages > 0 else 0
    
    print(f"     ✅ Saved {filtered_chars:,} characters")
    print(f"     📉 Reduction: ~{reduction:.1f}%")
    
    return True


def main():
    """Main function: unified policy filtering"""
    print("\n" + "="*80)
    print("UNIFIED POLICY FILTER - PAGE REDUCTION")
    print("="*80)
    print()
    
    # Parse command line arguments
    input_name = None
    
    for arg in sys.argv[1:]:
        if not arg.startswith('--'):
            input_name = arg
            break
    
    # Default input if not provided
    if input_name is None:
        input_name = "arrr"
        print(f"⚠️  No input provided, using default: {input_name}")
        print()
    
    # Carrier directory (change this to switch between nationwideop, encovaop, etc.)
    carrier_dir = "usgnonop"
    
    # Extract base name
    base_name = extract_base_name(input_name)
    print(f"Base name: {base_name}\n")
    
    # Set up input/output paths
    output_dir = Path(carrier_dir)
    output_dir.mkdir(exist_ok=True)
    
    tesseract_input = output_dir / f"{base_name}_pol1.txt"
    pymupdf_input = output_dir / f"{base_name}_pol2.txt"
    
    tesseract_output = output_dir / f"{base_name}_fil1.txt"
    pymupdf_output = output_dir / f"{base_name}_fil2.txt"
    
    print("="*80)
    print("FILTERING POLICY FILES")
    print("="*80)
    print()
    
    # Filter Tesseract file
    print("STEP 1: Filtering Tesseract extraction")
    print("-" * 80)
    tesseract_success = filter_policy_file(tesseract_input, tesseract_output)
    print()
    
    # Filter PyMuPDF file
    print("STEP 2: Filtering PyMuPDF extraction")
    print("-" * 80)
    pymupdf_success = filter_policy_file(pymupdf_input, pymupdf_output)
    print()
    
    # Summary
    print("="*80)
    print("FILTERING COMPLETE")
    print("="*80)
    print()
    print("Output files:")
    if tesseract_success:
        print(f"  ✅ Tesseract: {tesseract_output}")
    if pymupdf_success:
        print(f"  ✅ PyMuPDF:   {pymupdf_output}")
    print("="*80)


if __name__ == "__main__":
    main()

