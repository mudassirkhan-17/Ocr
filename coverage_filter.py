# """
# Coverage Keyword Filter
# Filters policy text to only include pages containing coverage keywords from certificate
# """

# import os
# import json
# import re
# from typing import List, Set, Dict


# def extract_keywords_smart(coverage_name: str) -> List[str]:
#     """
#     Extract meaningful keywords from coverage name, filtering out noise
    
#     Examples:
#         "Wind and Hail" → ["wind", "hail"]
#         "Barns #1 & 2" → ["barns"]
#         "Business Income" → ["business", "income"]
#         "Wind & Hail Deductible (3% subject to $25,000 min)" → ["wind", "hail", "deductible"]
#     """
#     # Remove anything in parentheses (details/explanations)
#     coverage_name = re.sub(r'\([^)]*\)', '', coverage_name)
    
#     # Stop words to ignore
#     stop_words = {
#         "and", "or", "the", "of", "&", "#", "a", "an", "in", "on", "at", 
#         "to", "for", "with", "by", "from", "as", "is", "was", "are", "be"
#     }
    
#     # Remove special characters and split
#     clean_text = re.sub(r'[^\w\s]', ' ', coverage_name.lower())
#     words = clean_text.split()
    
#     keywords = []
#     for word in words:
#         # Skip stop words
#         if word in stop_words:
#             continue
        
#         # Skip ALL numbers (pure digits)
#         if word.isdigit():
#             continue
        
#         # Skip words that are mostly numbers
#         if re.match(r'^\d+[a-z]*$', word):  # Like "1st", "2nd", "000"
#             continue
        
#         # Skip very short words (< 3 characters)
#         if len(word) < 3:
#             continue
        
#         keywords.append(word)
    
#     return keywords


# def extract_all_keywords(coverages) -> Set[str]:
#     """
#     Extract all unique keywords from all coverages
    
#     Args:
#         coverages: Either a dict of coverage_name: value OR 
#                    a list of dicts with 'coverage_name' field
        
#     Returns:
#         Set of unique keywords
#     """
#     all_keywords = set()
    
#     # Handle both dict and list formats
#     if isinstance(coverages, dict):
#         # Dictionary format: {"Building": "$1,320,000", ...}
#         coverage_names = list(coverages.keys())
#     elif isinstance(coverages, list):
#         # List format: [{"coverage_name": "Building", ...}, ...]
#         coverage_names = [c.get("coverage_name", "") for c in coverages]
#     else:
#         return all_keywords
    
#     # Extract keywords from each coverage name
#     for coverage_name in coverage_names:
#         if not coverage_name:
#             continue
            
#         keywords = extract_keywords_smart(coverage_name)
#         all_keywords.update(keywords)
    
#     return all_keywords


# def parse_policy_pages(policy_text: str) -> Dict[int, str]:
#     """
#     Parse policy text into pages
    
#     Args:
#         policy_text: Full policy text with page markers
        
#     Returns:
#         Dictionary mapping page number to page content
#     """
#     pages = {}
#     current_page = None
#     current_lines = []
    
#     lines = policy_text.split('\n')
#     i = 0
    
#     while i < len(lines):
#         line = lines[i]
        
#         # Check for page marker format:
#         # ================================================================================
#         # PAGE 2
#         # ================================================================================
#         if line.startswith('=' * 40):  # Line of equals signs
#             # Check if next line has PAGE number
#             if i + 1 < len(lines):
#                 page_match = re.match(r'PAGE\s+(\d+)', lines[i + 1], re.IGNORECASE)
#                 if page_match:
#                     # Save previous page if exists
#                     if current_page is not None:
#                         pages[current_page] = '\n'.join(current_lines)
                    
#                     # Start new page
#                     current_page = int(page_match.group(1))
#                     current_lines = []
                    
#                     # Skip the page header (3 lines: ===, PAGE X, ===)
#                     i += 3
#                     continue
        
#         # Add line to current page
#         if current_page is not None:
#             current_lines.append(line)
        
#         i += 1
    
#     # Save last page
#     if current_page is not None:
#         pages[current_page] = '\n'.join(current_lines)
    
#     return pages


# def find_pages_with_keywords(pages: Dict[int, str], keywords: Set[str]) -> Set[int]:
#     """
#     Find all page numbers that contain any of the keywords
    
#     Args:
#         pages: Dictionary mapping page number to page content
#         keywords: Set of keywords to search for
        
#     Returns:
#         Set of page numbers containing at least one keyword
#     """
#     matching_pages = set()
    
#     for page_num, page_content in pages.items():
#         # Convert page content to lowercase for case-insensitive search
#         page_lower = page_content.lower()
        
#         # Check if any keyword appears in this page
#         for keyword in keywords:
#             if keyword.lower() in page_lower:
#                 matching_pages.add(page_num)
#                 break  # Found at least one keyword, move to next page
    
#     return matching_pages


# def extract_filtered_text(pages: Dict[int, str], page_numbers: Set[int]) -> str:
#     """
#     Extract and combine pages by page numbers (in order)
    
#     Args:
#         pages: Dictionary mapping page number to page content
#         page_numbers: Set of page numbers to extract
        
#     Returns:
#         Combined text of all specified pages with page markers
#     """
#     # Sort page numbers
#     sorted_pages = sorted(page_numbers)
    
#     # Combine pages with headers
#     filtered_lines = []
#     for page_num in sorted_pages:
#         if page_num in pages:
#             # Add page header
#             filtered_lines.append('=' * 80)
#             filtered_lines.append(f'PAGE {page_num}')
#             filtered_lines.append('=' * 80)
#             filtered_lines.append('')
#             # Add page content
#             filtered_lines.append(pages[page_num])
#             filtered_lines.append('')  # Blank line between pages
    
#     return '\n'.join(filtered_lines)


# def load_certificate_json(json_path: str) -> Dict:
#     """Load certificate JSON file"""
#     with open(json_path, 'r', encoding='utf-8') as f:
#         return json.load(f)


# def load_policy_text(text_path: str) -> str:
#     """Load policy text file"""
#     with open(text_path, 'r', encoding='utf-8') as f:
#         return f.read()


# def save_filtered_text(filtered_text: str, output_path: str):
#     """Save filtered text to file"""
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(filtered_text)


# def coverage_filter(cert_json_path: str, policy_combo_path: str, output_path: str):
#     """
#     Main function: Filter policy text to only pages containing coverage keywords
    
#     Args:
#         cert_json_path: Path to certificate JSON file
#         policy_combo_path: Path to policy combo text file
#         output_path: Path to save filtered output
#     """
#     print(f"\n{'='*60}")
#     print("Coverage Keyword Filter")
#     print(f"{'='*60}\n")
    
#     # 1. Load certificate JSON
#     print(f"[1/6] Loading certificate: {cert_json_path}")
#     cert_data = load_certificate_json(cert_json_path)
#     coverages = cert_data.get("coverages", [])
#     num_coverages = len(coverages) if coverages else 0
#     print(f"      Found {num_coverages} coverages")
    
#     # 2. Extract keywords
#     print(f"\n[2/6] Extracting keywords from coverages...")
#     keywords = extract_all_keywords(coverages)
#     print(f"      Keywords: {sorted(keywords)}")
#     print(f"      Total unique keywords: {len(keywords)}")
    
#     # 3. Load policy text
#     print(f"\n[3/6] Loading policy: {policy_combo_path}")
#     policy_text = load_policy_text(policy_combo_path)
#     print(f"      Policy size: {len(policy_text)} characters, {len(policy_text.split(chr(10)))} lines")
    
#     # 4. Parse policy into pages
#     print(f"\n[4/6] Parsing policy into pages...")
#     pages = parse_policy_pages(policy_text)
#     print(f"      Found {len(pages)} pages")
    
#     # 5. Find pages with keywords
#     print(f"\n[5/6] Finding pages with coverage keywords...")
#     matching_pages = find_pages_with_keywords(pages, keywords)
#     print(f"      Matching pages: {sorted(matching_pages)}")
#     print(f"      Total pages to include: {len(matching_pages)}")
    
#     # 6. Extract and save filtered text
#     print(f"\n[6/6] Extracting filtered text...")
#     filtered_text = extract_filtered_text(pages, matching_pages)
#     save_filtered_text(filtered_text, output_path)
    
#     # Summary
#     original_lines = len(policy_text.split('\n'))
#     filtered_lines = len(filtered_text.split('\n'))
#     reduction_pct = ((original_lines - filtered_lines) / original_lines) * 100 if original_lines > 0 else 0
    
#     print(f"\n{'='*60}")
#     print("Summary")
#     print(f"{'='*60}")
#     print(f"Original policy:  {original_lines} lines ({len(pages)} pages)")
#     print(f"Filtered policy:  {filtered_lines} lines ({len(matching_pages)} pages)")
#     print(f"Reduction:        {reduction_pct:.1f}%")
#     print(f"Output saved to:  {output_path}")
#     print(f"{'='*60}\n")


# if __name__ == "__main__":
#     # ========== EDIT THESE VALUES ==========
#     cert_prefix = "stay"                      # Certificate name prefix
#     carrier_dir = "nationwideop"              # Carrier directory
#     cert_json_filename = "stay_pl_extracted_real.json"  # Certificate JSON filename
#     # =======================================
    
#     # Construct paths
#     cert_json_path = os.path.join(carrier_dir, cert_json_filename)
#     policy_combo_path = os.path.join(carrier_dir, f"{cert_prefix}_pol_combo.txt")
#     output_path = os.path.join(carrier_dir, f"{cert_prefix}_cov_filtered.txt")
    
#     # Check if files exist
#     if not os.path.exists(cert_json_path):
#         print(f"Error: Certificate JSON not found: {cert_json_path}")
#         exit(1)
    
#     if not os.path.exists(policy_combo_path):
#         print(f"Error: Policy combo text not found: {policy_combo_path}")
#         exit(1)
    
#     # Run filter
#     coverage_filter(cert_json_path, policy_combo_path, output_path)

