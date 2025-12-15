"""
QC System - Heading Detection & Page Extraction with Persistence
Complete regex-based extraction with full output storage and validation
"""
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class HeadingMatch:
    """Data class for heading match"""
    coverage: str
    pattern: str
    match_text: str
    char_position: int
    line_number: int
    page_number: int


@dataclass
class ExtractedSection:
    """Data class for extracted section"""
    coverage: str
    heading_match: dict
    start_page: int
    end_page: int
    page_count: int
    char_start: int
    char_end: int
    content_length: int
    content_preview: str
    validation: dict
    content: str = ""  # Full extracted content


class PolicyPageExtractor:
    """Extract pages from policy OCR text with full validation and logging"""
    
    def __init__(self, policy_text: str, filename: str):
        """Initialize with policy text and filename"""
        self.policy_text = policy_text
        self.filename = filename
        self.pages = self._split_into_pages()
        self.page_boundaries = self._calculate_page_boundaries()
        self.extraction_log = {
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'total_characters': len(policy_text),
            'total_pages': len(self.pages),
            'headings_found': {},
            'sections_extracted': {},
            'errors': []
        }
    
    def _split_into_pages(self) -> List[str]:
        """Count actual pages (not split by markers)"""
        # Count "PAGE X" markers to get actual page count
        page_markers = re.findall(r'PAGE\s+\d+', self.policy_text)
        return page_markers  # Return actual page count (used for len() only)
    
    def _calculate_page_boundaries(self) -> Dict[int, Tuple[int, int]]:
        """Calculate character positions for each page - robust version with multiple fallbacks"""
        boundaries = {}
        
        # Try multiple patterns for PAGE markers (different formats)
        page_patterns = [
            # Standard format: ==========...\nPAGE X\n==========...
            r'={50,}\s*\nPAGE\s+(\d+)\s*\n={50,}',
            # Alternative: PAGE X with just one separator before
            r'={50,}\s*\nPAGE\s+(\d+)\s*\n',
            # Alternative: PAGE X without separators (just newlines)
            r'\nPAGE\s+(\d+)\s*\n',
            # Alternative: Page X (lowercase)
            r'={50,}\s*\nPage\s+(\d+)\s*\n={50,}',
        ]
        
        matches = []
        for pattern in page_patterns:
            found = list(re.finditer(pattern, self.policy_text, re.MULTILINE | re.IGNORECASE))
            if found:
                matches = found
                break  # Use first pattern that finds matches
        
        if not matches:
            # Fallback: try to find any "PAGE X" pattern
            fallback_pattern = r'PAGE\s+(\d+)'
            fallback_matches = list(re.finditer(fallback_pattern, self.policy_text, re.IGNORECASE))
            if fallback_matches:
                matches = fallback_matches
            else:
                # Last resort: treat entire document as page 1
                boundaries[1] = (0, len(self.policy_text))
                self.extraction_log['errors'].append('No PAGE markers found, treating as single page')
                return boundaries
        
        # Process each page marker
        for i, match in enumerate(matches):
            try:
                page_num = int(match.group(1))
            except (ValueError, IndexError):
                continue  # Skip invalid page numbers
            
            # Start of page content is after the PAGE marker
            page_start = match.end()
            
            # End of page is start of next page marker (or end of document)
            if i < len(matches) - 1:
                next_match = matches[i + 1]
                page_end = next_match.start()
            else:
                page_end = len(self.policy_text)
            
            # Handle duplicate page numbers (keep the first occurrence, extend if needed)
            if page_num not in boundaries:
                boundaries[page_num] = (page_start, page_end)
            else:
                # If duplicate, extend the existing boundary if this one is larger
                existing_start, existing_end = boundaries[page_num]
                if page_end > existing_end:
                    boundaries[page_num] = (existing_start, page_end)
        
        # Validate boundaries (ensure no overlaps and proper ordering)
        if boundaries:
            sorted_pages = sorted(boundaries.items(), key=lambda x: x[1][0])
            # Fix any overlaps
            for i in range(len(sorted_pages) - 1):
                current_num, (current_start, current_end) = sorted_pages[i]
                next_num, (next_start, next_end) = sorted_pages[i + 1]
                
                if current_end > next_start:
                    # Overlap detected, adjust
                    boundaries[current_num] = (current_start, next_start)
        
        return boundaries
    
    def get_page_from_char_position(self, char_pos: int) -> int:
        """Get page number from character position - robust version"""
        if not self.page_boundaries:
            return 1
        
        # Sort pages by start position to check in order
        sorted_pages = sorted(self.page_boundaries.items(), key=lambda x: x[1][0])
        
        # Check each page boundary
        for page_num, (start, end) in sorted_pages:
            if start <= char_pos < end:
                return page_num
        
        # If position is before first page, return first page
        if sorted_pages and char_pos < sorted_pages[0][1][0]:
            return sorted_pages[0][0]
        
        # If position is after last page, return last page
        if sorted_pages:
            return sorted_pages[-1][0]
        
        return 1
    
    def get_line_number(self, char_pos: int) -> int:
        """Get line number from character position"""
        return self.policy_text[:char_pos].count('\n') + 1
    
    def extract_pages_after_heading(self, heading_char_pos: int, 
                                   num_pages: int = 4) -> Tuple[str, Dict]:
        """Extract N pages starting from heading position - robust version"""
        start_page = self.get_page_from_char_position(heading_char_pos)
        
        validation = {
            'heading_page': start_page,
            'start_page': start_page,
            'end_page': start_page,
            'pages_requested': num_pages,
            'status': 'success',
            'warnings': []
        }
        
        # Get char positions for requested pages
        if start_page not in self.page_boundaries:
            validation['status'] = 'error'
            validation['error'] = f'Start page {start_page} not found in boundaries'
            return '', validation
        
        start_char = self.page_boundaries[start_page][0]
        
        # Find end position - handle missing pages gracefully
        available_pages = sorted([p for p in self.page_boundaries.keys() if p >= start_page])
        
        if len(available_pages) >= num_pages:
            # We have enough pages
            end_page = available_pages[num_pages - 1]
            end_char = self.page_boundaries[end_page][1]
            validation['end_page'] = end_page
        else:
            # Not enough pages available, extract what we can
            end_page = available_pages[-1] if available_pages else start_page
            end_char = self.page_boundaries[end_page][1] if end_page in self.page_boundaries else len(self.policy_text)
            validation['end_page'] = end_page
            validation['warnings'].append(
                f'Requested {num_pages} pages but only {len(available_pages)} available. Extracted pages {start_page}-{end_page}'
            )
            validation['pages_actually_extracted'] = len(available_pages)
        
        # Ensure end_char is valid
        if end_char > len(self.policy_text):
            end_char = len(self.policy_text)
        
        if start_char >= end_char:
            validation['status'] = 'error'
            validation['error'] = f'Invalid boundaries: start_char {start_char} >= end_char {end_char}'
            return '', validation
        
        extracted_text = self.policy_text[start_char:end_char]
        
        validation['char_start'] = start_char
        validation['char_end'] = end_char
        validation['extracted_length'] = len(extracted_text)
        validation['page_count'] = len(available_pages) if len(available_pages) < num_pages else num_pages
        
        return extracted_text, validation
    
    def find_pages_with_dollar_amounts(self) -> List[int]:
        """
        NEW APPROACH: Find pages that contain dollar amounts >= $200.
        This is carrier-agnostic - every policy uses $ for limits.
        Filters out small amounts (page numbers, references, etc.)
        
        Returns:
            List of page numbers that have at least one $ amount >= $200
        """
        pages_with_dollars = set()
        min_amount = 200  # Only consider amounts >= $200
        
        # Skip patterns (instructional/example pages)
        skip_patterns = ['EXAMPLE', 'CALCULATION', 'HOW TO', 'SAMPLE', 'ILLUSTRATION']
        
        for page_num, (page_start, page_end) in self.page_boundaries.items():
            page_text = self.policy_text[page_start:page_end]
            page_text_upper = page_text.upper()
            
            # Skip if page is instructional/example
            if any(skip in page_text_upper for skip in skip_patterns):
                continue
            
            # Find all dollar amounts on this page
            dollar_matches = re.finditer(r'\$\s*([0-9,]+)', page_text)
            
            # Check if any amount is >= min_amount
            for match in dollar_matches:
                try:
                    # Extract numeric value (remove commas)
                    amount_str = match.group(1).replace(',', '')
                    amount = int(amount_str)
                    
                    # Only add page if amount is significant (>= $200)
                    if amount >= min_amount:
                        pages_with_dollars.add(page_num)
                        break  # Found one significant amount, no need to check more
                except (ValueError, AttributeError):
                    # Skip invalid matches
                    continue
        
        return sorted(pages_with_dollars)

    def merge_page_ranges(self, pages: List[int], buffer: int = 3) -> List[Tuple[int, int]]:
        """
        Add buffer pages and merge overlapping ranges.
        
        Args:
            pages: List of page numbers with $ amounts
            buffer: Number of pages to add before/after each page
        
        Returns:
            List of (start_page, end_page) tuples with no overlaps
        """
        if not pages:
            return []
        
        # Get min and max page numbers in document
        all_pages = sorted(self.page_boundaries.keys())
        min_page = all_pages[0] if all_pages else 1
        max_page = all_pages[-1] if all_pages else 1
        
        # Create ranges with buffer
        ranges = []
        for page in pages:
            start = max(min_page, page - buffer)
            end = min(max_page, page + buffer)
            ranges.append((start, end))
        
        # Sort by start page
        ranges.sort(key=lambda x: x[0])
        
        # Merge overlapping ranges
        merged = []
        for start, end in ranges:
            if merged and start <= merged[-1][1] + 1:
                # Overlaps or adjacent - extend previous range
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                # New separate range
                merged.append((start, end))
        
        return merged

    def find_headings(self) -> Dict[str, List[HeadingMatch]]:
        """
        NEW APPROACH: Find pages with $ amounts + 3-page buffer.
        Carrier-agnostic - works with any policy format.
        No keyword dependency - won't miss endorsements or schedules.
        """
        
        # Step 1: Find all pages with dollar amounts
        dollar_pages = self.find_pages_with_dollar_amounts()
        print(f"[DEBUG] Found {len(dollar_pages)} pages with $ amounts")
        
        # Step 2: Add 3-page buffer and merge overlapping ranges
        merged_ranges = self.merge_page_ranges(dollar_pages, buffer=1)
        print(f"[DEBUG] After 1-page buffer + merge: {len(merged_ranges)} range(s)")
        for start, end in merged_ranges:
            print(f"         Pages {start}-{end} ({end - start + 1} pages)")
        
        # Step 3: Calculate total pages to extract
        total_pages = sum(end - start + 1 for start, end in merged_ranges)
        total_doc_pages = len(self.page_boundaries)
        reduction = ((total_doc_pages - total_pages) / total_doc_pages) * 100 if total_doc_pages > 0 else 0
        print(f"[DEBUG] Total: {total_pages} pages from {total_doc_pages} ({reduction:.1f}% reduction)")
        
        # Step 4: Create HeadingMatch objects for each range
        # All coverages get the same content - LLM will separate
        all_matches = {
            'GL': [],
            'PROPERTY': []
        }
        
        for i, (start_page, end_page) in enumerate(merged_ranges):
            # Get char position for start of this range
            if start_page in self.page_boundaries:
                char_pos = self.page_boundaries[start_page][0]
                line_num = self.get_line_number(char_pos)
                
                for coverage in ['GL', 'PROPERTY']:
                    heading_match = HeadingMatch(
                        coverage=coverage,
                        pattern=f'$ Amount Range {i+1}/{len(merged_ranges)}',
                        match_text=f'Pages {start_page}-{end_page}',
                        char_position=char_pos,
                        line_number=line_num,
                        page_number=start_page
                    )
                    all_matches[coverage].append(heading_match)
        
        for coverage in all_matches:
            self.extraction_log['headings_found'][coverage] = len(merged_ranges)
        
        return all_matches
    
    def process_all_headings(self) -> Dict:
        """Find headings and extract sections"""
        
        print(f"\n{'='*80}")
        print(f"Processing: {self.filename}")
        print(f"{'='*80}\n")
        
        # Find all headings
        headings = self.find_headings()
        
        print(f"[HEADINGS FOUND - TOP 5 matches per coverage]:")
        for coverage, matches in headings.items():
            if matches:
                print(f"  [OK] {coverage}: {len(matches)} match(es) (showing top 5)")
                for match in matches:
                    print(f"     - Page {match.page_number}, Line {match.line_number}: '{match.match_text}'")
            else:
                print(f"  [NOT FOUND] {coverage}: Not found")
        
        # Extract sections
        print(f"\n[EXTRACTING SECTIONS - $ amount pages + 1-page buffer]:\n")
        
        sections = {}
        
        for coverage, matches in headings.items():
            if not matches:
                sections[coverage] = None
                continue
            
            print(f"  {coverage}:")
            print(f"    Found {len(matches)} page range(s): ", end="")
            
            # Extract pages for EACH range and combine them
            # Each match now represents a merged range (e.g., "Pages 9-25")
            combined_text = ""
            page_ranges = []
            char_start = float('inf')
            char_end = 0
            start_page = float('inf')
            end_page = 0
            all_warnings = []
            
            for i, match in enumerate(matches):
                # Parse the range from match_text (format: "Pages X-Y")
                try:
                    range_parts = match.match_text.replace('Pages ', '').split('-')
                    range_start = int(range_parts[0])
                    range_end = int(range_parts[1])
                    num_pages = range_end - range_start + 1
                except:
                    # Fallback if parsing fails
                    range_start = match.page_number
                    range_end = match.page_number + 4
                    num_pages = 5
                
                print(f"{match.match_text}", end=" ")
                
                # Extract the specific page range
                extracted_text, validation = self.extract_pages_after_heading(
                    match.char_position, 
                    num_pages=num_pages
                )
                
                # Combine with previous extractions
                if extracted_text:
                    combined_text += f"\n\n{'='*80}\n[Match {i+1}] Page {match.page_number}\n{'='*80}\n\n"
                    combined_text += extracted_text
                    
                    # Track page ranges
                    page_ranges.append(f"{validation['start_page']}-{validation['end_page']}")
                    char_start = min(char_start, validation['char_start'])
                    char_end = max(char_end, validation['char_end'])
                    start_page = min(start_page, validation['start_page'])
                    end_page = max(end_page, validation['end_page'])
                    
                    if validation.get('warnings'):
                        all_warnings.extend(validation['warnings'])
            
            print()
            
            # Create validation record
            validation = {
                'heading_page': matches[0].page_number,
                'start_page': start_page if start_page != float('inf') else 0,
                'end_page': end_page if end_page != float('inf') else 0,
                'pages_requested': 5,
                'status': 'success' if combined_text else 'failed',
                'warnings': all_warnings,
                'char_start': char_start if char_start != float('inf') else 0,
                'char_end': char_end,
                'extracted_length': len(combined_text),
                'page_count': (end_page - start_page + 1) if start_page != float('inf') else 0,
                'page_ranges': page_ranges,
                'total_matches': len(matches)
            }
            
            # Create section record
            section = ExtractedSection(
                coverage=coverage,
                heading_match=asdict(matches[0]),
                start_page=validation['start_page'],
                end_page=validation['end_page'],
                page_count=validation['page_count'],
                char_start=validation['char_start'],
                char_end=validation['char_end'],
                content_length=len(combined_text),
                content_preview=combined_text[:500],
                validation=validation,
                content=combined_text  # Store COMBINED extracted content from all matches
            )
            
            sections[coverage] = section
            self.extraction_log['sections_extracted'][coverage] = {
                'status': validation['status'],
                'pages': ', '.join(page_ranges) if page_ranges else 'N/A',
                'content_length': len(combined_text),
                'warnings': validation.get('warnings', []),
                'total_matches': len(matches)
            }
            
            print(f"    [OK] Extracted from {len(matches)} matches")
            print(f"    [OK] Page ranges: {', '.join(page_ranges)}")
            print(f"    [OK] Total content length: {len(combined_text):,} chars")
            if validation.get('warnings'):
                print(f"    [WARN] Warnings: {validation['warnings']}")
            print()
        
        return sections
    
    def validate_extractions(self, sections: Dict) -> Dict:
        """Validate all extractions"""
        
        print(f"[VALIDATION RESULTS]:\n")
        
        validation_report = {}
        
        for coverage, section in sections.items():
            if section is None:
                validation_report[coverage] = {
                    'status': 'not_found',
                    'content_length': 0,
                    'valid': False
                }
                print(f"  {coverage}: [NOT FOUND] in policy")
                continue
            
            # Check content length
            is_valid = section.content_length > 100  # At least 100 chars
            
            # Check for expected keywords
            keywords = {
                'GL': ['limit', 'aggregate', 'occurrence'],
                'PROPERTY': ['building', 'property', 'coverage']
            }
            
            found_keywords = []
            if coverage in keywords:
                # Search in FULL content, not just preview
                section_text = section.content.lower()
                for kw in keywords[coverage]:
                    if kw in section_text:
                        found_keywords.append(kw)
            
            validation_report[coverage] = {
                'status': 'extracted',
                'content_length': section.content_length,
                'valid': is_valid,
                'pages_extracted': f"{section.start_page}-{section.end_page}",
                'keywords_found': found_keywords,
                'pages_count': section.page_count
            }
            
            status_icon = "[OK]" if is_valid else "[WARN]"
            print(f"  {coverage}: {status_icon}")
            print(f"    Pages: {section.start_page}-{section.end_page} ({section.page_count} pages)")
            print(f"    Content: {section.content_length:,} chars")
            print(f"    Keywords: {', '.join(found_keywords) if found_keywords else 'none'}")
        
        print()
        return validation_report


def save_results(all_results: Dict):
    """Save ALL extraction results to SINGLE JSON file AND TXT content file"""
    
    output_file = Path("divine_cna_extraction2.json")
    txt_file = Path("divine_cna_extraction2.txt")
    
    # FIRST: Save TXT file (before converting to dict)
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("QC SYSTEM - EXTRACTED CONTENT (5 PAGES PER COVERAGE)\n")
        f.write("="*80 + "\n\n")
        
        for filename, data in all_results.items():
            f.write(f"\nPOLICY FILE: {filename}\n")
            f.write("="*80 + "\n\n")
            
            for coverage in ['GL', 'PROPERTY']:
                section = data['sections'].get(coverage)
                
                if section:
                    heading_page = section.heading_match['page_number'] if isinstance(section.heading_match, dict) else section.heading_match.page_number
                    
                    f.write(f"\n[{coverage}]\n")
                    f.write(f"Heading Page: {heading_page}\n")
                    f.write(f"Extracted Pages: {section.start_page}-{section.end_page} ({section.page_count} pages)\n")
                    f.write(f"Content Length: {section.content_length:,} chars\n")
                    f.write("-"*80 + "\n")
                    f.write(section.content)
                    f.write("\n" + "-"*80 + "\n")
                else:
                    f.write(f"\n[{coverage}]\n")
                    f.write("[NOT FOUND IN POLICY]\n")
                    f.write("-"*80 + "\n")
    
    print(f"\n[OK] Saved extracted content to TXT: {txt_file}")
    print(f"     File size: {txt_file.stat().st_size:,} bytes")
    
    # SECOND: Prepare and save JSON file (converting objects to dicts)
    consolidated = {
        'timestamp': datetime.now().isoformat(),
        'files_processed': len(all_results),
        'results': {}
    }
    
    for filename, data in all_results.items():
        sections_data = {}
        
        for coverage, section in data['sections'].items():
            if section:
                section_dict = asdict(section)
                sections_data[coverage] = section_dict
            else:
                sections_data[coverage] = None
        
        consolidated['results'][filename] = {
            'log': data['log'],
            'sections': sections_data,
            'validation': data['validation']
        }
    
    # Save JSON file
    with open(output_file, 'w') as f:
        json.dump(consolidated, f, indent=2, default=str)
    
    print(f"[OK] Saved extraction results to JSON: {output_file}")
    print(f"     File size: {output_file.stat().st_size:,} bytes")
    print()


def extract_structured_fields_OLD_REMOVED(content: str, coverage: str) -> Dict:
    """REMOVED - Field extraction should be done by LLM on extracted content"""
    fields = {
        # Policy Information
        'policy_number': None,
        'named_insured': None,
        'dba': None,
        'mailing_address': None,
        'policy_period': None,
        'effective_date': None,
        'expiration_date': None,
        'issue_date': None,
        
        # Insurer Information
        'insurer_name': None,
        'insurer_naic': None,
        'insuring_company': None,
        
        # Producer/Agency Information
        'producer_name': None,
        'producer_address': None,
        'producer_contact': None,
        'producer_phone': None,
        'producer_email': None,
        
        # Certificate Holder
        'certificate_holder': None,
        'additional_insured': None,
        'loss_payee': None,
        'mortgagee': None,
        
        # Coverage Limits
        'limits': {},
        
        # Deductibles
        'deductibles': {},
        
        # Premiums
        'premiums': {},
        
        # Locations/Operations
        'locations': [],
        'description_of_operations': None,
        'vehicles': [],
        
        # Coverage Details
        'coverage_form': None,
        'occurrence_or_claims_made': None,
        'aggregate_applies_per': {},  # POLICY, PROJECT, LOC
        'additional_subr_insd_wvd': {},  # INSD, WVD checkboxes
        
        # Certificate Specific
        'certificate_number': None,
        'revision_number': None,
        
        # Workers Compensation
        'workers_compensation': {},
        
        # Auto Liability
        'auto_liability': {},
        
        # Umbrella/Excess
        'umbrella_excess': {},
        
        # Special Provisions
        'special_provisions': [],
        'remarks': None,
        'cancellation_provisions': None,
        
        # Forms and Endorsements
        'forms_endorsements': [],
        
        # Property Certificate Specific
        'perils_insured': {},  # Basic, Broad, Special, Replacement Cost
        'loan_number': None,
        'continued_until_terminated': None,
        'replaces_prior_evidence_dated': None,
        
        # Other
        'classifications': [],
        'premium_basis': None,
    }
    
    content_upper = content.upper()
    
    # Extract Policy Number
    policy_patterns = [
        r'POLICY\s+(?:NUMBER|NO\.?|#)\s*[:_]?\s*([A-Z0-9\-_]+)',
        r'POLICY\s+(?:NUMBER|NO\.?|#)\s*([A-Z0-9\-_]+)',
    ]
    for pattern in policy_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            fields['policy_number'] = match.group(1).strip()
            break
    
    # Extract Named Insured
    insured_patterns = [
        r'NAMED\s+INSURED[:\s]+([^\n]+(?:\n[^\n]+){0,3})',
        r'INSURED[:\s]+([^\n]+(?:\n[^\n]+){0,2})',
    ]
    for pattern in insured_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            insured_text = match.group(1).strip()
            # Split into named insured and DBA if present
            if 'DBA' in insured_text.upper():
                parts = re.split(r'\s+DBA\s*:?\s*', insured_text, flags=re.IGNORECASE)
                fields['named_insured'] = parts[0].strip()
                if len(parts) > 1:
                    fields['dba'] = parts[1].strip()
            else:
                fields['named_insured'] = insured_text
            break
    
    # Extract Mailing Address
    address_patterns = [
        r'MAILING\s+ADDRESS[:\s]+([^\n]+(?:\n[^\n]+){0,2})',
        r'ADDRESS[:\s]+([^\n]+(?:\n[^\n]+){0,2})',
    ]
    for pattern in address_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            fields['mailing_address'] = ' '.join(match.group(1).strip().split())
            break
    
    # Extract Policy Period and Dates
    period_patterns = [
        r'POLICY\s+PERIOD[:\s]+(?:FROM|FROM:)\s+([^\n]+)',
        r'EFFECTIVE\s+DATE[:\s]+([^\n]+)',
        r'ISSUE\s+DATE[:\s]+([^\n]+)',
    ]
    for pattern in period_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            fields['policy_period'] = match.group(1).strip()
            break
    
    # Extract Effective and Expiration Dates
    date_patterns = [
        r'EFFECTIVE\s+DATE[:\s]+([0-9\/\-]+)',
        r'EXPIRATION\s+DATE[:\s]+([0-9\/\-]+)',
        r'POLICY\s+EXP[:\s]+([0-9\/\-]+)',
        r'FROM\s+([0-9\/\-]+)\s+TO\s+([0-9\/\-]+)',
    ]
    for pattern in date_patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            if 'EFFECTIVE' in match.group(0).upper() or 'FROM' in match.group(0).upper():
                fields['effective_date'] = match.group(1).strip()
            elif 'EXPIRATION' in match.group(0).upper() or 'EXP' in match.group(0).upper() or 'TO' in match.group(0).upper():
                if len(match.groups()) > 1:
                    fields['expiration_date'] = match.group(2).strip()
                else:
                    fields['expiration_date'] = match.group(1).strip()
    
    # Extract Issue Date (for certificates)
    issue_date_match = re.search(r'ISSUE\s+DATE[:\s]+([0-9\/\-]+)', content, re.IGNORECASE)
    if issue_date_match:
        fields['issue_date'] = issue_date_match.group(1).strip()
    
    # Extract Loan Number (for property certificates)
    loan_number_match = re.search(r'LOAN\s+NUMBER[:\s]+([^\n]+)', content, re.IGNORECASE)
    if loan_number_match:
        loan_num = loan_number_match.group(1).strip()
        if loan_num and loan_num.upper() not in ['TBD', 'N/A', 'NONE', '']:
            fields['loan_number'] = loan_num
    
    # Extract Limits based on coverage type
    if coverage == 'GL':
        # GL Certificate specific fields - comprehensive extraction
        limit_patterns = {
            'each_occurrence': [
                r'EACH\s+OCCURRENCE\s+LIMIT[^\$]*\$?\s*([0-9,]+)',
                r'EACH\s+OCCURRENCE[^\$]*\$?\s*([0-9,]+)',
            ],
            'general_aggregate': [
                r'GENERAL\s+AGGREGATE\s+LIMIT\s*\([^\)]*\)[^\$]*\$\.?\s*([0-9,]+)',  # With parentheses like "(Other than Products...)"
                r'GENERAL\s+AGGREGATE\s+LIMIT[^\$]*\$\.?\s*([0-9,]+)',  # Must have $ sign
                r'GENERAL\s+AGGREGATE[^\$]*\$\.?\s*([0-9,]+)',  # Must have $ sign
            ],
            'products_completed_operations': [
                r'PRODUCTS\s*[-]?\s*COMP[/]?OP\s+AGG[^\$]*\$?\s*([0-9,]+|INCLUDED)',
                r'PRODUCTS[/]?\s*COMPLETED\s+OPERATIONS[^\$]*\$?\s*([0-9,]+|INCLUDED)',
                r'PRODUCTS[^\$]*AGGREGATE[^\$]*\$?\s*([0-9,]+|INCLUDED)',
            ],
            'personal_advertising_injury': [
                r'PERSONAL\s+[&]?\s*ADV[^\$]*INJURY[^\$]*\$?\s*([0-9,]+)',
                r'PERSONAL\s+[&]?\s*ADVERTISING\s+INJURY[^\$]*\$?\s*([0-9,]+)',
            ],
            'damage_to_rented_premises': [
                r'DAMAGE\s+TO\s+RENTED\s+PREMISES[^\$]*\$?\s*([0-9,]+)',
                r'DAMAGE\s+TO\s+PREMISES\s+RENTED[^\$]*\$?\s*([0-9,]+)',
                r'DAMAGE\s+TO\s+(?:PREMISES\s+)?RENTED[^\$]*\$?\s*([0-9,]+)',
            ],
            'medical_expense': [
                r'MED\s+EXP[^\$]*\$?\s*([0-9,]+)',
                r'MEDICAL\s+EXPENSE\s+LIMIT[^\$]*\$?\s*([0-9,]+)',
                r'MEDICAL\s+EXPENSE[^\$]*\$?\s*([0-9,]+)',
            ],
        }
        
        for limit_name, patterns in limit_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    fields['limits'][limit_name] = value
                    break
        
        # Extract General Aggregate Limit Applies Per (POLICY, PROJECT, LOC)
        aggregate_applies_per = {}
        if re.search(r'AGGREGATE.*APPLIES\s+PER.*POLICY', content, re.IGNORECASE):
            aggregate_applies_per['policy'] = True
        if re.search(r'AGGREGATE.*APPLIES\s+PER.*PROJECT', content, re.IGNORECASE):
            aggregate_applies_per['project'] = True
        if re.search(r'AGGREGATE.*APPLIES\s+PER.*LOC', content, re.IGNORECASE):
            aggregate_applies_per['loc'] = True
        if aggregate_applies_per:
            fields['aggregate_applies_per'] = aggregate_applies_per
        
        # Extract Additional Subr Insd WVD checkboxes
        if re.search(r'ADDL\s+SUBR\s+INSD\s+WVD', content, re.IGNORECASE):
            addl_subr = {}
            if re.search(r'ADDL\s+SUBR.*INSD.*[X✓√]', content, re.IGNORECASE | re.DOTALL):
                addl_subr['insd'] = True
            if re.search(r'ADDL\s+SUBR.*WVD.*[X✓√]', content, re.IGNORECASE | re.DOTALL):
                addl_subr['wvd'] = True
            if addl_subr:
                fields['additional_subr_insd_wvd'] = addl_subr
        
        # Extract Certificate Number and Revision Number
        cert_number_match = re.search(r'CERTIFICATE\s+NUMBER[:\s]+([^\n]+)', content, re.IGNORECASE)
        if cert_number_match:
            cert_num = cert_number_match.group(1).strip()
            if cert_num and cert_num.upper() not in ['TBD', 'N/A', 'NONE', '']:
                fields['certificate_number'] = cert_num
        
        revision_number_match = re.search(r'REVISION\s+NUMBER[:\s]+([^\n]+)', content, re.IGNORECASE)
        if revision_number_match:
            rev_num = revision_number_match.group(1).strip()
            if rev_num and rev_num.upper() not in ['TBD', 'N/A', 'NONE', '']:
                fields['revision_number'] = rev_num
    
    elif coverage == 'PROPERTY':
        # Property certificate specific fields - extract coverage table
        # Extract all coverage types with their amounts and deductibles
        coverage_table_patterns = {
            'building': {
                'amount': r'BUILDING[^\d]*AMOUNT\s+OF\s+INSURANCE[:\s]*([0-9,]+|INCLUDED)',
                'deductible': r'BUILDING[^\d]*DEDUCTIBLE[:\s]*([0-9,]+|[\d%]+)',
                'simple_amount': r'BUILDING[:\s]*([0-9,]+)',
                'simple_deductible': r'BUILDING.*?DEDUCTIBLE[:\s]*([0-9,]+|[\d%]+)',
            },
            'business_personal_property': {
                'amount': r'BUSINESS\s+PERSONAL\s+PROPERTY[^\d]*AMOUNT\s+OF\s+INSURANCE[:\s]*([0-9,]+|INCLUDED)',
                'deductible': r'BUSINESS\s+PERSONAL\s+PROPERTY[^\d]*DEDUCTIBLE[:\s]*([0-9,]+|[\d%]+)',
                'simple_amount': r'BUSINESS\s+PERSONAL\s+PROPERTY[:\s]*([0-9,]+)',
            },
            'business_income': {
                'amount': r'BUSINESS\s+INCOME[^\d]*AMOUNT\s+OF\s+INSURANCE[:\s]*([0-9,]+|ACTUAL\s+LOSS\s+SUSTAINED|INCLUDED)',
                'deductible': r'BUSINESS\s+INCOME[^\d]*DEDUCTIBLE[:\s]*([0-9,]+|[\d%]+)',
                'simple_amount': r'BUSINESS\s+INCOME[:\s]*([0-9,]+|ACTUAL\s+LOSS)',
            },
            'equipment_breakdown': {
                'amount': r'EQUIPMENT\s+BREAKDOWN[^\d]*AMOUNT\s+OF\s+INSURANCE[:\s]*([0-9,]+|INCLUDED)',
                'deductible': r'EQUIPMENT\s+BREAKDOWN[^\d]*DEDUCTIBLE[:\s]*([0-9,]+|[\d%]+)',
            },
            'employee_dishonesty': {
                'amount': r'EMPLOYEE\s+DISHONESTY[^\d]*AMOUNT\s+OF\s+INSURANCE[:\s]*([0-9,]+|INCLUDED)',
                'deductible': r'EMPLOYEE\s+DISHONESTY[^\d]*DEDUCTIBLE[:\s]*([0-9,]+|[\d%]+)',
            },
            'money_securities': {
                'amount': r'MONEY\s+[&]?\s*SECURITIES[^\d]*AMOUNT\s+OF\s+INSURANCE[:\s]*([0-9,]+|INCLUDED)',
                'deductible': r'MONEY\s+[&]?\s*SECURITIES[^\d]*DEDUCTIBLE[:\s]*([0-9,]+|[\d%]+)',
            },
            'pumps_canopy': {
                'amount': r'PUMPS\s+[&]?\s*CANOPY[^\d]*AMOUNT\s+OF\s+INSURANCE[:\s]*([0-9,]+|INCLUDED)',
                'deductible': r'PUMPS\s+[&]?\s*CANOPY[^\d]*DEDUCTIBLE[:\s]*([0-9,]+|[\d%]+)',
            },
            'outdoor_signs': {
                'amount': r'OUTDOOR\s+SIGNS[^\d]*AMOUNT\s+OF\s+INSURANCE[:\s]*([0-9,]+|INCLUDED)',
                'deductible': r'OUTDOOR\s+SIGNS[^\d]*DEDUCTIBLE[:\s]*([0-9,]+|[\d%]+)',
            },
            'windstorm_hail': {
                'amount': r'WINDSTORM\s+OR\s+HAIL[^\d]*AMOUNT\s+OF\s+INSURANCE[:\s]*([0-9,]+|INCLUDED)',
                'deductible': r'WINDSTORM\s+OR\s+HAIL[^\d]*DEDUCTIBLE[:\s]*([0-9,]+|[\d%]+)',
            },
        }
        
        # Extract coverage details
        for coverage_name, patterns in coverage_table_patterns.items():
            coverage_data = {}
            
            # Try to extract amount
            for pattern_key in ['amount', 'simple_amount']:
                if pattern_key in patterns:
                    match = re.search(patterns[pattern_key], content, re.IGNORECASE)
                    if match:
                        coverage_data['amount_of_insurance'] = match.group(1).strip()
                        break
            
            # Try to extract deductible
            for pattern_key in ['deductible', 'simple_deductible']:
                if pattern_key in patterns:
                    match = re.search(patterns[pattern_key], content, re.IGNORECASE)
                    if match:
                        coverage_data['deductible'] = match.group(1).strip()
                        break
            
            if coverage_data:
                fields['limits'][coverage_name] = coverage_data
        
        # Extract Perils Insured (Basic, Broad, Special, Replacement Cost)
        perils_insured = {}
        if re.search(r'\bBASIC\b', content, re.IGNORECASE):
            perils_insured['basic'] = True
        if re.search(r'\bBROAD\b', content, re.IGNORECASE):
            perils_insured['broad'] = True
        if re.search(r'\bSPECIAL\b', content, re.IGNORECASE):
            perils_insured['special'] = True
        if re.search(r'REPLACEMENT\s+COST', content, re.IGNORECASE):
            perils_insured['replacement_cost'] = True
        
        if perils_insured:
            fields['perils_insured'] = perils_insured
        
        # Extract general deductibles if not found in coverage table
        if not fields['deductibles']:
            deductible_patterns = {
                'property_deductible': r'DEDUCTIBLE[:\s]+\$?\s*([0-9,]+)',
                'windstorm_deductible': r'WINDSTORM[^\$]*DEDUCTIBLE[:\s]*([0-9,]+|[\d%]+)',
            }
            for ded_name, pattern in deductible_patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    fields['deductibles'][ded_name] = value
    
    # Extract Locations / Property Description (for property certificates)
    location_patterns = [
        r'LOCATION[/]?DESCRIPTION[:\s]+([^\n]+(?:\n[^\n]+){0,2})',
        r'LOCATION[:\s]+([^\n]+(?:\n[^\n]+){0,2})',
        r'PROPERTY[:\s]+([^\n]+(?:\n[^\n]+){0,2})',
        r'ADDRESS[:\s]+([^\n]+(?:\n[^\n]+){0,2})',
    ]
    for pattern in location_patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            loc_text = match.group(1).strip()
            if loc_text and len(loc_text) > 5:  # Valid address length
                # Avoid duplicates
                if loc_text not in fields['locations']:
                    fields['locations'].append(loc_text)
    
    # Extract Premiums
    premium_patterns = [
        r'PREMIUM[:\s]+\$?\s*([0-9,]+\.?\d*)',
        r'TOTAL[:\s]+\$?\s*([0-9,]+\.?\d*)',
        r'ADVANCE\s+PREMIUM[:\s]+\$?\s*([0-9,]+\.?\d*)',
    ]
    for pattern in premium_patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            match_text = match.group(0).upper()
            if 'ADVANCE' in match_text:
                fields['premiums']['advance_premium'] = match.group(1).strip()
            elif 'PREMIUM' in match_text:
                fields['premiums']['total_premium'] = match.group(1).strip()
            elif 'TOTAL' in match_text:
                fields['premiums']['total'] = match.group(1).strip()
    
    # Extract Insurer Information
    insurer_patterns = [
        r'INSURER[:\s]+([A-Z0-9\s&\-\.]+)',
        r'INSURING\s+COMPANY[:\s]+([A-Z0-9\s&\-\.]+)',
        r'COMPANY[:\s]+([A-Z0-9\s&\-\.]+)',
    ]
    for pattern in insurer_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            fields['insurer_name'] = match.group(1).strip()
            break
    
    # Extract NAIC Number
    naic_patterns = [
        r'NAIC\s+(?:#|NUMBER|NO\.?)[:\s]*([0-9A-Z]+)',
        r'NAIC[:\s]+([0-9A-Z]+)',
    ]
    for pattern in naic_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            fields['insurer_naic'] = match.group(1).strip()
            break
    
    # Extract Producer/Agency Information
    producer_patterns = [
        r'PRODUCER[:\s]+([^\n]+(?:\n[^\n]+){0,2})',
        r'AGENCY[:\s]+([^\n]+(?:\n[^\n]+){0,2})',
    ]
    for pattern in producer_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            fields['producer_name'] = match.group(1).strip()
            break
    
    # Extract Producer Contact Info
    phone_pattern = r'PHONE[:\s]+([0-9\-\(\)\s]+)'
    phone_match = re.search(phone_pattern, content, re.IGNORECASE)
    if phone_match:
        fields['producer_phone'] = phone_match.group(1).strip()
    
    email_pattern = r'E[-]?MAIL[:\s]+([^\s\n]+)'
    email_match = re.search(email_pattern, content, re.IGNORECASE)
    if email_match:
        fields['producer_email'] = email_match.group(1).strip()
    
    # Extract Certificate Holder / Additional Insured
    holder_patterns = [
        r'CERTIFICATE\s+HOLDER[:\s]+([^\n]+)',
        r'ADDITIONAL\s+INSURED[:\s]+([^\n]+)',
        r'LOSS\s+PAYEE[:\s]+([^\n]+)',
        r'MORTGAGEE[:\s]+([^\n]+)',
    ]
    for pattern in holder_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            if 'CERTIFICATE HOLDER' in match.group(0).upper():
                fields['certificate_holder'] = match.group(1).strip()
            elif 'ADDITIONAL INSURED' in match.group(0).upper():
                fields['additional_insured'] = match.group(1).strip()
            elif 'LOSS PAYEE' in match.group(0).upper():
                fields['loss_payee'] = match.group(1).strip()
            elif 'MORTGAGEE' in match.group(0).upper():
                fields['mortgagee'] = match.group(1).strip()
    
    # Extract Occurrence vs Claims-Made
    occurrence_match = re.search(r'(OCCUR|OCCURRENCE)', content, re.IGNORECASE)
    claims_made_match = re.search(r'CLAIMS[-]?MADE', content, re.IGNORECASE)
    if occurrence_match:
        fields['occurrence_or_claims_made'] = 'Occurrence'
    elif claims_made_match:
        fields['occurrence_or_claims_made'] = 'Claims-Made'
    
    # Extract Aggregate Applies Per
    aggregate_per_match = re.search(r'AGGREGATE\s+(?:LIMIT\s+)?APPLIES\s+PER[:\s]+([^\n]+)', content, re.IGNORECASE)
    if aggregate_per_match:
        fields['aggregate_applies_per'] = aggregate_per_match.group(1).strip()
    
    # Extract Description of Operations
    operations_patterns = [
        r'DESCRIPTION\s+OF\s+OPERATIONS[:\s]+([^\n]+(?:\n[^\n]+){0,5})',
        r'OPERATIONS[:\s]+([^\n]+(?:\n[^\n]+){0,3})',
    ]
    for pattern in operations_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            fields['description_of_operations'] = match.group(1).strip()
            break
    
    # Extract Workers Compensation fields
    if 'WORKERS' in content_upper or 'COMPENSATION' in content_upper:
        wc_fields = {}
        wc_excluded_match = re.search(r'PROPRIETOR|PARTNER|EXECUTIVE\s+OFFICER|MEMBER\s+EXCLUDED[:\s]+([YN/A]+)', content, re.IGNORECASE)
        if wc_excluded_match:
            wc_fields['excluded'] = wc_excluded_match.group(1).strip()
        if wc_fields:
            fields['workers_compensation'] = wc_fields
    
    # Extract Auto Liability fields
    if 'AUTO' in content_upper or 'AUTOMOBILE' in content_upper:
        auto_fields = {}
        auto_types = ['ANY AUTO', 'OWNED AUTOS', 'HIRED AUTOS', 'SCHEDULED AUTOS', 'NON-OWNED AUTOS']
        for auto_type in auto_types:
            if auto_type in content_upper:
                auto_fields[auto_type.lower().replace(' ', '_')] = True
        if auto_fields:
            fields['auto_liability'] = auto_fields
    
    # Extract Classifications
    classification_match = re.search(r'CLASS[:\s]+([^\n]+)', content, re.IGNORECASE)
    if classification_match:
        fields['classifications'].append(classification_match.group(1).strip())
    
    # Extract Premium Basis
    premium_basis_match = re.search(r'PREMIUM\s+BASIS[:\s]+([^\n]+)', content, re.IGNORECASE)
    if premium_basis_match:
        fields['premium_basis'] = premium_basis_match.group(1).strip()
    
    # Extract Special Provisions / Remarks
    remarks_patterns = [
        r'REMARKS[:\s]+([^\n]+(?:\n[^\n]+){0,10})',
        r'SPECIAL\s+PROVISIONS[:\s]+([^\n]+(?:\n[^\n]+){0,5})',
        r'SPECIAL\s+CONDITIONS[:\s]+([^\n]+(?:\n[^\n]+){0,5})',
    ]
    for pattern in remarks_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            fields['remarks'] = match.group(1).strip()
            break
    
    # Extract Cancellation Provisions
    cancellation_match = re.search(r'CANCELLATION[:\s]+([^\n]+(?:\n[^\n]+){0,3})', content, re.IGNORECASE)
    if cancellation_match:
        fields['cancellation_provisions'] = cancellation_match.group(1).strip()
    
    # Clean up empty dictionaries/lists and None values
    cleaned_fields = {}
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, dict) and not value:
            continue
        if isinstance(value, list) and not value:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        cleaned_fields[key] = value
    
    return cleaned_fields


# Removed duplicate save_results - keeping only the one below
# All field extraction code above was removed - should be done by LLM


def save_results(all_results: Dict, base_filename: str = None):
    """Save ALL extraction results to SINGLE JSON file AND TXT content file
    
    Args:
        all_results: Dictionary of extraction results
        base_filename: Base filename from OCR input (e.g., "divine_cna_Package_ocr_output3.txt")
                      Will generate output names like: divine_cna_extraction.json
    """
    
    # Generate unique output filenames based on input OCR file
    if base_filename:
        # Extract base name and clean it up
        base_name = Path(base_filename).stem  # Gets filename without extension
        
        # Extract just the carrier name (remove common suffixes like _policy2, _policy, etc.)
        carrier_name = base_name
        # Remove common suffixes
        for suffix in ['_policy2', '_policy', '_ocr', '_Package']:
            if suffix in carrier_name:
                carrier_name = carrier_name.split(suffix)[0]
                break
        
        # Save to encovaop folder
        # Path("encovaop").mkdir(exist_ok=True)
        output_file = Path(f"nationwideop/{carrier_name}_extraction1.json")
        txt_file = Path(f"nationwideop/{carrier_name}_extraction1.txt")
    else:
        # Fallback to default names - save to encovaop folder
        # Path("encovaop").mkdir(exist_ok=True)
        output_file = Path("nationwideop/evergreen_extraction1.json")
        txt_file = Path("nationwideop/evergreen_extraction1.txt")
    
    # FIRST: Save TXT file (before converting to dict)
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("QC SYSTEM - EXTRACTED CONTENT (5 PAGES PER COVERAGE)\n")
        f.write("="*80 + "\n\n")
        
        for filename, data in all_results.items():
            f.write(f"\nPOLICY FILE: {filename}\n")
            f.write("="*80 + "\n\n")
            
            for coverage in ['GL', 'PROPERTY']:
                section = data['sections'].get(coverage)
                
                if section:
                    heading_page = section.heading_match['page_number'] if isinstance(section.heading_match, dict) else section.heading_match.page_number
                    
                    f.write(f"\n[{coverage}]\n")
                    f.write(f"Heading Page: {heading_page}\n")
                    f.write(f"Extracted Pages: {section.start_page}-{section.end_page} ({section.page_count} pages)\n")
                    f.write(f"Content Length: {section.content_length:,} chars\n")
                    f.write("-"*80 + "\n")
                    f.write(section.content)
                    f.write("\n" + "-"*80 + "\n")
                else:
                    f.write(f"\n[{coverage}]\n")
                    f.write("[NOT FOUND IN POLICY]\n")
                    f.write("-"*80 + "\n")
    
    print(f"\n[OK] Saved extracted content to TXT: {txt_file}")
    print(f"     File size: {txt_file.stat().st_size:,} bytes")
    
    # SECOND: Prepare and save JSON file (converting objects to dicts)
    consolidated = {
        'timestamp': datetime.now().isoformat(),
        'files_processed': len(all_results),
        'results': {}
    }
    
    for filename, data in all_results.items():
        sections_data = {}
        
        for coverage, section in data['sections'].items():
            if section:
                section_dict = asdict(section)
                sections_data[coverage] = section_dict
            else:
                sections_data[coverage] = None
        
        consolidated['results'][filename] = {
            'log': data['log'],
            'sections': sections_data,
            'validation': data['validation']
        }
    
    # Save JSON file
    with open(output_file, 'w') as f:
        json.dump(consolidated, f, indent=2, default=str)
    
    print(f"[OK] Saved extraction results to JSON: {output_file}")
    print(f"     File size: {output_file.stat().st_size:,} bytes")
    print()


def main():
    """Main execution"""
    
    print("\n" + "QC SYSTEM - HEADING EXTRACTION & STORAGE ".center(80, "="))
    
    # Test ALL available carrier files
    # Can pass filename as command line argument: python qc_head2.py taj_berkshire_Package_ocr_output.txt
    import sys
    if len(sys.argv) > 1:
        files_to_test = [sys.argv[1]]
    else:
        files_to_test = [
            # NEW OPTIMIZED TESSERACT (output3.txt with 300 DPI + PSM 6)
            "nationwideop/evergreen_ocrp6.txt",            # Aaniya (NEW)
            # OLD OCR OUTPUTS (for comparison)
            # "divine_cna_Package_ocr_output2.txt",          # CNA (OLD - 300 DPI no preprocess)
            # "divine_cna_Package_ocr_output.txt",        Windstorm or Hail Includedmore   # CNA (OLDEST - 200 DPI)
            # "taj_berkshire_Package_ocr_output.txt",        # Berkshire
            # "westside_Package_ocr_output.txt",             # Westside
        ]
    
    all_results = {}
    
    for filename in files_to_test:
        filepath = Path(filename)
        
        if not filepath.exists():
            print(f"[ERROR] File not found: {filename}\n")
            continue
        
        # Load file
        with open(filepath, 'r', encoding='utf-8') as f:
            policy_text = f.read()
        
        # Process
        extractor = PolicyPageExtractor(policy_text, filename)
        sections = extractor.process_all_headings()
        validation = extractor.validate_extractions(sections)
        
        all_results[filename] = {
            'sections': sections,
            'validation': validation,
            'log': extractor.extraction_log
        }
    
    # Save ALL results to SINGLE file (with unique names per carrier)
    # Use first filename as base for output naming
    base_filename = files_to_test[0] if files_to_test else None
    save_results(all_results, base_filename=base_filename)
    
    print("="*80)
    print("[DONE] EXTRACTION COMPLETE - All results saved to files")
    print("="*80 + "\n")
    
    return all_results


if __name__ == "__main__":
    results = main()

