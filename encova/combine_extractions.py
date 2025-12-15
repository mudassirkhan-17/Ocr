"""
Combine Tesseract and PyMuPDF extraction files for LLM processing
Takes two extraction files and combines them with clear source markers
Supports two modes: simple concatenation or page-by-page interleaving
"""

import sys
import re
from pathlib import Path
from typing import Optional, List, Tuple


def extract_pages_from_content(content: str) -> List[Tuple[int, str]]:
    """
    Extract individual pages from extraction content
    Detects ALL page markers simultaneously:
    - Standard: ==========...\nPAGE X\n==========...
    - Match format: [Match N] Page X
    
    Returns list of (page_number, page_content) tuples
    """
    # Find ALL page markers of ALL types simultaneously
    all_markers = []
    
    # Pattern 1: Standard PAGE X with equals separators (most common in OCR files)
    for match in re.finditer(r'={50,}\s*\nPAGE\s+(\d+)\s*\n={50,}', content, re.MULTILINE | re.IGNORECASE):
        all_markers.append((match.start(), match.end(), int(match.group(1))))
    
    # Pattern 2: [Match N] Page X with equals (from QC head scripts)
    for match in re.finditer(r'={50,}\s*\n\[Match\s+\d+\]\s+Page\s+(\d+)\s*\n={50,}', content, re.MULTILINE | re.IGNORECASE):
        all_markers.append((match.start(), match.end(), int(match.group(1))))
    
    if not all_markers:
        # Fallback: try simpler patterns
        for match in re.finditer(r'\nPAGE\s+(\d+)\s*\n', content, re.MULTILINE | re.IGNORECASE):
            all_markers.append((match.start(), match.end(), int(match.group(1))))
    
    if not all_markers:
        # No page markers found, treat as single page
        return [(1, content)]
    
    # Sort markers by position in file
    all_markers.sort(key=lambda x: x[0])
    
    # Extract pages - keep first occurrence of each page number
    pages = []
    seen_pages = set()
    
    for i, (marker_start, marker_end, page_num) in enumerate(all_markers):
        # Skip if we've already seen this page number
        if page_num in seen_pages:
            continue
        seen_pages.add(page_num)
        
        # Get content from AFTER this marker to next marker (or end of file)
        if i < len(all_markers) - 1:
            page_end = all_markers[i + 1][0]  # Start of next marker
        else:
            page_end = len(content)
        
        page_content = content[marker_end:page_end].strip()
        pages.append((page_num, page_content))
    
    return pages


def combine_extraction_files(
    tesseract_file: str,
    pymupdf_file: str,
    output_file: Optional[str] = None,
    interleave_pages: bool = True
) -> str:
    """
    Combine two extraction files with clear source markers
    
    Args:
        tesseract_file: Path to Tesseract extraction file
        pymupdf_file: Path to PyMuPDF extraction file
        output_file: Output file path (auto-generated if None)
    
    Returns:
        Path to combined output file
    """
    
    tesseract_path = Path(tesseract_file)
    pymupdf_path = Path(pymupdf_file)
    
    # Validate input files
    if not tesseract_path.exists():
        raise FileNotFoundError(f"Tesseract file not found: {tesseract_file}")
    if not pymupdf_path.exists():
        raise FileNotFoundError(f"PyMuPDF file not found: {pymupdf_file}")
    
    # Read both files
    print(f"Reading Tesseract extraction: {tesseract_file}")
    with open(tesseract_path, 'r', encoding='utf-8') as f:
        tesseract_content = f.read()
    
    print(f"Reading PyMuPDF extraction: {pymupdf_file}")
    with open(pymupdf_path, 'r', encoding='utf-8') as f:
        pymupdf_content = f.read()
    
    # Generate output filename if not provided
    if output_file is None:
        # Save to encovaop folder
        # Path("encovaop").mkdir(exist_ok=True)
        output_file = Path("encovaop/salem_combined.txt")
    else:
        # Save to encovaop folder
        # Path("encovaop").mkdir(exist_ok=True)
        output_file = Path(f"encovaop/{output_file}")
    
    output_path = Path(output_file)
    
    # Combine with clear markers
    print(f"\nCombining files...")
    combined_content = []
    
    # Header
    combined_content.append("="*80)
    combined_content.append("COMBINED EXTRACTION - TESSERACT + PYMUPDF")
    combined_content.append("="*80)
    combined_content.append("")
    combined_content.append("This document contains two extraction sources:")
    combined_content.append("1. TESSERACT (OCR with buffer=1)")
    combined_content.append("2. PYMUPDF (OCRmyPDF extraction with buffer=0)")
    combined_content.append("")
    combined_content.append("Use the most complete/accurate version when sources differ.")
    combined_content.append("")
    combined_content.append("="*80)
    combined_content.append("")
    
    if interleave_pages:
        # Page-by-page interleaving mode
        print("Mode: Page-by-page interleaving")
        
        # Extract pages from both sources
        tesseract_pages = extract_pages_from_content(tesseract_content)
        pymupdf_pages = extract_pages_from_content(pymupdf_content)
        
        # Create page lookup dictionaries
        tesseract_dict = {page_num: content for page_num, content in tesseract_pages}
        pymupdf_dict = {page_num: content for page_num, content in pymupdf_pages}
        
        # Get all unique page numbers
        all_pages = sorted(set(list(tesseract_dict.keys()) + list(pymupdf_dict.keys())))
        
        print(f"   Found {len(tesseract_pages)} Tesseract pages")
        print(f"   Found {len(pymupdf_pages)} PyMuPDF pages")
        print(f"   Combining {len(all_pages)} unique pages")
        
        # Interleave pages
        for page_num in all_pages:
            combined_content.append("="*80)
            combined_content.append(f"PAGE {page_num}")
            combined_content.append("="*80)
            combined_content.append("")
            
            # Tesseract version
            if page_num in tesseract_dict:
                combined_content.append("--- TESSERACT (Buffer=1) ---")
                combined_content.append("")
                combined_content.append(tesseract_dict[page_num])
                combined_content.append("")
            else:
                combined_content.append("--- TESSERACT (Buffer=1) ---")
                combined_content.append("[Page not found in Tesseract extraction]")
                combined_content.append("")
            
            # PyMuPDF version
            if page_num in pymupdf_dict:
                combined_content.append("--- PYMUPDF (Buffer=0) ---")
                combined_content.append("")
                combined_content.append(pymupdf_dict[page_num])
                combined_content.append("")
            else:
                combined_content.append("--- PYMUPDF (Buffer=0) ---")
                combined_content.append("[Page not found in PyMuPDF extraction]")
                combined_content.append("")
            
            combined_content.append("")
    else:
        # Simple concatenation mode
        print("Mode: Simple concatenation (all Tesseract, then all PyMuPDF)")
        
        # Tesseract section
        combined_content.append("="*80)
        combined_content.append("SOURCE 1: TESSERACT EXTRACTION (Buffer=1)")
        combined_content.append("="*80)
        combined_content.append("")
        combined_content.append(tesseract_content)
        combined_content.append("")
        combined_content.append("="*80)
        combined_content.append("END OF TESSERACT EXTRACTION")
        combined_content.append("="*80)
        combined_content.append("")
        combined_content.append("")
        
        # PyMuPDF section
        combined_content.append("="*80)
        combined_content.append("SOURCE 2: PYMUPDF EXTRACTION (Buffer=0)")
        combined_content.append("="*80)
        combined_content.append("")
        combined_content.append(pymupdf_content)
        combined_content.append("")
        combined_content.append("="*80)
        combined_content.append("END OF PYMUPDF EXTRACTION")
        combined_content.append("="*80)
    
    # Write combined file
    print(f"Writing combined file: {output_file}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(combined_content))
    
    # Calculate stats
    tesseract_chars = len(tesseract_content)
    pymupdf_chars = len(pymupdf_content)
    combined_chars = len('\n'.join(combined_content))
    tesseract_tokens = tesseract_chars // 4
    pymupdf_tokens = pymupdf_chars // 4
    combined_tokens = combined_chars // 4
    
    print(f"\n✅ Combination complete!")
    print(f"   Tesseract: {tesseract_chars:,} chars (~{tesseract_tokens:,} tokens)")
    print(f"   PyMuPDF:   {pymupdf_chars:,} chars (~{pymupdf_tokens:,} tokens)")
    print(f"   Combined:  {combined_chars:,} chars (~{combined_tokens:,} tokens)")
    print(f"   Output:     {output_path.absolute()}")
    print()
    
    return str(output_path)


def main():
    """Main function with command-line argument parsing"""
    
    print("\n" + "="*80)
    print("COMBINE EXTRACTION FILES - TESSERACT + PYMUPDF")
    print("="*80)
    print()
    
    # Parse command line arguments
    interleave = True  # Default to page-by-page interleaving
    
    if len(sys.argv) < 3:
        print("Usage: python combine_extractions.py <tesseract_file> <pymupdf_file> [output_file] [--simple]")
        print()
        print("Options:")
        print("  --simple    Use simple concatenation instead of page-by-page interleaving")
        print()
        print("Examples:")
        print("  python combine_extractions.py tesseract_extraction.txt pymupdf_extraction.txt")
        print("  python combine_extractions.py tesseract_extraction.txt pymupdf_extraction.txt combined.txt")
        print("  python combine_extractions.py tesseract_extraction.txt pymupdf_extraction.txt combined.txt --simple")
        print()
        
        # Try to find default files if not provided
        default_tesseract = "encovaop/salem_extraction1.txt"  # Tesseract with buffer=1
        default_pymupdf = "encovaop/salem_extraction0.txt"  # PyMuPDF with buffer=0
        
        tesseract_path = Path(default_tesseract)
        pymupdf_path = Path(default_pymupdf)
        
        if tesseract_path.exists() and pymupdf_path.exists():
            print(f"Found default files, using:")
            print(f"  Tesseract: {default_tesseract}")
            print(f"  PyMuPDF:   {default_pymupdf}")
            print()
            
            # Check for --simple flag
            interleave = '--simple' not in sys.argv
            
            output_file = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else None
            combine_extraction_files(default_tesseract, default_pymupdf, output_file, interleave_pages=interleave)
        else:
            print("No default files found. Please provide file paths.")
            return
    else:
        tesseract_file = sys.argv[1]
        pymupdf_file = sys.argv[2]
        
        # Parse remaining arguments
        output_file = None
        interleave = True
        
        for arg in sys.argv[3:]:
            if arg == '--simple':
                interleave = False
            elif not arg.startswith('--'):
                output_file = arg
        
        combine_extraction_files(tesseract_file, pymupdf_file, output_file, interleave_pages=interleave)


if __name__ == "__main__":
    main()

