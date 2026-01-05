"""
Unified Policy OCR + Extraction Pipeline
Takes a single input name, runs both Tesseract and PyMuPDF extractions,
saves with consistent naming in carrier directory

Usage:
    python policy_extract.py aaniya_policy
    python policy_extract.py aaniya_policy.pdf
    python policy_extract.py nationwide/aaniya_policy.pdf --max-pages 25

Outputs:
    - {carrier_dir}/{base_name}_pol1.txt (Tesseract extraction)
    - {carrier_dir}/{base_name}_pol2.txt (PyMuPDF extraction)
    Note: "_policy" suffix is automatically stripped from base name
"""

import os
import sys
import subprocess
import time
import re
from pathlib import Path
from typing import Optional, List, Tuple

# Try to import joblib for parallelization
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Try to import required libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("⚠️  PyMuPDF not installed. Install with: pip install pymupdf")

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️  Tesseract/PIL not installed. Install with: pip install pytesseract pillow")

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def extract_base_name(input_path: str) -> str:
    """
    Extract base name from input (removes .pdf, paths, etc.)
    
    Examples:
        "aaniya_policy" -> "aaniya_policy"
        "aaniya_policy.pdf" -> "aaniya_policy"
        "nationwide/aaniya_policy.pdf" -> "aaniya_policy"
    """
    path = Path(input_path)
    # Remove extension and path
    base_name = path.stem
    return base_name


def find_pdf_file(base_name: str, pdf_dir: str = "nationwide") -> Optional[Path]:
    """
    Find PDF file by trying different locations and extensions
    
    Tries:
        1. {base_name}.pdf (current dir)
        2. {pdf_dir}/{base_name}.pdf
        4. {base_name} (as-is, if it's already a full path)
    """
    candidates = [
        Path(f"{base_name}.pdf"),
        Path(f"{pdf_dir}/{base_name}.pdf"),
        Path(base_name),  # In case it's already a full path
    ]
    
    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() == '.pdf':
            return candidate
    
    return None


def pdf_page_to_image(pdf_path: str, page_num: int, dpi: int = 100):
    """Convert a single PDF page to PIL Image"""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)  # 0-indexed
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def process_single_page_tesseract(pdf_path: str, page_num: int):
    """Process a single page with Tesseract OCR (for parallel processing)"""
    try:
        image = pdf_page_to_image(pdf_path, page_num, dpi=100)
        custom_config = r'--oem 1 --psm 6 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(image, config=custom_config)
        return (page_num, text, None)
    except Exception as e:
        return (page_num, None, str(e))


def process_single_page_tesseract_psm3(pdf_path: str, page_num: int):
    """Process a single page with Tesseract OCR using PSM 3 (for fallback)"""
    try:
        image = pdf_page_to_image(pdf_path, page_num, dpi=100)
        custom_config = r'--oem 1 --psm 3 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(image, config=custom_config)
        return (page_num, text, None)
    except Exception as e:
        return (page_num, None, str(e))


def _is_text_gibberish_or_empty(text: str, min_length: int = 10, gibberish_threshold: float = 0.3) -> bool:
    """Check if text is null, empty, gibberish, or mostly whitespace"""
    if not text or len(text.strip()) < min_length:
        return True
    
    sample = text[:500] if len(text) > 500 else text
    
    # Check for mostly whitespace (spaces, tabs, newlines)
    non_whitespace = sum(1 for c in sample if not c.isspace())
    if non_whitespace / len(sample) < 0.1:  # Less than 10% actual content = mostly whitespace
        return True
    
    # Check for non-printable characters (excluding newlines/tabs)
    non_printable = sum(1 for c in sample if ord(c) < 32 and c not in '\n\r\t')
    ratio = non_printable / len(sample) if sample else 0
    
    if ratio > gibberish_threshold:
        return True
    
    # Check for actual English letters (a-z, A-Z) - most important for readability
    english_letters = sum(1 for c in sample if c.isalpha() and ord(c) < 128)
    if english_letters / len(sample) < 0.2:  # Less than 20% English letters = likely corrupted
        return True
    
    # Check for too many special/weird characters (corrupted OCR artifacts)
    # Count characters that are not: letters, digits, common punctuation, spaces
    weird_chars = sum(1 for c in sample if not (c.isalnum() or c in '.,;:!?()-\'" \n\r\t' or ord(c) < 128))
    if weird_chars / len(sample) > 0.3:  # More than 30% weird chars = corrupted
        return True
    
    return False


def extract_tesseract(pdf_path: Path, output_path: Path, n_jobs: int = -1, max_pages: Optional[int] = None) -> bool:
    """
    Extract text from PDF using Tesseract OCR
    
    Args:
        pdf_path: Path to PDF file
        output_path: Output text file path
        n_jobs: Number of parallel workers (-1 for all cores)
        max_pages: Process only first N pages (None = all pages)
    """
    if not TESSERACT_AVAILABLE:
        print("❌ Tesseract extraction skipped (dependencies not available)")
        return False
    
    print("="*60)
    print("TESSERACT EXTRACTION")
    print("="*60)
    print(f"Input:  {pdf_path}")
    print(f"Output: {output_path}\n")
    
    start_time = time.time()
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        # Limit pages if requested
        num_pages = min(total_pages, max_pages) if max_pages else total_pages
        
        if max_pages and total_pages > max_pages:
            print(f"✅ PDF has {total_pages} pages (processing first {num_pages} only)\n")
        else:
            print(f"✅ PDF has {num_pages} pages\n")
    except Exception as e:
        print(f"❌ Error opening PDF: {e}")
        return False
    
    print(f"Extracting text using Tesseract OCR ({n_jobs if n_jobs != -1 else 'all'} workers)...")
    
    # Process pages in parallel
    if JOBLIB_AVAILABLE:
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_single_page_tesseract)(str(pdf_path), page_num)
            for page_num in range(1, num_pages + 1)
        )
    else:
        # Sequential fallback
        results = [process_single_page_tesseract(str(pdf_path), page_num) 
                   for page_num in range(1, num_pages + 1)]
    
    # Sort results by page number and build output
    results.sort(key=lambda x: x[0])
    all_text = []
    
    for page_num, text, error in results:
        all_text.append(f"\n{'='*80}\n")
        all_text.append(f"PAGE {page_num}\n")
        all_text.append(f"{'='*80}\n")
        
        if error:
            all_text.append(f"\n[ERROR ON PAGE {page_num}: {error}]\n")
            print(f"⚠️  Error on page {page_num}: {error}")
        else:
            all_text.append(text)
    
    # Save to file
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(all_text))
        
        elapsed = time.time() - start_time
        file_size = len(''.join(all_text))
        print(f"\n✅ Tesseract extraction completed in {elapsed:.2f} seconds")
        print(f"   Saved {file_size:,} characters ({file_size/1024:.2f} KB)")
        return True
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False


def run_ocrmypdf(input_pdf: Path, output_pdf: Optional[Path] = None, 
                 force_ocr: bool = False, smart_mode: bool = True) -> Optional[Path]:
    """
    Step 1: Convert scanned PDF to OCR-able PDF using OCRmyPDF
    """
    if output_pdf is None:
        # Create temp OCR'd PDF in same directory
        output_pdf = input_pdf.parent / f"{input_pdf.stem}_ocr{input_pdf.suffix}"
    
    print("="*60)
    print("OCR PROCESSING (OCRmyPDF)")
    print("="*60)
    print(f"Input:  {input_pdf}")
    print(f"Output: {output_pdf}\n")
    
    cmd = [sys.executable, '-m', 'ocrmypdf']
    
    if force_ocr:
        cmd.append('--force-ocr')
        print("Mode: Force OCR (replaces ALL text)")
    elif smart_mode:
        cmd.append('--skip-text')
        print("Mode: Smart OCR (auto-detects pages needing OCR)")
    
    cmd.extend([str(input_pdf), str(output_pdf)])
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.time() - start_time
    
    if result.returncode == 0 or (result.returncode in [6, 10] and output_pdf.exists()):
        print(f"✅ OCR completed in {elapsed:.2f} seconds\n")
        return output_pdf
    else:
        print(f"⚠️  OCR completed with warnings (exit code {result.returncode})")
        if output_pdf.exists():
            print("✅ Output file created, continuing...\n")
            return output_pdf
        else:
            print("❌ No output file created\n")
            return None


def extract_pymupdf(pdf_path: Path, output_path: Path, 
                    use_ocr: bool = True, force_ocr: bool = False,
                    max_pages: Optional[int] = None,
                    n_jobs: int = -1) -> bool:
    """
    Extract text from PDF using PyMuPDF
    Falls back to Tesseract PSM 3 if PyMuPDF extraction is corrupted/gibberish
    
    Args:
        pdf_path: Path to PDF file
        output_path: Output text file path
        use_ocr: If True, run OCRmyPDF first to make PDF searchable
        force_ocr: If True, force OCR on all pages
        max_pages: Process only first N pages (None = all pages)
        n_jobs: Number of parallel workers (-1 for all cores) for fallback
    """
    if not PYMUPDF_AVAILABLE:
        print("❌ PyMuPDF extraction skipped (dependencies not available)")
        return False
    
    print("="*60)
    print("PYMUPDF EXTRACTION")
    print("="*60)
    
    # Strategy: Try original PDF first (may already have text)
    # Only use OCR if original has no/minimal text
    # This avoids OCRmyPDF corruption issues
    print(f"Input:  {pdf_path}")
    print(f"Output: {output_path}\n")
    print("Strategy: Extract from original PDF (skip OCRmyPDF to avoid corruption)")
    print("Note: Tesseract pipeline already handles OCR separately\n")
    
    start_time = time.time()
    
    try:
        # Extract directly from original PDF (no OCR preprocessing)
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # Limit pages if requested
        num_pages = min(total_pages, max_pages) if max_pages else total_pages
        
        if max_pages and total_pages > max_pages:
            print(f"✅ PDF has {total_pages} pages (processing first {num_pages} only)\n")
        else:
            print(f"✅ PDF has {num_pages} pages\n")
        
        all_text = []
        corrupted_pages = 0
        for page_num in range(num_pages):
            print(f"Extracting page {page_num + 1}/{num_pages}...", end=" ", flush=True)
            page = doc[page_num]
            
            # Try multiple extraction methods for better OCR text handling
            text = None
            
            # Method 1: Try "dict" mode - most reliable for OCR'd PDFs
            try:
                text_dict = page.get_text("dict")
                text_blocks = []
                for block in text_dict.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text = " ".join(span.get("text", "") for span in line.get("spans", []))
                            if line_text.strip():
                                text_blocks.append(line_text)
                text = "\n".join(text_blocks)
                # Validate text quality - check for too many non-printable characters
                if text:
                    sample = text[:500] if len(text) > 500 else text
                    non_printable = sum(1 for c in sample if ord(c) < 32 and c not in '\n\r\t')
                    if non_printable > len(sample) * 0.2:  # More than 20% non-printable = likely corrupted
                        text = None
            except Exception as e:
                text = None
            
            # Method 2: Try explicit "text" mode
            if not text or len(text.strip()) == 0:
                try:
                    text = page.get_text("text")
                    # Validate text quality
                    if text:
                        sample = text[:500] if len(text) > 500 else text
                        non_printable = sum(1 for c in sample if ord(c) < 32 and c not in '\n\r\t')
                        if non_printable > len(sample) * 0.2:
                            text = None
                except:
                    text = None
            
            # Method 3: Fallback to default get_text()
            if not text or len(text.strip()) == 0:
                try:
                    text = page.get_text()
                except:
                    text = ""
            
            # Clean up any remaining encoding issues
            if text:
                # Remove or replace problematic characters
                text = ''.join(c if (ord(c) >= 32 or c in '\n\r\t') else ' ' for c in text)
                # Collapse multiple spaces
                text = re.sub(r' +', ' ', text)
            
            # Check if this page's text is corrupted/gibberish
            if _is_text_gibberish_or_empty(text):
                corrupted_pages += 1
            
            all_text.append(f"\n{'='*60}\nPAGE {page_num + 1}\n{'='*60}\n{text}")
            print(f"✅ ({len(text)} chars)")
        
        doc.close()
        
        # Check overall quality
        total_chars = sum(len(t) for t in all_text)
        corruption_ratio = corrupted_pages / num_pages if num_pages > 0 else 0
        
        # If PyMuPDF is corrupted, fallback to Tesseract PSM 3
        if corruption_ratio > 0.5 or total_chars < 1000:
            print(f"\n⚠️  PyMuPDF extraction appears corrupted:")
            print(f"   - {corrupted_pages}/{num_pages} pages corrupted ({corruption_ratio*100:.1f}%)")
            print(f"   - Total characters: {total_chars:,}")
            print(f"   - Falling back to Tesseract PSM 3...")
            
            if not TESSERACT_AVAILABLE:
                print("❌ Tesseract not available for fallback")
                return False
            
            # Run Tesseract PSM 3 extraction
            print(f"\nRunning Tesseract PSM 3 extraction (fallback)...")
            
            if JOBLIB_AVAILABLE:
                results = Parallel(n_jobs=n_jobs, verbose=10)(
                    delayed(process_single_page_tesseract_psm3)(str(pdf_path), page_num)
                    for page_num in range(1, num_pages + 1)
                )
            else:
                results = [process_single_page_tesseract_psm3(str(pdf_path), page_num) 
                           for page_num in range(1, num_pages + 1)]
            
            # Sort and build output
            results.sort(key=lambda x: x[0])
            all_text = []
            
            for page_num, text, error in results:
                all_text.append(f"\n{'='*80}\n")
                all_text.append(f"PAGE {page_num}\n")
                all_text.append(f"{'='*80}\n")
                
                if error:
                    all_text.append(f"\n[ERROR ON PAGE {page_num}: {error}]\n")
                else:
                    all_text.append(text)
            
            # Save Tesseract PSM 3 output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(''.join(all_text))
            
            elapsed = time.time() - start_time
            file_size = len(''.join(all_text))
            print(f"\n✅ Tesseract PSM 3 fallback completed in {elapsed:.2f} seconds")
            print(f"   Saved {file_size:,} characters ({file_size/1024:.2f} KB)")
            return True
        
        # PyMuPDF was good, save it
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_text))
        
        elapsed = time.time() - start_time
        print(f"\n✅ PyMuPDF extraction completed in {elapsed:.2f} seconds")
        print(f"   Saved {total_chars:,} characters ({total_chars/1024:.2f} KB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try Tesseract PSM 3 fallback even on exception
        if TESSERACT_AVAILABLE:
            print(f"\n⚠️  Attempting Tesseract PSM 3 fallback...")
            try:
                doc = fitz.open(pdf_path)
                num_pages = min(len(doc), max_pages) if max_pages else len(doc)
                doc.close()
                
                if JOBLIB_AVAILABLE:
                    results = Parallel(n_jobs=n_jobs, verbose=10)(
                        delayed(process_single_page_tesseract_psm3)(str(pdf_path), page_num)
                        for page_num in range(1, num_pages + 1)
                    )
                else:
                    results = [process_single_page_tesseract_psm3(str(pdf_path), page_num) 
                               for page_num in range(1, num_pages + 1)]
                
                results.sort(key=lambda x: x[0])
                all_text = []
                for page_num, text, error in results:
                    all_text.append(f"\n{'='*80}\n")
                    all_text.append(f"PAGE {page_num}\n")
                    all_text.append(f"{'='*80}\n")
                    
                    if error:
                        all_text.append(f"\n[ERROR ON PAGE {page_num}: {error}]\n")
                    else:
                        all_text.append(text)
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(''.join(all_text))
                
                print(f"✅ Tesseract PSM 3 fallback successful")
                return True
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
        
        return False


def extract_pages_from_content(content: str) -> List[Tuple[int, str]]:
    """
    Extract individual pages from extraction content
    Detects page markers: ==========...\nPAGE X\n==========...
    """
    all_markers = []
    
    # Pattern 1: Standard PAGE X with equals separators
    for match in re.finditer(r'={50,}\s*\nPAGE\s+(\d+)\s*\n={50,}', content, re.MULTILINE | re.IGNORECASE):
        all_markers.append((match.start(), match.end(), int(match.group(1))))
    
    if not all_markers:
        # Fallback: try simpler patterns
        for match in re.finditer(r'\nPAGE\s+(\d+)\s*\n', content, re.MULTILINE | re.IGNORECASE):
            all_markers.append((match.start(), match.end(), int(match.group(1))))
    
    if not all_markers:
        return [(1, content)]
    
    all_markers.sort(key=lambda x: x[0])
    
    pages = []
    seen_pages = set()
    
    for i, (marker_start, marker_end, page_num) in enumerate(all_markers):
        if page_num in seen_pages:
            continue
        seen_pages.add(page_num)
        
        if i < len(all_markers) - 1:
            page_end = all_markers[i + 1][0]
        else:
            page_end = len(content)
        
        page_content = content[marker_end:page_end].strip()
        pages.append((page_num, page_content))
    
    return pages


def combine_extractions(tesseract_file: Path, pymupdf_file: Path, 
                       output_file: Path, interleave_pages: bool = True) -> bool:
    """
    Combine Tesseract and PyMuPDF extraction files
    """
    if not tesseract_file.exists():
        print(f"⚠️  Tesseract file not found: {tesseract_file}")
        return False
    if not pymupdf_file.exists():
        print(f"⚠️  PyMuPDF file not found: {pymupdf_file}")
        return False
    
    print("="*60)
    print("COMBINING EXTRACTIONS")
    print("="*60)
    print(f"Tesseract: {tesseract_file}")
    print(f"PyMuPDF:   {pymupdf_file}")
    print(f"Output:    {output_file}\n")
    
    # Read both files
    with open(tesseract_file, 'r', encoding='utf-8') as f:
        tesseract_content = f.read()
    
    with open(pymupdf_file, 'r', encoding='utf-8') as f:
        pymupdf_content = f.read()
    
    combined_content = []
    
    # Header
    combined_content.append("="*80)
    combined_content.append("COMBINED EXTRACTION - TESSERACT + PYMUPDF")
    combined_content.append("="*80)
    combined_content.append("")
    
    if interleave_pages:
        # Page-by-page interleaving mode
        print("Mode: Page-by-page interleaving")
        
        tesseract_pages = extract_pages_from_content(tesseract_content)
        pymupdf_pages = extract_pages_from_content(pymupdf_content)
        
        tesseract_dict = {page_num: content for page_num, content in tesseract_pages}
        pymupdf_dict = {page_num: content for page_num, content in pymupdf_pages}
        
        all_pages = sorted(set(list(tesseract_dict.keys()) + list(pymupdf_dict.keys())))
        
        print(f"   Found {len(tesseract_pages)} Tesseract pages")
        print(f"   Found {len(pymupdf_pages)} PyMuPDF pages")
        print(f"   Combining {len(all_pages)} unique pages\n")
        
        for page_num in all_pages:
            combined_content.append("="*80)
            combined_content.append(f"PAGE {page_num}")
            combined_content.append("="*80)
            combined_content.append("")
            
            if page_num in tesseract_dict:
                combined_content.append("--- TESSERACT (Buffer=1) ---")
                combined_content.append("")
                combined_content.append(tesseract_dict[page_num])
                combined_content.append("")
            else:
                combined_content.append("--- TESSERACT (Buffer=1) ---")
                combined_content.append("[Page not found in Tesseract extraction]")
                combined_content.append("")
            
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
        print("Mode: Simple concatenation\n")
        
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
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(combined_content))
    
    combined_chars = len('\n'.join(combined_content))
    print(f"✅ Combined file saved: {combined_chars:,} characters ({combined_chars/1024:.2f} KB)")
    return True


def main():
    """
    Main function: unified policy extraction pipeline
    """
    print("\n" + "="*80)
    print("UNIFIED POLICY OCR + EXTRACTION PIPELINE")
    print("="*80)
    print()
    
    # Parse command line arguments
    input_name = None
    force_ocr = False
    skip_ocr = False
    n_jobs = -1
    max_pages = None
    
    i = 0
    while i < len(sys.argv[1:]):
        arg = sys.argv[1:][i]
        if arg == '--force-ocr':
            force_ocr = True
        elif arg == '--skip-ocr':
            skip_ocr = True
        elif arg == '--max-pages' and i + 1 < len(sys.argv[1:]):
            try:
                max_pages = int(sys.argv[1:][i + 1])
                i += 1  # Skip next argument
            except ValueError:
                print(f"⚠️  Invalid --max-pages value, processing all pages")
        elif arg.startswith('--jobs'):
            try:
                n_jobs = int(arg.split('=')[1])
            except:
                pass
        elif not arg.startswith('--'):
            input_name = arg
        i += 1
    
    # Default input if not provided
    if input_name is None:
        input_name = "arrr_policy"
        print(f"⚠️  No input provided, using default: {input_name}")
        print()
    
    # Input/output directories
    # - pdf_dir: where PDFs live (e.g., encova/, nationwide/, hartford/, traveler/)
    # - output_dir: where extracted txt files are written (e.g., encovaop/, nationwideop/, ...)
    pdf_dir = "usgnon"
    carrier_dir = "usgnonop"
    
    # Extract base name
    base_name = extract_base_name(input_name)
    
    # Strip "_policy" suffix if present for cleaner output naming
    output_base = base_name
    if output_base.endswith("_policy"):
        output_base = output_base[:-7]  # Remove "_policy"
    
    print(f"Base name: {base_name}\n")
    
    # Find PDF file
    pdf_path = find_pdf_file(base_name, pdf_dir=pdf_dir)
    if not pdf_path:
        print(f"❌ PDF file not found for: {base_name}")
        print("   Tried:")
        print(f"     - {base_name}.pdf")
        print(f"     - {pdf_dir}/{base_name}.pdf")
        return
    
    print(f"✅ Found PDF: {pdf_path}\n")
    
    # Set up output paths
    output_dir = Path(carrier_dir)
    output_dir.mkdir(exist_ok=True)
    
    tesseract_output = output_dir / f"{output_base}_pol1.txt"
    pymupdf_output = output_dir / f"{output_base}_pol2.txt"
    
    total_start = time.time()
    
    # Run Tesseract extraction
    print("\n" + "="*80)
    print("STEP 1: TESSERACT EXTRACTION")
    print("="*80)
    tesseract_success = extract_tesseract(pdf_path, tesseract_output, n_jobs=n_jobs, max_pages=max_pages)
    
    # Run PyMuPDF extraction
    print("\n" + "="*80)
    print("STEP 2: PYMUPDF EXTRACTION")
    print("="*80)
    pymupdf_success = extract_pymupdf(pdf_path, pymupdf_output, 
                                      use_ocr=not skip_ocr, force_ocr=force_ocr,
                                      max_pages=max_pages, n_jobs=n_jobs)
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print()
    print("Output files:")
    if tesseract_success:
        print(f"  ✅ Tesseract: {tesseract_output}")
    if pymupdf_success:
        print(f"  ✅ PyMuPDF:   {pymupdf_output}")
    print("="*80)


if __name__ == "__main__":
    main()
