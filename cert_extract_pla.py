"""
Unified Certificate OCR + Extraction Pipeline
Takes a single input name, runs both Tesseract and PyMuPDF extractions,
saves with consistent naming in carrier directory

Usage:
    python cert_extract.py aaniya_pl
    python cert_extract.py aaniya_pl.pdf
    python cert_extract.py encova/aaniya_pl.pdf

Outputs:
    - {carrier_dir}/{base_name}1.txt (pdfplumber extraction with tables)
    - {carrier_dir}/{base_name}1.tables.json (structured table data from pdfplumber)
    - {carrier_dir}/{base_name}2.txt (PyMuPDF extraction)
    - {carrier_dir}/{base_name}3.txt (Tesseract extraction)
    - {carrier_dir}/{base_name}_combo.txt (auto-combined file with all three)
"""

import os
import sys
import subprocess
import time
import re
import json
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

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("⚠️  pdfplumber not installed. Install with: pip install pdfplumber")

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def extract_base_name(input_path: str) -> str:
    """
    Extract base name from input (removes .pdf, paths, etc.)
    
    Examples:
        "aaniya_pl" -> "aaniya_pl"
        "aaniya_pl.pdf" -> "aaniya_pl"
        "encova/aaniya_pl.pdf" -> "aaniya_pl"
    """
    path = Path(input_path)
    # Remove extension and path
    base_name = path.stem
    return base_name


def find_pdf_file(base_name: str, pdf_dir: str = "encova") -> Optional[Path]:
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


def extract_with_pdfplumber(pdf_path: Path, output_path: Path) -> bool:
    """
    Extract text and tables from PDF using pdfplumber (preserves table structure)
    
    Args:
        pdf_path: Path to PDF file
        output_path: Output text file path
    """
    if not PDFPLUMBER_AVAILABLE:
        print("❌ pdfplumber extraction skipped (dependencies not available)")
        print("   Install with: pip install pdfplumber")
        return False
    
    print("="*60)
    print("PDFPLUMBER EXTRACTION")
    print("="*60)
    print(f"Input:  {pdf_path}")
    print(f"Output: {output_path}\n")
    
    start_time = time.time()
    
    try:
        all_content = []
        all_tables = []
        
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            print(f"✅ PDF has {num_pages} pages\n")
            
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"Extracting page {page_num}/{num_pages}...", end=" ", flush=True)
                
                # Extract regular text
                text = page.extract_text()
                
                # Extract tables (preserves structure!)
                tables = page.extract_tables()
                
                # Build page content
                page_content = []
                page_content.append(f"\n{'='*80}\n")
                page_content.append(f"PAGE {page_num}\n")
                page_content.append(f"{'='*80}\n")
                page_content.append("\n--- TEXT ---\n")
                
                if text:
                    page_content.append(text)
                else:
                    page_content.append("[No text found on this page]")
                
                # Add tables if found
                if tables:
                    page_content.append(f"\n\n--- TABLES ({len(tables)} found) ---\n")
                    for table_idx, table in enumerate(tables, 1):
                        page_content.append(f"\nTABLE {table_idx}:\n")
                        # Format table as readable text
                        for row_idx, row in enumerate(table):
                            if row:
                                # Filter out None values and join with tabs
                                clean_row = [str(cell) if cell is not None else "" for cell in row]
                                page_content.append("\t".join(clean_row) + "\n")
                        page_content.append("\n")
                    
                    # Also save tables as JSON for structured access
                    all_tables.append({
                        "page": page_num,
                        "tables": tables
                    })
                
                all_content.append(''.join(page_content))
                print(f"✅ ({len(text or '')} chars, {len(tables)} tables)")
        
        # Save extracted text
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(all_content))
        
        # Save tables as JSON (structured data)
        if all_tables:
            # Save JSON with same base name but different suffix
            json_path = output_path.parent / f"{output_path.stem}.tables.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_tables, f, indent=2, ensure_ascii=False)
            print(f"   Also saved tables to: {json_path.name}")
        
        elapsed = time.time() - start_time
        total_chars = len(''.join(all_content))
        
        print(f"\n✅ pdfplumber extraction completed in {elapsed:.2f} seconds")
        print(f"   Saved {total_chars:,} characters ({total_chars/1024:.2f} KB)")
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_tesseract(pdf_path: Path, output_path: Path, n_jobs: int = -1) -> bool:
    """
    Extract text from PDF using Tesseract OCR
    
    Args:
        pdf_path: Path to PDF file
        output_path: Output text file path
        n_jobs: Number of parallel workers (-1 for all cores)
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
        num_pages = len(doc)
        doc.close()
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
                    use_ocr: bool = True, force_ocr: bool = False) -> bool:
    """
    Extract text from PDF using PyMuPDF (optionally with OCRmyPDF preprocessing)
    
    Args:
        pdf_path: Path to PDF file
        output_path: Output text file path
        use_ocr: If True, run OCRmyPDF first to make PDF searchable
        force_ocr: If True, force OCR on all pages
    """
    if not PYMUPDF_AVAILABLE:
        print("❌ PyMuPDF extraction skipped (dependencies not available)")
        return False
    
    print("="*60)
    print("PYMUPDF EXTRACTION")
    print("="*60)
    
    # Step 1: OCR if needed
    if use_ocr:
        ocr_pdf = run_ocrmypdf(pdf_path, force_ocr=force_ocr)
        if not ocr_pdf:
            print("⚠️  OCR failed, trying direct extraction...")
            ocr_pdf = pdf_path
    else:
        ocr_pdf = pdf_path
    
    print(f"Input:  {ocr_pdf}")
    print(f"Output: {output_path}\n")
    
    start_time = time.time()
    
    try:
        doc = fitz.open(ocr_pdf)
        num_pages = len(doc)
        print(f"✅ PDF has {num_pages} pages\n")
        
        all_text = []
        for page_num in range(num_pages):
            print(f"Extracting page {page_num + 1}/{num_pages}...", end=" ", flush=True)
            page = doc[page_num]
            text = page.get_text()
            all_text.append(f"\n{'='*60}\nPAGE {page_num + 1}\n{'='*60}\n{text}")
            print(f"✅ ({len(text)} chars)")
        
        doc.close()
        
        # Save extracted text
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_text))
        
        elapsed = time.time() - start_time
        total_chars = sum(len(t) for t in all_text)
        
        print(f"\n✅ PyMuPDF extraction completed in {elapsed:.2f} seconds")
        print(f"   Saved {total_chars:,} characters ({total_chars/1024:.2f} KB)")
        
        # Clean up temp OCR file if we created one
        if use_ocr and ocr_pdf != pdf_path and ocr_pdf.exists():
            try:
                ocr_pdf.unlink()
            except:
                pass
        
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
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


def combine_extractions(pdfplumber_file: Path, pymupdf_file: Path, tesseract_file: Path,
                       output_file: Path, interleave_pages: bool = True) -> bool:
    """
    Combine pdfplumber, PyMuPDF, and Tesseract extraction files
    """
    files_to_check = []
    if pdfplumber_file.exists():
        files_to_check.append(("pdfplumber", pdfplumber_file))
    if pymupdf_file.exists():
        files_to_check.append(("pymupdf", pymupdf_file))
    if tesseract_file.exists():
        files_to_check.append(("tesseract", tesseract_file))
    
    if not files_to_check:
        print("⚠️  No extraction files found to combine")
        return False
    
    print("="*60)
    print("COMBINING EXTRACTIONS")
    print("="*60)
    if pdfplumber_file.exists():
        print(f"pdfplumber: {pdfplumber_file}")
    if pymupdf_file.exists():
        print(f"PyMuPDF:    {pymupdf_file}")
    if tesseract_file.exists():
        print(f"Tesseract:  {tesseract_file}")
    print(f"Output:     {output_file}\n")
    
    # Read all available files
    pdfplumber_content = ""
    pymupdf_content = ""
    tesseract_content = ""
    
    if pdfplumber_file.exists():
        with open(pdfplumber_file, 'r', encoding='utf-8') as f:
            pdfplumber_content = f.read()
    
    if pymupdf_file.exists():
        with open(pymupdf_file, 'r', encoding='utf-8') as f:
            pymupdf_content = f.read()
    
    if tesseract_file.exists():
        with open(tesseract_file, 'r', encoding='utf-8') as f:
            tesseract_content = f.read()
    
    combined_content = []
    
    # Header
    sources = []
    if pdfplumber_content:
        sources.append("PDFPLUMBER")
    if pymupdf_content:
        sources.append("PYMUPDF")
    if tesseract_content:
        sources.append("TESSERACT")
    
    combined_content.append("="*80)
    combined_content.append(f"COMBINED EXTRACTION - {' + '.join(sources)}")
    combined_content.append("="*80)
    combined_content.append("")
    
    if interleave_pages:
        # Page-by-page interleaving mode
        print("Mode: Page-by-page interleaving")
        
        pdfplumber_pages = extract_pages_from_content(pdfplumber_content) if pdfplumber_content else []
        pymupdf_pages = extract_pages_from_content(pymupdf_content) if pymupdf_content else []
        tesseract_pages = extract_pages_from_content(tesseract_content) if tesseract_content else []
        
        pdfplumber_dict = {page_num: content for page_num, content in pdfplumber_pages}
        pymupdf_dict = {page_num: content for page_num, content in pymupdf_pages}
        tesseract_dict = {page_num: content for page_num, content in tesseract_pages}
        
        all_pages = sorted(set(
            list(pdfplumber_dict.keys()) + 
            list(pymupdf_dict.keys()) + 
            list(tesseract_dict.keys())
        ))
        
        print(f"   Found {len(pdfplumber_pages)} pdfplumber pages")
        print(f"   Found {len(pymupdf_pages)} PyMuPDF pages")
        print(f"   Found {len(tesseract_pages)} Tesseract pages")
        print(f"   Combining {len(all_pages)} unique pages\n")
        
        for page_num in all_pages:
            combined_content.append("="*80)
            combined_content.append(f"PAGE {page_num}")
            combined_content.append("="*80)
            combined_content.append("")
            
            if page_num in pdfplumber_dict:
                combined_content.append("--- PDFPLUMBER (Table-aware) ---")
                combined_content.append("")
                combined_content.append(pdfplumber_dict[page_num])
                combined_content.append("")
            elif pdfplumber_content:
                combined_content.append("--- PDFPLUMBER (Table-aware) ---")
                combined_content.append("[Page not found in pdfplumber extraction]")
                combined_content.append("")
            
            if page_num in pymupdf_dict:
                combined_content.append("--- PYMUPDF (Text layer) ---")
                combined_content.append("")
                combined_content.append(pymupdf_dict[page_num])
                combined_content.append("")
            elif pymupdf_content:
                combined_content.append("--- PYMUPDF (Text layer) ---")
                combined_content.append("[Page not found in PyMuPDF extraction]")
                combined_content.append("")
            
            if page_num in tesseract_dict:
                combined_content.append("--- TESSERACT (OCR) ---")
                combined_content.append("")
                combined_content.append(tesseract_dict[page_num])
                combined_content.append("")
            elif tesseract_content:
                combined_content.append("--- TESSERACT (OCR) ---")
                combined_content.append("[Page not found in Tesseract extraction]")
                combined_content.append("")
            
            combined_content.append("")
    else:
        # Simple concatenation mode
        print("Mode: Simple concatenation\n")
        
        if pdfplumber_content:
            combined_content.append("="*80)
            combined_content.append("SOURCE 1: PDFPLUMBER EXTRACTION (Table-aware)")
            combined_content.append("="*80)
            combined_content.append("")
            combined_content.append(pdfplumber_content)
            combined_content.append("")
            combined_content.append("="*80)
            combined_content.append("END OF PDFPLUMBER EXTRACTION")
            combined_content.append("="*80)
            combined_content.append("")
            combined_content.append("")
        
        if pymupdf_content:
            combined_content.append("="*80)
            combined_content.append("SOURCE 2: PYMUPDF EXTRACTION (Text layer)")
            combined_content.append("="*80)
            combined_content.append("")
            combined_content.append(pymupdf_content)
            combined_content.append("")
            combined_content.append("="*80)
            combined_content.append("END OF PYMUPDF EXTRACTION")
            combined_content.append("="*80)
            combined_content.append("")
            combined_content.append("")
        
        if tesseract_content:
            combined_content.append("="*80)
            combined_content.append("SOURCE 3: TESSERACT EXTRACTION (OCR)")
            combined_content.append("="*80)
            combined_content.append("")
            combined_content.append(tesseract_content)
            combined_content.append("")
            combined_content.append("="*80)
            combined_content.append("END OF TESSERACT EXTRACTION")
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
    Main function: unified certificate extraction pipeline
    """
    print("\n" + "="*80)
    print("UNIFIED CERTIFICATE OCR + EXTRACTION PIPELINE")
    print("="*80)
    print()
    
    # Parse command line arguments
    input_name = None
    force_ocr = False
    skip_ocr = False
    n_jobs = -1
    
    for arg in sys.argv[1:]:
        if arg == '--force-ocr':
            force_ocr = True
        elif arg == '--skip-ocr':
            skip_ocr = True
        elif arg.startswith('--jobs'):
            try:
                n_jobs = int(arg.split('=')[1])
            except:
                pass
        elif not arg.startswith('--'):
            input_name = arg
    
    # Default input if not provided
    if input_name is None:
        input_name = "westside_pla"
        print(f"⚠️  No input provided, using default: {input_name}")
        print()
    
    # Input/output directories
    # - pdf_dir: where PDFs live (e.g., encova/, nationwide/, hartford/, traveler/)
    # - output_dir: where extracted txt files are written (e.g., encovaop/, nationwideop/, ...)
    pdf_dir = "nationwide"
    output_dir = "nationwideop"
    
    # Extract base name
    base_name = extract_base_name(input_name)
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
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    pdfplumber_output = output_dir / f"{base_name}1.txt"
    pymupdf_output = output_dir / f"{base_name}2.txt"
    tesseract_output = output_dir / f"{base_name}3.txt"
    combined_output = output_dir / f"{base_name}_combo.txt"
    
    total_start = time.time()
    
    # Run pdfplumber extraction
    print("\n" + "="*80)
    print("STEP 1: PDFPLUMBER EXTRACTION")
    print("="*80)
    pdfplumber_success = extract_with_pdfplumber(pdf_path, pdfplumber_output)
    
    # Run PyMuPDF extraction
    print("\n" + "="*80)
    print("STEP 2: PYMUPDF EXTRACTION")
    print("="*80)
    pymupdf_success = extract_pymupdf(pdf_path, pymupdf_output, 
                                      use_ocr=not skip_ocr, force_ocr=force_ocr)
    
    # Run Tesseract extraction
    print("\n" + "="*80)
    print("STEP 3: TESSERACT EXTRACTION")
    print("="*80)
    tesseract_success = extract_tesseract(pdf_path, tesseract_output, n_jobs=n_jobs)
    
    # Auto-combine if at least one extraction succeeded
    if pdfplumber_success or pymupdf_success or tesseract_success:
        print("\n" + "="*80)
        print("STEP 4: COMBINING EXTRACTIONS")
        print("="*80)
        combine_extractions(pdfplumber_output, pymupdf_output, tesseract_output, combined_output)
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print()
    print("Output files:")
    if pdfplumber_success:
        print(f"  ✅ pdfplumber: {pdfplumber_output}")
        json_file = pdfplumber_output.with_suffix('.tables.json')
        if json_file.exists():
            print(f"  ✅ Tables JSON: {json_file}")
    if pymupdf_success:
        print(f"  ✅ PyMuPDF:     {pymupdf_output}")
    if tesseract_success:
        print(f"  ✅ Tesseract:   {tesseract_output}")
    if pdfplumber_success or pymupdf_success or tesseract_success:
        print(f"  ✅ Combined:    {combined_output}")
    print("="*80)


if __name__ == "__main__":
    main()

