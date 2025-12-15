# ocr_extract_pipeline.py
"""
Complete Pipeline: OCR PDF → Extract Text
1. Uses OCRmyPDF to make scanned PDF searchable
2. Uses pdfplumber/PyMuPDF to extract text from OCR'd PDF
3. Parallelized for faster processing
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from functools import partial

# Try to import joblib for parallelization
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Multiprocessing is always available in Python stdlib
try:
    from multiprocessing import Pool, cpu_count
    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    MULTIPROCESSING_AVAILABLE = False

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def run_ocrmypdf(input_pdf, output_pdf=None, force_ocr=False, enhance_images=False, n_jobs=None, smart_mode=True):
    """
    Step 1: Convert scanned PDF to OCR-able PDF using OCRmyPDF
    
    Args:
        input_pdf: Input PDF file path
        output_pdf: Output PDF file path (optional)
        force_ocr: If True, use --force-ocr (replaces all text). 
        enhance_images: If True, use --deskew and --clean (only works with --force-ocr)
        n_jobs: Number of parallel workers
        smart_mode: If True, uses --skip-text flag (auto-detects pages needing OCR, skips pages with text)
                    If False, uses --redo-ocr (improves existing text)
    """
    if output_pdf is None:
        # Save to same directory, just add "2" before .pdf extension
        # e.g., shelby_policy.pdf -> shelby_policy2.pdf
        pdf_path = Path(input_pdf)
        output_pdf = str(pdf_path.parent / f"{pdf_path.stem}2.pdf")
    
    print("="*60)
    print("STEP 1: OCR Processing with OCRmyPDF")
    print("="*60)
    print(f"Input:  {input_pdf}")
    print(f"Output: {output_pdf}\n")
    
    # Build OCRmyPDF command
    cmd = [sys.executable, '-m', 'ocrmypdf']
    
    # Choose OCR mode
    if force_ocr:
        cmd.append('--force-ocr')
        print("Mode: Force OCR (replaces ALL text on ALL pages)")
    elif smart_mode:
        # Smart mode - OCRs pages without text, skips pages with text
        # Use --skip-text to prevent aborting when pages already have text
        cmd.append('--skip-text')
        print("Mode: Smart OCR (auto-detects pages needing OCR)")
        print("   ✅ Pages WITH text: Skipped (preserved)")
        print("   🔍 Pages WITHOUT text: OCR'd automatically")
    else:
        cmd.append('--redo-ocr')
        print("Mode: Redo OCR (improves existing text on ALL pages)")
    
    # Image enhancement options (only compatible with --force-ocr)
    if enhance_images and force_ocr:
        cmd.extend(['--deskew', '--clean'])
        print("Enhancements: Deskew + Clean enabled")
    elif enhance_images and not force_ocr:
        print("⚠️  Warning: --deskew and --clean are not compatible with --redo-ocr")
        print("   Skipping image enhancements (use --force-ocr to enable)")
    
    # Add parallel processing for OCR (OCRmyPDF supports --jobs flag)
    if n_jobs is not None and n_jobs != 1:
        if n_jobs == -1:
            # Use all available cores
            try:
                import multiprocessing
                n_jobs = multiprocessing.cpu_count()
            except:
                n_jobs = 4  # Default fallback
        cmd.extend(['--jobs', str(n_jobs)])
        print(f"Parallel OCR: {n_jobs} workers")
    
    cmd.extend([input_pdf, output_pdf])
    
    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False  # Don't raise on non-zero exit codes
    )
    elapsed = time.time() - start_time
    
    # OCRmyPDF exit codes:
    # 0 = success
    # 1-9 = errors
    # 6 = PriorOcrFoundError (page already has text - can be handled with --skip-text)
    # 10 = warning (e.g., PDF/A metadata issue, but file created successfully)
    
    # Check if output file was created (even with warnings)
    output_exists = os.path.exists(output_pdf)
    
    # Check for specific error messages
    error_output = result.stderr if result.stderr else ""
    has_prior_ocr_error = "PriorOcrFoundError" in error_output or "page already has text" in error_output.lower()
    
    if result.returncode == 0:
        print(f"✅ OCR completed successfully in {elapsed:.2f} seconds\n")
        return output_pdf
    elif result.returncode == 6 and has_prior_ocr_error:
        # Exit code 6 = PriorOcrFoundError - page already has text
        # This shouldn't happen with --skip-text, but handle it gracefully
        print(f"⚠️  OCR encountered pages with existing text (exit code 6)")
        if output_exists:
            print("✅ Output file was created, continuing...\n")
            return output_pdf
        else:
            print("❌ No output file created. Try using --force-ocr or --redo-ocr")
            print("Error details:")
            if error_output:
                # Show relevant error lines
                error_lines = [line for line in error_output.split('\n') if line.strip() and ('error' in line.lower() or 'abort' in line.lower())]
                for line in error_lines[:3]:
                    print(f"  {line}")
            print()
            return None
    elif result.returncode == 10 and output_exists:
        # Exit code 10 = warning (like PDF/A metadata), but file was created
        print(f"⚠️  OCR completed with warnings in {elapsed:.2f} seconds")
        if result.stderr:
            # Print warnings but don't fail
            warning_lines = [line for line in result.stderr.split('\n') if line.strip()]
            if warning_lines:
                print("Warnings:")
                for line in warning_lines[-3:]:  # Show last 3 warning lines
                    if line.strip():
                        print(f"  {line}")
        print("✅ Output file created successfully\n")
        return output_pdf
    elif output_exists:
        # File was created despite non-zero exit code - treat as success
        print(f"⚠️  OCR completed (exit code {result.returncode}) in {elapsed:.2f} seconds")
        print("✅ Output file exists, continuing...\n")
        return output_pdf
    else:
        # Actual failure - no output file created
        print(f"❌ OCR failed (exit code {result.returncode})")
        if error_output:
            print("Error output:")
            # Show most relevant error lines
            error_lines = [line for line in error_output.split('\n') if line.strip()]
            if error_lines:
                for line in error_lines[-5:]:  # Show last 5 error lines
                    if line.strip():
                        print(f"  {line}")
        print()
        return None

def _extract_page_pdfplumber(pdf_path, page_num, page_index):
    """
    Helper function to extract a single page (for parallelization)
    """
    try:
        import pdfplumber
        
        with pdfplumber.open(pdf_path) as pdf:
            if page_index >= len(pdf.pages):
                return None
            
            page = pdf.pages[page_index]
            
            # Extract text
            text = page.extract_text()
            
            # Extract tables
            tables = page.extract_tables()
            
            page_content = []
            page_content.append(f"\n{'='*60}\nPAGE {page_num}\n{'='*60}\n")
            
            if text:
                page_content.append("TEXT CONTENT:\n")
                page_content.append(text)
                page_content.append("\n")
            
            table_data = []
            if tables:
                page_content.append(f"\nTABLES FOUND: {len(tables)}\n")
                for table_idx, table in enumerate(tables, 1):
                    page_content.append(f"\n--- TABLE {table_idx} ---\n")
                    # Format table as markdown-style
                    for row in table:
                        if row:
                            # Filter out None values and join with | separator
                            clean_row = [str(cell) if cell else "" for cell in row]
                            page_content.append("| " + " | ".join(clean_row) + " |\n")
                    page_content.append("\n")
                    table_data.append({
                        'page': page_num,
                        'table_num': table_idx,
                        'data': table
                    })
            
            page_text = ''.join(page_content)
            chars = len(text) if text else 0
            tables_count = len(tables) if tables else 0
            
            return {
                'page_num': page_num,
                'text': page_text,
                'tables': table_data,
                'chars': chars,
                'tables_count': tables_count
            }
    except Exception as e:
        return {
            'page_num': page_num,
            'text': f"\n{'='*60}\nPAGE {page_num}\n{'='*60}\nError: {e}\n",
            'tables': [],
            'chars': 0,
            'tables_count': 0
        }

def extract_text_pdfplumber(pdf_path, output_txt=None, n_jobs=None):
    """
    Step 2: Extract text and tables from OCR'd PDF using pdfplumber
    Best for preserving table structure and formatting
    
    Args:
        pdf_path: Path to PDF file
        output_txt: Output text file path
        n_jobs: Number of parallel jobs (-1 for all cores, None for sequential)
    """
    try:
        import pdfplumber
    except ImportError:
        print("❌ pdfplumber not installed. Install with: pip install pdfplumber")
        return None
    
    if output_txt is None:
        output_txt = pdf_path.replace('.pdf', '_extracted.txt')
    
    print("="*60)
    print("STEP 2: Text & Table Extraction with pdfplumber")
    print("="*60)
    print(f"Reading: {pdf_path}\n")
    
    start_time = time.time()
    
    try:
        # First, get page count
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            print(f"✅ Opened PDF with {num_pages} pages")
        
        # Determine if we should parallelize
        use_parallel = False
        if n_jobs is not None and n_jobs != 1 and num_pages > 1:
            if JOBLIB_AVAILABLE:
                use_parallel = True
                if n_jobs == -1:
                    n_jobs = None  # Use all cores
                print(f"🚀 Using parallel processing ({n_jobs or 'all'} workers)\n")
            elif MULTIPROCESSING_AVAILABLE:
                use_parallel = True
                if n_jobs == -1:
                    n_jobs = None
                print(f"🚀 Using multiprocessing ({n_jobs or 'all'} workers)\n")
            else:
                print("⚠️  Parallel processing not available (install joblib: pip install joblib)\n")
        
        if not use_parallel:
            print("📄 Processing pages sequentially\n")
            # Sequential processing (original method)
            all_text = []
            all_tables = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    print(f"Extracting page {page_num}/{num_pages}...", end=" ", flush=True)
                    
                    text = page.extract_text()
                    tables = page.extract_tables()
                    
                    page_content = []
                    page_content.append(f"\n{'='*60}\nPAGE {page_num}\n{'='*60}\n")
                    
                    if text:
                        page_content.append("TEXT CONTENT:\n")
                        page_content.append(text)
                        page_content.append("\n")
                    
                    if tables:
                        page_content.append(f"\nTABLES FOUND: {len(tables)}\n")
                        for table_idx, table in enumerate(tables, 1):
                            page_content.append(f"\n--- TABLE {table_idx} ---\n")
                            for row in table:
                                if row:
                                    clean_row = [str(cell) if cell else "" for cell in row]
                                    page_content.append("| " + " | ".join(clean_row) + " |\n")
                            page_content.append("\n")
                            all_tables.append({
                                'page': page_num,
                                'table_num': table_idx,
                                'data': table
                            })
                    
                    page_text = ''.join(page_content)
                    all_text.append(page_text)
                    
                    chars = len(text) if text else 0
                    tables_count = len(tables) if tables else 0
                    print(f"✅ ({chars} chars, {tables_count} tables)")
        else:
            # Parallel processing
            if JOBLIB_AVAILABLE:
                results = Parallel(n_jobs=n_jobs, backend='threading')(
                    delayed(_extract_page_pdfplumber)(pdf_path, page_num, page_index)
                    for page_num, page_index in enumerate(range(num_pages), 1)
                )
            else:  # multiprocessing
                with Pool(processes=n_jobs) as pool:
                    results = pool.starmap(
                        _extract_page_pdfplumber,
                        [(pdf_path, page_num, page_index) 
                         for page_num, page_index in enumerate(range(num_pages), 1)]
                    )
            
            # Sort results by page number and combine
            results = sorted([r for r in results if r], key=lambda x: x['page_num'])
            all_text = [r['text'] for r in results]
            all_tables = []
            for r in results:
                all_tables.extend(r['tables'])
            
            # Print summary
            for r in results:
                print(f"Page {r['page_num']}: ✅ ({r['chars']} chars, {r['tables_count']} tables)")
        
        # Save extracted text with tables
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_text))
        
        elapsed = time.time() - start_time
        total_chars = sum(len(t) for t in all_text)
        total_tables = len(all_tables)
        
        print(f"\n✅ Extraction completed in {elapsed:.2f} seconds")
        print(f"📄 Total characters: {total_chars:,}")
        print(f"📊 Total tables found: {total_tables}")
        if use_parallel:
            print(f"⚡ Speedup: {num_pages / elapsed:.1f} pages/second")
        print(f"💾 Saved to: {output_txt}\n")
        
        return output_txt
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_text_pymupdf(pdf_path, output_txt=None):
    """
    Step 2: Extract text from OCR'd PDF using PyMuPDF
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("❌ PyMuPDF not installed. Install with: pip install pymupdf")
        return None
    
    if output_txt is None:
        # Save to encovaop folder
        # Extract base name (e.g., "shelby_policy2" from "shelby_policy2.pdf")
        pdf_name = Path(pdf_path).stem
        output_txt = f"nationwideop/{pdf_name}.txt"
        # Ensure encovaop directory exists
        # Path("hartfordop").mkdir(exist_ok=True)
    
    print("="*60)
    print("STEP 2: Text Extraction with PyMuPDF")
    print("="*60)
    print(f"Reading: {pdf_path}\n")
    
    start_time = time.time()
    
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        print(f"✅ Opened PDF with {num_pages} pages\n")
        
        all_text = []
        for page_num in range(num_pages):
            print(f"Extracting page {page_num + 1}/{num_pages}...", end=" ", flush=True)
            page = doc[page_num]
            text = page.get_text()
            all_text.append(f"\n{'='*60}\nPAGE {page_num + 1}\n{'='*60}\n{text}")
            print(f"✅ ({len(text)} chars)")
        
        doc.close()
        
        # Save extracted text
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_text))
        
        elapsed = time.time() - start_time
        total_chars = sum(len(t) for t in all_text)
        
        print(f"\n✅ Extraction completed in {elapsed:.2f} seconds")
        print(f"📄 Total characters: {total_chars:,}")
        print(f"💾 Saved to: {output_txt}\n")
        
        return output_txt
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_text_pypdf2(pdf_path, output_txt=None):
    """
    Alternative: Extract text using PyPDF2 (fallback if PyMuPDF not available)
    """
    try:
        import PyPDF2
    except ImportError:
        print("❌ PyPDF2 not installed. Install with: pip install PyPDF2")
        return None
    
    if output_txt is None:
        output_txt = pdf_path.replace('.pdf', '_extracted.txt')
    
    print("="*60)
    print("STEP 2: Text Extraction with PyPDF2 (fallback)")
    print("="*60)
    print(f"Reading: {pdf_path}\n")
    
    start_time = time.time()
    
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            print(f"✅ Opened PDF with {num_pages} pages\n")
            
            all_text = []
            for page_num, page in enumerate(reader.pages, 1):
                print(f"Extracting page {page_num}/{num_pages}...", end=" ", flush=True)
                text = page.extract_text()
                all_text.append(f"\n{'='*60}\nPAGE {page_num}\n{'='*60}\n{text}")
                print(f"✅ ({len(text)} chars)")
        
        # Save extracted text
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_text))
        
        elapsed = time.time() - start_time
        total_chars = sum(len(t) for t in all_text)
        
        print(f"\n✅ Extraction completed in {elapsed:.2f} seconds")
        print(f"📄 Total characters: {total_chars:,}")
        print(f"💾 Saved to: {output_txt}\n")
        
        return output_txt
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Main pipeline: OCR → Extract
    
    Usage:
        python ocr_extract_pipeline.py <input_pdf> [--force-ocr] [--enhance] [--skip-ocr] [--redo-ocr] [--jobs N]
    
    Arguments:
        input_pdf: Path to input PDF file (REQUIRED)
        
        OCR Options (default: SMART mode - auto-detects pages needing OCR):
        --smart: Smart mode (default) - Uses --skip-text flag to OCR pages without text,
                 skips pages with existing text (prevents PriorOcrFoundError)
                 PERFECT for mixed PDFs (some pages OCR'd, some not)
        --force-ocr: Force OCR on ALL pages (replaces existing text)
        --redo-ocr: Redo OCR on ALL pages (improves existing text)
        --enhance: Enable image enhancements (deskew, clean) - only works with --force-ocr
        --skip-ocr: Skip OCR step entirely (ONLY use if PDF already has searchable text)
        
        Performance:
        --jobs N: Number of parallel workers (-1 for all cores, 1 for sequential, default: auto)
    
    Examples:
        # Most common: Scanned PDF needs OCR (SMART mode - default)
        python ocr_extract_pipeline.py scanned_document.pdf
        
        # Mixed PDF: Some pages OCR'd, some not (SMART mode handles this!)
        python ocr_extract_pipeline.py mixed_document.pdf
        # ✅ Pages with text: Preserved
        # 🔍 Pages without text: OCR'd automatically
        
        # Force OCR on all pages (even if they have text)
        python ocr_extract_pipeline.py document.pdf --force-ocr
        
        # PDF already has text (rare case)
        python ocr_extract_pipeline.py already_ocr_document.pdf --skip-ocr
        
        # Use all CPU cores for speed
        python ocr_extract_pipeline.py large_document.pdf --jobs -1
    """
    # Parse command line arguments
    input_pdf = None
    force_ocr = False
    enhance_images = False
    skip_ocr = False
    redo_ocr = False
    smart_mode = True  # Default: smart mode for mixed PDFs
    n_jobs = None  # Auto-detect
    
    i = 0
    while i < len(sys.argv[1:]):
        arg = sys.argv[1:][i]
        if arg == '--force-ocr':
            force_ocr = True
            smart_mode = False
        elif arg == '--redo-ocr':
            redo_ocr = True
            smart_mode = False
        elif arg == '--enhance':
            enhance_images = True
        elif arg == '--skip-ocr':
            skip_ocr = True
            smart_mode = False
        elif arg == '--smart':
            smart_mode = True
        elif arg == '--jobs' and i + 1 < len(sys.argv[1:]):
            try:
                n_jobs = int(sys.argv[1:][i + 1])
                i += 1  # Skip next argument
            except ValueError:
                print(f"⚠️  Invalid --jobs value, using auto")
        elif not arg.startswith('--') and input_pdf is None:
            input_pdf = arg
        i += 1
    
    # Default input PDF if not provided
    if input_pdf is None:
        input_pdf = "nationwide/evergreen_policy.pdf"
    
    if not os.path.exists(input_pdf):
        print(f"❌ File not found: {input_pdf}")
        return
    
    print("\n" + "="*60)
    print("COMPLETE OCR + EXTRACTION PIPELINE")
    print("="*60)
    print(f"Input PDF: {input_pdf}")
    
    # OCR mode explanation
    if skip_ocr:
        print("⚠️  Mode: Skip OCR (PDF already has text layer)")
        print("   Use this ONLY if your PDF already has searchable text!")
    elif force_ocr:
        print("Mode: Force OCR (replaces ALL text on ALL pages)")
    elif redo_ocr:
        print("Mode: Redo OCR (improves existing text on ALL pages)")
    else:
        print("Mode: Smart OCR (default - auto-detects pages needing OCR)")
        print("   ✅ Pages WITH text: Preserved (skipped)")
        print("   🔍 Pages WITHOUT text: OCR'd automatically")
        print("   💡 Perfect for mixed PDFs (some OCR'd, some not)!")
    
    if enhance_images:
        print("Enhancements: Enabled (deskew + clean)")
    if n_jobs is not None:
        print(f"Parallel workers: {n_jobs if n_jobs != -1 else 'all'}")
    elif JOBLIB_AVAILABLE or MULTIPROCESSING_AVAILABLE:
        print("Parallel processing: Auto (enabled for multi-page PDFs)")
    print()
    
    total_start = time.time()
    
    # Step 1: OCR with OCRmyPDF (skip ONLY if --skip-ocr is explicitly set)
    if skip_ocr:
        print("⏭️  Skipping OCR step (PDF already has text layer)")
        print("   ⚠️  Make sure your PDF is already searchable!\n")
        ocr_pdf = input_pdf
    else:
        print("🔍 Running OCR to make PDF searchable...")
        ocr_pdf = run_ocrmypdf(
            input_pdf, 
            force_ocr=force_ocr, 
            enhance_images=enhance_images, 
            n_jobs=n_jobs,
            smart_mode=smart_mode
        )
        if not ocr_pdf:
            print("❌ Pipeline stopped: OCR failed")
            return
    
    # Step 2: Extract text (try PyMuPDF first for clean LLM-friendly output)
    output_txt = None
    
    # Try PyMuPDF first (cleanest output - no labels, no artifacts, plain text)
    try:
        import fitz
        output_txt = extract_text_pymupdf(ocr_pdf)
    except ImportError:
        print("⚠️  PyMuPDF not available, trying pdfplumber...\n")
        # Try pdfplumber as fallback (if you need table extraction)
        try:
            import pdfplumber
            output_txt = extract_text_pdfplumber(ocr_pdf, n_jobs=n_jobs)
        except ImportError:
            print("⚠️  pdfplumber not available, using PyPDF2 fallback...\n")
            output_txt = extract_text_pypdf2(ocr_pdf)
    
    if not output_txt:
        print("❌ Pipeline stopped: Text extraction failed")
        return
    
    total_time = time.time() - total_start
    
    # Summary
    print("="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"✅ OCR'd PDF:  {ocr_pdf}")
    print(f"✅ Extracted:   {output_txt}")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print("="*60)

if __name__ == "__main__":
    main()

    