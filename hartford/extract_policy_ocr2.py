"""
Extract OCR text from policy PDF using Tesseract with parallel processing
Uses PyMuPDF (fitz) to convert PDF to images - no external dependencies needed
"""
import sys
import fitz  # PyMuPDF
import pytesseract
from pathlib import Path
import time
from joblib import Parallel, delayed
from PIL import Image
import io

# Set Tesseract path (adjust if needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def pdf_page_to_image(pdf_path, page_num, dpi=100):
    """Convert a single PDF page to PIL Image
    
    Args:
        dpi: Resolution for rendering (300 DPI recommended for tables)
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)  # 0-indexed
    
    # Render page to image at higher DPI for better table OCR
    mat = fitz.Matrix(dpi/72, dpi/72)  # 300 DPI default
    pix = page.get_pixmap(matrix=mat)
    
    # Convert to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    
    return img

def process_single_page(pdf_path, page_num):
    """Process a single page with OCR (for parallel processing)
    
    Optimized Tesseract config for insurance docs with tables:
    - PSM 6: Uniform block of text (best for structured documents)
    - OEM 1: LSTM neural net mode (most accurate)
    - preserve_interword_spaces: Maintains table column spacing
    """
    try:
        # Convert PDF page to image at 300 DPI for better table OCR
        image = pdf_page_to_image(pdf_path, page_num, dpi=100)
        
        # Tesseract config optimized for documents with tables
        custom_config = r'--oem 1 --psm 6 -c preserve_interword_spaces=1'
        
        # Run OCR with custom config
        text = pytesseract.image_to_string(image, config=custom_config)
        
        return (page_num, text, None)
    except Exception as e:
        return (page_num, None, str(e))

def extract_pdf_text(pdf_path, output_path, n_jobs=-1, max_pages=None):
    """Extract text from PDF using OCR
    
    Args:
        max_pages: Process only first N pages (None = all pages, 20 = first 20 for declarations)
    """
    print(f"Starting OCR extraction from: {pdf_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Get page count
    print("\nOpening PDF...")
    start_time = time.time()
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        # Limit pages if requested
        num_pages = min(total_pages, max_pages) if max_pages else total_pages
        
        if max_pages and total_pages > max_pages:
            print(f"✓ PDF has {total_pages} pages (processing first {num_pages} only)")
        else:
            print(f"✓ PDF has {num_pages} pages")
    except Exception as e:
        print(f"✗ Error opening PDF: {e}")
        return False
    
    # Extract text from each page in parallel
    print(f"\nExtracting text from {num_pages} pages using {n_jobs} parallel workers...")
    ocr_start = time.time()
    
    # Process pages in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_page)(str(pdf_path), page_num) 
        for page_num in range(1, num_pages + 1)
    )
    
    print(f"\n✓ Extracted text from {num_pages} pages in {time.time() - ocr_start:.2f}s")
    print(f"Total time: {time.time() - start_time:.2f}s")
    
    # Sort results by page number and build output
    results.sort(key=lambda x: x[0])
    all_text = []
    
    for page_num, text, error in results:
        all_text.append(f"\n{'='*80}\n")
        all_text.append(f"PAGE {page_num}\n")
        all_text.append(f"{'='*80}\n")
        
        if error:
            all_text.append(f"\n[ERROR ON PAGE {page_num}: {error}]\n")
            print(f"✗ Error on page {page_num}: {error}")
        else:
            all_text.append(text)
    
    # Save to file
    print(f"\nSaving to {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(all_text))
        
        file_size = len(''.join(all_text))
        print(f"✓ Saved {file_size:,} characters ({file_size/1024:.2f} KB)")
        return True
    except Exception as e:
        print(f"✗ Error saving file: {e}")
        return False

if __name__ == "__main__":
    # pdf_path = Path("../encova/shelby_policy.pdf")
    pdf_path = Path("hartford/ameen_policy.pdf")
    output_path = Path("hartfordop/ameen_ocrp6.txt")
    
    if not pdf_path.exists():
        print(f"✗ PDF not found: {pdf_path}")
        sys.exit(1)
    
    # Use all CPU cores
    n_jobs = -1
    
    # Process only first 20 pages (for declarations) - set to None for all pages
    max_pages = None  # Change to None to process entire policy
    
    print(f"Using all CPU cores for OCR processing")
    print(f"Optimized for documents with TABLES:")
    print(f"  • 300 DPI rendering (better resolution)")
    print(f"  • PSM 6 (uniform block segmentation)")
    print(f"  • Preserve interword spaces (table structure)\n")
    if max_pages:
        print(f"Will process only first {max_pages} pages (declarations)\n")
    else:
        print(f"Will process entire policy\n")
    
    success = extract_pdf_text(pdf_path, output_path, n_jobs=n_jobs, max_pages=max_pages)
    
    if success:
        print("\n" + "="*80)
        print("✓ OCR EXTRACTION COMPLETE!")
        print("="*80)
        print(f"Output saved to: {output_path.absolute()}")
    else:
        print("\n✗ OCR extraction failed!")
        sys.exit(1)
