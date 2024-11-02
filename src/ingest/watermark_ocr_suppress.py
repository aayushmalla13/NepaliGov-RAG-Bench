#!/usr/bin/env python3
"""
Watermark suppression for OCR preprocessing.

Implements deskewing, Sauvola binarization, denoising, and diagonal watermark
suppression using morphological operations to improve OCR accuracy.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, filters
from skimage.transform import rotate
import fitz  # PyMuPDF


def detect_skew_angle(image: np.ndarray, angle_range: int = 45) -> float:
    """
    Detect skew angle of document image using Hough line transform.
    
    Args:
        image: Input grayscale image
        angle_range: Range of angles to check in degrees
        
    Returns:
        Detected skew angle in degrees
    """
    # Edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    # Hough line transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is None:
        return 0.0
    
    # Calculate angles
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        if abs(angle) <= angle_range:
            angles.append(angle)
    
    if not angles:
        return 0.0
    
    # Return median angle
    return float(np.median(angles))


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Deskew image by rotating to correct detected skew angle.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Deskewed image
    """
    angle = detect_skew_angle(image)
    
    if abs(angle) < 0.5:  # Skip rotation for very small angles
        return image
    
    # Rotate image
    rotated = rotate(image, angle, resize=True, preserve_range=True)
    return rotated.astype(np.uint8)


def sauvola_binarize(image: np.ndarray, window_size: int = 15, k: float = 0.2) -> np.ndarray:
    """
    Apply Sauvola binarization for better text extraction.
    
    Args:
        image: Input grayscale image
        window_size: Local window size for threshold calculation
        k: Sauvola parameter controlling sensitivity
        
    Returns:
        Binary image (0 or 255)
    """
    # Apply Sauvola thresholding
    threshold = filters.threshold_sauvola(image, window_size=window_size, k=k)
    binary = image > threshold
    
    # Convert to 0/255 format
    return (binary * 255).astype(np.uint8)


def denoise_image(image: np.ndarray) -> np.ndarray:
    """
    Remove noise from binary image using morphological operations.
    
    Args:
        image: Input binary image
        
    Returns:
        Denoised binary image
    """
    # Remove small noise with opening operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    denoised = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    # Fill small holes with closing operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    
    return denoised


def suppress_diagonal_watermarks(image: np.ndarray, 
                               angle: float = 45.0,
                               kernel_size: int = 50,
                               apply_dilation: bool = False,
                               apply_erosion: bool = True) -> np.ndarray:
    """
    Suppress diagonal watermarks using morphological operations.
    
    Args:
        image: Input binary image
        angle: Angle of diagonal elements to suppress (degrees)
        kernel_size: Size of morphological kernel
        apply_dilation: Whether to apply dilation after suppression
        apply_erosion: Whether to apply erosion after suppression
        
    Returns:
        Image with diagonal watermarks suppressed
    """
    # Create diagonal kernel for morphological opening
    angle_rad = np.radians(angle)
    
    # Create line kernel at specified angle
    kernel_length = kernel_size
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    
    # Draw diagonal line in kernel
    center = kernel_size // 2
    for i in range(-center, center + 1):
        x = int(center + i * np.cos(angle_rad))
        y = int(center + i * np.sin(angle_rad))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1
    
    # Also create kernel for opposite diagonal
    opposite_angle_rad = np.radians(angle + 90)
    kernel2 = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    for i in range(-center, center + 1):
        x = int(center + i * np.cos(opposite_angle_rad))
        y = int(center + i * np.sin(opposite_angle_rad))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel2[y, x] = 1
    
    # Apply morphological opening to remove diagonal elements
    opened1 = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    opened2 = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel2)
    
    # Combine results - keep pixels that survived both operations
    result = cv2.bitwise_and(opened1, opened2)
    
    # Optional post-processing
    if apply_erosion:
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        result = cv2.erode(result, erosion_kernel, iterations=1)
    
    if apply_dilation:
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.dilate(result, dilation_kernel, iterations=1)
    
    return result


def detect_noise_level(image: np.ndarray) -> float:
    """
    Detect noise level in image using gradient analysis.
    
    Args:
        image: Input image (grayscale)
        
    Returns:
        Noise level estimate (0.0 = clean, 1.0 = very noisy)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Calculate gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # High frequency noise detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_variance = np.var(laplacian)
    
    # Normalize noise estimate (empirically determined thresholds)
    noise_level = min(noise_variance / 10000.0, 1.0)
    
    return noise_level


def adaptive_noise_removal(image: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Apply adaptive noise removal based on detected noise level.
    
    Args:
        image: Input image
        noise_level: Detected noise level (0.0-1.0)
        
    Returns:
        Denoised image
    """
    if noise_level < 0.1:
        # Clean image, minimal processing
        return image
    elif noise_level < 0.3:
        # Light noise, gentle denoising
        return cv2.bilateralFilter(image, 5, 50, 50)
    elif noise_level < 0.6:
        # Moderate noise, standard denoising
        return cv2.bilateralFilter(image, 9, 75, 75)
    else:
        # Heavy noise, aggressive denoising
        denoised = cv2.bilateralFilter(image, 13, 100, 100)
        # Additional morphological cleaning
        kernel = np.ones((2, 2), np.uint8)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        return denoised


def preprocess_image_for_ocr(image: np.ndarray,
                           deskew: bool = True,
                           sauvola_window: int = 15,
                           sauvola_k: float = 0.2,
                           suppress_diagonals: bool = True,
                           diagonal_angle: float = 45.0,
                           kernel_size: int = 50,
                           apply_dilation: bool = False,
                           apply_erosion: bool = True,
                           adaptive_noise_removal_enabled: bool = True) -> np.ndarray:
    """
    Complete enhanced preprocessing pipeline for OCR with adaptive noise removal.
    
    Args:
        image: Input grayscale image
        deskew: Whether to apply deskewing
        sauvola_window: Window size for Sauvola binarization
        sauvola_k: K parameter for Sauvola binarization
        suppress_diagonals: Whether to suppress diagonal watermarks
        diagonal_angle: Angle for diagonal suppression
        kernel_size: Kernel size for morphological operations
        apply_dilation: Whether to apply dilation
        apply_erosion: Whether to apply erosion
        adaptive_noise_removal_enabled: Whether to apply adaptive noise removal
        
    Returns:
        Preprocessed binary image ready for OCR
    """
    processed = image.copy()
    
    # Step 0: Adaptive noise removal (before other processing)
    if adaptive_noise_removal_enabled:
        noise_level = detect_noise_level(processed)
        if noise_level > 0.1:  # Only apply if significant noise detected
            processed = adaptive_noise_removal(processed, noise_level)
    
    # Step 1: Deskewing
    if deskew:
        processed = deskew_image(processed)
    
    # Step 2: Sauvola binarization
    processed = sauvola_binarize(processed, window_size=sauvola_window, k=sauvola_k)
    
    # Step 3: Standard denoising
    processed = denoise_image(processed)
    
    # Step 4: Diagonal watermark suppression
    if suppress_diagonals:
        processed = suppress_diagonal_watermarks(
            processed, 
            angle=diagonal_angle,
            kernel_size=kernel_size,
            apply_dilation=apply_dilation,
            apply_erosion=apply_erosion
        )
    
    return processed


def extract_page_as_image(pdf_path: Path, page_num: int, dpi: int = 300) -> np.ndarray:
    """
    Extract PDF page as grayscale image.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        dpi: Resolution for image extraction
        
    Returns:
        Grayscale image as numpy array
    """
    doc = fitz.open(str(pdf_path))
    
    if page_num >= len(doc):
        raise ValueError(f"Page {page_num} not found in PDF with {len(doc)} pages")
    
    page = doc[page_num]
    
    # Render page as image
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # Scale factor for DPI
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # Convert to numpy array
    img_data = pix.tobytes("ppm")
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    doc.close()
    return gray


def save_debug_images(original: np.ndarray, 
                     processed: np.ndarray, 
                     output_dir: Path, 
                     prefix: str) -> None:
    """
    Save debug images showing preprocessing steps.
    
    Args:
        original: Original grayscale image
        processed: Preprocessed binary image
        output_dir: Directory to save debug images
        prefix: Prefix for output filenames
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original
    cv2.imwrite(str(output_dir / f"{prefix}_original.png"), original)
    
    # Save processed
    cv2.imwrite(str(output_dir / f"{prefix}_processed.png"), processed)
    
    # Save side-by-side comparison
    h, w = original.shape
    comparison = np.zeros((h, w * 2), dtype=np.uint8)
    comparison[:, :w] = original
    comparison[:, w:] = processed
    cv2.imwrite(str(output_dir / f"{prefix}_comparison.png"), comparison)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess images for OCR with watermark suppression"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Input PDF file"
    )
    parser.add_argument(
        "--page",
        type=int,
        required=True,
        help="Page number to process (0-indexed)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output preprocessed image path"
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        help="Directory to save debug images"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for image extraction (default: 300)"
    )
    parser.add_argument(
        "--no-deskew",
        action="store_true",
        help="Skip deskewing step"
    )
    parser.add_argument(
        "--sauvola-window",
        type=int,
        default=15,
        help="Window size for Sauvola binarization (default: 15)"
    )
    parser.add_argument(
        "--sauvola-k",
        type=float,
        default=0.2,
        help="K parameter for Sauvola binarization (default: 0.2)"
    )
    parser.add_argument(
        "--no-diagonal-suppress",
        action="store_true",
        help="Skip diagonal watermark suppression"
    )
    parser.add_argument(
        "--diagonal-angle",
        type=float,
        default=45.0,
        help="Angle for diagonal suppression (default: 45.0)"
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=50,
        help="Kernel size for morphological operations (default: 50)"
    )
    parser.add_argument(
        "--apply-dilation",
        action="store_true",
        help="Apply dilation after suppression"
    )
    parser.add_argument(
        "--no-erosion",
        action="store_true",
        help="Skip erosion after suppression"
    )
    
    args = parser.parse_args()
    
    try:
        # Extract page as image
        print(f"Extracting page {args.page} from {args.pdf.name}...")
        original_image = extract_page_as_image(args.pdf, args.page, args.dpi)
        
        # Preprocess image
        print("Preprocessing image for OCR...")
        processed_image = preprocess_image_for_ocr(
            original_image,
            deskew=not args.no_deskew,
            sauvola_window=args.sauvola_window,
            sauvola_k=args.sauvola_k,
            suppress_diagonals=not args.no_diagonal_suppress,
            diagonal_angle=args.diagonal_angle,
            kernel_size=args.kernel_size,
            apply_dilation=args.apply_dilation,
            apply_erosion=not args.no_erosion
        )
        
        # Save processed image
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), processed_image)
        print(f"✅ Saved preprocessed image to {args.output}")
        
        # Save debug images if requested
        if args.debug_dir:
            prefix = f"{args.pdf.stem}_page_{args.page:03d}"
            save_debug_images(original_image, processed_image, args.debug_dir, prefix)
            print(f"✅ Saved debug images to {args.debug_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()



