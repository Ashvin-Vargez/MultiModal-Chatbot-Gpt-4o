import streamlit as st
import base64
import requests
import os
from dotenv import load_dotenv
import fitz
import tempfile
from pathlib import Path
import math
from PIL import Image
import io
import datetime
from typing import Dict, List

# Load environment variables
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create a directory for saving converted images
OUTPUT_DIR = Path("converted_pdfs")
OUTPUT_DIR.mkdir(exist_ok=True)

def initialize_session_state():
    """Initialize session state variables"""
    session_vars = {
        "messages": [],
        "uploaded_files": None,
        "processed_pdfs": [],
        "quality_level": 3,
        "conversion_folders": [],
        "current_conversion": None,
        "selected_images": {},  # Dict to track selected/deselected images
        "image_quality": 2,     # Default to medium quality (1=low, 2=medium, 3=high)
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

def resize_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    """Resize image maintaining aspect ratio with maximum dimension of max_size"""
    ratio = min(max_size / image.size[0], max_size / image.size[1])
    if ratio < 1:  # Only resize if image is larger than max_size
        new_size = tuple(int(dim * ratio) for dim in image.size)
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image

def process_image_quality(image: Image.Image, quality_level: int) -> Image.Image:
    """Process image based on quality level (1=low, 2=medium, 3=high)"""
    if quality_level == 3:  # High - original size
        return image
    elif quality_level == 2:  # Medium - half size
        new_size = tuple(dim // 2 for dim in image.size)
        return image.resize(new_size, Image.Resampling.LANCZOS)
    else:  # Low - quarter size
        new_size = tuple(dim // 4 for dim in image.size)
        return image.resize(new_size, Image.Resampling.LANCZOS)

def convert_pdf_to_images(pdf_file, quality_level):
    """Convert PDF to images with 512x512 max resolution"""
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        total_pages = len(pdf_document)
        pages_per_image = get_pages_per_image(quality_level)
        processed_images = []
        
        for i in range(0, total_pages, pages_per_image):
            end_idx = min(i + pages_per_image, total_pages)
            current_pages = end_idx - i
            
            if current_pages < pages_per_image and i + pages_per_image < total_pages:
                continue
                
            first_page = pdf_document[i]
            rect = first_page.rect
            zoom = 2
            mat = fitz.Matrix(zoom, zoom)
            
            combined_width = int(rect.width * zoom)
            combined_height = int(rect.height * zoom * current_pages)
            combined_image = Image.new('RGB', (combined_width, combined_height), 'white')
            
            for j, page_idx in enumerate(range(i, end_idx)):
                page = pdf_document[page_idx]
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                combined_image.paste(img, (0, j * int(rect.height * zoom)))
            
            # Resize to max 512x512
            combined_image = resize_image(combined_image, 512)
            
            filename = f"{pdf_file.name.rsplit('.', 1)[0]}_pages_{i+1}-{end_idx}.jpg"
            
            img_byte_arr = io.BytesIO()
            combined_image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            
            processed_images.append({
                'name': filename,
                'data': img_byte_arr,
                'selected': True  # Default to selected
            })
            
            # Initialize selection state for this image
            if filename not in st.session_state.selected_images:
                st.session_state.selected_images[filename] = True
        
        pdf_document.close()
        
        if processed_images:
            save_dir, saved_paths = save_images_locally(processed_images, pdf_file.name)
        
        return processed_images
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return []

def display_image_selection():
    """Display images with selection checkboxes"""
    st.write("### Select Images to Analyze")
    
    # Create columns for the grid
    cols = st.columns(3)
    col_idx = 0
    
    # Display uploaded images
    if st.session_state.uploaded_files:
        for idx, file in enumerate(st.session_state.uploaded_files):
            with cols[col_idx]:
                if hasattr(file, 'name'):
                    # Original image upload
                    key = f"img_{file.name}"
                    st.image(file, use_column_width=True)
                    selected = st.checkbox("Include in analysis", 
                                        key=key,
                                        value=st.session_state.selected_images.get(key, True))
                    st.session_state.selected_images[key] = selected
                else:
                    # PDF-converted image
                    pdf_info = st.session_state.processed_pdfs[idx]
                    key = pdf_info['name']
                    st.image(pdf_info['data'], use_column_width=True)
                    selected = st.checkbox("Include in analysis",
                                        key=key,
                                        value=st.session_state.selected_images.get(key, True))
                    st.session_state.selected_images[key] = selected
                
                col_idx = (col_idx + 1) % 3

def get_selected_images():
    """Return only the selected images for analysis"""
    selected_files = []
    if st.session_state.uploaded_files:
        for file in st.session_state.uploaded_files:
            key = f"img_{file.name}" if hasattr(file, 'name') else file.name
            if st.session_state.selected_images.get(key, True):
                selected_files.append(file)
    return selected_files

def main():
    st.title("Multi-Image and PDF Analysis Chat App")
    
    initialize_session_state()
    
    with st.sidebar:
        st.header("Upload Files")
        
        # Image quality slider for direct uploads
        st.session_state.image_quality = st.select_slider(
            "Image Upload Quality",
            options=[1, 2, 3],
            value=st.session_state.image_quality,
            format_func=lambda x: {1: "Low", 2: "Medium", 3: "High"}[x],
            help="Affects only directly uploaded images, not PDF conversions"
        )
        
        # PDF Quality Slider
        st.session_state.quality_level = st.slider(
            "PDF Pages per Image",
            min_value=1,
            max_value=5,
            value=st.session_state.quality_level,
            help="5: 1 page per image\n4: 2 pages per image\n3: 4 pages per image\n2: 6 pages per image\n1: 8 pages per image"
        )
        
        # File uploaders with icons
        st.markdown("### 📸 Upload Images")
        uploaded_images = st.file_uploader(
            "Choose images to analyze",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        st.markdown("### 📄 Upload PDFs")
        uploaded_pdfs = st.file_uploader(
            "Choose PDFs to analyze",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        # Process uploaded files
        all_files = []
        
        # Handle image files with quality processing
        if uploaded_images:
            for img_file in uploaded_images:
                img = Image.open(img_file)
                processed_img = process_image_quality(img, st.session_state.image_quality)
                
                # Convert back to bytes
                img_byte_arr = io.BytesIO()
                processed_img.save(img_byte_arr, format=img.format or 'JPEG')
                img_byte_arr.seek(0)
                all_files.append(img_byte_arr)
        
        # Handle PDF files
        if uploaded_pdfs:
            with st.spinner("Converting PDFs to images..."):
                for pdf_file in uploaded_pdfs:
                    processed_images = convert_pdf_to_images(
                        pdf_file,
                        st.session_state.quality_level
                    )
                    st.session_state.processed_pdfs.extend(processed_images)
                    all_files.extend([img['data'] for img in processed_images])
        
        if all_files:
            st.session_state.uploaded_files = all_files
    
    # Display image selection interface
    if st.session_state.uploaded_files:
        display_image_selection()
    
    # Chat interface
    st.write("""
    Upload images and PDFs using the sidebar and start a conversation about them.
    Select which images to include in the analysis using the checkboxes below each image.
    """)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about the files..."):
        if not st.session_state.uploaded_files:
            st.error("Please upload at least one image or PDF first.")
            return
        
        if not api_key:
            st.error("OpenAI API key not found. Please check your .env file.")
            return
        
        # Get only selected images for analysis
        selected_files = get_selected_images()
        
        if not selected_files:
            st.warning("Please select at least one image for analysis.")
            return
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = analyze_images(selected_files, st.session_state.messages)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    if st.button("Clear All"):
        st.session_state.messages = []
        st.session_state.processed_pdfs = []
        st.session_state.uploaded_files = None
        st.session_state.selected_images = {}
        st.rerun()

if __name__ == "__main__":
    main()