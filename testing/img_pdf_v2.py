import streamlit as st
import base64
import requests
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
import tempfile
from pathlib import Path
import math
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

def encode_image(image_bytes):
    """Encode image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode('utf-8')

def get_pages_per_image(quality_level):
    """Determine number of PDF pages per output image based on quality level"""
    pages_map = {
        5: 1,    # 1 page per image
        4: 2,    # 2 pages per image
        3: 4,    # 4 pages per image
        2: 6,    # 6 pages per image
        1: 8     # 8 pages per image
    }
    return pages_map.get(quality_level, 1)

def convert_pdf_to_images(pdf_file, quality_level):
    """Convert PDF to images based on quality level using PyMuPDF"""
    try:
        # Read PDF file
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        total_pages = len(pdf_document)
        pages_per_image = get_pages_per_image(quality_level)
        processed_images = []
        
        # Process images according to quality level
        for i in range(0, total_pages, pages_per_image):
            end_idx = min(i + pages_per_image, total_pages)
            current_pages = end_idx - i
            
            # Skip if we don't have enough pages for a complete set, unless it's the last set
            if current_pages < pages_per_image and i + pages_per_image < total_pages:
                continue
                
            # Calculate dimensions for the combined image
            # Get the first page to determine dimensions
            first_page = pdf_document[i]
            rect = first_page.rect
            zoom = 2  # Increase quality of the image
            mat = fitz.Matrix(zoom, zoom)
            
            # Create a new PIL Image to hold all pages
            combined_width = int(rect.width * zoom)
            combined_height = int(rect.height * zoom * current_pages)
            combined_image = Image.new('RGB', (combined_width, combined_height), 'white')
            
            # Add each page to the combined image
            for j, page_idx in enumerate(range(i, end_idx)):
                page = pdf_document[page_idx]
                pix = page.get_pixmap(matrix=mat)
                
                # Convert PyMuPDF pixmap to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Paste into the combined image
                combined_image.paste(img, (0, j * int(rect.height * zoom)))
            
            # Generate filename with page range
            filename = f"{pdf_file.name.rsplit('.', 1)[0]}_pages_{i+1}-{end_idx}.jpg"
            
            # Save the combined image to bytes
            img_byte_arr = io.BytesIO()
            combined_image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            
            # Add to processed images
            processed_images.append({
                'name': filename,
                'data': img_byte_arr
            })
        
        pdf_document.close()
        return processed_images
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return []

def analyze_images(images, messages):
    """Analyze images with chat history context"""
    if not images:
        return "Please upload at least one image or PDF."
    
    # Initialize API messages with system message
    api_messages = [
        {
            "role": "system",
            "content": "You are an AI assistant analyzing images and PDFs, engaging in conversation about them."
        }
    ]
    
    # Add the first message with images
    first_content = [
        {
            "type": "text",
            "text": messages[0]["content"]
        }
    ]
    
    # Add all images to the first content
    for img in images:
        base64_image = encode_image(img.getvalue())
        first_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    # Add first message with images
    api_messages.append({
        "role": "user",
        "content": first_content
    })
    
    # Add subsequent messages from chat history
    for message in messages[1:]:
        api_messages.append({
            "role": message["role"],
            "content": message["content"]
        })
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": api_messages,
        "max_tokens": 300
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error making request to OpenAI API: {str(e)}"
    except KeyError as e:
        return f"Error parsing API response: {str(e)}"

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = None
    if "processed_pdfs" not in st.session_state:
        st.session_state.processed_pdfs = []
    if "quality_level" not in st.session_state:
        st.session_state.quality_level = 3

def main():
    st.title("Multi-Image and PDF Analysis Chat App")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Upload Files")
        
        # PDF Quality Slider
        st.session_state.quality_level = st.slider(
            "PDF Quality Level",
            min_value=1,
            max_value=5,
            value=st.session_state.quality_level,
            help="5: 1 page per image\n4: 2 pages per image\n3: 4 pages per image\n2: 6 pages per image\n1: 8 pages per image"
        )
        
        # File uploaders
        uploaded_images = st.file_uploader(
            "Choose images to analyze",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        uploaded_pdfs = st.file_uploader(
            "Choose PDFs to analyze",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        # Process uploaded files
        all_files = []
        
        # Handle image files
        if uploaded_images:
            all_files.extend(uploaded_images)
        
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
            st.write("### Uploaded Files")
            
            # Display images in a grid
            cols = st.columns(2)
            for idx, file in enumerate(all_files):
                with cols[idx % 2]:
                    if hasattr(file, 'name'):
                        caption = file.name
                    else:
                        caption = st.session_state.processed_pdfs[idx]['name']
                    st.image(file, caption=caption, use_column_width=True)
    
    # Main chat interface
    st.write("""
    Upload images and PDFs using the sidebar and start a conversation about them. 
    Adjust the PDF quality level to control how pages are combined into images.
    The AI will maintain context of the conversation while analyzing the files.
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
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = analyze_images(
                    st.session_state.uploaded_files,
                    st.session_state.messages
                )
                st.write(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Add a button to clear chat history and processed files
    if st.button("Clear All"):
        st.session_state.messages = []
        st.session_state.processed_pdfs = []
        st.session_state.uploaded_files = None
        st.rerun()

if __name__ == "__main__":
    main()