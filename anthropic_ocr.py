import streamlit as st
import base64
import requests
import os
from dotenv import load_dotenv
from PIL import Image
import io
import fitz  # PyMuPDF
import math
import anthropic

# Load environment variables
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=api_key)

# [Previous functions remain the same up to analyze_images]
def get_zoom_factor(quality):
    """Convert quality setting to zoom factor"""
    quality_map = {
        "Very High": 1.0,
        "High": 0.7,
        "Medium": 0.45,
        "Low": 0.3,
        "Very Low": 0.2
    }
    return quality_map[quality]

def get_pages_per_image(pdf_quality):
    """Convert PDF quality setting to pages per image"""
    quality_map = {
        "Very High": 1,
        "High": 2,
        "Medium": 4,
        "Low": 6,
        "Very Low": 8
    }
    return quality_map[pdf_quality]

def combine_images_vertically(images):
    """Combine multiple PIL images vertically"""
    if not images:
        return None
    
    # Calculate total height and maximum width
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)
    
    # Create new image with white background
    combined = Image.new('RGB', (max_width, total_height), 'white')
    
    # Paste images
    y_offset = 0
    for img in images:
        # Center image if width is less than max_width
        x_offset = (max_width - img.width) // 2
        combined.paste(img, (x_offset, y_offset))
        y_offset += img.height
    
    return combined

def process_image(image_file, quality):
    """Process image according to quality setting"""
    # Read image into PIL
    img = Image.open(image_file)
    
    # Convert RGBA to RGB if necessary
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    if quality == "Very High":
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        return img_byte_arr
    
    # Calculate new dimensions based on quality
    zoom = get_zoom_factor(quality)
    new_width = int(img.width * zoom)
    new_height = int(img.height * zoom)
    
    # Resize image
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    resized_img.save(img_byte_arr, format='JPEG', quality=95)
    img_byte_arr.seek(0)
    return img_byte_arr

def pdf_to_images(pdf_file, quality, pdf_quality):
    """Convert PDF to images with specified quality and pages per image"""
    images = []
    captions = []
    
    try:
        # Read the PDF content into memory
        pdf_content = pdf_file.read()
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        zoom_factor = get_zoom_factor(quality)
        pages_per_image = get_pages_per_image(pdf_quality)
        
        # Calculate number of combined images needed
        total_pages = pdf_document.page_count
        num_combined_images = math.ceil(total_pages / pages_per_image)
        
        # Process each group of pages
        for group_idx in range(num_combined_images):
            start_page = group_idx * pages_per_image
            end_page = min(start_page + pages_per_image, total_pages)
            page_images = []
            
            # Convert each page in the group to image
            for page_num in range(start_page, end_page):
                page = pdf_document[page_num]
                matrix = fitz.Matrix(zoom_factor, zoom_factor)
                pix = page.get_pixmap(matrix=matrix)
                img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_images.append(img_data)
            
            # Combine pages into one image
            combined_image = combine_images_vertically(page_images)
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            combined_image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            
            images.append(img_byte_arr)
            
            # Create caption with page range
            page_range = f"Pages {start_page + 1}-{end_page}"
            if end_page == start_page + 1:
                page_range = f"Page {start_page + 1}"
            captions.append(f"{pdf_file.name} - {page_range}")
        
        pdf_document.close()
        
    except Exception as e:
        st.error(f"Error processing PDF {pdf_file.name}: {str(e)}")
        return [], []
    
    return images, captions

def encode_image(image_bytes):
    """Encode image bytes to base64 string"""
    if isinstance(image_bytes, io.BytesIO):
        return base64.b64encode(image_bytes.getvalue()).decode('utf-8')
    return base64.b64encode(image_bytes).decode('utf-8')

def analyze_images(images, messages):
    """Analyze images with chat history context using Anthropic API"""
    if not images:
        return "Please upload at least one image."
    
    # Prepare system message
    system_message = "You are Claude, an AI assistant analyzing images and engaging in conversation about them."
    
    # Prepare the message content
    content = []
    
    # Add system message
    content.append({
        "type": "text",
        "text": system_message
    })
    
    # Add the first user message with images
    content.append({
        "type": "text",
        "text": messages[0]["content"]
    })
    
    # Add all images
    for img in images:
        base64_image = encode_image(img)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64_image
            }
        })
    
    # Add subsequent messages
    for message in messages[1:]:
        content.append({
            "type": "text",
            "text": message["content"]
        })
    
    try:
        # Create the message using Anthropic's API
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            temperature=0.7,
            messages=[{
                "role": "user",
                "content": content
            }]
        )
        
        # Extract the response text
        return message.content[0].text
        
    except anthropic.APIError as e:
        return f"Error making request to Anthropic API: {str(e)}"
    except Exception as e:
        return f"Error processing request: {str(e)}"

# [Rest of the code remains the same]
def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = None
    if "processed_images" not in st.session_state:
        st.session_state.processed_images = []
    if "image_captions" not in st.session_state:
        st.session_state.image_captions = []

def main():
    st.title("AIQ Testbed - PDF Analysis")
    
    initialize_session_state()
    
    with st.sidebar:
        st.header("Upload Settings")
        
        # Overall quality selector with info icon
        col1, col2 = st.columns([5, 1])
        with col1:
            quality = st.select_slider(
                "Overall Quality",
                options=["Very Low", "Low", "Medium", "High", "Very High"],
                value="High"
            )
        # with col2:
        #     st.info("ℹ️ Higher quality settings will incur higher processing charges, use only when performance is not satisfactory with lower settings")
        
        # PDF quality selector with info icon
        col3, col4 = st.columns([5, 1])
        with col3:
            pdf_quality = st.select_slider(
                "PDF Quality",
                options=["Very Low", "Low", "Medium", "High", "Very High"],
                value="High"
            )
        # with col4:
        #     st.info("ℹ️ Try lower settings if any errors are encountered")
        
        st.header("Upload Files")
        uploaded_files = st.file_uploader(
            "Upload PDFs or images",
            type=["jpg", "jpeg", "png", "pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.session_state.processed_images = []
            st.session_state.image_captions = []
            
            for file in uploaded_files:
                try:
                    if file.type == "application/pdf":
                        # Handle PDF
                        pdf_images, pdf_captions = pdf_to_images(file, quality, pdf_quality)
                        st.session_state.processed_images.extend(pdf_images)
                        st.session_state.image_captions.extend(pdf_captions)
                    else:
                        # Handle image
                        processed_img = process_image(file, quality)
                        st.session_state.processed_images.append(processed_img)
                        st.session_state.image_captions.append(file.name)
                except Exception as e:
                    st.error(f"Error processing file {file.name}: {str(e)}")
                    continue
            
            st.write("### Processed Files")
            for idx, (image, caption) in enumerate(zip(st.session_state.processed_images, st.session_state.image_captions)):
                st.image(image, caption=caption, use_column_width=True)
    
    st.write("""
    Upload images or PDFs using the sidebar and start a conversation about them. 
    Use the quality sliders to control the resolution and page grouping of uploaded files.
    """)
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask about the images..."):
        if not st.session_state.processed_images:
            st.error("Please upload at least one file first.")
            return
        
        if not api_key:
            st.error("OpenAI API key not found. Please check your .env file.")
            return
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = analyze_images(
                    st.session_state.processed_images,
                    st.session_state.messages
                )
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()