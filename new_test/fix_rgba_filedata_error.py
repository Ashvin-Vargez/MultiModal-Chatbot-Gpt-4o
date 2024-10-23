import streamlit as st
import base64
import requests
import os
from dotenv import load_dotenv
from PIL import Image
import io
import fitz  # PyMuPDF
import tempfile

# Load environment variables
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

def get_zoom_factor(quality):
    """Convert quality setting to zoom factor"""
    quality_map = {
        "Very High": 1.0,
        "High": 0.5,
        "Medium": 0.33,
        "Low": 0.25,
        "Very Low": 1/6
    }
    return quality_map[quality]

def process_image(image_file, quality):
    """Process image according to quality setting"""
    # Read image into PIL
    img = Image.open(image_file)
    
    # Convert RGBA to RGB if necessary
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        # Create a white background image
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        # Paste the image on the background
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    if quality == "Very High":
        # Convert to bytes
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

def pdf_to_images(pdf_file, quality):
    """Convert PDF to images with specified quality"""
    images = []
    captions = []
    
    # Read the PDF content into memory
    pdf_content = pdf_file.read()
    
    try:
        # Open PDF from memory
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        zoom_factor = get_zoom_factor(quality)
        
        # Convert each page to image
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            matrix = fitz.Matrix(zoom_factor, zoom_factor)
            pix = page.get_pixmap(matrix=matrix)
            
            # Convert to PIL Image
            img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            img_data.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            
            images.append(img_byte_arr)
            captions.append(f"{pdf_file.name} - Page {page_num + 1}")
        
        pdf_document.close()
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return [], []
    
    return images, captions

def encode_image(image_bytes):
    """Encode image bytes to base64 string"""
    if isinstance(image_bytes, io.BytesIO):
        return base64.b64encode(image_bytes.getvalue()).decode('utf-8')
    return base64.b64encode(image_bytes).decode('utf-8')

def analyze_images(images, messages):
    """Analyze images with chat history context"""
    if not images:
        return "Please upload at least one image."
    
    api_messages = [
        {
            "role": "system",
            "content": "You are an AI assistant analyzing images and engaging in conversation about them."
        }
    ]
    
    first_content = [
        {
            "type": "text",
            "text": messages[0]["content"]
        }
    ]
    
    for img in images:
        base64_image = encode_image(img)
        first_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    api_messages.append({
        "role": "user",
        "content": first_content
    })
    
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
        "max_tokens": 5000
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
    if "processed_images" not in st.session_state:
        st.session_state.processed_images = []
    if "image_captions" not in st.session_state:
        st.session_state.image_captions = []

def main():
    st.title("Multi-Image Analysis")
    
    initialize_session_state()
    
    with st.sidebar:
        st.header("Upload Settings")
        
        # Quality selector
        quality = st.select_slider(
            "Select Image Quality",
            options=["Very Low", "Low", "Medium", "High", "Very High"],
            value="High"
        )
        
        st.header("Upload Files")
        uploaded_files = st.file_uploader(
            "Choose images or PDFs to analyze",
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
                        pdf_images, pdf_captions = pdf_to_images(file, quality)
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
    Use the quality slider to control the resolution of uploaded files.
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