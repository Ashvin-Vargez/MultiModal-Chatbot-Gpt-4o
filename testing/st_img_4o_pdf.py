import streamlit as st
import base64
import requests
import os
from dotenv import load_dotenv
import pdf2image
import tempfile
from pathlib import Path
import math

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
    """Convert PDF to images based on quality level"""
    # Create a temporary directory for PDF processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded PDF temporarily
        pdf_path = Path(temp_dir) / pdf_file.name
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getvalue())
        
        # Convert PDF to images
        pdf_images = pdf2image.convert_from_path(
            pdf_path,
            dpi=200,
            fmt="jpeg"
        )
        
        pages_per_image = get_pages_per_image(quality_level)
        total_pages = len(pdf_images)
        processed_images = []
        
        # Process images according to quality level
        for i in range(0, total_pages, pages_per_image):
            end_idx = min(i + pages_per_image, total_pages)
            if end_idx - i == pages_per_image:  # Only process full sets of pages
                # Calculate dimensions for the combined image
                width = pdf_images[i].width
                height = pdf_images[i].height * pages_per_image
                
                # Create a new image with the calculated dimensions
                combined_image = Image.new('RGB', (width, height))
                
                # Paste each page into the combined image
                for j, page_idx in enumerate(range(i, end_idx)):
                    combined_image.paste(pdf_images[page_idx], (0, j * height // pages_per_image))
                
                # Generate filename with page range
                filename = f"{pdf_file.name.rsplit('.', 1)[0]}_pages_{i+1}-{end_idx}.jpg"
                
                # Save the combined image to bytes
                img_byte_arr = io.BytesIO()
                combined_image.save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)
                
                # Create a named temporary file
                processed_images.append({
                    'name': filename,
                    'data': img_byte_arr
                })
        
        return processed_images

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