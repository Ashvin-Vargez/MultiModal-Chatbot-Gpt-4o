import streamlit as st
import base64
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

def encode_image(image_bytes):
    """Encode image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode('utf-8')

def analyze_images(images, messages):
    """Analyze images with chat history context"""
    if not images:
        return "Please upload at least one image."
    
    # Initialize API messages with system message
    api_messages = [
        {
            "role": "system",
            "content": "You are an AI assistant analyzing images and engaging in conversation about them."
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
        "max_tokens": 5000
    }
    # print(api_messages)
    
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

def main():
    st.title("Multi-Image Analysis ")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Images")
        uploaded_files = st.file_uploader(
            "Choose images to analyze",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.write("### Uploaded Images")
            for idx, image in enumerate(uploaded_files):
                st.image(image, caption=f"Image {idx + 1}", use_column_width=True)
    
    # Main chat interface
    st.write("""
    Upload images using the sidebar and start a conversation about them. 
    The AI will maintain context of the conversation while analyzing the images.
    """)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about the images..."):
        if not st.session_state.uploaded_files:
            st.error("Please upload at least one image first.")
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
    
    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()