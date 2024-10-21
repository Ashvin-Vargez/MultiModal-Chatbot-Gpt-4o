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

def analyze_images(images, question):
    """Analyze multiple images with OpenAI API"""
    if not images:
        return "Please upload at least one image."
    
    # Prepare the messages content
    content = [
        {
            "type": "text",
            "text": question if question else "What's in these images?"
        }
    ]
    
    # Add all images to the content
    for img in images:
        base64_image = encode_image(img.getvalue())
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
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

def main():
    st.title("Multi-Image Analysis App")
    
    # Add some information about the app
    st.write("""
    Upload multiple images and ask questions about them using OpenAI's GPT-4 Vision model.
    """)
    
    # File uploader for multiple images
    uploaded_files = st.file_uploader(
        "Choose images to analyze",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    # Display uploaded images in a horizontal layout
    if uploaded_files:
        cols = st.columns(len(uploaded_files))
        for idx, image in enumerate(uploaded_files):
            with cols[idx]:
                st.image(image, caption=f"Image {idx + 1}", use_column_width=True)
    
    # Text input for the question
    question = st.text_input(
        "What would you like to know about the images?",
        placeholder="e.g., 'What objects do you see in these images?'"
    )
    
    # Analysis button
    if st.button("Analyze Images"):
        if not uploaded_files:
            st.warning("Please upload at least one image first.")
        elif not api_key:
            st.error("OpenAI API key not found. Please check your .env file.")
        else:
            with st.spinner("Analyzing images..."):
                result = analyze_images(uploaded_files, question)
                st.write("### Analysis Result")
                st.write(result)

if __name__ == "__main__":
    main()