import streamlit as st
import os
from PIL import Image
from io import BytesIO
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.tavily import TavilyTools
from tempfile import NamedTemporaryFile
from constants import SYSTEM_PROMPT, INSTRUCTIONS
import requests

# Load API keys from Streamlit secrets
os.environ['TAVILY_API_KEY'] = st.secrets['TAVILY_API_KEY']
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

MAX_IMAGE_WIDTH = 300  # Display size limit

def resize_image_for_display(image_file):
    """Resize image for display only, returns bytes"""
    img = Image.open(image_file)
    aspect_ratio = img.height / img.width
    new_height = int(MAX_IMAGE_WIDTH * aspect_ratio)
    img = img.resize((MAX_IMAGE_WIDTH, new_height), Image.Resampling.LANCZOS)
    
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@st.cache_resource
def get_agent():
    """Load the AI agent once"""
    return Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        system_prompt=SYSTEM_PROMPT,
        instructions=INSTRUCTIONS,
        tools=[TavilyTools(api_key=os.getenv("TAVILY_API_KEY"))],
        markdown=True,
    )

def analyze_image(image_path):
    """Run AI analysis on the given image"""
    agent = get_agent()
    with st.spinner('🔍 Analyzing image...'):
        response = agent.run("Analyze the given image", images=[image_path])
        st.success("✅ Analysis Complete!")
        st.markdown(response.content)

def save_uploaded_file(uploaded_file):
    """Save uploaded file temporarily"""
    with NamedTemporaryFile(dir='.', suffix='.jpg', delete=False) as f:
        f.write(uploaded_file.getbuffer())
        return f.name

def main():
    st.title("📸 AI Product Image Analyzer")

    tab_upload, tab_url, tab_camera = st.tabs([
        "📤 Upload Image", 
        "🌐 Provide URL", 
        "📸 Capture Photo"
    ])
    
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file:
            resized_image = resize_image_for_display(uploaded_file)
            st.image(resized_image, caption="Uploaded Image", use_column_width=False, width=MAX_IMAGE_WIDTH)
            if st.button("🔍 Analyze Uploaded Image"):
                temp_path = save_uploaded_file(uploaded_file)
                analyze_image(temp_path)
                os.unlink(temp_path)  # Delete temp file
    
    with tab_url:
        image_url = st.text_input("Enter Image URL:")
        if image_url:
            try:
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                resized_image = resize_image_for_display(img)
                st.image(resized_image, caption="Image from URL", use_column_width=False, width=MAX_IMAGE_WIDTH)
                if st.button("🔍 Analyze URL Image"):
                    temp_path = "temp_url_image.jpg"
                    img.save(temp_path)
                    analyze_image(temp_path)
                    os.unlink(temp_path)
            except Exception as e:
                st.error("❌ Error loading image from URL")

    with tab_camera:
        captured_image = st.camera_input("Take a picture")
        if captured_image:
            resized_image = resize_image_for_display(captured_image)
            st.image(resized_image, caption="Captured Photo", use_column_width=False, width=MAX_IMAGE_WIDTH)
            if st.button("🔍 Analyze Captured Photo"):
                temp_path = save_uploaded_file(captured_image)
                analyze_image(temp_path)
                os.unlink(temp_path)

if __name__ == "__main__":
    st.set_page_config(
        page_title="AI Product Analyzer",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    main()
