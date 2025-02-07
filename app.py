import streamlit as st
import os
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.tavily import TavilyTools
from constants import SYSTEM_PROMPT, INSTRUCTIONS
from PIL import Image
import requests
from io import BytesIO

# Load API keys from Streamlit secrets
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Initialize AI agent
agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[TavilyTools()],
    system_prompt=SYSTEM_PROMPT,
    instructions=INSTRUCTIONS,
)

st.title("📸 Image Analysis with AI")

# Option to upload, provide URL, or capture photo
option = st.radio("Choose an option:", ["Upload Image", "Provide URL", "Capture Photo"])

image = None
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="📤 Uploaded Image", use_column_width=True)

elif option == "Provide URL":
    image_url = st.text_input("Enter Image URL:")
    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="🌐 Image from URL", use_column_width=True)
        except Exception as e:
            st.error("❌ Error loading image from URL")

elif option == "Capture Photo":
    captured_image = st.camera_input("📷 Take a photo")
    if captured_image:
        image = Image.open(captured_image)
        st.image(image, caption="📸 Captured Image", use_column_width=True)

# Process image if available
if image and st.button("🔍 Analyze Image"):
    image_path = "temp_image.jpg"
    image.save(image_path)

    with st.spinner("Analyzing the image... 🔄"):
        response = agent.run(
            "Analyze the product image", 
            images=[image_path]
        )

    st.success("✅ Analysis Complete!")
    st.write("### 🔍 AI Analysis Result:")
    st.write(response)
