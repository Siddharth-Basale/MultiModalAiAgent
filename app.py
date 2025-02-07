import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.tavily import TavilyTools
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Product Analyzer",
    page_icon=":mag:",
    layout="centered",
)

# Get secrets from Streamlit
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

# System prompts (replace with your actual prompts from constants.py)
SYSTEM_PROMPT = """You are a professional product analyst..."""
INSTRUCTIONS = """Analyze products based on..."""

@st.cache_resource
def get_agent():
    """Create and cache the Agent with Gemini model"""
    return Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[TavilyTools()],
        system_prompt=SYSTEM_PROMPT,
        instructions=INSTRUCTIONS,
    )

def main():
    st.title(":mag: Product Image Analyzer")
    st.markdown("Analyze product images using Google Gemini and web search")

    # Get the cached agent
    agent = get_agent()

    # User inputs
    query = st.text_input(
        "Enter your analysis request:",
        placeholder="Analyze the product image..."
    )
    uploaded_file = st.file_uploader(
        "Upload product image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )

    if st.button("Analyze", type="primary"):
        if not query:
            st.error("Please enter an analysis request")
            return
        if not uploaded_file:
            st.error("Please upload a product image")
            return

        with st.spinner("Analyzing..."):
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                image_path = tmp_file.name

            try:
                # Create a placeholder for streaming response
                response_placeholder = st.empty()
                full_response = ""

                # Get streaming response
                response_stream = agent.print_response(
                    query,
                    images=[image_path],
                    stream=True
                )

                # Display streamed response
                for chunk in response_stream:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                # Final update without cursor
                response_placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
            finally:
                os.unlink(image_path)

if __name__ == "__main__":
    main()
