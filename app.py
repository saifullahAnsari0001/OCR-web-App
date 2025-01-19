import streamlit as st
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torch

# Ensure the use of CPU only
device = torch.device("cpu")

# Load the tokenizer and model on CPU only
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0',
                                  trust_remote_code=True,
                                  low_cpu_mem_usage=True,  # Optimize for CPU
                                  use_safetensors=True, 
                                  pad_token_id=tokenizer.eos_token_id)

# Make sure the model stays on CPU
model = model.eval().to(device)

# Streamlit app definition
st.title('OCR Application: Extract and Search Text from Images')

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    image.save("uploaded_image.jpg")
    image_file = "uploaded_image.jpg"

    with st.spinner('Extracting text from the image...'):
        try:
            # OCR extraction
            res = model.chat(tokenizer, image_file, ocr_type='ocr')
            extracted_text = res['text']  # Assuming this structure from the model output
        except Exception as e:
            st.error(f"Error during OCR extraction: {str(e)}")
            extracted_text = ""

    if extracted_text:
        st.subheader("Extracted Text:")
        st.write(extracted_text)

        keyword = st.text_input("Enter a keyword to search in the extracted text")
        if keyword:
            search_results = [line for line in extracted_text.splitlines() if keyword.lower() in line.lower()]
            if search_results:
                st.subheader(f"Search Results for '{keyword}':")
                for result in search_results:
                    st.write(f"**{result}**")
            else:
                st.write(f"No results found for '{keyword}'")
    else:
        st.warning("No text extracted. Please upload a valid image containing text.")
