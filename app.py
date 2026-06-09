import streamlit as st
from PIL import Image 
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch 

# 1. Page Configuration
st.set_page_config(
    page_title="AI DETECTOR", 
    layout="centered"
) 
# 2. Load the AI DETECTION Model 
st.cache_resource 
def load_model():
    #Utilizing a specialized Vision Transformer model for AI generation detection
    model_name = "umm-maybe/AI-image-detector"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

try:
    processor, model = load_model()
except Exception as e:
    st.error("AI DETECTOR")
    st.error("Error loading the AI Detection model.Please check your internet connection.")
    st.stop()

# 3. UI Header
st.title("AI DETECTOR")
st.subheader("Detect whether a human photo is real or AI-Generated")
st.write("upload a photo below, and our vision Transformer model will analyse it for non_realistic patterns.")

st.markdown("---")

# 4. File Uploader
uploaded_file= st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    #Display the uploaded image 
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("")

    # 5. Prediction Logic
    with st.spinner("Analyzing the image...Please wait.."):
        #Preprocess the image for the model
        inputs = processor(images=image, return_tensors="pt")

        #Forward pass through the network (without calculating gradients)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Convert raw scores (logits) into probability percentages
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        labels = model.config.id2label


        # Determine the highest scoring predection
        top_predection_idx = torch.argmax(probabilities).item()
        confidence = probabilities[top_predection_idx].item() *100
        predicted_label = labels[top_predection_idx]   


    # 6. Display Results
    st.markdown("### Detection Result")

    # Check if the highest scoring label indicates AI/Synthetics origin
    is_ai= any(x in predicted_label.lower() for x in["artificial", "fake" "synthetic", "ai"])


    if is_ai:
        st.error(f"**Result: AIGENERATED / NON-REALISTIC**")
        st.write(f"**Confidence Level:** {confidence:2f}%")
        st.progress(int(confidence))
        st.warning("This photo shows structural anomalies or synthetic patterns typical of AI generators.")
    else:
        st.success(f"**Result: REAL HUMAN PHOTO**")
        st.write(f"**Confidence Level:** {confidence:2f}%")
        st.progress(int(confidence))
        st.info("The photo appears to be authentic and captured by a traditional camera.")


    # Show full breakdown
    with st.expander("See Detailed Probabality Breakdown"):
        for idx, prob in enumerate(probabilities):
            st.write(f"**{labels[idx].upper()}**:{prob.item()*100:2f}%")


# 7. Lyout Footer
st.divider()
st.caption("AI DETECTOR v1.0 . Powered by Hugging Face ViT")

