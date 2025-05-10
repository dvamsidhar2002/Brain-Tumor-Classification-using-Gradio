import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# Load the trained model
model_path = 'Residual_Xception_parallel_model_integration.h5'
model = load_model(model_path)

# Define image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    # If image is a file path
    if isinstance(image, str):
        img = load_img(image, target_size=target_size)
        img_array = img_to_array(img)
    # If image is already loaded by Gradio
    else:
        img = image.resize(target_size)
        img_array = img_to_array(img)
    
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Define the prediction function
def predict_tumor(image):
    if image is None:
        return "No image uploaded!"
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)[0][0]
    
    # Format the result
    confidence = prediction if prediction > 0.5 else 1 - prediction
    confidence_percentage = f"{confidence * 100:.2f}%"
    
    if prediction > 0.5:
        result = f"Brain Tumor Detected (Confidence: {confidence_percentage})"
    else:
        result = f"No Brain Tumor Detected (Confidence: {confidence_percentage})"
        
    return result

# Create Gradio interface
with gr.Blocks(title="Brain Tumor Detection") as demo:
    gr.Markdown("# Brain Tumor Detection")
    gr.Markdown("Upload an MRI scan image to detect the presence of a brain tumor.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload MRI Scan")
            submit_btn = gr.Button("Analyze Image")
        
        with gr.Column():
            output_text = gr.Textbox(label="Result")
            
    submit_btn.click(
        fn=predict_tumor,
        inputs=input_image,
        outputs=output_text
    )
    
    gr.Examples(
        examples=[
            "example_images/tumor_example.jpg",
            "example_images/no_tumor_example.jpg"
        ],
        inputs=input_image
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)  # Set share=False in production