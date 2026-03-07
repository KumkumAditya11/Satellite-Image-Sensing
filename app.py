import gradio as gr
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch

# 1. Point to your local folder where the model is stored
model_path = "Kumkum-5/Satellite-Model-Weights"

print("Loading your fine-tuned model... please wait.")

# 2. Load the model and tools
model = VisionEncoderDecoderModel.from_pretrained(model_path).to("cpu")
feature_extractor = ViTImageProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

def predict_caption(input_image):
    # Ensure image is in RGB format
    img = input_image.convert("RGB")
    
    # Preprocess the image and convert to PyTorch tensors
    pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values
    
    # Generate the caption using beam search and sampling
    output_ids = model.generate(
        pixel_values,
        max_length=50,
        num_beams=5,
        repetition_penalty=3.5,
        temperature=1.2,
        do_sample=True,
        early_stopping=True
    )
    
    # Decode the generated IDs back into a readable string
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return caption

# 4. Create the User Interface (UI)
app = gr.Interface(
    fn=predict_caption, 
    inputs=gr.Image(type="pil", label="Upload Satellite Image"), 
    outputs=gr.Textbox(label="AI Generated Caption"),
    title="Satellite Intelligence System",
    description="Fine-tuned ViT-GPT2 model for Remote Sensing Imagery. Developed for ISRO Outreach prep."
)

# 5. Launch the app!
if __name__ == "__main__":
    app.launch()