import gradio as gr
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch

# 1. Load model and tools
model_path = "Kumkum-5/Satellite-Model-Weights"
model = VisionEncoderDecoderModel.from_pretrained(model_path).to("cpu")
feature_extractor = ViTImageProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

def predict_caption(input_image):
    if input_image.mode != "RGB":
        input_image = input_image.convert(mode="RGB")
        
    pixel_values = feature_extractor(images=input_image, return_tensors="pt").pixel_values
    
    # 2. Generate detailed technical description
    output_ids = model.generate(
        pixel_values.to("cpu"),
        max_length=100,         # Increased for formal detail
        min_length=35,         # Forces a full technical report
        num_beams=5,
        no_repeat_ngram_size=3,
        repetition_penalty=6.5, # High penalty to avoid simple loops
        temperature=0.8,
        do_sample=True,
        early_stopping=True
    )

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    caption = preds[0].strip().lower()

    # 3. FORMAL LANGUAGE TRANSLATOR & BIAS KILLER
    # This replaces casual words with professional geospatial terminology
    formal_map = {
        "airport": "specialized infrastructure zone",
        "runway": "linear transit corridor",
        "planes": "stationary logistical units",
        "plane": "unit",
        "flying": "aerial surveying",
        "trees": "dense vegetation cover",
        "grass": "low-lying flora",
        "water": "hydrographic feature",
        "beach": "coastal sediment zone",
        "houses": "residential structures",
        "sand": "arid geological substrate"
    }

    # Apply the mapping
    for casual, formal in formal_map.items():
        if casual in caption:
            caption = caption.replace(casual, formal)

    # Final cleanup for the professional look
    return f"Geospatial Analysis: {caption.capitalize()}."

# 4. Create the User Interface
app = gr.Interface(
    fn=predict_caption, 
    inputs=gr.Image(type="pil", label="Input Satellite Imagery"), 
    outputs=gr.Textbox(label="Technical Terrain Analysis"),
    title="Satellite Intelligence System (GEOINT)",
    description="Advanced ViT-GPT2 architecture optimized for formal remote sensing terrain classification."
)

if __name__ == "__main__":
    app.launch()
