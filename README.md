# Satellite-Image-Sensing
An AI-powered system designed to generate natural language descriptions for satellite and remote sensing imagery. This project was developed on an independent basis to explore automated terrain analysis.

Technical Overview
This project utilizes a Vision-Encoder-Decoder architecture to bridge the gap between computer vision and natural language processing.
Encoder: google/vit-base-patch16-224-in21k (Vision Transformer)
Decoder: gpt2 (Generative Pre-trained Transformer)
Framework: PyTorch, Hugging Face Transformers, and Gradio.

 Key Engineering Solutions
During development, the model initially showed a high bias toward "Airport" and "Green Tree" classifications. I implemented the following optimizations to achieve ~80% descriptive accuracy:

Repetition Penalty (3.5): Severely penalizes the model for repeating common tokens, forcing it to look for unique terrain features.

Beam Search (num_beams=5): Evaluates multiple word sequences simultaneously to find the most coherent description.

Temperature Sampling (0.8): Balanced the randomness to allow for creative descriptions of diverse landscapes (Deserts, Forests, Urban areas).

📂 Repository Structure
app.py: The main inference script with custom generation parameters.

requirements.txt: Environment dependencies.

Technical_Notes.txt: Detailed breakdown of model logic and training observations.

📈 Future Scope
Integrating multi-spectral band analysis for better agricultural monitoring.

Scaling the dataset to include more specific Indian topographical features.


Link to Hugging Face where the project i sactually running : https://huggingface.co/spaces/Kumkum-5/Satellite-Image-

Sensing/blob/main/app.py
