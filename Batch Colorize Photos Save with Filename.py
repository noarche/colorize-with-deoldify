import os
import torch
import fastai
from deoldify import device
from deoldify.device import get_default_device
from deoldify.visualize import get_image_colorizer
from PIL import Image


device = get_default_device()

def get_model_path():
    default_model_path = 'ddcolor_modelscope.pth'
    if os.path.exists(default_model_path):
        print(f"Using default model path: {default_model_path}")
        return default_model_path
    else:
        return input("Enter the path to the DeOldify model: ")

def colorize_image(colorizer, input_path, output_path):
    
    colorizer.plot_transformed_image(
        path=input_path, 
        results_dir=os.path.dirname(output_path), 
        render_factor=35, 
        watermarked=False
    )
    
    original_filename = os.path.basename(input_path)
    os.rename(
        os.path.join(os.path.dirname(output_path), "colorized.png"),
        output_path
    )

def process_images(input_dir, output_dir, model_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    colorizer = get_image_colorizer(artistic=True, model_path=model_path)

    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.svg', '.tif', '.webp']
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file)
                
                try:
                    colorize_image(colorizer, input_path, output_path)
                    print(f"Processed and saved: {output_path}")
                except Exception as e:
                    print(f"Failed to process image {input_path}: {e}")

input_dir = input("Enter the path to the input directory: ")
output_dir = input("Enter the path to the output directory: ")
model_path = get_model_path()

process_images(input_dir, output_dir, model_path)
