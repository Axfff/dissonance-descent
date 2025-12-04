import os
import glob
from PIL import Image
import re

def create_gif(experiment_dir, output_filename="optimization_process.gif", duration=200):
    # Find all spectrum.png files in checkpoint folders
    search_pattern = os.path.join(experiment_dir, "checkpoint_*", "spectrum.png")
    image_paths = glob.glob(search_pattern)
    
    if not image_paths:
        print(f"No spectrum.png files found in {experiment_dir}")
        return

    # Sort files by checkpoint number
    def extract_checkpoint_num(path):
        match = re.search(r'checkpoint_(\d+)', path)
        return int(match.group(1)) if match else 0
    
    image_paths.sort(key=extract_checkpoint_num)
    
    print(f"Found {len(image_paths)} images.")
    
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    if images:
        output_path = os.path.join(experiment_dir, output_filename)
        # Save as GIF
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved to {output_path}")
    else:
        print("No valid images to create GIF.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        experiment_dirs = sys.argv[1:]
        for experiment_dir in experiment_dirs:
            print(f"Processing {experiment_dir}...")
            create_gif(experiment_dir)
    else:
        print("Please provide experiment directories as arguments.")
