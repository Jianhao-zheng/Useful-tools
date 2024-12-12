import subprocess
import os

def create_gif_ffmpeg(image_folder1, image_folder2, output_path, fps=30, width=-1):
    """
    Generate a GIF from frames using FFmpeg with high quality and good compression.
    
    Args:
        image_folder1 (str): Path to the first folder containing frames
        image_folder2 (str): Path to the second folder containing frames
        output_path (str): Output path for the final GIF
        fps (int): Frames per second for the output GIF
        width (int): Output width. -1 means original size. Height will scale proportionally.
    """
    # Create temporary folder for all frames
    temp_folder = "temp_frames"
    os.makedirs(temp_folder, exist_ok=True)
    
    # Create symbolic links or copy files to temp folder
    frame_number = 0
    for folder in [image_folder1, image_folder2]:
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for file in files:
            src = os.path.join(folder, file)
            # Using format to ensure correct ordering
            dst = os.path.join(temp_folder, f"frame_{frame_number:06d}.png")
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst) if hasattr(os, 'symlink') else os.link(src, dst)
            frame_number += 1

    # Method 1: Direct conversion (faster but larger file size)
    def direct_conversion():
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', f'{temp_folder}/frame_%06d.png',
            '-vf', f'fps={fps}{"" if width <= 0 else f",scale={width}:-1"}',
            output_path
        ]
        subprocess.run(cmd)

    # Method 2: Palette method (smaller file size, better quality)
    def palette_conversion():
        # Generate palette
        palette_path = "palette.png"
        cmd1 = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', f'{temp_folder}/frame_%06d.png',
            '-vf', f'fps={fps}{"" if width <= 0 else f",scale={width}:-1"},palettegen=max_colors=256:stats_mode=single',
            palette_path
        ]
        subprocess.run(cmd1)
        
        # Create GIF using palette
        cmd2 = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', f'{temp_folder}/frame_%06d.png',
            '-i', palette_path,
            '-filter_complex', 
            f'fps={fps}{"" if width <= 0 else f",scale={width}:-1"}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
            output_path
        ]
        subprocess.run(cmd2)
        
        # Clean up palette
        if os.path.exists(palette_path):
            os.remove(palette_path)

    # Use palette method by default (better quality)
    try:
        palette_conversion()
        print(f"Created GIF using palette method: {output_path}")
    except Exception as e:
        print(f"Palette method failed: {e}")
        print("Trying direct conversion...")
        try:
            direct_conversion()
            print(f"Created GIF using direct conversion: {output_path}")
        except Exception as e:
            print(f"Direct conversion failed: {e}")
    
    # Clean up temp folder
    for file in os.listdir(temp_folder):
        os.remove(os.path.join(temp_folder, file))
    os.rmdir(temp_folder)
    
    # Print file size
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"GIF size: {size_mb:.2f} MB")

# Example usage
if __name__ == "__main__":
    image_folder1 = "output/sliding_bar_frames_up"
    image_folder2 = "output/sliding_bar_frames_down"
    output_gif = "output/sliding_bar.gif"
    
    # Create GIF with default settings
    create_gif_ffmpeg(image_folder1, image_folder2, output_gif)
    
    # Or with custom settings
    # create_gif_ffmpeg(image_folder1, image_folder2, output_gif, fps=24, width=800)