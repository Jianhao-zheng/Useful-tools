# Diagonal Sliding Line Transition with Text Overlay

This script generates a video with a diagonal sliding line transition effect between two images. It allows adding customizable text overlays to each image, using different font styles, sizes, and colors.

## Usage

### 1. Prepare Input Images
Place the source images, `ours.png` and `splat-slam.png` (or custom name, but the name in the python script should be consistent with the file name) in the `src_imgs` directory.

### 2. Run Scipts
Set the parameters in the scripts and run the following code to generate videos or frames

```bash
python gen_video.py # Generate a video where the sliding bar moves from bottom right to top left

python gen_video_frames_up.py # Same as {gen_video.py}, but output each frame instead of a full mp4
python gen_video_frames_down.py # Sliding bar goes down
