import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def add_text_to_image(
    img,
    text,
    position,
    color=(255, 255, 255),
    font_size=32,
    font_path="Roboto-Regular.ttf",
):
    """
    Helper function to add text using PIL with specified Roboto font style
    """
    # Convert OpenCV image (BGR) to PIL image (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Load specified Roboto font style
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Font not found: {font_path}")
        return img

    # For multiline text
    lines = text.split("\n")
    line_height = font_size + 10

    # Convert color from BGR to RGB
    color_rgb = (color[2], color[1], color[0])

    # Add text with outline
    for i, line in enumerate(lines):
        x, y = position
        text_y = y + i * line_height

        # Draw black outline
        outline_color = (240, 163, 10)
        outline_width = 0.4
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            draw.text(
                (x + dx * outline_width, text_y + dy * outline_width),
                line,
                font=font,
                fill=outline_color,
            )

        # Draw main text
        draw.text((x, text_y), line, font=font, fill=color_rgb)

    # Convert back to OpenCV format (RGB to BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def diagonal_sliding_line(
    img1,
    img2,
    text1,
    text2,
    output_folder,
    line_thickness=5,
    step=5,
    text1_color=(255, 255, 255),
    text2_color=(255, 255, 255),
    text1_size=32,
    text2_size=32,
    text1_font="Roboto-Bold.ttf",  # Can be different Roboto style
    text2_font="Roboto-Medium.ttf",
    text1_bottom_right_margin=(80, -100),
    text2_top_left_margin=(10, 10),
):  # Can be different Roboto style
    """
    Create a diagonal sliding line effect transitioning from img1 to img2,
    with text in different Roboto font styles. Save frames as separate images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    h, w = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h))

    # Add text to images with different Roboto styles
    bottom_margin = text1_bottom_right_margin[0]
    right_margin = text1_bottom_right_margin[1]
    img1_with_text = add_text_to_image(
        img1,
        text1,
        (w - 250 - right_margin, h - bottom_margin),
        color=text1_color,
        font_size=text1_size,
        font_path=text1_font,
    )

    top_margin = text2_top_left_margin[0]
    left_margin = text2_top_left_margin[1]
    img2_with_text = add_text_to_image(
        img2,
        text2,
        (left_margin, top_margin),
        color=text2_color,
        font_size=text2_size,
        font_path=text2_font,
    )

    combined = img1_with_text.copy()
    frame_idx = 0  # Initialize frame index for naming files

    for pos in range(w + h, -h, -step):
        combined = img1_with_text.copy()
        mask = np.zeros((h, w), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                if x + y < pos:
                    mask[y, x] = 255

        combined[mask > 0] = img2_with_text[mask > 0]

        for y in range(h):
            x = pos - y
            if 0 <= x < w and 0 <= y < h:
                for thickness in range(-line_thickness // 2, line_thickness // 2):
                    x_thick = x + thickness
                    if 0 <= x_thick < w:
                        combined[y, x_thick] = [255, 255, 255]

        # Save the current frame as an image
        frame_path = os.path.join(output_folder, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(frame_path, combined)
        frame_idx += 1
        if frame_idx >= 202:
            break

    # Save the final frame as well
    # final_frame_path = os.path.join(output_folder, f"frame_{frame_idx:04d}.png")
    # cv2.imwrite(final_frame_path, img2_with_text)

    cv2.destroyAllWindows()


# Example usage with different Roboto styles
if __name__ == "__main__":
    img1 = cv2.imread("src_imgs/ours.png")
    img2 = cv2.imread("src_imgs/splat-slam.png")
    text1 = "WildGS-SLAM\n      (Ours)"
    text2 = "Splat-SLAM"
    output_folder = "output/sliding_bar_frames_up"
    text1_color = (10, 163, 240)
    text2_color = (10, 163, 240)
    text1_size = 25
    text2_size = 25
    text1_bottom_right_margin = (80, -100)
    text2_top_left_margin = (10, 10)

    diagonal_sliding_line(
        img1,
        img2,
        text1=text1,
        text2=text2,
        output_folder=output_folder,
        text1_color=text1_color,
        text2_color=text2_color,
        text1_size=text1_size,
        text2_size=text2_size,
        text1_font="RobotoCondensed-Light.ttf",
        text2_font="RobotoCondensed-Light.ttf",
        text1_bottom_right_margin=text1_bottom_right_margin,
        text2_top_left_margin=text2_top_left_margin,
    )
    print(f"Frames saved in: {output_folder}")
