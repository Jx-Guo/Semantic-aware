from maploc.utils.viz_2d import features_to_RGB
import matplotlib.pyplot as plt
from maploc.demo import Demo
from maploc.utils.viz_2d import plot_images
import pandas as pd
import numpy as np
import cv2
import os
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

semantic_colors = {
    "building": (84, 155, 255),
    "parking": (255, 229, 145),
    "playground": (150, 133, 125),
    "grass": (188, 255, 143),
    "park": (0, 158, 16),
    "forest": (0, 92, 9),
    "water": (184, 213, 255),
    "fence": (238, 0, 255),
    "wall": (0, 0, 0),
    "hedge": (107, 68, 48),
    "kerb": (255, 234, 0),
    "building_outline": (0, 0, 255),
    "cycleway": (0, 251, 255),
    "path": (8, 237, 0),
    "road": (255, 0, 0),
    "tree_row": (0, 92, 9),
    "busway": (255, 128, 0),
    "void": [int(255 * 0.9)] * 3,
}

semantic_lab_values = [
    convert_color(sRGBColor(*rgb, is_upscaled=True), LabColor)
    for rgb in semantic_colors.values()
]


demo = Demo(num_rotations=256)

base_filename = "1-1"
input_dir = "scenes"
output_dir = "bev"
os.makedirs(output_dir, exist_ok=True)

image_path = os.path.join(input_dir, f"{base_filename}.jpg")
output_image_path = os.path.join(output_dir, f"{base_filename}_mapped2.png")
output_excel_path = os.path.join(output_dir, f"{base_filename}_info.xlsx")

image, camera, gravity = demo.read_input_image(image_path)
image_rectified, feats_q, mask_bev, conf_q = demo.localize(image, camera, gravity=gravity)
(feats_q_rgb,) = features_to_RGB(feats_q.numpy(), masks=[mask_bev.numpy()])

plot_images(
    [feats_q_rgb],
    dpi=14.7,
    cmaps="jet",
)
plt.savefig(f'bev/{base_filename}.jpg', bbox_inches='tight', pad_inches=0)

feats_q_rgb = (feats_q_rgb[..., :3] * 255).astype(np.uint8)
mask_bev = mask_bev.numpy()
conf_q = conf_q.squeeze(0).cpu().numpy()

valid_mask = (mask_bev != 0) & ~np.all(feats_q_rgb == [255, 255, 255], axis=-1)
valid_pixels = feats_q_rgb[valid_mask]
valid_pixels_lab = [
    convert_color(sRGBColor(*rgb, is_upscaled=True), LabColor)
    for rgb in valid_pixels
]

# Color difference calculation
mapped_colors = []
for pixel_lab in valid_pixels_lab:
    min_delta_e = float('inf')
    best_match_idx = -1
    for idx, semantic_lab in enumerate(semantic_lab_values):
        delta_e = delta_e_cie2000(pixel_lab, semantic_lab)
        if delta_e < min_delta_e:
            min_delta_e = delta_e
            best_match_idx = idx
    mapped_colors.append(list(semantic_colors.values())[best_match_idx])

mapped_image = np.copy(feats_q_rgb)
mapped_image[valid_mask] = mapped_colors
mapped_image[~valid_mask] = [255, 255, 255]

mapped_image_rgba = np.zeros((mapped_image.shape[0], mapped_image.shape[1], 4), dtype=np.uint8)
mapped_image_rgba[..., :3] = mapped_image
mapped_image_rgba[..., 3] = 255
mapped_image_rgba[~valid_mask, 3] = 0

cv2.imwrite(output_image_path, mapped_image_rgba)
print(f"Mapped image saved to: {output_image_path}")

pixel_ids = np.arange(1, conf_q.size + 1).reshape(conf_q.shape)
pixel_colors = mapped_image.reshape(-1, 3)
data = {
    "PixelID": pixel_ids.flatten(),
    "RGB": [f"({r}, {g}, {b})" for r, g, b in pixel_colors],
    "Confidence": conf_q.flatten(),
}
df = pd.DataFrame(data)
df.to_excel(output_excel_path, index=False)
print(f"Excel file saved to: {output_excel_path}")


