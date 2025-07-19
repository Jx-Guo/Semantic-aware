from maploc.utils.viz_2d import features_to_RGB
import matplotlib.pyplot as plt
from maploc.demo import Demo
from maploc.utils.viz_2d import plot_images

demo = Demo(num_rotations=256, device='cpu')
image_path = "scenes/1-1.jpg"

image, camera, gravity = demo.read_input_image(image_path,)

image_rectified, feats_q, mask_bev, conf_q = demo.localize(
    image, camera, gravity=gravity
)

conf_q = conf_q.squeeze(0)

(feats_q_rgb,) = features_to_RGB(feats_q.numpy(), masks=[mask_bev.numpy()])

plot_images(
    [image, feats_q_rgb, conf_q],
    titles=["input image", "BEV features", "BEV confidence"],
    dpi=80,
    cmaps="jet",
)

# plt.savefig('bev/1-1neturalBEV.jpg', bbox_inches='tight', pad_inches=0)
plt.show()
