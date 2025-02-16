import glob
import shutil
import os

images = glob.glob("gpt_caching/*/best_assembly/isometric_image.png")

os.makedirs("images", exist_ok=True)

# copy all images to the new directory
for image in images:
    folder_name = image.split(os.sep)[1]

    new_filename = f"images/{folder_name}.png"

    shutil.copy(image, new_filename)
print(len(images), "copied")
