import os
import random
from PIL import Image

#path = kagglehub.dataset_download("arnaud58/flickrfaceshq-dataset-ffhq", local)

input_folder = "./images/"
output_folder = "./compressed/"

quality_levels = [10, 30, 50]

for file in os.listdir(input_folder):

    if file.lower().endswith((".jpg", ".jpeg", ".png")):

        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        with Image.open(input_path) as img:

            if img.mode in ("RGBA", "P"):

                img = img.convert("RGB")
                
            quality = random.choice(quality_levels)

            img.save(output_path, "JPEG", quality = quality)


# Primjer za samo jednu sliku uz alternativu smanjenja veličine
#with Image.open(test_input) as img:
#    quality_level = 1
#    img.save(test_output, "JPEG", quality = quality_level)
#    #size = (img.width // 2, img.height // 2)
#    #resized = img.resize(size, Image.LANCZOS)
#    #resized.save(test_resize)
#    # doslovno smanjuje veličinu ali nema pretjerani tujecaj na samu kvalitetu



