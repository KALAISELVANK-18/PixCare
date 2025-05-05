import cv2
from PIL import Image
import os

def resize_image(image_path, new_size):
    # Open the image using Pillow
    image = Image.open(image_path)
    size=image.size


    # #Resize the image
    resized_image = image.resize(new_size)
    resized_image=resized_image.convert('RGB')

    # Get the directory and filename of the original image
    directory, filename = os.path.split(image_path)


    os.remove(image_path)

    #print(output_folder+fr"\{filename}")
    resized_image.save(image_path)

    # Close the image
    resized_image.close()

# Set the path to the folder containing the images
folder_path = r"C:\Users\kalai\Pictures\characterize\test\random_data"

# Set the new size for the resized images
new_size = (224, 224)

# Traverse through the folder and its subfolders
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith(('.jpg')):
            # Get the full path of the image file
            image_path = os.path.join(root, file)

            # Resize the image
            resize_image(image_path, new_size)