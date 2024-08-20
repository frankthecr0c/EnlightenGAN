import time
import os
import sys
from torchvision import transforms
import numpy as np
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from pathlib import Path
# from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
import torchvision
from PIL import Image as PImage
import matplotlib.pyplot as plt


def image_display(visuals):
    for label, image_numpy in visuals.items():
        img = PImage.fromarray(image_numpy, 'RGB')
        plt.figure()
        plt.imshow(img)
        plt.show()
        # plt.show(block=False)

def display_image_pil(image_tensor):
    """Displays an image tensor using PIL (Pillow).

    Args:
        image_tensor: The image tensor to be displayed.
    """

    image_np = image_tensor.numpy()

    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)

    image_np = (image_np * 255).astype(np.uint8)

    image = PImage.fromarray(image_np)
    image.show()


def save_image(visuals, file_path):
    """
    Save the given image to the specified file path.

    Args:
        image (PIL.Image.Image): The image to be saved.
        file_path (str): The path to save the image to.
    """

    try:
        image_numpy = visuals["fake_B"]
        img = PImage.fromarray(image_numpy, 'RGB')
        img.save(file_path)
        print("Image saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving the image: {e}")


opt = TestOptions().parse()
opt.nThreads = 0   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
# visualizer = Visualizer(opt)
# create website
# web_dir = os.path.join("./ablation/", opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
# webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
print(len(dataset))
out_directory = Path("out")
root_directory = Path(".", "test_dataset")
star_t = time.time()
for i, data in enumerate(dataset):
    model.set_input(data)
    star_t = time.time()
    visuals = model.predict()
    avg_time = time.time() - star_t
    img_path = model.get_image_paths()[0]
    filename = img_path.split("/", -1)[-1]
    print('process image... %s' % img_path)
    print('avg_speed= {}'.format(avg_time))
    final_path_save = Path(root_directory, out_directory, filename)
    save_image(visuals, str(final_path_save))
    # visualizer.save_images(webpage, visuals, img_path)
    # image_display(visuals)
    # webpage.save()

