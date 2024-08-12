import time
import os
from torchvision import transforms
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
# from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
import torchvision
from PIL import Image
import matplotlib.pyplot as plt


def image_display(visuals):
    for label, image_numpy in visuals.items():
        img = Image.fromarray(image_numpy, 'RGB')
        plt.figure()
        plt.imshow(img)
        plt.show()
        # plt.show(block=False)


opt = TestOptions().parse()
opt.nThreads = 0   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)

avg_time = 0
count = 0
print(len(dataset))
star_t = time.time()
for i, data in enumerate(dataset):
    model.set_input(data)
    star_t = time.time()
    visuals = model.predict()
    avg_time += time.time() - star_t
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    print('avg_speed= {}'.format(avg_time))
    # visualizer.save_images(webpage, visuals, img_path)
    #image_display(visuals)
    # webpage.save()


