import re
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import os

SAVE_PATH = './image_collection/'
LOAD_PATH = './DATASET_sample/'

# Function to search all subdirectories


def dir_image_processing(dirname):
    filenames = os.listdir(dirname)
    for filename in tqdm(filenames):
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        # DNM data has two type, each type calls a differnet function
        if ext == '.css':
            image_decoding(full_filename)
        elif ext == '.jpg':
            image_cut(full_filename)

# Funtion to cut image and save


def image_cut(fn):
    filename = fn
    img = Image.open(filename)
    filename = os.path.split(filename)[-1]
    filename = filename.replace('.jpg', '')
    list_filename = filename.split('-')
    width, height = img.size[0], img.size[1]
    for i in range(len(list_filename)):
        try:
            croppedimg = img.crop(
                ((width/len(list_filename)) * i, 0, (width/len(list_filename))*(i+1), height))
            # all white image is passed
            if np.average(np.asarray(croppedimg)) >= 254:
                return 0
            croppedimg.save(SAVE_PATH + str(list_filename[i])+'.jpg', 'JPEG')
        except:
            continue

# Function to decoding image


def image_decoding(filename):
    with open(filename, "rb") as f:
        data = f.read().decode('ISO-8859-1')
    # Only text #image_XX {~~~} format is selected and put label image list
    p = re.compile('\#[^#]*\}')
    label_image_list = p.findall(data)
    for label_image in label_image_list:
        label = label_image[label_image.find('_')+1: label_image.find('{')-1]
        label = label.replace(':', '_')
        # Beacase (~~~) is real image part
        p = re.compile('\([^)]*\)')
        image = p.findall(label_image)[0]
        image = image[25:-2]
        img = Image.open(BytesIO(base64.b64decode(image)))
        # all white image is passed
        if np.average(np.asarray(img)) >= 254:
            return 0
        img.save(SAVE_PATH+str(label)+'.jpg', 'JPEG')


if __name__ == '__main__':
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    filename = LOAD_PATH+'2014-11-01/images'
    dir_image_processing(filename)

    filename = LOAD_PATH+'2014-11-05/images'
    dir_image_processing(filename)

    filename = LOAD_PATH+'2014-11-06/images'
    dir_image_processing(filename)
