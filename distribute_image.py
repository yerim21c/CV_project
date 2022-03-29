import re
import os
from tqdm import tqdm
from PIL import Image

SAVE_PATH = './user_collection/'
LOAD_PATH = './image_collection/'
USER_PATH = './DATASET_sample/'
user_image_dict = {}

# Distributed only when there are at least 15 images for train and 15 images for test.
THRESHOLD = 2*15

# Function to search all subdirectories


def dir_user_processing(dirname):
    filenames = os.listdir(dirname)
    for filename in tqdm(filenames):
        full_filename = os.path.join(dirname, filename)
        # because we need item page in directory others are feedback page
        if os.path.isdir(full_filename):
            username = os.path.split(filename)[-1]
            item_list = os.listdir(full_filename)
            for item in item_list:
                full_itemname = os.path.join(full_filename, item)
                user_image_collecting(username, full_itemname)

# Function to collect user's image number


def user_image_collecting(username, dirname):
    global user_image_dict
    with open(dirname, 'r', encoding='UTF-8') as f:
        data = f.read()
        p = re.compile('<img src[^<]*.jpg')
        sequence_image_list = p.findall(data)
        for sequence_image in sequence_image_list:
            sequence_image = sequence_image[18:sequence_image.find('.')]
            sequence_image = sequence_image.split('-')
            for image_number in sequence_image:
                if username in user_image_dict:
                    user_image_dict[username].add(image_number)
                else:
                    user_image_dict[username] = set([image_number])

# Function to distribute image to train set and test set
# If user's image count is bigger than 2*Tr


def image_distribution(dict):
    # Make train user folder
    for username in user_image_dict:
        os.makedirs(SAVE_PATH+'train/'+username)

    # Make test folder
    for username in user_image_dict:
        os.makedirs(SAVE_PATH+'test/'+username)

    train_cnt = 0
    test_cnt = 0

    for username in tqdm(user_image_dict):
        real_cnt = 0
        half_cnt = 0

        for image in user_image_dict[username]:
            existing_imagenames = os.listdir(LOAD_PATH)
            for existing_image in existing_imagenames:
                if image == existing_image.replace('.jpg', '').split('_')[0]:
                    real_cnt += 1

        # test case
        if real_cnt >= THRESHOLD:
            for image in user_image_dict[username]:
                existing_imagenames = os.listdir(LOAD_PATH)
                for existing_image in existing_imagenames:
                    if image == existing_image.replace('.jpg', '').split('_')[0]:
                        # half of image is used for test
                        if half_cnt % 2 == 0:
                            copy_image = Image.open(LOAD_PATH+existing_image)
                            copy_image.save(SAVE_PATH+'test/' +
                                            username+'/'+existing_image)
                            test_cnt += 1
                            half_cnt += 1
                        # half of image is used for train
                        else:
                            copy_image = Image.open(LOAD_PATH+existing_image)
                            copy_image.save(
                                SAVE_PATH+'train/'+username+'/'+existing_image)
                            train_cnt += 1
                            half_cnt += 1
        # train case
        else:
            for image in user_image_dict[username]:
                existing_imagenames = os.listdir(LOAD_PATH)
                for existing_image in existing_imagenames:
                    if image == existing_image.replace('.jpg', '').split('_')[0]:
                        copy_image = Image.open(LOAD_PATH+existing_image)
                        copy_image.save(SAVE_PATH+'train/' +
                                        username+'/'+existing_image)
                        train_cnt += 1

    summary = open(SAVE_PATH+'summary.txt', 'w')
    summary.write(f'usernumber : {len(user_image_dict)}\n')
    summary.write(f'image_cnt : {train_cnt+test_cnt}\n')
    summary.write(f'train_image_cnt : {train_cnt}\n')
    summary.write(f'test_image_cnt : {test_cnt}\n')
    summary.close()


if __name__ == '__main__':
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    filename = USER_PATH+'2014-11-01/users'
    dir_user_processing(filename)

    filename = USER_PATH+'2014-11-05/users'
    dir_user_processing(filename)

    filename = USER_PATH+'2014-11-06/users'
    dir_user_processing(filename)

    with open("./user_image_dict.txt", "w") as f:
        f.write(str(user_image_dict))

    image_distribution(user_image_dict)
