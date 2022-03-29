import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import os
from SR_models import SRCNN
from SR_utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr
from tqdm import tqdm

SAVE_PATH = './image_collection_sr/'


def super_resolution(image_file_root):

    weights_file = 'srcnn_x2.pth'
    image_file = image_file_root
    # scale = 4

    # input의 크기와 사용중인 hardware에 따라 cudnn에서 사용하는 알고리즘을 바꿔준다.
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image = pil_image.open(image_file).convert('RGB')
    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    # psnr = calc_psnr(y, preds)
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]
                      ).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    name = os.path.splitext(image_file)[-2].split('/')[-1]
    output.save(SAVE_PATH+name+'.jpg')

# Function to search all subdirectories


def dir_image_processing(dirname):
    filenames = os.listdir(dirname)
    for filename in tqdm(filenames):
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext.lower() == '.jpg' or ext.lower() == '.bmp':
            super_resolution(full_filename)


if __name__ == '__main__':
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    dir_image_processing('./image_collection/')
