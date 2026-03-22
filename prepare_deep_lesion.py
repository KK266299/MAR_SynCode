import os
import numpy as np
from scipy import io as sio
import PIL
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from adn.utils import get_config
from adn.build_gemotry import initialization, build_gemotry
from simulate_data import simulate_metal_artifact
from util_func import get_mar_params

from multiprocessing.dummy import Pool as ThreadPool
#from multiprocessing import Pool as ThreadPool
pool=ThreadPool(4)

if __name__ == '__main__':
    config = get_config('config/dataset_py_640geo.yaml')
    config = config['deep_lesion']
    CTpara = config['CTpara']
    MARpara = get_mar_params('data/deep_lesion/metal_masks')
    splits = ['train']

    # Load meta dataD
    data_dict = sio.loadmat(os.path.join(config['mar_dir'], 'SampleMasks'))
    CT_samples_bwMetal =data_dict['CT_samples_bwMetal']
    metal_masks = CT_samples_bwMetal

    with open(config['data_list'], 'r') as f:
        data_list = f.readlines()

    param = initialization()
    reco_space, ray_trafo, FBPOper = build_gemotry(param)

    # Generate MAR data
    for phase in splits:
        phase_dir = os.path.join(config['dataset_dir'], phase+'_640geo')
        image_indices = eval(CTpara[phase+'_indices'])
        mask_indices = CTpara[phase+'_mask_indices']
        if type(mask_indices) is str:
            mask_indices = list(eval(mask_indices))
        image_size = [CTpara['imPixNum'], CTpara['imPixNum'], len(mask_indices)]
        sinogram_size = [CTpara['sinogram_size_x'], CTpara['sinogram_size_y'], len(mask_indices)]

        # prepare metal masks
        print('Preparing metal masks...\n')
        selected_metal = metal_masks[:, :, mask_indices]
        # mask_resize = np.array(Image.fromarray(selected_metal).resize((CTpara['imPixNum'], CTpara['imPixNum']), PIL.Image.BILINEAR))
        np.save('trainmask.npy',  selected_metal)

        ## handle with image
        for ii in tqdm(range(len(image_indices))):
            image_name = data_list[image_indices[ii]][:-1]
            output_dir = os.path.join(phase_dir, image_name[:-4])

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if os.path.exists(os.path.join(output_dir, 'gt.h5')):
                continue

            print('[{}][{}/{}] Processing {}'.format(phase, ii, len(image_indices), image_name))
            raw_image = plt.imread(os.path.join(config['raw_dir'], image_name))
            image = raw_image*2**16 - 32768
            image = np.array(Image.fromarray(image).resize((CTpara['imPixNum'], CTpara['imPixNum']), PIL.Image.BILINEAR))
            image[image < -1000] = -1000

            simulate_metal_artifact(image, selected_metal, CTpara, MARpara, output_dir, param, ray_trafo, FBPOper, pool)
