import os
import numpy as np
from skimage import io
from tqdm import tqdm

def subtract_background(images_dir,output_dir, n_frames=100):
    """
    The following function removes background from smlm experimental data by subtracting minimum map of n consecutive frames
    to reduce background noise.
    """
    os.makedirs(output_dir, exist_ok=True) 
    im_files = sorted([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f)) and not f.startswith('.')])  # make sure the names are sortable
    n_ims = len(im_files)
    pointer = 0
    for i in tqdm(range(n_ims//n_frames), desc="Processing frames"):
        im_names = [im_files[pointer+j] for j in range(n_frames)]
        im_stack = [io.imread(os.path.join(images_dir, im_files[pointer+j])) for j in range(n_frames)]
        pointer += n_frames
        im_stack = np.array(im_stack)
        im_stack = im_stack-np.min(im_stack, axis=0)

        for j in range(n_frames):  # save
            io.imsave(os.path.join(output_dir, im_names[j]), im_stack[j], check_contrast=False)

    # remainder of n_ims/num
    im_stack = [io.imread(os.path.join(images_dir, im_files[-j])) for j in range(n_frames)]
    im_stack = np.array(im_stack)
    im_min = np.min(im_stack, axis=0)
    for j in tqdm(range(pointer, n_ims), desc="Processing remainder frames"):
        im = io.imread(os.path.join(images_dir, im_files[j]))
        im = im-im_min
        io.imsave(os.path.join(output_dir, im_files[j]), im, check_contrast=False)
    


if __name__ == "__main__":
    images_dir = "/zjbd/zd1/shechtmanlab/omer/DoDeAbr/experiments/FOV6/mask_off_005_exp50__illuminationx1_010/" # specify your images directory here
    output_dir = "/zjbd/zd1/shechtmanlab/omer/DoDeAbr/experiments/FOV6/mask_off_005_exp50__illuminationx1_010_bg_subtracted/"
    subtract_background(images_dir, output_dir, n_frames=100)