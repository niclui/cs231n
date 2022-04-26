# Decode segmentation masks and save them as npy files in a mask folder
# Run this once only. It should take <6mins
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
tqdm.pandas()
from PIL import Image
import sys

# Ref: https://www.kaggle.com/code/fabiendaniel/image-with-masks-quick-overview
def rle_decode(rle, height, width , fill=255):
    mask = np.zeros(height*width, dtype=np.uint8) # Generate an array of 0s corresponding to the picture size
    if rle == rle: # Only do decoding if my segmentation is not NaN        
        s = rle.split() # Split the pixels up
        start, length = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        start -= 1        
        for i, l in zip(start, length):
            mask[i:i+l] = fill
        mask = mask.reshape(width,height).T
        mask = np.ascontiguousarray(mask)
    return mask

if __name__ == '__main__':
    # usage: python decode.py [masks_folder] [combined_csv_path] [output_csv_path]
    masks_folder_path = sys.argv[1]
    combined_csv_path = sys.argv[2]
    output_csv_path = sys.argv[3]
    combined_df = pd.read_csv(combined_csv_path)

    mask_paths = []
    id_list = list(combined_df["id"])
    class_list = list(combined_df["class"])
    seg_list = list(combined_df["segmentation"])
    height_list = list(combined_df["slice_height"])
    width_list = list(combined_df["slice_width"])

    for i in tqdm(range(len(combined_df))):
        decoded_mask = rle_decode(seg_list[i], height_list[i], width_list[i])
        mask_path = masks_folder_path + "/" + id_list[i] + "_" + class_list[i] + '.npy'
        np.save(mask_path, decoded_mask)
        mask_paths.append(mask_path)

    combined_df["mask_path"] = mask_paths
    combined_df.to_csv(output_csv_path + '.csv')