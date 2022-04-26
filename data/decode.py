# Decode segmentation masks and save them as npy files in a mask folder
# Run this once only. It should take <5mins
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
tqdm.pandas()
from PIL import Image
import sys
import os

# Reference: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    if mask_rle==mask_rle:
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
    else: # If segmentation is NaN, just return an array of 0s
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    return img.reshape(shape)  # Needed to align to RLE direction

if __name__ == '__main__':
    # usage: python decode.py [masks_folder_path] [combined_csv_path] [final_csv_path]
    masks_folder_path = sys.argv[1]
    combined_csv_path = sys.argv[2]
    final_csv_path = sys.argv[3]

    # Make masks folder path if it doesn't already exist
    if not os.path.exists(masks_folder_path):
        os.mkdir(masks_folder_path)

    # Read in the combined df
    combined_df = pd.read_csv(combined_csv_path)

    mask_paths = []
    id_list = list(combined_df["id"])
    class_list = list(combined_df["class"])
    seg_list = list(combined_df["segmentation"])
    height_list = list(combined_df["slice_height"])
    width_list = list(combined_df["slice_width"])

    # Decode!
    for i in tqdm(range(len(combined_df))):
        decoded_mask = rle_decode(seg_list[i], [height_list[i], width_list[i]])
        mask_path = masks_folder_path + "/" + id_list[i] + "_" + class_list[i] + '.npy'
        np.save(mask_path, decoded_mask)
        mask_paths.append(mask_path)

    combined_df["mask_path"] = mask_paths
    combined_df.to_csv(final_csv_path + '.csv')