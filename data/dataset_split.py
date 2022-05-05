import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def dataset_split(dataset_path, output_folder, train_prop=70, val_prop=20, test_prop=10):
    # Default split is 70/20/10
    # Check if the splits add up to 1
    total = train_prop + val_prop + test_prop
    if abs(total - 1) > 0.0000001:
        print("Train, validation, and test proportions must add up to 1. Instead, they are", total)

    # Create output folder if its not already created
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    data = pd.read_csv(dataset_path, index_col=0)
    
    # Split into train and val+test datasets
    train, val_test = train_test_split(data, test_size=val_prop+test_prop, random_state=0)

    # Split the val+test datasets into validation and test
    val, test = train_test_split(val_test, test_size=test_prop/(val_prop+test_prop), random_state=0)
        
    # Output train, val, test datasets
    train.to_csv(output_folder + "train_dataset.csv", index=False)
    val.to_csv(output_folder + "val_dataset.csv", index=False)
    test.to_csv(output_folder + "test_dataset.csv", index=False)

if __name__ == '__main__':
    # usage: python data/data_split.py [dataset path] [output folder] [train percent] [val percent] [test percent]
    dataset_path = sys.argv[1]
    output_folder = sys.argv[2]
    train_prop = int(sys.argv[3]) / 100
    val_prop = int(sys.argv[4]) / 100
    test_prop = int(sys.argv[5]) / 100

    dataset_split(dataset_path, output_folder, train_prop, val_prop, test_prop)