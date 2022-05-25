import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pdb
import numpy as np

def odds_pairing(odds, idx):
    odds['dummy'] = 1
    odds['pair'] = odds.sort_values(['case','slice_id'],ascending=False).groupby(['case'])['slice_id'].shift()
    odds = odds[odds['slice_id'] % 2 == 1]
    odds['pair_idx'] = odds['dummy'].cumsum()
    odds['pair_idx'] = odds['pair_idx'].apply(lambda x: idx + str(x))
    odds['pair'] = odds['pair'].astype('int')
    odds = pd.concat([odds[['case', 'slice_id', 'day', 'pair_idx']], odds[['case', 'pair', 'day', 'pair_idx']].rename({'pair': 'slice_id'}, axis = 'columns')])
    return odds

def evens_pairing(evens, idx):
    evens['dummy'] = 1

    pdb.set_trace()
    group1 = evens[evens['day_idx'] % 2 == 1][['case', 'slice_id', 'day']]
    group2 = evens[evens['day_idx'] % 2 == 0][['case', 'slice_id', 'day']]

    evens = pd.merge(group1, group2, on = ['case', 'slice_id'])

    evens['pair_idx'] = evens['dummy'].cumsum()
    evens['pair_idx'] = evens['pair_idx'].apply(lambda x: idx + str(x))
    evens['pair'] = evens['pair'].astype('int')
    evens = pd.concat([evens[['case', 'slice_id', 'day', 'pair_idx']], evens[['case', 'pair', 'day', 'pair_idx']].rename({'pair': 'slice_id'}, axis = 'columns')])
    return evens

def dataset_split(dataset_path, output_folder, train_prop=0.7, val_prop=0.2, test_prop=0.1):
    # Default split is 70/20/10
    # Check if the splits add up to 1
    total = train_prop + val_prop + test_prop
    if abs(total - 1) > 0.0000001:
        print("Train, validation, and test proportions must add up to 1. Instead, they are", round(total, 3))

    # Create output folder if its not already created
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    data = pd.read_csv(dataset_path, index_col=0)

    #Find pairs
    data['dummy'] = 1

    #Denote special cases where a day in the case does not have the same number of scans
    data['day_slices'] = data.groupby(['case', 'day'])['dummy'].transform(lambda x: x.sum())
    data['days'] = data.groupby(['case'])['day'].transform(lambda x: x.nunique())
    data['case_slices'] = data.groupby(['case'])['dummy'].transform(lambda x: x.sum())
    data['special'] = (data['case_slices']/data['days'] != data['day_slices']) &\
     ((data['case_slices']/data['days'] > 112) & (data['day_slices'] == 80) | (data['case_slices']/data['days'] < 112) & (data['day_slices'] == 144) )

    special_ids = odds_pairing(data[data['special'] == True], 'S')

    data_normal = data[data['special'] == False]
    data_normal['idx'] = data_normal.groupby(['case', 'slice_id'])['dummy'].cumsum()

    #Indicate odds
    to_odd = data_normal[['case', 'day']].drop_duplicates().groupby(['case']).count().reset_index()
    to_odd['odd_idx'] = to_odd.apply(lambda x: np.random.randint(x.day) + 1, axis = 1)
    data_normal = pd.merge(data_normal, to_odd[['case', 'odd_idx']], how = 'left', on = 'case')

    #Odd cases
    odds = data_normal[data_normal['odd_idx'] == data_normal['idx']]
    odd_ids = odds_pairing(odds, 'O')

    #Evens
    evens = data_normal[data_normal['odd_idx'] != data_normal['idx']]
    evens['day_idx'] = evens.groupby(['case', 'slice_id'])['dummy'].cumsum()
    evens['groups_int'] = evens['day_idx'].apply(lambda x: int(x<=2))

    pdb.set_trace()
    
    evens_a = evens[evens['groups_int'] == 0][['case', 'slice_id', 'day', 'day_idx']]
    evens_a_ids = evens_pairing(evens_a, 'EA') 

    evens_b = evens[evens['groups_int'] != 0][['case', 'slice_id', 'day', 'day_idx']]

    #Back to data
    
    # Split into train and val+test datasets
    train, val_test = train_test_split(data, test_size=val_prop+test_prop, random_state=0)

    # Split the val+test datasets into validation and test
    val, test = train_test_split(val_test, test_size=test_prop/(val_prop+test_prop), random_state=0)
        
    # Output train, val, test datasets
    train.to_csv(os.path.join(output_folder, "train_dataset.csv"), index=False)
    val.to_csv(os.path.join(output_folder, "val_dataset.csv"), index=False)
    test.to_csv(os.path.join(output_folder, "test_dataset.csv"), index=False)

if __name__ == '__main__':
    # usage: python data/dataset_split.py [final.csv] [output folder] [train percent] [val percent] [test percent]
    dataset_path = sys.argv[1]
    output_folder = sys.argv[2]
    try:
        train_prop = int(sys.argv[3]) / 100
        val_prop = int(sys.argv[4]) / 100
        test_prop = int(sys.argv[5]) / 100
        #print(f"Using train-val-test split of {sys.argv[3]}%-{sys.argv[4]}%-{sys.argv[5]}%")
        dataset_split(dataset_path, output_folder, train_prop, val_prop, test_prop)
    except:
        #print("Using default train-val-test split of 70%-20%-10%")
        dataset_split(dataset_path, output_folder)