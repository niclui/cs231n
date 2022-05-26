import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pdb
import numpy as np
import math

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
    
    group1 = evens[evens['day_idx'] % 2 == 1][['case', 'slice_id', 'day']]
    group2 = evens[evens['day_idx'] % 2 == 0][['case', 'slice_id', 'day']]
    evens = pd.merge(group1, group2, on = ['case', 'slice_id'])
    evens['dummy'] = 1

    evens['pair_idx'] = evens['dummy'].cumsum()
    evens['pair_idx'] = evens['pair_idx'].apply(lambda x: idx + str(x))

    evens = pd.concat([evens[['case', 'slice_id', 'day_x', 'pair_idx']].rename({'day_x': 'day'}, axis = 'columns'), evens[['case', 'slice_id', 'day_y', 'pair_idx']].rename({'day_y': 'day'}, axis = 'columns')])
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
    data_normal['days'] = data_normal.groupby(['case'])['day'].transform(lambda x: x.nunique())
    data_normal['idx'] = data_normal.groupby(['case', 'slice_id'])['dummy'].cumsum()

    #Indicate odds
    data_normal_odds = data_normal[data_normal['days']%2==1]
    to_odd = data_normal_odds[['case', 'day']].drop_duplicates().groupby(['case']).count().reset_index()
    to_odd['odd_idx'] = to_odd.apply(lambda x: np.random.randint(x.day) + 1, axis = 1)
    data_normal_odds = pd.merge(data_normal_odds, to_odd[['case', 'odd_idx']], how = 'left', on = 'case')

    #Odd cases
    odds = data_normal_odds[data_normal_odds['odd_idx'] == data_normal_odds['idx']]
    odd_ids = odds_pairing(odds, 'O')

    #Evens
    evens = pd.concat([data_normal_odds[data_normal_odds['odd_idx'] != data_normal_odds['idx']], data_normal[data_normal['days']%2==0]])
    evens['day_idx'] = evens.groupby(['case', 'slice_id'])['dummy'].cumsum()
    evens['groups_int'] = evens['day_idx'].apply(lambda x: int(math.ceil(float(x)/2.0)))

    evens_a = evens[evens['groups_int'] == 1][['case', 'slice_id', 'day', 'day_idx']]
    evens_a_ids = evens_pairing(evens_a, 'EA') 

    evens_b = evens[evens['groups_int'] == 2][['case', 'slice_id', 'day', 'day_idx']]
    evens_b_ids = evens_pairing(evens_b, 'EB') 

    evens_c = evens[evens['groups_int'] == 3][['case', 'slice_id', 'day', 'day_idx']]
    evens_c_ids = evens_pairing(evens_c, 'EC')

    #Combine all data and batch
    ids = pd.concat([special_ids, odd_ids, evens_a_ids, evens_b_ids, evens_c_ids])
    data = pd.merge(ids, data.drop(['day_slices', 'days', 'case_slices', 'special'], axis = 1), on = ['case', 'slice_id', 'day'])
    data['dummy'] = data.groupby(['pair_idx'])['dummy'].cumsum()
    data1 = data[data['dummy'] == 1]

    #Random sort, make sure no more than one repeat in a batch
    min_check = 0
    while min_check <= 2:
        data1 = data1.sample(frac=1).reset_index(drop=True)
        data1['batch'] = data1.index // 16
        check = data1.groupby('batch')['case'].count()
        min_check = min(check)
    
    data2 = pd.merge(data[data['dummy'] == 2], data1[['pair_idx', 'batch']], on = 'pair_idx')
    data = pd.concat([data1, data2]).reset_index(drop=True).sort_values(['batch', 'pair_idx'])

    n_batches = len(data)/32.0
    train_batch = int(math.ceil(train_prop * n_batches))
    
    # Split into train and val+test datasets
    #train, val_test = train_test_split(data, test_size=val_prop+test_prop, random_state=0)
    train = data[data['batch'] <= train_batch]
    others = data[data['batch'] > train_batch]

    # Split the val+test datasets into validation and test
    val, test = train_test_split(others, test_size=test_prop/(val_prop+test_prop), random_state=0)
        
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