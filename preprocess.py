import os
import pickle
import pandas as pd
from utils.data import get_vocabs, get_args, df_to_dict, TRAIN_DATA, TEST_DATA


def save_data_dict(data_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


path = (get_args()).path
vocabs = get_vocabs(os.path.join(path, 'data', 'vocabs'))
for data_name in TRAIN_DATA + TEST_DATA:
    folder = 'train' if 'train' in data_name else 'test'
    data_path = os.path.join(path, 'data', folder, data_name)
    data_df = pd.read_parquet(data_path)
    data_dict = df_to_dict(data_df, vocabs)
    new_data_name = data_name.replace('.parquet', '.pkl')
    new_data_path = os.path.join(path, 'data', 'cached', new_data_name)
    save_data_dict(data_dict, new_data_path)
