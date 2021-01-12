import os
import pandas as pd

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/Saria"
data = "twitter_29_liwc_label.csv"
data = os.path.join(root_dir, data)

data_df = pd.read_csv(data)
data_df['u_id'] = data_df.index
copn_df = data_df[['u_id', 'cOPN']]
print('finished')