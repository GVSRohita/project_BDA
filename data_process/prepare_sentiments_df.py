import os
import pandas as pd
from utils.file_utils import write_json_dict

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/BDA_Project"
data_path = os.path.join(root_dir, "Sentiment_Financial_Data/all-data.csv")
final_data = os.path.join(root_dir, "Sentiment_Financial_Data/sentiment_data_processed.csv")
mapping_dict = os.path.join(root_dir, "Sentiment_Financial_Data/label_mapping_dict.json")

sentiment_df = pd.read_csv(data_path, encoding="ISO-8859-1", header=None)
sentiment_df.rename(columns={0: 'label', 1: 'desc'}, inplace=True)

label_mapping_dict = {}
counter = 0
for each in sorted(list(sentiment_df.label.unique())):
    label_mapping_dict[str(counter)] = each
    label_mapping_dict[each] = counter
    counter += 1

write_json_dict(input_dict=label_mapping_dict, file_name=mapping_dict)

sentiment_df.label = sentiment_df.label.apply(lambda x: label_mapping_dict[x])
sentiment_df.to_csv(final_data, header=True, index=True, index_label='u_id')
print("dataframe processed")
