from models.sentiments.dataset.data_handler import load_datasets, load_test_datasets
from models.sentiments.models.model_train_test import start_epochs, load_model
import os
import pandas as pd
import numpy as np

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/Saria's"
final_data = os.path.join(root_dir, "twitter_29_liwc_label.csv")
load_model_path = ""
# list_vals = ["cOPN", "cCON", "cEXT", "cAGR", "cNEU"]
list_vals = ["cOPN", "cCON", "cEXT", "cAGR", "cNEU"]


def process_df(input_df, input_str):
    input_df['u_id'] = input_df.index
    input_df.rename(columns={"STATUS": "desc", input_str: "label"}, inplace=True)
    return input_df


def train_financial_news_sentiments():
    for each in list_vals:
        classification_df = pd.read_csv(final_data)
        classification_df = process_df(classification_df, each)
        number_of_classes = len(list(classification_df['label'].unique()))
        model_directory = os.path.join(root_dir, "classify_dict")
        metrics_json = os.path.join(root_dir, "accuracy_metrics_" + each + ".json")
        training_loader, testing_loader = load_datasets(classification_df, train_size=0.8,
                                                        number_of_classes=number_of_classes)
        start_epochs(training_loader, testing_loader, metrics_json, model_directory, epochs=20,
                     number_of_classes=number_of_classes)


def inference_classification():
    inference_df = pd.read_csv(final_data)
    test_loader = load_test_datasets(inference_df, 3)
    unique_ids, predictions = load_model(load_model_path, test_loader, 3)
    out_numpy = np.concatenate((unique_ids.reshape(-1, 1), predictions.reshape(-1, 1)), axis=1)
    dept_df = pd.DataFrame(out_numpy, columns=['id', 'classification'])
    dept_df.to_csv(os.path.join(root_dir, "news_data/processed_sentiments.csv"), index=False, header=True)


if __name__ == '__main__':
    train_financial_news_sentiments()

# load_model_path = os.path.join(root_dir, 'prob_state_dict2.pt')
# load_model(load_model_path, training_loader, testing_loader)
