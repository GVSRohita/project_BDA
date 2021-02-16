from models.sentiments.dataset.data_handler import load_datasets, load_test_datasets
from models.sentiments.models.model_train_test import start_epochs, load_model
import os
import pandas as pd
import numpy as np

from utils.file_utils import read_json

root_dir = "/home/charan/DATA/311_Data/multi-level-classification"
final_data = os.path.join(root_dir, "balanced_multi-level.csv")
updated_data = os.path.join(root_dir, "balanced_multi-level_update.csv")
cat_json = os.path.join(root_dir, "category_class.json")
type_json = os.path.join(root_dir, "type_class.json")
cat_json = read_json(cat_json)
type_json = read_json(type_json)
load_model_path = ""


def setup_data(input_df):
    input_df["label"] = input_df["TYPE"].apply(lambda x: type_json[x])
    input_df["u_id"] = input_df.index
    input_df.rename(columns={"Description": "desc"}, inplace=True)
    input_df.to_csv(updated_data, index=False)
    return input_df


def train_classification():
    classification_df = pd.read_csv(final_data)
    classification_df = setup_data(classification_df)
    number_of_classes = len(list(classification_df['label'].unique()))
    model_directory = os.path.join(root_dir, "classify_state_dict")
    metrics_json = os.path.join(root_dir, "accuracy_metrics.json")
    training_loader, testing_loader = load_datasets(classification_df, train_size=0.8,
                                                    number_of_classes=number_of_classes)
    unique_ids, val_targets, val_outputs = start_epochs(training_loader, testing_loader, metrics_json, model_directory,
                                                        epochs=50, number_of_classes=number_of_classes)
    out_numpy = np.concatenate((unique_ids.reshape(-1, 1), val_targets.reshape(-1, 1), val_outputs.reshape(-1, 1)),
                               axis=1)
    predicted_df = pd.DataFrame(out_numpy, columns=['id', 'original', 'predicted'])
    predicted_df.to_csv(os.path.join(root_dir, "predicted.csv"), index=False, header=True)


def inference_classification():
    inference_df = pd.read_csv(final_data)
    test_loader = load_test_datasets(inference_df, 3)
    unique_ids, predictions = load_model(load_model_path, test_loader, 3)
    out_numpy = np.concatenate((unique_ids.reshape(-1, 1), predictions.reshape(-1, 1)), axis=1)
    dept_df = pd.DataFrame(out_numpy, columns=['id', 'classification'])
    dept_df.to_csv(os.path.join(root_dir, "news_data/processed_sentiments.csv"), index=False, header=True)


if __name__ == '__main__':
    train_classification()
