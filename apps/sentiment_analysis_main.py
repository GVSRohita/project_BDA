from models.sentiments.dataset.data_handler import load_datasets, load_test_datasets
from models.sentiments.models.model_train_test import start_epochs, load_model
import os
import pandas as pd
import numpy as np

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/BDA_Project"
final_data = os.path.join(root_dir, "Sentiment_Financial_Data/sentiment_data_processed.csv")
mapping_dict = os.path.join(root_dir, "Sentiment_Financial_Data/label_mapping_dict.json")
load_model_path = os.path.join(root_dir, 'sentiment_state_dict_9.pt')
news_classify_path = os.path.join(root_dir, "news_data/news_classification.csv")


def train_financial_news_sentiments():
    classification_df = pd.read_csv(final_data)
    number_of_classes = len(list(classification_df['label'].unique()))
    model_directory = os.path.join(root_dir, "sentiment_state_dict")
    metrics_json = os.path.join(root_dir, "accuracy_metrics.json")
    training_loader, testing_loader = load_datasets(classification_df, train_size=0.8,
                                                    number_of_classes=number_of_classes)
    start_epochs(training_loader, testing_loader, metrics_json, model_directory, epochs=10,
                 number_of_classes=number_of_classes)


def inference_classification():
    inference_df = pd.read_csv(news_classify_path)
    test_loader = load_test_datasets(inference_df, 3)
    unique_ids, predictions = load_model(load_model_path, test_loader, 3)
    out_numpy = np.concatenate((unique_ids.reshape(-1, 1), predictions.reshape(-1, 1)), axis=1)
    dept_df = pd.DataFrame(out_numpy, columns=['id', 'classification'])
    dept_df.to_csv(os.path.join(root_dir, "news_data/processed_sentiments.csv"), index=False, header=True)


if __name__ == '__main__':
    inference_classification()

# load_model_path = os.path.join(root_dir, 'prob_state_dict2.pt')
# load_model(load_model_path, training_loader, testing_loader)
