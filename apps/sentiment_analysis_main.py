from models.sentiments.dataset.data_handler import load_datasets
from models.sentiments.models.model_train_test import start_epochs, load_model
import os
import pandas as pd


root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/BDA_Project"
final_data = os.path.join(root_dir, "Sentiment_Financial_Data/sentiment_data_processed.csv")
mapping_dict = os.path.join(root_dir, "Sentiment_Financial_Data/label_mapping_dict.json")


def train_financial_news_sentiments():
    classification_df = pd.read_csv(final_data)
    number_of_classes = len(list(classification_df['label'].unique()))
    model_directory = os.path.join(root_dir, "sentiment_state_dict")
    metrics_json = os.path.join(root_dir, "accuracy_metrics.json")
    training_loader, testing_loader = load_datasets(classification_df, train_size=0.8,
                                                    number_of_classes=number_of_classes)
    start_epochs(training_loader, testing_loader, metrics_json, model_directory, epochs=10,
                 number_of_classes=number_of_classes)
    # load_model_path = os.path.join(root_dir, 'prob_state_dict2.pt')
    # load_model(load_model_path, training_loader, testing_loader)


if __name__ == '__main__':
    train_financial_news_sentiments()
