from models.sentiments.dataset.data_handler import load_datasets, load_test_datasets
from models.sentiments.models.model_train_test import start_epochs, load_model
import os
import pandas as pd
import numpy as np

"""
Ignore this file, as its part of some of our experiments to understand sentiments better by doing this 
parameters based on OCEAN metrics
"""

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/Saria"
final_data = os.path.join(root_dir, "twitter_29_liwc_label.csv")
formatted_data = os.path.join(root_dir, "formatted.csv")
load_model_path = ""
list_vals = ["cOPN", "cCON", "cEXT", "cAGR", "cNEU"]
# list_vals = ["cOPN"]

input_dict = {}

liwc_list = ['Analytic', 'Clout', 'Authentic', 'Tone', 'WPS', 'Sixltr', 'Dic', 'function', 'pronoun', 'ppron', 'i',
             'we', 'you', 'shehe', 'they', 'ipron', 'article', 'prep', 'auxverb', 'adverb', 'conj', 'negate', 'verb',
             'adj', 'compare', 'interrog', 'number', 'quant', 'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad',
             'social', 'family', 'friend', 'female', 'male', 'cogproc', 'insight', 'cause', 'discrep', 'tentat',
             'certain', 'differ', 'percept', 'see', 'hear', 'feel', 'bio', 'body', 'health', 'sexual', 'ingest',
             'drives', 'affiliation', 'achieve', 'power', 'reward', 'risk', 'focuspast', 'focuspresent', 'focusfuture',
             'relativ', 'motion', 'space', 'time', 'work', 'leisure', 'home', 'money', 'relig', 'death', 'informal',
             'swear', 'netspeak', 'assent', 'nonflu', 'filler', 'AllPunc', 'Period', 'Comma', 'Colon', 'SemiC', 'QMark',
             'Exclam', 'Dash', 'Quote', 'Apostro', 'Parenth', 'OtherP']


def setup_dict():
    for each in list_vals:
        file_name = "twitter_uc_{}.csv".format(each)
        input_dict[each] = os.path.join(root_dir, file_name)


def apply_liwc(input_record):
    return_list = []
    for each in liwc_list:
        if input_record[each] > 0:
            return_list.append(each)
    return " ".join(return_list)


def process_df(input_df):
    input_df['liwc_features'] = input_df.apply(apply_liwc, axis=1)
    input_df['STATUS'] = input_df['liwc_features'] + input_df['STATUS']
    # input_df.rename(columns={"TOPICS": "desc", input_str: "label"}, inplace=True)
    input_df.to_csv(formatted_data, index=False, header=True)
    return input_df


def train_for_classification():
    for each in list_vals:
        classification_df = pd.read_csv(input_dict[each])
        classification_df['u_id'] = classification_df.index
        classification_df = process_df(classification_df)
        classification_df.rename(columns={"STATUS": "desc", each: "label"}, inplace=True)
        number_of_classes = len(list(classification_df['label'].unique()))
        model_directory = os.path.join(root_dir, "classify_dict_" + each)
        metrics_json = os.path.join(root_dir, "accuracy_metrics_" + each + ".json")
        training_loader, testing_loader = load_datasets(classification_df, train_size=0.8,
                                                        number_of_classes=number_of_classes)
        unique_ids, val_targets, val_outputs = start_epochs(training_loader, testing_loader, metrics_json,
                                                            model_directory, epochs=5,
                                                            number_of_classes=number_of_classes)
        out_numpy = np.concatenate((unique_ids.reshape(-1, 1), val_targets.reshape(-1, 1), val_outputs.reshape(-1, 1)),
                                   axis=1)
        predicted_df = pd.DataFrame(out_numpy, columns=['id', 'original', 'predicted'])
        predicted_df.to_csv(os.path.join(root_dir, "predicted_" + each + ".csv"), index=False, header=True)


def inference_classification():
    inference_df = pd.read_csv(final_data)
    test_loader = load_test_datasets(inference_df, 3)
    unique_ids, predictions = load_model(load_model_path, test_loader, 3)
    out_numpy = np.concatenate((unique_ids.reshape(-1, 1), predictions.reshape(-1, 1)), axis=1)
    dept_df = pd.DataFrame(out_numpy, columns=['id', 'classification'])
    dept_df.to_csv(os.path.join(root_dir, "news_data/processed_sentiments.csv"), index=False, header=True)


if __name__ == '__main__':
    setup_dict()
    train_for_classification()
