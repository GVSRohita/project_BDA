import pandas as pd
import os
from utils.file_utils import write_json_dict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/Saria/twittter_LIWC"
root_dir = "/home/charan/DATA/OCEAN_DATA/Results/LIWC/"
list_prediction = ["predicted_cOPN.csv", "predicted_cCON.csv", "predicted_cEXT.csv", "predicted_cAGR.csv",
                   "predicted_cNEU.csv"]


# list_prediction = ["predicted_cOPN.csv"]


def manual_check():
    for each in list_prediction:
        current = os.path.join(root_dir, each)
        current_data = pd.read_csv(current)
        current_data.original, current_data.predicted = current_data.original.astype(
            'int64'), current_data.predicted.astype('int64')
        confusion_dict = {"0": {"true": 0, "false": 0}, "1": {"true": 0, "false": 0}}
        for each in current_data.to_dict("records"):
            each["original"], each["predicted"] = str(each["original"]), str(each["predicted"])
            if each["original"] == each["predicted"]:
                confusion_dict[each["original"]]["true"] += 1
            else:
                confusion_dict[each["original"]]["false"] += 1
        write_json_dict(confusion_dict, current.replace(".csv", ".json"))


def calculate_scores():
    list_scores = []
    for each in list_prediction:
        current = os.path.join(root_dir, each)
        current_data = pd.read_csv(current)
        current_data.original, current_data.predicted = current_data.original.astype(
            'int64'), current_data.predicted.astype('int64')
        print(f'analysis for - {each}')
        predicted = {
            "Class": each,
            "F1 Score": round(f1_score(current_data.original, current_data.predicted), 2),
            "Precision": round(precision_score(current_data.original, current_data.predicted), 2),
            "Recall": round(recall_score(current_data.original, current_data.predicted), 2)
        }
        list_scores.append(predicted)
        print(f'F1 Score {round(f1_score(current_data.original, current_data.predicted), 2)}')
        print(f'Precision {round(precision_score(current_data.original, current_data.predicted), 2)}')
        print(f'Recall {round(recall_score(current_data.original, current_data.predicted), 2)}')
    pd.DataFrame(list_scores).to_csv(os.path.join(root_dir, "scores.csv"), index=False)


if __name__ == '__main__':
    calculate_scores()
