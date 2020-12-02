import matplotlib.pyplot as plt
import os
import json

# root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/Saria's"
root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/BDA_Project"


def plot_accuracy(values1, values2, x_label, y_label, plot_title, file_name):
    plt.plot(values1)
    plt.plot(values2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.legend(['training_accuracy', 'validation_accuracy'], loc='upper left')
    plt.savefig(file_name)
    plt.cla()


def get_values(input_metrics, plot_title, file_name):
    train_accuracy = []
    validation_accuracy = []
    with open(input_metrics, 'r') as f:
        metrics_dict = json.load(f)
        f.close()
    for i in range(10):
        train_accuracy.append(metrics_dict['train_accuracy_' + str(i)])
        validation_accuracy.append(metrics_dict['val_accuracy_' + str(i)])
    plot_accuracy(train_accuracy, validation_accuracy, 'Epoch', 'Accuracy', plot_title, file_name)


# accuracy_metrics = ["accuracy_metrics_cEXT.json", "accuracy_metrics_cNEU.json", "accuracy_metrics_cOPN.json",
#                     "accuracy_metrics_cCON.json", "accuracy_metrics_cAGR.json"]

accuracy_metrics = ["accuracy_metrics.json"]

if __name__ == '__main__':
    for department_metrics in accuracy_metrics:
        department_metrics = os.path.join(root_dir, department_metrics)
        get_values(department_metrics, "Accuracy", department_metrics.replace(".json", ".png"))
