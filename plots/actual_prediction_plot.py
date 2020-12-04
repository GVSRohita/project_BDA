import matplotlib.pyplot as plt


def plot_prediction(y_axis, actual, predictions):
    plt.plot(actual, y_axis)
    plt.plot(predictions, y_axis)
    plt.show()
