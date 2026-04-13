import matplotlib.pyplot as plt

def plot_results(y_test, predictions):
    plt.plot(y_test.values, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.savefig("images/prediction.png")
    plt.show()