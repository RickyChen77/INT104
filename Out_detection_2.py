'''
# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

def outlier_detection_and_classification_2(X_train, y_train, X_test, y_test, C_num):
    # Outlier detection using LOF algorithm
    lof = LocalOutlierFactor(n_neighbors=10, contamination=0.03)
    y_pred_train = lof.fit_predict(X_train)

    # Find samples that are identified as outliers
    outliers_train = X_train[y_pred_train == -1]

    # Remove these samples from the training set
    X_train_clean = X_train[y_pred_train != -1]
    y_train_clean = y_train[y_pred_train != -1]

    # define SVM model
    clf = svm.SVC(kernel='linear', C=C_num)
    clf.fit(X_train_clean, y_train_clean)

    # Calculate and display the model classification accuracy
    y_pred_test = clf.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    plt.text(0.99, 0.01, f"Test Accuracy: {accuracy_test:.2f}\nC={C_num}", transform=plt.gca().transAxes, fontsize=14, color='blue', ha='right', va='bottom')

    # Plotting decision boundaries and support vectors
    plt.xlim([4, 8.5])
    plt.ylim([1.7, 4.5])
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    # Mapping decision boundaries
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Cover the entire drawing area with a grid
    x_grid = np.linspace(xlim[0], xlim[1], 30)
    y_grid = np.linspace(ylim[0], ylim[1], 30)
    X_label, Y_label = np.meshgrid(x_grid, y_grid)

    # Predicting the labels of grid points
    xy = np.vstack([X_label.ravel(), Y_label.ravel()]).T
    Z = clf.decision_function(xy).reshape(X_label.shape)

    # Plotting decision boundaries and support vectors
    ax.contour(X_label, Y_label, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')

    # Plotting the test data points
    colors = ['red' if label == 1 else 'blue' for label in y_test]
    plt.scatter(X_test[:, 0], X_test[:, 1], c=colors)

    plt.legend(handles=[plt.scatter([],[], c='blue', label='Iris Setosa'),
                         plt.scatter([],[], c='red', label='Iris Versicolor')],
               loc='upper right')
    plt.title('Test Data')
    plt.show()

    # Calculate the distance of the support vector to the decision boundary
    sv_count = len(clf.support_vectors_)
    distances = clf.decision_function(X_train_clean[clf.support_])
    avg_distance = abs(distances).mean()
    return accuracy_test, avg_distance, sv_count
'''
# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

def outlier_detection_and_classification_2(X_train, y_train, X_test, y_test, C_num):
    # Outlier detection using LOF algorithm
    lof = LocalOutlierFactor(n_neighbors=10, contamination=0.03)
    y_pred_train = lof.fit_predict(X_train)

    # Find samples that are identified as outliers
    outliers_train = X_train[y_pred_train == -1]

    # Remove these samples from the training set
    X_train_clean = X_train[y_pred_train != -1]
    y_train_clean = y_train[y_pred_train != -1]

    # define SVM model
    clf = svm.SVC(kernel='linear', C=C_num)
    clf.fit(X_train_clean, y_train_clean)

    # Calculate and display the model classification accuracy
    y_pred_test = clf.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    plt.text(0.99, 0.01, f"Test Accuracy: {accuracy_test:.2f}\nC={C_num}", transform=plt.gca().transAxes, fontsize=14, color='blue', ha='right', va='bottom')

    # Plotting decision boundaries and support vectors
    plt.xlim([4, 8.5])
    plt.ylim([1.7, 4.5])
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    # Mapping decision boundaries
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Cover the entire drawing area with a grid
    x_grid = np.linspace(xlim[0], xlim[1], 30)
    y_grid = np.linspace(ylim[0], ylim[1], 30)
    X_label, Y_label = np.meshgrid(x_grid, y_grid)

    # Predicting the labels of grid points
    xy = np.vstack([X_label.ravel(), Y_label.ravel()]).T
    Z = clf.decision_function(xy).reshape(X_label.shape)

    # Plotting decision boundaries
    ax.contour(X_label, Y_label, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # Plotting the test data points
    colors = ['red' if label == 1 else 'blue' for label in y_test]
    plt.scatter(X_test[:, 0], X_test[:, 1], c=colors)

    plt.legend(handles=[plt.scatter([],[], c='blue', label='Iris Setosa'),
                         plt.scatter([],[], c='red', label='Iris Versicolor')],
               loc='upper right')
    plt.title('Test Data')
    plt.show()

    # Calculate the distance of the support vector to the decision boundary
    sv_count = len(clf.support_vectors_)
    distances = clf.decision_function(X_train_clean[clf.support_])
    avg_distance = abs(distances).mean()
    return accuracy_test, avg_distance, sv_count
