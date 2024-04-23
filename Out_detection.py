# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

def outlier_detection_and_classification(X_train, y_train, X_test, y_test, C_num):

    colors = ['red' if label == 1 else 'blue' for label in y_train]
    # Outlier detection using LOF algorithm
    lof = LocalOutlierFactor(n_neighbors=10, contamination=0.03) #更改neighbors了原来10

    #X_train = np.array(X_train).reshape(1, -1)  #自行加的因为报错

    y_pred = lof.fit_predict(X_train)
    
    # Find samples that are identified as outliers
    outliers = X_train[y_pred == -1]

    # Mark the outlier points in the figure and bold the borders
    plt.scatter(outliers[:, 0], outliers[:, 1], s=100, facecolors='none', edgecolors='g', linewidths=2)

    # Find the index of the samples identified as outliers
    outlier_indexes = np.where(y_pred == -1)[0]

    # Remove these samples from the training set
    #X_train = np.delete(X_train, outlier_indexes, axis=0)
    #y_train = np.delete(y_train, outlier_indexes, axis=0)

    # define SVM model
    clf = svm.SVC(kernel='linear', C=C_num)
    clf.fit(X_train, y_train)

    # Plotting decision boundaries and support vectors
    plt.xlim([4, 8.5])
    plt.ylim([1.7, 4.5])
    plt.scatter(X_train[:, 0], X_train[:, 1], c=colors)
    plt.xlabel('Sepal length')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=colors)
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

    # Calculate and display the model classification correct rate
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    #print('training sets classification accuracy：', accuracy_train)
    #print('testing sets classification accuracy：', accuracy_test)
    plt.text(0.99, 0.01, f"Accuracy: {accuracy_train:.2f}\nC={C_num}", transform=plt.gca().transAxes, fontsize=14, color='blue', ha='right', va='bottom')
    #plt.text(0.99, 0.01, f"Train Acc: {accuracy_train:.2f}\nTest Acc: {accuracy_test:.2f}\nC={C_num}", transform=plt.gca().transAxes, fontsize=14, color='blue', ha='right', va='bottom')  
    plt.legend(handles=[plt.scatter([],[], c='blue', label='Iris Setosa'),
                    plt.scatter([],[], c='red', label='Iris Versicolor')],
           loc='upper right')
    plt.title('Training Data')
    plt.show()
    # Calculate the distance of the support vector to the decision boundary

    sv_count = len(clf.support_vectors_)
    distances = clf.decision_function(X_train[clf.support_])
    avg_distance = abs(distances).mean()
    #print('Distance of support vector to decision boundary：', avg_distance)
    #print('support vector数量：', sv_count)
    return accuracy_train, avg_distance, sv_count




