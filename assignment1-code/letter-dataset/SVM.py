"""
This file is part of assignment1.
This file implements the SVM algorithm on the dataset 2.
@author: vmarois
"""
import pandas as pd
import sys

from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Set python path to find the local functions file
sys.path.insert(0, 'pwd')
from functions import *

# import dataset
letters = pd.read_csv('letter.csv')

# separate attributes & labels
X = letters.drop('lettr', axis=1)
y = letters['lettr']

####### Measure impact of data preprocessing with Standard Scaler #####
# the following will return a centered & scaled copy of X (no need to do so on y since that's labels)
X = StandardScaler().fit_transform(X, y)

# split dataset into training set & testing set, using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def perform_grid_search():
    """
    Perform grid search on the kernel type and the degree (for the polynomial kernel) to find the best pair.
    """
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    degree = np.arange(1, 7)
    param_grid = {"kernel": kernel, "degree": degree}
    print('Performing grid search using the following parameters & ranges: \n', param_grid)
    grid_search = GridSearchCV(SVC(C=1.0, gamma='auto', shrinking=True, probability=False, tol=0.001,
                                   cache_size=200, class_weight=None, max_iter=200000),
                               param_grid=param_grid,
                               cv=10)  # 10-fold cross validation
    grid_search.fit(X_train, y_train)
    print('Optimal parameters: ', grid_search.best_params_)
    print('Best score: ', grid_search.best_score_)

    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    mean_test_scores = grid_search.cv_results_['mean_test_score'].reshape(len(degree), len(kernel))
    plt.figure(figsize=(len(degree), len(kernel)))
    sns.heatmap(mean_test_scores, cmap='Blues')
    plt.xlabel('kernel')
    plt.xticks(np.arange(len(kernel)) + 0.5, grid_search.param_grid['kernel'], rotation=0)
    plt.ylabel('degree')
    plt.yticks(np.arange(len(degree)) + 0.5, grid_search.param_grid['degree'], rotation=89)
    plt.savefig('letter-dataset-plots/grid_search-svm.png', bbox_inches='tight')


def perform_grid_search_rbf():
    """
    Perform grid search on the penalty parameter C (for the error term) & kernel coefficient gamma for a rbf kernel svc.
    """
    C = [0.1, 1, 10, 100, 1000]
    gamma = [1, 0.1, 0.01, 0.001, 0.0001]
    param_grid = {"C": C, "gamma": gamma}
    print('Performing grid search using the following parameters & ranges: \n', param_grid)
    grid_search = GridSearchCV(SVC(kernel='rbf', shrinking=True, probability=False, tol=0.001,
                                   cache_size=200, class_weight=None, max_iter=200000),
                               param_grid=param_grid,
                               cv=10)  # 10-fold cross validation
    grid_search.fit(X_train, y_train)
    print('Optimal parameters: ', grid_search.best_params_)
    print('Best score: ', grid_search.best_score_)

    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    mean_test_scores = grid_search.cv_results_['mean_test_score'].reshape(len(C), len(gamma))
    plt.figure(figsize=(len(gamma), len(C)))
    sns.heatmap(mean_test_scores, cmap='Blues')
    plt.xlabel('$\gamma$')
    plt.xticks(np.arange(len(gamma)) + 0.5, grid_search.param_grid['gamma'], rotation=0)
    plt.ylabel('C')
    plt.yticks(np.arange(len(C)) + 0.5, grid_search.param_grid['C'], rotation=89)
    plt.savefig('letter-dataset-plots/grid_search-svm_rbf.png', bbox_inches='tight')


if __name__ == '__main__':
    initial_training_wall_clock_time(estimator=SVC(C=1.0, kernel='rbf', gamma='auto', shrinking=True, probability=False,
                                             tol=0.001, cache_size=200, class_weight=None, max_iter=200000),
                                     X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    cross_validation_dist_plot(estimator=SVC(C=1.0, kernel='rbf', gamma='auto', shrinking=True, probability=False,
                                             tol=0.001, cache_size=200, class_weight=None, max_iter=200000), title="svm",
                               location="letter-dataset-plots", X=X_train, y=y_train, cv=10)

    perform_grid_search()

    perform_grid_search_rbf()

    plot_iterations_curve(SVC(kernel='rbf', C=10, gamma=0.1, shrinking=True, probability=False, tol=0.001,
                              cache_size=200, class_weight=None), "svm", "letter-dataset-plots", X,
                          y, ylim=(0.4, 1.01), cv=ShuffleSplit(n_splits=10, test_size=0.3, random_state=42),
                          iterations=np.arange(1, 200000, 10000))

    plot_learning_curve(SVC(kernel='rbf', C=10, gamma=0.1, shrinking=True, probability=False, tol=0.001,
                            cache_size=200, class_weight=None), "svm", "letter-dataset-plots", X, y, ylim=(0.4, 1.01),
                        cv=ShuffleSplit(n_splits=10, test_size=0.3, random_state=42), n_jobs=4,
                        train_sizes=np.linspace(.1, 1.0, 5))

    plot_validation_curve(SVC(kernel='rbf', gamma=0.1, shrinking=True, probability=False, tol=0.001,
                              cache_size=200, class_weight=None), "svm", "letter-dataset-plots", "C", "Score", X, y,
                          "C", ylim=(0.4, 1.01), cv=ShuffleSplit(n_splits=10, test_size=0.3, random_state=42), n_jobs=4,
                          param_range=[0.01, 0.1, 1, 10, 100, 1000])