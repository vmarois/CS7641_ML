"""
This file is part of assignment1.
This file implements the multi-layer perceptron algorithm on the dataset 2.
@author: vmarois
"""
import pandas as pd
import sys

from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.neural_network import MLPClassifier

# Set python path to find the local functions file
sys.path.insert(0, 'pwd')
from functions import *

# import dataset
letters = pd.read_csv('letter.csv')

# separate attributes & labels
X = letters.drop('lettr', axis=1)
y = letters['lettr']

# split dataset into training set & testing set, using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def perform_global_grid_search():
    """
    Perform grid search on the hidden_layer_sizes, activation, solver, learning_rate.
    """
    param_grid = {"hidden_layer_sizes": [(100,), (100, 100,), (100, 100, 100,)],
                  "activation": ['identity', 'logistic', 'tanh', 'relu'],
                  "solver": ['lbfgs', 'sgd', 'adam'],
                  "learning_rate": ['constant', 'invscaling', 'adaptive']}
    print('Performing grid search using the following parameters & ranges: \n', param_grid)
    grid_search = GridSearchCV(MLPClassifier(), param_grid=param_grid, cv=10)  # 10-fold cross validation
    grid_search.fit(X_train, y_train)
    print('Optimal parameters: ', grid_search.best_params_)
    print('Best score: ', grid_search.best_score_)

    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))


def perform_grid_search():
    """
    Perform grid search on the hidden_layer_sizes & activation function.
    """
    hidden_layer_sizes = [(100,), (100, 100,), (100, 100, 100,)]
    activation = ['identity', 'logistic', 'tanh', 'relu']
    param_grid = {"hidden_layer_sizes": hidden_layer_sizes,
                  "activation": activation}

    print('Performing grid search using the following parameters & ranges: \n', param_grid)
    grid_search = GridSearchCV(MLPClassifier(solver='adam', learning_rate='constant', learning_rate_init=0.001),
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

    mean_test_scores = grid_search.cv_results_['mean_test_score'].reshape(len(activation), len(hidden_layer_sizes))
    plt.figure(figsize=(len(activation) + 1, len(hidden_layer_sizes) + 1))
    sns.heatmap(mean_test_scores, cmap='Blues')
    plt.xlabel('hidden_layer_sizes')
    plt.xticks(np.arange(len(hidden_layer_sizes)) + 0.5, grid_search.param_grid['hidden_layer_sizes'], rotation=0)
    plt.ylabel('activation')
    plt.yticks(np.arange(len(activation)) + 0.5, grid_search.param_grid['activation'], rotation=89)
    plt.savefig('letter-dataset-plots/grid_search-neuralnet.png', bbox_inches='tight')


if __name__ == '__main__':
    initial_training_wall_clock_time(estimator=MLPClassifier(hidden_layer_sizes=(100,100,100,), activation='tanh',
                                                             solver='adam', learning_rate='constant',
                                                             learning_rate_init=0.001), X_train=X_train,
                                     y_train=y_train, X_test=X_test, y_test=y_test)

    cross_validation_dist_plot(estimator=MLPClassifier(), title="neuralnet", location="letter-dataset-plots",
                               X=X_train, y=y_train, cv=10)

    perform_global_grid_search()

    perform_grid_search()

    plot_iterations_curve(MLPClassifier(hidden_layer_sizes=(100,100,100,), activation='tanh', solver='adam',
                                        learning_rate='constant', learning_rate_init=0.001), "neural network",
                          "letter-dataset-plots", X, y, ylim=(0.4, 1.01),
                          cv=ShuffleSplit(n_splits=10, test_size=0.3, random_state=42), iterations=np.arange(1, 100, 10))

    print('Done')
