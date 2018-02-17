"""
This file is part of assignment1.
This file implements functions used recurrently to evaluate the 5 algorithms.
@author: vmarois
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from sklearn.model_selection import learning_curve, validation_curve, GridSearchCV, cross_val_score


def initial_training_wall_clock_time(estimator, X_train, y_train, X_test, y_test):
    """
    Performs a first initial training & test of the estimator using both training & testing set. We are also measuring
    the wall clock time taken to train the estimator & get the training / testing scores.
    """
    start_time = time.time()
    # train on training set
    estimator.fit(X_train, y_train)

    # evaluate training score
    score_training = estimator.score(X_train, y_train)
    print('The training score is : ', score_training)

    # evaluate training score
    score_test = estimator.score(X_test, y_test)
    print('The testing score is : ', score_test)
    # get wall clock time taken
    print("\nWall clock time: %s" % (time.time() - start_time))


def cross_validation_dist_plot(estimator, title, location, X, y, cv=10):
    """
    Plot the distribution of the cross validation score.
    """
    plt.figure()
    cv_scores = cross_val_score(estimator, X, y, cv=cv)
    sns.set()
    sns.distplot(cv_scores, bins=5)
    plt.title('Mean score: %0.2f ; standard deviation : %0.3f' % (np.mean(cv_scores), np.std(cv_scores)))
    plt.savefig('{}/10_fold_cv_score-{}.png'.format(location, title), bbox_layout='tight')


def plot_learning_curve(estimator, title, location, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Learning curve. Determines cross-validated training and test scores for different training set sizes.
    """
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.grid()
    plt.title("Learning curves ({})".format(title))

    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig('{}/learning_curve_{}.png'.format(location, title), bbox_inches='tight')


def plot_validation_curve(estimator, title, location, xlabel, ylabel, X, y, param_name, ylim=None, cv=None, n_jobs=4,
                          param_range=np.linspace(1, 1, 10)):
    """
    Validation curve. Determine training and test scores for varying parameter values.
    Compute scores for an estimator with different values of a specified parameter.
    """
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range,
                                                 cv=cv, scoring="accuracy", n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Validation curves ({})".format(title))

    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=2)

    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=2)

    plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=2)
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=2)

    plt.legend(loc="best")
    plt.savefig('{}/validation_curve_{}.png'.format(location, title), bbox_inches='tight')


def plot_iterations_curve(estimator, title, location, X, y, ylim=None, cv=None, iterations=np.arange(1, 200000, 10000)):
    """
    Learning curve. Determines cross-validated training and test scores for different number of iterations.
    """
    param_grid = {'max_iter': iterations}
    grid_search = GridSearchCV(estimator, param_grid=param_grid, cv=cv)
    grid_search.fit(X, y)

    train_scores_mean = grid_search.cv_results_['mean_train_score']
    train_scores_std = grid_search.cv_results_['std_train_score']
    test_scores_mean = grid_search.cv_results_['mean_test_score']
    test_scores_std = grid_search.cv_results_['std_test_score']

    plt.figure()
    plt.grid()
    plt.title("Iterative Learning curves ({})".format(title))

    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel("Iterations")
    plt.ylabel("Score")

    plt.fill_between(iterations, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")

    plt.fill_between(iterations, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")

    plt.plot(iterations, train_scores_mean, 'o-', color="r", label="Training score")

    plt.plot(iterations, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig('{}/learning_curve_{}.png'.format(location, title), bbox_inches='tight')
