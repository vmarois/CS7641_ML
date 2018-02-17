"""
This file is part of assignment1.
This file implements the decision tree algorithm on the dataset 2.
@author: vmarois
"""
import pandas as pd
import sys

from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier

# Set python path to find the local functions file
sys.path.insert(0, 'pwd')
from functions import *

# import dataset
letters = pd.read_csv('letter.csv')

# separate attributes & labels
X = letters.drop('lettr', axis=1)
y = letters['lettr']

# split dataset into training set & testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def perform_grid_search():
    """
    Perform grid search on max_depth & min_samples_split to find the best pair.
    """
    max_depth = np.arange(1, 20)
    min_samples_split = np.arange(2, 15)
    param_grid = [{"max_depth": max_depth, "min_samples_split": min_samples_split}]
    print('Performing grid search using the following parameters & ranges: \n', param_grid)
    grid_search = GridSearchCV(DecisionTreeClassifier(criterion='entropy'),
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

    mean_test_scores = grid_search.cv_results_['mean_test_score'].reshape(len(max_depth), len(min_samples_split))
    plt.figure()
    sns.heatmap(mean_test_scores, cmap='Blues')
    plt.xlabel('min_samples_split')
    plt.xticks(np.arange(len(min_samples_split)) + 0.5, min_samples_split)
    plt.ylabel('max_depth')
    plt.yticks(np.arange(len(max_depth)) + 0.5, max_depth)
    plt.savefig('letter-dataset-plots/grid_search-dtree.png', bbox_inches='tight')


if __name__ == '__main__':
    initial_training_wall_clock_time(estimator=DecisionTreeClassifier(criterion='entropy'), X_train=X_train,
                                     y_train=y_train, X_test=X_test, y_test=y_test)

    cross_validation_dist_plot(estimator=DecisionTreeClassifier(), title="dtree", location="letter-dataset-plots",
                               X=X_train, y=y_train, cv=10)

    perform_grid_search()

    plot_learning_curve(DecisionTreeClassifier(criterion='entropy'), "dtree", "letter-dataset-plots", X, y,
                        ylim=(0.4, 1.01), cv=ShuffleSplit(n_splits=10, test_size=0.3, random_state=42), n_jobs=4,
                        train_sizes=np.linspace(.1, 1.0, 5))

    plot_validation_curve(estimator=DecisionTreeClassifier(criterion='entropy'),
                          title="dtree", location="letter-dataset-plots", xlabel="max_depth", ylabel="Score",
                          X=X, y=y, param_name="max_depth", ylim=(0.4, 1.01),
                          cv=ShuffleSplit(n_splits=10, test_size=0.3, random_state=42),
                          n_jobs=4, param_range=np.arange(1, 15))

    print('Done')
