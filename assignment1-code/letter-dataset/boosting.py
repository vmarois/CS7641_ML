"""
This file is part of assignment1.
This file implements the boosting algorithm on the dataset 2.
@author: vmarois
"""
import pandas as pd
import sys

from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
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
    Perform grid search on n_estimators & learning_rate to find the best pair.
    """
    n_estimators = np.arange(1, 200, 20)
    learning_rate = np.linspace(0.1, 2, 11)
    param_grid = {"n_estimators": n_estimators, "learning_rate": learning_rate}
    print('Performing grid search using the following parameters & ranges: \n', param_grid)
    grid_search = GridSearchCV(AdaBoostClassifier(base_estimator=dtree),
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

    mean_test_scores = grid_search.cv_results_['mean_test_score'].reshape(len(learning_rate), len(n_estimators))
    plt.figure()
    sns.heatmap(mean_test_scores, cmap='Blues')
    plt.xlabel('n_estimators')
    plt.xticks(np.arange(len(n_estimators)) + 0.5, n_estimators)
    plt.ylabel('learning_rate')
    plt.yticks(np.arange(len(learning_rate)) + 0.5, learning_rate, rotation=0)
    plt.savefig('letter-dataset-plots/grid_search-boosting.png', bbox_inches='tight')


if __name__ == '__main__':
    initial_training_wall_clock_time(
        estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=10),
                                     n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None),
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    cross_validation_dist_plot(
        estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=10),
                                     n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None),
        title="boosting", location="letter-dataset-plots", X=X_train, y=y_train, cv=10)

    perform_grid_search()

    plot_learning_curve(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=10),
                                           n_estimators=161, learning_rate=0.86), "boosting", "letter-dataset-plots", X,
                        y, ylim=(0.4, 1.01), cv=ShuffleSplit(n_splits=10, test_size=0.3, random_state=42), n_jobs=4,
                        train_sizes=np.linspace(.1, 1.0, 5))

    plot_validation_curve(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=10),
                                             learning_rate=0.86), "boosting", "letter-dataset-plots", "n_estimators",
                          "Score", X, y, "n_estimators", ylim=(0.4, 1.01),
                          cv=ShuffleSplit(n_splits=10, test_size=0.3, random_state=42), n_jobs=4,
                          param_range=np.arange(1, 200, 20))

    print('Done')

