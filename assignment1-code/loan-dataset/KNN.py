"""
This file is part of assignment1.
This file implements the k nearest neighbors algorithm on the dataset 1.
@author: vmarois
"""
import pandas as pd
import sys

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

# Set python path to find the local functions file
sys.path.insert(0, 'pwd')
from functions import *


# import dataset
loans = pd.read_csv('loan_data.csv')
# drop the 'purpose' attribute as it is text-based
loans.drop('purpose', axis=1, inplace=True)

# separate attributes & labels
X = loans.drop('not.fully.paid', axis=1)
y = loans['not.fully.paid']

# split dataset into training set & testing set, using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


def perform_grid_search():
    """
    Perform grid search on n_neighbors & weights to find the best pair.
    """
    n_neighbors = np.arange(1, 16)
    weights = ["distance", "uniform"]
    p = [1, 2]
    param_grid = {"n_neighbors": n_neighbors, "weights": weights}
    print('Performing grid search using the following parameters & ranges: \n', param_grid)
    grid_search = GridSearchCV(KNeighborsClassifier(algorithm='auto', p=2, metric='minkowski', n_jobs=4),
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

    mean_test_scores = grid_search.cv_results_['mean_test_score'].reshape(len(n_neighbors), len(weights))
    plt.figure(figsize=(len(weights), len(n_neighbors)))
    sns.heatmap(mean_test_scores, cmap='Blues')
    plt.xlabel('weights')
    plt.xticks(np.arange(len(weights)) + 0.5, grid_search.param_grid['weights'], rotation=45)
    plt.ylabel('n_neighbors')
    plt.yticks(np.arange(len(n_neighbors)) + 0.5, grid_search.param_grid['n_neighbors'], rotation=89)
    plt.savefig('loan-dataset-plots/grid_search-knn.png', bbox_inches='tight')


if __name__ == '__main__':
    initial_training_wall_clock_time(estimator=KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto',
                                                                    p=2, metric='minkowski', n_jobs=4), X_train=X_train,
                                     y_train=y_train, X_test=X_test, y_test=y_test)

    cross_validation_dist_plot(estimator=KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', p=2,
                                                              metric='minkowski', n_jobs=4), title="knn",
                               location="loan-dataset-plots", X=X_train, y=y_train, cv=10)

    perform_grid_search()

    plot_learning_curve(KNeighborsClassifier(n_neighbors=14, weights="uniform", algorithm='auto', p=2,
                                             metric='minkowski', n_jobs=4), "knn",
                        "loan-dataset-plots", X, y, ylim=(0.4, 1.01), cv=StratifiedKFold(n_splits=10, random_state=42),
                        n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5))

    plot_validation_curve(KNeighborsClassifier(weights="uniform", algorithm='auto', p=2, metric='minkowski', n_jobs=4), "knn",
                          "loan-dataset-plots", "n_neighbors", "Score", X, y, "n_neighbors", ylim=(0.4, 1.01),
                          cv=StratifiedKFold(n_splits=10, random_state=42), n_jobs=4, param_range=np.arange(1, 21))

    print('Done')
