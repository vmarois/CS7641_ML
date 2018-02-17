import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# import dataset
loans = pd.read_csv('loan_data.csv')
# drop the 'purpose' attribute as it is text-based
loans.drop('purpose', axis=1, inplace=True)

# print some initial info about the dataset
print(loans.info())

# print the first 5 lines of the dataset
print(loans.head())

# see the distribution of the outputs : quite unbalanced
plt.figure()
sns.countplot(x='not.fully.paid', data=loans)
plt.title("Distribution of the binary output 'not.fully.paid'")
plt.show()


def scatter_matrix():
    """
    Displays the scatterplot matrix to give an idea of the correlation between the attributes.
    """
    plt.figure()
    sns.pairplot(loans, hue='not.fully.paid')
    plt.title('Scatter matrix of the loans dataset')
    plt.show()


def corr_matrix():
    """
    Visualize the correlation matrix.
    """
    corr = loans.corr()
    plt.figure()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('Correlation heatmap for dataset 1')
    plt.savefig('loan-dataset-plots/corr_heatmap.png', bbox_layout='tight')


def histograms_train_test():
    """
    Displays histograms of the classes in the training set & test set to verify if stratified sampling has been done
    properly.
    """
    # stratified split
    X = loans.drop('not.fully.paid', axis=1)
    y = loans['not.fully.paid']
    _, _, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    train_split = y_train.value_counts(normalize=True)
    test_split = y_test.value_counts(normalize=True)
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    train_split.plot(kind='bar', ax=ax1, title='Training set')
    test_split.plot(kind='bar', ax=ax2, title='Testing set')
    plt.savefig('loan-dataset-plots/stratified_sampling.png', bbox_layout='tight')


if __name__ == '__main__':
    #corr_matrix()
    histograms_train_test()
