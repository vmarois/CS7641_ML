import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import dataset
letters = pd.read_csv('letter.csv')

# print some initial info about the dataset
print(letters.info())

# print the first 5 lines of the dataset
print(letters.head())


# see the distribution of the outputs : very evenly split
plt.figure()
sns.countplot(x='lettr', data=letters)
plt.title("Distribution of the 26 classes")
plt.show()


def scatter_matrix():
    """
    Displays the plot of the scatter matrix to give an idea of the correlation between the attributes.
    """
    plt.figure()
    sns.pairplot(letters, hue='lettr')
    plt.title('Scatter matrix of the letters dataset')
    plt.show()


def corr_matrix():
    """
    Visualize the correlation matrix.
    """
    # visualize the correlation matrix
    corr = letters.corr()
    plt.figure()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('Correlation heatmap for dataset 1')
    plt.show()


if __name__ == '__main__':
    scatter_matrix()

    corr_matrix()
