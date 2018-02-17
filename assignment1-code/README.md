This assignment was implemented using Python 3.6. The packages requirements are the following :
-sklearn
-numpy
-matplotlib
-seaborn
-pandas

It is possible to install missing packages with the command "pip3 install <package-name>".

The code is structured as follows :
- functions.py contains several methods which are recurrently used for the algorithms analysis,
- letter-dataset : KNN.py (k nearest neighbors method)			
		   dataset.py (print some info about the letter dataset)
		   letter.csv (source data file)
		   SVM.py (Support Vector Machine method)
		   decision_tree.py (Decision Tree method)
		   neural_network.py (Neural Network method)
		   boosting.py (Boosting method)
		   letter-dataset-plots (folder which will contain produced plots)

- loan-dataset : KNN.py (k nearest neighbors method)			
		 dataset.py (print some info about the letter dataset)
		 loan_data.csv (source data file)
		 SVM.py (Support Vector Machine method)
		 decision_tree.py (Decision Tree method)
		 neural_network.py (Neural Network method)
		 boosting.py (Boosting method)
		 loan-dataset-plots (folder which will contain produced plots)

Note : each .py script contains an entry section ("if __name__ == '__main__':") at the bottom of the script which gathers the function calls. Some functions are computationnaly heavy, so you might want to comment these out if needed.