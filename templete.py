#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def load_dataset(dataset_path):
	loadfile = pd.read_csv(dataset_path)

def dataset_stat(dataset_df):
        X = dataset_df.drop(columns="target", axis=1)
        y = dataset_df["target"]
        print("Number of features: ", X) 
        print("Number of data for class 0: ", dataset_df.groupby("target").size()[0])
        print("Number of data for class 1: ", dataset_df.groupby("target").size()[1])
        

def split_dataset(dataset_df, testset_size):
        print("Splitting the dataset with the test size of ", testset_size)
        x_train, x_test, y_train, y_test = train_test_split(dataset_df.data, dataset_df.target, test_size = testset_size)

def decision_tree_train_test(x_train, x_test, y_train, y_test):
        print("Decision Tree Performances")
        dt_cls = DecisionTreeClassifier()
        dt_cls.fit(x_train, y_train)
        print("Accuracy: ",accuracy_score(y_test, dt_cls.predict(x_test)))
        print("Precision: ", precision_score(y_test, dt_cls.predict(x_test)))
        print("Recall: ", recall_score(y_test, dt_cls.predict(x_test)))

def random_forest_train_test(x_train, x_test, y_train, y_test):
	print("Random Forest Performances")
	rf_cls = RandomForestClassifier()
	rf_cls.fit(x_train, y_train)
	print("Accuracy: ", accuracy_score(rf_cls.predict(x_test), y_test))
	print("Precision : ", precision_score(rf_cls.predict(x_test), y_test))
	print("Recall: ", recall_score(rf_cls.predict(x_test), y_test))

def svm_train_test(x_train, x_test, y_train, y_test):
	print("SVM Performances")
	svm_cls = SVC()
	svm_cls.fit(x_train, y_train)
	print("Accuracy: ", accuracy_score(y_test, svm_cls.predict(x_test)))
	print("Precision: ", precision_score(y_test, svm_cls.predict(x_test)))
	print("Recall: ", recall_score(y_test, svm_cls.predict(x_test)))

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)
