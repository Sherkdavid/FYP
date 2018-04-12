from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.feature_selection import RFECV
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display, HTML

class AlgorithmPredictionSet():
    def __init__(self, algo, results, actual, selector):
        self.algo = algo
        self.difference = []
        self.percentage = []
        self.score = metrics.r2_score(actual,results)
        self.selector = selector
        i = 0
        for act in actual:
            self.difference.append(results[i] - act)
            self.percentage.append(((self.difference[i])/(act))*100)
            i+=1




def corr_matrix():
    corr = data.corr()
    map = sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                      square=True)
    map.set_xticklabels(labels, rotation=90)
    map.set_yticklabels(labels, rotation=0)
    sns.set(font_scale=1.5)
    plt.show()

def boxplot(result, x_label = 'x',y_label = 'y',title= "Figure",save = False):
    sns.boxplot(result)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(title+".jpg")
    plt.show()

def jointplot(feature, comparator,mode = 'reg'):
    sns.jointplot(feature,comparator,kind=mode)
    plt.savefig(mode+'_'+feature.name+'_'+comparator.name+'.jpg')
    plt.close()

def jointplot_all():
    for label in labels:
        for label_2 in labels:
            if label_2!=label:
                jointplot(data[label],data[label_2])
                jointplot(data[label], data[label_2],'kde')
                plt.show()

def simple_plot(y_list,title = 'Figure', save = False):
    plt.figure()
    plt.plot(range(0, len(y_list)), y_list, 'ro')
    plt.plot([0, len(y_list)], [0, 0], linewidth=3)
    plt.title(title)
    if save:
        plt.savefig(title + ".jpg")
    plt.show()

def principal_component_analysis(n):
    pca = PCA(n_components=n)
    data = pca.fit_transform(x)
    predictions = cross_val_predict(clf, data, y, cv=10)
    print(metrics.r2_score(y, predictions))
    simple_plot(predictions,"PCA n="+str(n))

def cv_predict(X,Y,algo):
    selector = RFECV(algorithms[algo], cv=10)
    selector.fit(x, y)
    predictions = selector.predict(x)
    result_set = AlgorithmPredictionSet(algo, predictions, y, selector)
    return result_set

def algorithm_tests():
    #Stage one get results

    results = pd.DataFrame()
    for algo in algorithms:
        print(algo)
        selector = RFECV(algorithms[algo], cv=10)
        selector.fit(x, y)
        print('Optimal number of features is ', selector.n_features_)
        predictions = selector.predict(x)
        result_set = AlgorithmPredictionSet(algo, predictions, y)
        results[algo] = predictions
        results[algo + "_difference"] = result_set.difference
        results[algo + "_percent"] = result_set.percentage
        print("r2 score:",metrics.r2_score(y,predictions),'\n')
     #   boxplot(results[algo + "_percent"], title="Boxplot (Percentage Error) " +algo+ " Std="+str(round(np.std(results[algo + "_percent"]),2)), y_label='', save=True)
     #   simple_plot(results[algo + "_percent"], "Percentage Error per instance " + algo, save=True)
    #Stage Two process full result set

    #get results > 100% error
    targetted = results[results["Linear Regression_percent"]>100]
    print("Linear Regression targetted test: train with instances showing > 100% error")
    print("Number of instances : ", len(targetted))

    #Fetch instances
    instances = data.iloc[targetted.index]
    #Join instances (for now just having a look to see if there's patterns in percentile error bands)
    targetted = targetted.join(instances)
    selector = RFECV(clf,cv=10)

    #Get subset of targetted indices in dataset
    X = x[targetted.index.values]
    Y = y[targetted.index.values]
    selector.fit(X,Y)
    print('Optimal number of features is ', selector.n_features_)
    clf.fit(X[:,selector.support_],Y)
    predictions = clf.predict(X[:,selector.support_])
    print("r2 score:", metrics.r2_score(Y,predictions))

    file = open('results.html', 'w')
    file.write(targetted.to_html())
    file.close()
    file = open('percentages.html', 'w')
    p = []
    for result in results:
        if result.__contains__('percent'):
            p.append(result)
    file.write(targetted[p].to_html())
    file.close()

'''What main is doing will depend on what needs to be done. Code from here is regularly made into a function call and a new step taken'''
def main():
    algorithm_tests()


'''Load Data and Call Main'''

#Load CSV
data = pd.read_csv("fulldataset.csv", delimiter=';')

#Drop all NaN and non informative days
data = data.dropna(axis=0)
data = data[data['Power Generated'] != 0]
#reset index for dropped rows
data.reset_index(inplace=True, drop=True)
#Grab column names
labels = list(data.columns.values)

#seperate data into target data and feature data
y = data["Power Generated"]
x = data.drop(["Power Generated","Date"],axis=1)
#scale data
scaler = StandardScaler()
x=scaler.fit_transform(x)

clf = linear_model.LinearRegression()
algorithms = {"Linear Regression": linear_model.LinearRegression(), "Ridge Regression" : linear_model.Ridge(), "Lasso Regression" : linear_model.Lasso(),"ElasticNet": linear_model.ElasticNet()}


main()






