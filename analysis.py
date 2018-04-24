from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.feature_selection import RFECV
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statistics import mode
from IPython.display import display, HTML
import timeit

class ResultSet():
    def __init__(self, title, predicted, actual, selector = None):
        self.title = title
        self.difference = []
        self.predicted = predicted
        self.actual = actual
        self.percentage = []
        self.score = metrics.r2_score(actual,predicted)
        self.selector = selector
        i = 0
        for act in actual:
            self.difference.append(predicted[i] - act)
            self.percentage.append(((self.difference[i])/(act))*100)
            i+=1




def corr_matrix():
    corr = data.corr()
    map = sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                      square=True)
    map.set_xticklabels(labels, rotation=90)
    map.set_yticklabels(labels, rotation=0)
    sns.set(font_scale=1.5)
    #plt.show()

def boxplot(result, x_label = 'x',y_label = 'y',title= "Figure",save = False):
    sns.boxplot(result)
    plt.xlabel(x_label)
    plt.xticks(np.arange(-1000,2500,step=250))
    plt.xlim([-1000,2500])
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.minorticks_on()
    sns.swarmplot(result, color='orange')
    if save:
        plt.savefig(title+".jpg")
    #plt.show()

def jointplot(feature, comparator,mode = 'reg'):
    sns.jointplot(feature,comparator,kind=mode)
    plt.savefig(mode+'_'+feature.name+'_'+comparator.name+'.jpg')
    #plt.close()

def jointplot_all():
    for label in labels:
        for label_2 in labels:
            if label_2!=label:
                jointplot(data[label],data[label_2])
                jointplot(data[label], data[label_2],'kde')
                #plt.show()

def simple_plot(y_list,title = 'Figure', save = False):
    plt.figure()
    plt.plot(range(0, len(y_list)), y_list, 'ro')
    plt.plot([0, len(y_list)], [0, 0], linewidth=3)
    plt.title(title)
    if save:
        plt.savefig(title + ".jpg")
    #plt.show()

def principal_component_analysis(n):
    pca = PCA(n_components=n)
    data = pca.fit_transform(x)
    predictions = cross_val_predict(clf, data, y, cv=10)
    print(metrics.r2_score(y, predictions))
    simple_plot(predictions,"PCA n="+str(n))

def cv_predict(X,Y,algo):
    selector = RFECV(algorithms[algo], cv=10)
    selector.fit(X, Y)
    predictions = selector.predict(X)
    result_set = ResultSet(algo, predictions, Y, selector)
    return result_set

def algorithm_tests():
    #Stage one get results
    results = []
    rtable = pd.DataFrame()
    for algo in algorithms:
        rset = cv_predict(x,y,algo)
        results.append(rset)
        rtable[algo] = rset.predicted
        rtable[algo+'_difference'] = rset.difference
        rtable[algo+'_percent'] = rset.percentage
        print(algo)
        print('Optimal number of features is ', rset.selector.n_features_)
        print("r2 score:", rset.score, '\n')
        boxplot(rtable[algo + "_percent"], title="Boxplot (Percentage Error) " +algo+ " r2 score="+str(round(rset.score,4)) + " Std="+str(round(np.std(rtable[algo + "_percent"]),2)), y_label='', save=True)
        plt.show()
        
    #get results > 100% error
    targetted = rtable[rtable["Linear Regression_percent"]>90]
    print("Linear Regression targetted test: train with instances showing > 90% error")
    print("Number of instances : ", len(targetted))

    #Fetch instances
    instances = data.iloc[targetted.index]
    #Join instances (for now just having a look to see if there's patterns in percentile error bands)
    targetted = targetted.join(instances)
    selector = RFECV(clf,cv=10)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nFor isolated target instances\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for column in labels:
        print(column+" : \n\t","max="+str(max(targetted[column])),"\t\t\t\t\tmin="+str(min(targetted[column])),"\t\t\t\t\tmean="+str(np.mean(targetted[column])),"\t\t\t\t\tstd="+str(np.std(targetted[column])))
    #Get subset of targetted indices in dataset
    X = x[targetted.index.values]
    Y = y[targetted.index.values]
    rset = cv_predict(X,Y,"Linear Regression")
    boxplot(rset.percentage, title="Boxplot (Percentage Error) Overfitted Linear Regression model Std = "+str(round(np.std(rset.percentage),2)), y_label='', save=True)
    print("r2 score: ",rset.score)
    file = open('results.html', 'w')
    file.write(targetted.to_html())
    file.close()
    file = open('percentages.html', 'w')
    p = []
    for result in rtable:
        if result.__contains__('percent'):
            p.append(result)
    file.write(targetted[p].to_html())
    file.close()


def knr_fit_predict(knr,instance):
    # Get the index array for nearest neighbours to instance
    neighbors = (knr.kneighbors(instance.reshape(1, -1))[1][0])
    # exclude the 0th value, which is the instance itself in our data
    neighbors = neighbors[1:]
    # array index training data
    X = x[neighbors]
    Y = y[neighbors]
    # fit model
    rfecv = RFECV(clf)
    clf.fit(X,Y)
    return clf.predict(instance.reshape(1, -1))

'''Performs cross validationa across all instances in the data set using the n Nearest Neighbours'''
def param_KNearest_CV(n=178, weight="distance"):
    results = []
    knr = KNeighborsRegressor(n_neighbors=n, weights=weight)
    knr.fit(x, y)
    for instance in x:
        #add prediction to list of results
        results.append(knr_fit_predict(knr,instance)[0])
    #make result set
    rs = ResultSet("Linear Regression", results,y)
    return rs

def hpo_knn():
    results = []
    best = None
    best_n = 0
    #Distance weighting outperforms uniform
    weight = "distance"
    best_w = ""
    range_n = range(160 ,200,1)
    print("n values", list(range_n))
    for n in range_n:
        rs = param_KNearest_CV(n,weight)
        if best is None:
            print("score=", rs.score, "weight=", weight, "n=", n)
            best = rs
            best_n = n
            best_w = weight
        else:
            if best.score < rs.score:
                print("New best : score=",rs.score,"weight=",weight,"n=",n)
                best = rs
                best_n = n
                best_w = weight
        results.append(rs.score)
    sns.pointplot(x=list(range_n),y=results)
    plt.show()
    boxplot(best.percentage,title="Best KNN n="+str(best_n)+" w=" +str(best_w)+" score="+str(round(best.score,3))+" std="+str(round(np.std(best.percentage),2)))
    plt.show()

def timeit_call():
    knr = KNeighborsRegressor(n_neighbors=178, weights="distance")
    knr.fit(x, y)
    knr_fit_predict(knr,x[240])

def main():
    #Tests algorithms
    #algorithm_tests()

    #Time taken for single prediction
    #print("execution time",timeit.timeit(timeit_call,number=1))

    #Hyper parameter optimization of KNN similar day fitting for K = number of similar days used
    #hpo_knn()


    #KNN Cross Validation
    rs = param_KNearest_CV()
    boxplot(rs.percentage,title="KNN Trained Linear Regression r2 score="+str(round(rs.score,4)) + " Std="+str(round(np.std(rs.percentage),2)))
    plt.show()
    print(rs.score)


'''Load Data and Call Main'''

#Load CSV
data = pd.read_csv("fulldataset.csv", delimiter=';')

#Drop all NaN and non informative days
data = data.dropna(axis=0)
data = data[data['Power Generated'] != 0]
#reset index for dropped rows
data.reset_index(inplace=True, drop=True)


#Grab column names
labels = list(data.drop("Date",axis=1).columns.values)

#seperate data into target data and feature data
y = data["Power Generated"]
x = data.drop(["Power Generated","Date"],axis=1)


#scale data
scaler = StandardScaler()
x=scaler.fit_transform(x)

clf = linear_model.LinearRegression()


algorithms = {"Linear Regression": linear_model.LinearRegression() , "Ridge Regression" : linear_model.Ridge(), "Lasso Regression" : linear_model.Lasso(),"ElasticNet": linear_model.ElasticNet()}


main()






