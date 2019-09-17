#setting some starting
#and end dates for data that will be fetched
import datetime as dt
#creates new directories
import os
import pickle
#pickle serializes any python objects
#i did this in order to save the sp500 list
from collections import Counter
from statistics import mean

import bs4 as bs
#plt utilizes pyplot in order to create plots,charts graphs etc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# grab data from Yahoo Finance API
import pandas_datareader.data as web
import requests
import warnings
warnings.filterwarnings('ignore')
from matplotlib import style
#scikit learn contains all the classifiers
#Creating training and testing samples
from sklearn import model_selection
# model selection
from sklearn import svm, neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression


style.use('ggplot')
#pickle serializes any python objects
#i did this in order to save the sp500 list
def save_stock_symbols():
    rp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    # soup is a  beautiful soup object

    obg_sp = bs.BeautifulSoup(rp.text, 'lxml')
    #response is the text of the source code
    #lxml is the parser
    #to use beautiful soup in order to find certain things
    #i used soup.find
    #
    stock_table = obg_sp.find('table', {'class': 'wikitable sortable'})
    #specifying an empty tickers list
    symbols = []
    #iterating through this table
    #for each tr is a table row
    #fetching the first row of
    for row in stock_table.findAll('tr')[1:]:
        # .text because it is a soup object
        stock_smbl = row.findAll('td')[1].text
        symbols.append(stock_smbl)
    with open("stocksymbols.pickle", "wb") as i:
        pickle.dump(symbols, i)

        print(symbols)
        # wb = write bytes
    return symbols

#save_stock_symbols()

# save_sp500_tickers()
#it was defined as false because
#function will be invoked from this one
def fetch_stock_data(stock_loading=False):
    # geting my list
    if stock_loading:
        symbol_stocks = save_stock_symbols()
    else:
        # rb = readbytes
        with open("stocksymbols.pickle", "rb") as i:
            symbol_stocks = pickle.load(i)
            # convert all the stock data in a csv file

            # if this directory does not exist localy

    if not os.path.exists('all_stocks'):
        # then create it
        os.makedirs('all_stocks')

    dt_start = dt.datetime(2010, 1, 1)
    dt_end = dt.datetime(2018, 12, 31)
    for symbol in symbol_stocks[:70]:
        print (symbol)
        if not os.path.exists('all_stocks/{}.csv'.format(symbol)):
            datafr = web.DataReader(symbol, 'yahoo', dt_start, dt_end)

            datafr.to_csv('all_stocks/{}.csv'.format(symbol))
        else:
            print('Exist already {}'.format(symbol))



#fetch_stock_data()

def stock_compiling():
    with open("stocksymbols.pickle", "rb") as j:
        symbol_stocks = pickle.load(j)

    # a dataframe object that has no columns or indexes etc
    df_main = pd.DataFrame()
    # reads only 70
    for count, stock_symbol in enumerate(symbol_stocks[:70]):
        datafr = pd.read_csv('all_stocks/{}.csv'.format(stock_symbol))
        #In this case
        #we take date as an Index
        #
        datafr.set_index('Date', inplace=True)
        # rename some columns
        # renaming the Adjusted close column in to a stock_symbol
        # That column in this case will be the stock price or the Adj.close for that stock

        #w
        datafr.rename(columns={'Adj Close': stock_symbol}, inplace=True)

        #droping overlapping columns
        datafr.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
        # start joining all the data frames
#######
        if df_main.empty:
            df_main = datafr
        else:
            ####avoid losing data
            df_main = df_main.join(datafr, how='outer')
        ##10 20 30 40 etc

    print(df_main.head())
    #create file
    df_main.to_csv('seventy_Adjusted_Closes_joined.csv')

#stock_compiling()


def stock_processing(stock):
    # how many days in the future do we have we earn or we lose X percent
    # in the next seven days it will go either up or down by 2 percent
    day_interval = 7
    datafr = pd.read_csv('seventy_Adjusted_Closes_joined.csv', index_col=0)
    #####
    stocks_ = datafr.columns.values.tolist()

    datafr.fillna(0, inplace=True)
    for i in range(1, day_interval+1):
        # print (i)
        #processing the actual data and create future values
        datafr['{}_{}day'.format(stock, i)] = (datafr[stock].shift(-i) - datafr[stock]) / datafr[stock]
    datafr.fillna(0, inplace=True)
    return stocks_, datafr
#process_data_for_labels ('ADM')


#function that creates our labels
   #args let us passes any number of parameters
#y_train = buy,hold,sell
def decision(*args):
    columns = [i for i in args]
    #it will go basically row
    #by row
    #0.6 will be for example 60 percent
    #in this case is 2 percent
    #if the stock prices changes by
    #2 percent in seven days
    percentage = 0.02
    for column in columns:
        #if more than perc it is a buy
        if column > percentage:
            return 1
        #if negative perc it is a sell
        if column < -percentage:
            return -1
        #if nothing hold
    return 0




#
#map hold buy or sell to our dataframe
def feautureset(stock):
    stocks, datafr = stock_processing(stock)

    datafr['{}_stocktrgt'.format(stock)] = list(map( decision, datafr['{}_1day'.format(stock)],datafr['{}_2day'.format(stock)],datafr['{}_3day'.format(stock)],datafr['{}_4day'.format(stock)],
                                                      datafr['{}_5day'.format(stock)], datafr['{}_6day'.format(stock)], datafr['{}_7day'.format(stock)] ))
                                    #sevenday percent changes for prices in the future

#creating a new column and generate
    #the decision buy sell or hold




    values = datafr['{}_stocktrgt'.format(stock)].values.tolist()
    ####
    values_of_string = [str(j) for j in values]
    #gives us the distribution
    #in fact how data were spreaded
    print('Overall Stock Distribution:', Counter(values_of_string))

    datafr.fillna(0, inplace=True)
    datafr = datafr.replace([np.inf, -np.inf], np.nan)
    datafr.dropna(inplace=True)
    # X_train data
    #This represent the percent change data for all the stocks
    #including the company i will search

    #pct_change is for normalizing data
    dataframe_values = datafr[[stock for stock in stocks]].pct_change()
    # Creating feauture sets and the labels
    dataframe_values = dataframe_values.replace([np.inf, -np.inf], 0)
    dataframe_values.fillna(0, inplace=True)
    # X is future sets price changes, daily percent changes is the percent change data for all of the companies
    # y is label (1,0,-1)
    X = dataframe_values.values
    y = datafr['{}_stocktrgt'.format(stock)].values
    #print(df.head(5))



    #ax1.plot(dataframe_values.values)
    ####
    #ax2.plot(datafr['{}_stocktrgt'.format(stock)])

    #plt.show()

    return X, y, datafr

#X is future set
#y represents labels
#df dataframe


def apply_machine_learning(stock):
    X, y, datafr = feautureset(stock)
    # 25 percent represents the test data of our sample data
    #and 75 percent the training data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.25)



    algorithm = neighbors.KNeighborsClassifier()

    #clf = VotingClassifier ([('lsvs',svm.LinearSVC()),
                           #  ('knn',neighbors.KNeighborsClassifier()),
                            # ('rfor',RandomForestClassifier())])

    #y



    algorithm = neighbors.KNeighborsClassifier()
    #fit is similar to train
    algorithm.fit(X_train, y_train)
    #confidence is the acc
    #score evaluates the model performance
    Evaluation = algorithm.score(X_test, y_test)


    Prediction = algorithm.predict(X_test)
    print('Decision Prediction:', Counter(Prediction))




    return Evaluation
#average for each stock
with open("stocksymbols.pickle","rb") as i:
    stock_symbols = pickle.load(i)

accurs = []
for count,stock in enumerate(stock_symbols[:70]):

    #####

    accur = apply_machine_learning(stock)
    accurs.append(accur)
    print("{} Stock Accuracy: {}.\nAverage accuracy:{}".format(stock,accur,mean(accurs)))
    print()
    print()




#sum of average of all stocks
print('Sum of 70 stocks average accuracy using neighbors.KNeighborsClassifier')
print(sum(accurs)/70)













##########################






style.use('ggplot')

df = pd.read_csv("sum_stocks_accuracy.csv")
print(df)

#city = ['KNN', 'RFC', 'SVC']
#pos = np.arange(len(city))
#Happiness_Index = [0.393241, 0.413241, 0.333241]

#plt.barh(pos, Happiness_Index, color='cyan', edgecolor='black')
#plt.yticks(pos, city)
#plt.xlabel('Sum Stock Accuracy', fontsize=16)
#plt.ylabel('Algorithms', fontsize=16)
#plt.title('Algorithms Results', fontsize=20)
#plt.show()

objects = ('K-NEIGHBORS ', 'RANDOM FOREST', 'LINEAR DISCRIMINANT ANALYSIS',
           'DECISION TREE','PASSIVE AGGRESIVE','EXTRA TREES')
y_pos = np.arange(len(objects))
performance = [38.61, 41.83, 41.03,39.46,40.25,40.68]

plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Accuracy')
plt.title('Algorithm Used')

plt.show()
df.set_index()[['KNN','RFC', 'SVC',]].plot.bar()
plt.show()















































