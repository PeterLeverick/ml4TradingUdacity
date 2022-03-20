# ML4Trading 
# Peter Leverick March 2022 
# p6_sharp.py --> Sharpe ratio and other portfolio statistics  

# Importing the libraries
from turtle import forward
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
#import google_libs
#import kraken_libs

''' Importing CSV ''' 
def Import_CSV(symbol):

    # set date as index, as we'll join it in the master df   
    # parse_dates --> to convert dates to daytime index objects
    # usecols --> we only pick the columns we are interested in (need of Date for the join)
    # na_values --> to indicate that nan is nan and not an string (csv default)
    df = pd.read_csv(f"./data/{symbol}.csv", index_col="Date", 
                    parse_dates = True, usecols=['Date', 'Adj Close'],
                    na_values=['nan'])

    df.rename(columns = {'Adj Close':symbol}, inplace=True)         # change column name to symbol 
    df[symbol]=pd.to_numeric(df[symbol])                            # make Adj Close float (default str)
    
    return df


''' Create a new dataframe that will host a timeframr of the stocks we want to analyze ''' 
def Create_Mater_Dataframe(symbols, start_date, end_date):

    #if "SPY" not in symbols: symbols.insert(0,"SPY")

    # create a temporary df with selected dates 
    dates=pd.date_range(start_date,end_date)
    print(dates[0], dates[-1])

    # create master df,  with dates as index
    df_symbols = pd.DataFrame(index=dates)
    #print(df_symbols)
    #print("master df empty, only index\n")

    for symbol in symbols:
        df_new_symbol = Import_CSV(symbol)
        
        # join each symbol to the master df
        # make sure that df has date as index otherwise we will only het NaN (no common index)
        df_symbols = df_symbols.join(df_new_symbol, how='inner')        # inner --> join only common dates to both dataframes 
        #df_symbols = df_symbols.dropna()                                             # do we need it? join made the job?                                         

    print (df_symbols)
    print ('master df after join\n')
    print (f"describe symbols \n{df_symbols.describe()}")
    print (f"check for NaN --> {df_symbols.isnull().sum().sum()}")

    return df_symbols


''' Plot_Data ''' 
def Plot_Data(df_symbols):
    
    # all symbols in df
    ax = df_symbols.plot(title='plot testing')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()                  #must be called to show plots in some environments 

    # only two symbols
    #df_symbols[['IBM', 'GLD']].plot()
    #df_symbols.loc['2022-01-10':, ['IBM', 'GLD']].plot()
    # plt.show()

    return 


''' Compute Daily Returns ''' 
def Daily_Returns(df):
 
    print(f"df -->\n{df}")

    # create a new column with daily returns
    # daily returns are implicitly normallized 

    #method 1 --> equation 
    # we need to use .values, if no pandas uses index and we get wrong results 
    # loc uses index, we cannot simple put numbers (or index range, or call .index) 
    # df.loc[df.index[1:],'dayret'] = (df.loc[df.index[1:],'SPY']/df.loc[df.index[:-1],'SPY'].values) - 1
    # df.loc[df.index[0],'dayret'] = 0        #1st row = 0 we cannot compute itto replace NaN  

    #method 2 --> pandas 
    # compute daily returns using pandas instead equation 
    df['dayret'] = (df['SPY']/df['SPY'].shift(1)) -1  
    df.loc[df.index[0],'dayret'] = 0        #1st row = 0 we cannot compute it 
    
    print(f"df -->\n{df}") 

    # Plot closing prices, daily returns 
    ax = df['dayret'].plot(title="Returns", label='Daily Returns')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()

    return df


''' Daily returns histogram ''' 
def Daily_Returns_Hist(df):
    
    # compute mean
    my_mean = df['dayret'].mean()
    print(f"\n\ndaily returns mean --> {my_mean}")

    # compute sdt
    my_std = df['dayret'].std()
    print(f"daily returns standard deviation --> {my_std}")

    # compute kurtosis
    # Kurtosis --> tells us about the tails of the distribution 
    # positive kurtosis --> (fat tails) bigger tail in the normal gaussian distribution
    # negative kurtosis --> (skinny tails) less tail than in a normal gaussian distribution
    my_kurtosis = df['dayret'].kurtosis()
    print(f"daily returns kurtosis (if + fat tails), (if - skinny tails) --> {my_kurtosis}")


    # Histogram
    # bins --> number of bars, by default 10
    ax = df['dayret'].hist(bins = 20)
    ax.set_xlabel("Daily Returns")
    ax.set_ylabel("number of values for each bin")
    #ax.legend(loc='upper left')
    plt.axvline(my_mean, color='w',linestyle='dashed',linewidth=2)
    plt.axvline(my_std, color='r',linestyle='dashed',linewidth=2)
    plt.axvline(-my_std, color='r',linestyle='dashed',linewidth=2)

    plt.show()

    return


''' Daily returns two histograms together ''' 
def Daily_Returns_Two_Hist(df):
    
    print(f"df -->\n{df}")

    # create a new column with daily returns
    # daily returns are implicitly normallized 

    #method 1 --> equation 
    # we need to use .values, if no pandas uses index and we get wrong results 
    # loc uses index, we cannot simple put numbers (or index range, or call .index) 
    # df.loc[df.index[1:],'dayret'] = (df.loc[df.index[1:],'SPY']/df.loc[df.index[:-1],'SPY'].values) - 1
    # df.loc[df.index[0],'dayret'] = 0        #1st row = 0 we cannot compute itto replace NaN  

    #method 2 --> pandas 
    # compute daily returns using pandas instead equation 
    df['dayret1'] = (df['SPY']/df['SPY'].shift(1)) -1  
    df.loc[df.index[0],'dayret1'] = 0        #1st row = 0 we cannot compute it 

    df['dayret2'] = (df['IBM']/df['IBM'].shift(1)) -1  
    df.loc[df.index[0],'dayret2'] = 0        #1st row = 0 we cannot compute it 
    
    print(f"df -->\n{df}") 

    # Plot daily returns 
    ax = df['dayret1'].plot(title="Returns", label='SPY')
    df['dayret1'].plot(label='IBM', ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()

    # Histogram -> 2 subplots
    # bins --> number of bars, by default 10
    df[['dayret1', 'dayret2']].hist(bins = 20)
    #ax.set_xlabel("Daily Returns")
    #ax.set_ylabel("number of values for each bin")
    plt.show()

    # Histogram -> 2 together 
    df['dayret2'].hist(bins = 20, label='IBM')
    df['dayret1'].hist(bins = 20, label='SPY')
    plt.legend(loc='upper right')
    plt.show()

    return


''' Create a new df with only the Daily Returns ''' 
def Daily_Returns2(df):
    
    # slope = Beta --> 
    #   how reactive is the stock to the market, 
    #   if beta = 1, each time the market goes up 1% this stock goes up 1%
    #   if the slope/beta = 2, when the marker (SPY) goes up 1%, this stock goes up 2% 
    #   it can be the market or other stock 
    # Alpha --> 
    #   where the regresion line incerpets the vertical line at 0 X (returns)
    #   if alpha is positive (y > 0) the stock performs better than the market or other stock
    #   if alpha is negative (y < 0) the stock performs worse than the maket or th other stock we campe with 
    # Slope if not Correlation 
    #   When slope is 1, doesn't mean the stocks are correlated (correlation=1)
    #   The slope is the slope
    #   Correlation --> is how tight the dots (scatter plot) fit the line  
    #   We can have slope < 1 but highly correlated (dots tight together to regression)
    #   We can have slope > 1 but not correlated (dot spread far from the regression)
    #   Correlation values --> 0 = not correlated at all, 1 = very correlated
 
    # 1/ --------- Daily returns 
    daily_returns = df.copy()       # it does not make date as index 
    print(f"df -->\n{daily_returns}")

    #method 1 --> equation 
    # we need to use .values, if no pandas uses index and we get wrong results 
    # loc uses index, we cannot simple put numbers (or index range, or call .index) 
    # in this case we do not need to use .index as date is not index
    # daily_returns.loc[df.index[1:] = (df.loc[df.index[1:],'SPY']/df.loc[df.index[:-1],'SPY'].values) - 1
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    daily_returns.iloc[0, :] = 0        # set daily returns for row 0 to 0   

    #method 2 --> pandas 
    # compute daily returns using pandas instead equation 
    #df['dayret'] = (df['SPY']/df['SPY'].shift(1)) -1  
    #df.loc[df.index[0],'dayret'] = 0        #1st row = 0 we cannot compute it 
    
    print(f"daily_returns -->\n{daily_returns}") 

    # Plot daily returns 
    ax = daily_returns.plot(title="Returns", label='Daily Returns')
    ax.set_xlabel("Date")
    ax.set_ylabel("Returns")
    ax.legend(loc='upper left')
    plt.show()

    # 2/ --------- Beta and Alpha
    # Scatterplot daily returns (we can only two stocks x and y ) 

    # 2.1/ SPY vs IBM
    daily_returns.plot(kind='scatter',x='SPY', y='IBM') # --> relatively correlated 
    # 1 = the degree of our function --> y = mx + b (m = coeficient, b = intercept)
    # this returns the polinomial coeficient, the intercept 
    beta_IBM, alpha_IBM = np.polyfit(daily_returns['SPY'], daily_returns['IBM'],1)

    # for every value of x which is 'SPY' we find the value of y by the equation y = mx + b
    # y = m (beta, the coeficient) * x (daily_returns['SPY']) + b (alpha, the intercept)
    plt.plot(daily_returns['SPY'], beta_IBM*daily_returns['SPY'] + alpha_IBM, '-', color='r')
    plt.show()

    print(f"\nbeta IBM --> {beta_IBM}")     # 0.47 IBM more reactive to the market than GLD
    print(f"alpha IBM --> {alpha_IBM}")     # 0.00030

    # 2.2/ SPY vs GLD
    daily_returns.plot(kind='scatter',x='SPY', y='GLD') # --->  no correlation
    beta_GLD, alpha_GLD = np.polyfit(daily_returns['SPY'], daily_returns['GLD'],1)
    plt.plot(daily_returns['SPY'], beta_GLD*daily_returns['SPY'] + alpha_GLD, '-', color='r')
    plt.show()

    print(f"\nbeta GLD --> {beta_GLD}")     # 0.10
    print(f"alpha GLD --> {alpha_GLD}")     # 0,00032 GLD perform a bit better than IBM

    # 3/ --------- Correlation
    print(f"\ncorrelation pearson --> \n{daily_returns.corr(method='pearson')}")
    # correlation goes between 0 and 1 --> (0 = no correlation, 1= perfect correlation)
    # SPY/IBM = 0.31    --> correlation not to strong (0 = no correlation, 1= perfect correlation)
    # SPY/GLD = 0.11    --> very little correlation
    # IBM/GLD = 0.07    --> almost no correlation



    return daily_returns


'''-----------------------------------------------------------------'''
'''                 Main Function                                   '''
'''-----------------------------------------------------------------'''
def main():

    #------ 1
    print(f"\n\n---------------------- 1 plot histogram ")

    ''' Parameters  '''
    start_date = '2021-01-01'
    end_date = '2022-02-28'
    #symbols = ['SPY','GOOG','IBM','GLD']  #'TSLA'
    symbols = ['SPY'] 

    ''' Create a new dataframe that hosts the stocks we want to analyze ''' 
    df_symbols = Create_Mater_Dataframe(symbols, start_date, end_date)
    g = input("return from Create_Mater_Dataframe .... Press any key : ")

    ''' Plot_Data ''' 
    Plot_Data(df_symbols)
    g = input("return from Plot Dataframe .... Press any key : ")

    ''' Compute Daily Returns ''' 
    df_symbols = Daily_Returns(df_symbols)
    g = input("return from Compute Daily Returns .... Press any key : ")

    ''' Daily returns histogram ''' 
    Daily_Returns_Hist(df_symbols)
    g = input("return from Daily_Returns_Hist .... Press any key : ")

    
    #------ 2
    print(f"\n\n---------------------- 2 Plot two histograms together  ")  

    ''' Parameters  '''
    start_date = '2021-01-01'
    end_date = '2022-02-28'
    symbols = ['SPY','IBM'] 

    ''' Create a new dataframe that hosts the stocks we want to analyze ''' 
    df_symbols = Create_Mater_Dataframe(symbols, start_date, end_date)
    g = input("return from Create_Mater_Dataframe .... Press any key : ")

    ''' Plot_Data ''' 
    Plot_Data(df_symbols)
    g = input("return from Plot Dataframe .... Press any key : ")

    ''' Plot_Data ''' 
    Daily_Returns_Two_Hist(df_symbols)
    g = input("return from Daily_Returns_Two_Hist .... Press any key : ")

    #------ 3
    print(f"\n\n---------------------- 3 Scatter plots, slope, beta, correlation  ")  

    ''' Parameters  '''
    start_date = '2021-01-01'
    end_date = '2022-02-28'
    symbols = ['SPY','IBM','GLD'] 

    ''' Create a new dataframe that hosts the stocks we want to analyze ''' 
    df_symbols = Create_Mater_Dataframe(symbols, start_date, end_date)
    g = input("return from Create_Mater_Dataframe .... Press any key : ")

    ''' Plot_Data ''' 
    Plot_Data(df_symbols)
    g = input("return from Plot Dataframe .... Press any key : ")

    ''' Daily_Returns2 create a new df for the daily_returns  ''' 
    df_dret = Daily_Returns2(df_symbols)
    g = input("return from Daily_Returns2 .... Press any key : ")

    return 


    

    return



  
if __name__== "__main__":
  main()
  g = input("End Program  .... Press any key : "); print (g)



