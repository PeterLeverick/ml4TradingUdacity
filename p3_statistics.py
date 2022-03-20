# ML4Trading 
# Peter Leverick March 2022 
# p1_statistics.py --> Statistical analysis on time series 

# Importing the libraries
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

    # create a temporary df with selected dates 
    dates=pd.date_range(start_date,end_date)
    print(dates[0], dates[-1])

    # create master df,  with dates as index
    df_symbols = pd.DataFrame(index=dates)
    print(df_symbols)
    print("master df empty, only index\n")

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


''' compute global statistics for each stock ''' 
def Global_Statistics(df_symbols):
    
    print (f"mean --> \n{df_symbols.mean()}")
    print (f"median --> \n{df_symbols.median()}")
    print (f"standard deviation --> \n{df_symbols.std()}")

    return 


''' compute rolling statistics  ''' 
def Rolling_Statistics(df_symbols):
    
    symbol = 'SPY'
    df = df_symbols[[symbol]].copy()
    print(f"\n symbol df --> \n{df}")

    # --- compute SMA
    #plot data
    ax = df[symbol].plot(title="SPY rolling mean", label='SPY')
    
    #compute rolling mean 
    window=20
    sma = df['SPY'].rolling(window).mean()
    
    # add rolling mean to same plot 
    sma.plot(label='Rolling mean', ax=ax)
    
    #plotting 
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()

    # ---- Compute Bollinger Bands 
    # 1.Compute rolling mean
    df['sma']=df['SPY'].rolling(window).mean()

    # 2. compute rolling standard deviation 
    df['std']=df['SPY'].rolling(window).std()

    # 3. Compute upper and lower bands 
    df['upper']=df['SPY'] + (df['std'] * 2)
    df['lower']=df['SPY'] - (df['std'] * 2)

    # Plot closing prices, SMA, bollinger bands
    ax = df['SPY'].plot(title="Bollinger Bands", label='SPY')
    df['sma'].plot(label='SMA', ax=ax)
    df['upper'].plot(label='Upper Band', ax=ax)
    df['lower'].plot(label='Lower Band', ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()
    
    # 1 ---- Compute Daily returns  
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

    # Daily returns mean 
    print(f"daily returns mean --> {df['dayret'].mean()}")

    # Plot closing prices, daily returns 
    ax = df['dayret'].plot(title="Returns", label='Daily Returns')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()


    # 2 ---- Compute Cumulative returns 
    # 2.1 method 1 - equation 
    df.loc[df.index[1:],'cumret'] = (df.loc[df.index[1:],'SPY']/df.loc[df.index[0],'SPY']) - 1
    df.loc[df.index[0],'cumret'] = 0        #1st row = 0 we cannot compute it to replace NaN 

    print(f"df -->\n{df}") 

    #Cumulative returns (in this case from the beginning (index 1) to the end)
    print(f"Cumulative returns --> {(df.loc[df.index[-1],'SPY']/df.loc[df.index[0],'SPY']) - 1}")

    # Plot closing prices, daily returns 
    ax = df['cumret'].plot(title="Returns", label='Cumulative Returns')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()


    return 



'''-----------------------------------------------------------------'''
'''                 Main Function                                   '''
'''-----------------------------------------------------------------'''
def main():

    ''' Parameters  '''
    start_date = '2021-01-01'
    end_date = '2022-02-28'
    symbols = ['SPY','GOOG','IBM','GLD']  #'TSLA'

    ''' Create a new dataframe that hosts the stocks we want to analyze ''' 
    df_symbols = Create_Mater_Dataframe(symbols, start_date, end_date)
    g = input("return from Create_Mater_Dataframe .... Press any key : ")

    ''' Plot_Data ''' 
    Plot_Data(df_symbols)
    g = input("return from Plot Dataframe .... Press any key : ")

    ''' Global Statistics ''' 
    Global_Statistics(df_symbols)
    g = input("return from Global Statistics .... Press any key : ")

    ''' Rolling Statistics ''' 
    Rolling_Statistics(df_symbols)
    g = input("return from Rolling Statistics .... Press any key : ")

    return


  
if __name__== "__main__":
  main()
  g = input("End Program  .... Press any key : "); print (g)



