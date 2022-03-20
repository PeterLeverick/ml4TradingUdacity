# ML4Trading 
# Peter Leverick March 2022 
# p4_missingvalues.py --> Incomplete data 

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


''' fillna 1/forward 2/backward ''' 
def Fill_Missing_Data(df_symbols):
    
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html

    #1. always forward fill first  (we can also fill several columns at the same time)
    df_symbols.fillna(method='ffill', inplace=True)

    #2. backward (we can also fill several columns at the same time)
    df_symbols.fillna(method='bfill', inplace=True)

    return df_symbols





'''-----------------------------------------------------------------'''
'''                 Main Function                                   '''
'''-----------------------------------------------------------------'''
def main():

    ''' Parameters  '''
    start_date = '2021-01-01'
    end_date = '2022-02-28'
    #symbols = ['SPY','GOOG','IBM','GLD']  #'TSLA'
    symbols = ['TSLAmis'] 

    ''' Create a new dataframe that hosts the stocks we want to analyze ''' 
    df_symbols = Create_Mater_Dataframe(symbols, start_date, end_date)
    g = input("return from Create_Mater_Dataframe .... Press any key : ")

    ''' Plot_Data ''' 
    Plot_Data(df_symbols)
    g = input("return from Plot Dataframe .... Press any key : ")

    ''' fillna 1/forward 2/backward ''' 
    df_symbols = Fill_Missing_Data(df_symbols)
    g = input("return from Fill_Missing_Data .... Press any key : ")
  
    ''' Plot_Data ''' 
    Plot_Data(df_symbols)
    g = input("return from Plot Dataframe .... Press any key : ")


    return


  
if __name__== "__main__":
  main()
  g = input("End Program  .... Press any key : "); print (g)



