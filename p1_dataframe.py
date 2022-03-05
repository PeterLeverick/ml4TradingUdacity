# ML4Trading 
# Peter Leverick Feb 2022 
# p1_dataframe.py --> Dataframes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    
    #print(f"\n df dtypes --> {df.dtypes}")
    #print(df)
    #print()
    #print (df[5:8])
    #print('symbol dataframe indexed by date\n')
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
        df_symbols = df_symbols.join(df_new_symbol, how='inner')         # inner --> join only common dates to both dataframes 
        #df1 = df1.dropna()                                             

    print (df_symbols)
    print ('master df after join\n')

    return df_symbols


''' Slicing ''' 
def Slicing(df_symbols):
    
    # get some rows
    df2 = df_symbols.loc['2022-01-01':'2022-02-25']
    print(df2)

    #get some columns 
    df2 = df_symbols.loc[:, ['IBM', 'GLD']]     # all columns for IBM & GLD
    print(df2)
    
    #get some rows of some columns  
    df2 = df_symbols.loc['2022-01-15':, ['IBM', 'GLD']]     # all columns for IBM & GLD
    print(df2)

    return 


''' Normalize_Data_Symbols ''' 
def Normalize_Data_Symbols(df_symbols):

    # #divide all rows by 1st, all stoks will start as 1 (row 1 / row 1)
    # this will allow to see the evolution of each stock compared weith day one
    # it will allow also to compare the evlotuion of the stock even if the have different prices 
    return df_symbols / df_symbols.iloc[0,:]  


''' Plot_Data ''' 
def Plot_Data(df_symbols):
    
    # all symbols in df
    ax = df_symbols.plot(title='plot testing')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()                  #must be called to show plots in some environments 

    # only two symbols
    #df_symbols[['IBM', 'GLD']].plot()
    df_symbols.loc['2022-01-10':, ['IBM', 'GLD']].plot()
    plt.show()

    return 


''' Get max close of a symbol ''' 
def Get_Max_Close(df, symbol):
    #print(f" Max Close {symbol} --> {df['Close'].max()}")
    return df.loc[:,'TSLA'].max()

''' Get mean of a symbol ''' 
def Get_Mean_Close(df, symbol):
    return df['TSLA'].mean()

''' Plot a symbol ''' 
def Plot_Symbol(df, symbol, column):
    df[column].plot()
    plt.show()
    return 

''' Plot two columns of a symbol ''' 
def Plot_Columns_Symbol(df, symbol, col1, col2):
    df[[col1, col2]].plot()
    plt.show()
    return 




'''-----------------------------------------------------------------'''
'''                 Main Function                                   '''
'''-----------------------------------------------------------------'''
def main():

    ''' Import symbols  '''
    start_date = '2022-01-01'
    end_date = '2022-01-21'
    symbols = ['SPY','GOOG','IBM','GLD']  #'TSLA'

    ''' Create a new dataframe that will host the stocks we want to analyze ''' 
    df_symbols = Create_Mater_Dataframe(symbols, start_date, end_date)
    g = input("Create_Mater_Dataframe .... Press any key : ")

    ''' Slicing tests ''' 
    Slicing(df_symbols)     

    ''' Normalize_Data_Symbols ''' 
    df_symbols_norm = Normalize_Data_Symbols(df_symbols)

    ''' Plot normalized symbols vales ''' 
    Plot_Data(df_symbols_norm)


    #------ other tests 
    symbol = 'TSLA'
    ''' import csv  '''
    df = Import_CSV(symbol)
    g = input("Import_CSV  .... Press any key : ")

    ''' get max_close symbol  '''
    max_close = Get_Max_Close(df, symbol)
    print (f"\nmax close {symbol} --> {max_close}")
    g = input("Get_Max_Close  .... Press any key : ")

    ''' get mean symbol  '''
    mean_close = Get_Mean_Close(df, symbol)
    print (f"\nmean close {symbol} --> {mean_close}")
    g = input("Get_Mean_Close  .... Press any key : ")

    ''' plot a symbol by selecting a column  '''
    column = symbol
    Plot_Symbol(df, symbol, column)
    g = input("plot a symbol  .... Press any key : ")

    ''' Plot two columns of a symbol ''' 
    col1 = symbol
    col2 = symbol
    Plot_Columns_Symbol(df, symbol, col1, col2)
    g = input("Plot_Columns_Symbol  .... Press any key : ")


    return 


  
if __name__== "__main__":
  main()
  g = input("End Program  .... Press any key : "); print (g)



