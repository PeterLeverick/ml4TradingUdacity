# ML4Trading 
# Peter Leverick Feb 2022 
# p2_numpy.py --> numpy

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
#import google_libs
#import kraken_libs

''' create array, see specifications ''' 
def Create_Array():

    a = np.random.random((5,4))         # 5rows x 4columns  array of random numbers 
    print(a)
    print(f" shape --> {a.shape}")
    print(f" # of rows --> {a.shape[0]}")
    print(f" # of columns --> {a.shape[1]}")
    print(f" # of dimensions --> {len(a.shape)}")
    print(f" # of elements --> {a.size}")
    print(f" data type --> {a.dtype}")

    return a


''' operations with arrays  ''' 
def Arrays_Operations():

    np.random.seed(693)     #seed the random number generator 
    a = np.random.randint(0,10, size =(5,4))         # 5rows x 4columns  integers between 0 and 9
    print(f"\narray --> 5rows x 4columns  integers between 0 and 9 \n{a}")

    #sum all elements 
    print(f"\nsum all alements of the array --> {a.sum()}")

    #sum of each column (compute all rows --> axis = 0)
    print(f"\nsum of each column --> {a.sum(axis=0)}")

    #sum of each row (compute all columns --> axis = 1)
    print(f"\nsum of each row --> {a.sum(axis=1)}")

    #statistics: min, max, mean (across rows, cols, and overall)
    print(f"\nMinimun of each column --> {a.min(axis=0)}")
    print(f"\nMaximun of each row --> {a.max(axis=1)}")
    print(f"\nMean of all elements --> {a.mean()}")

    print(f"\nMaximun value --> {a.max()}")
    print(f"\nIndex Maximun value --> {a.argmax}")  #works with one dimension arrays 

    return a

''' comparete numpy vs manual process to compute the mean of an array  ''' 
def Time_Function():

    arr = np.random.random((1000, 10000))     #large array

    #mean --> manual process
    t1 = time.time()
    sum = 0 
    for i in range(0, arr.shape[0]):
        for j in range (0, arr.shape[1]):
            sum += arr[i,j]
    avg = sum/arr.size
    t2 = time.time()
    print(f"manual mean --> {avg}    time --> {t2-t1} seconds")

    #mean --> numpy
    t1 = time.time()
    avg = arr.mean()
    t2 = time.time()
    print(f"numpy mean --> {avg}    time --> {t2-t1} seconds")

    return

''' accesing array elements   ''' 
def Accesing_Array():

    arr = np.random.random((5, 4))  # rows, columns     
    print(f" Array --> \n{arr}")

    #accesing one element 
    print(f"one element at 3, 2  --> {arr[3,2]}")

    #slicing columns from a row
    print(f"from 1st row columns 2 and 3  --> {arr[0,1:3]}")

    #slicing columns and rows
    print(f"top left corner  --> \n{arr[0:2,0:2]}")

    #slicing n:m:t
    print(f"slicing n:m:t  --> \n{arr[:,0:3:2]}")     # 2 is the jumper

    return


''' assigning values  ''' 
def Assigning_Values_Array():

    arr = np.random.random((5, 4))  # rows, columns     
    print(f" Array --> \n{arr}")

    #assigning value to a particular location
    arr[0,0]=1.11111111 
    print(f"changing 0, 0  --> \n{arr}")

    #assigning a single value to a row
    arr[0,:]=2
    print(f"changing row 0, to 2  --> \n{arr}")

    #assigning a list to a column
    arr[:,3]= [1,2,3,4,5] 
    print(f"asigning a list to a column  --> \n{arr}")

    return


''' indexing an array with another array  ''' 
def Indexing_Array():

    arr = np.random.random(5)  # rows, columns     
    print(f"Array --> \n{arr}")

    #accesing using a list of index in another array 
    indices = np.array([1,1,0,3])     # these are the index we want to access 
    print(f"accesing values through index  --> \n{arr[indices]}")

    return


''' accesing elements   ''' 
def Accesing_Elements():

    arr = np.array([(20,25,10,23,26,32,10,5,0),(0,2,50,20,0,1,28,5,0)])
    print(f"Array --> \n{arr}")

    #calculating mean
    mean = arr.mean()
    print(f"Mean --> {mean}")

    #masking
    print(f"print values < Mean --> {arr[arr<mean]}")

    #replace all values < mean by mean 
    arr[arr<mean] = mean
    print(f"replace values by mean --> \n{arr}")

    return




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

    ''' test ''' 
    a = Create_Array()
    g = input("Create array  .... Press any key : ")

    ''' operations with arrays  ''' 
    a = Arrays_Operations()
    g = input("Arrays Operations  .... Press any key : ")

    ''' using time function  ''' 
    Time_Function()
    g = input("return from time function  .... Press any key : ")

    ''' accesing array elements   ''' 
    Accesing_Array()
    g = input("return from accesing array  .... Press any key : ")

    ''' assigning values  ''' 
    Assigning_Values_Array()
    g = input("return from accesing Assigning_Values_Array  .... Press any key : ")

    ''' indexing an array with another array  ''' 
    Indexing_Array()
    g = input("return from Indexing_Array  .... Press any key : ")

    ''' accesing elements   ''' 
    Accesing_Elements()
    g = input("return from accesing elements  .... Press any key : ")

    return 





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



######### sources 

'''
def Get_OHLCV_720():
    crypto_pair, interval =  google_libs.google_sheet_lib.Import_Export_Files().Get_Params_Mirror_Gsh()
    print(f"return from GSheet lib crypto_pair --> {crypto_pair}")
    print(f"return from GSheet lib interval --> {interval} \n")

    # -- library returns df and creates a csv file (we will use the csv, the lib returns a df anyway))
    asset_df = kraken_libs.kraken_ohlc_lib.main(crypto_pair, interval) 
    print (f"\nlast candle processed --> {asset_df.index[-1]}    Close --> {asset_df['Close'].iloc[-1]}\n")       #last index

    return 


Importing the dataset
def Importing_Dataset():
    dataset = pd.read_csv('./data/ohlcv.csv')
    #print(dataset)
    #g = input("Importing_Dataset  .... Press any key : ")
    return dataset


Preprocessing dataset
    # indexing, slicing, subsetting dataframes 
    # https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html
    # https://datacarpentry.org/python-ecology-lesson/03-index-slice-subset/index.html
    # https://datatofish.com/rows-with-nan-pandas-dataframe/
def Preprocessing_Dataset(dataset):
    
    #-- drop columns 
    #dataset.drop(['Volume'], axis = 1, inplace=True)     # don't drop it if we want volume in the plot 

    #-- check nan 
    print(pd.isnull(dataset).any(axis=0))                       #per columns 
    print("check nan\n")
    #nan_values = dataset[dataset['volume_pct'].isna()]         #select rows with nan 
    #print(nan_values)
    #g = input("pd.isnull(dataset)   .... Press any key : ")

    # -- get parameters for candles/dates subset extraction 
    #date_starts = '2021-09-29 10:15:00'   date_ends = '2021-09-29 15:15:00'
    date_starts, date_ends =  google_libs.google_sheet_lib.Import_Export_Files().Get_Dates_Mirror_Gsh()
    print (date_starts)
    print (date_ends)

    # -- extraxt subset from main df 
    dataset = dataset[dataset.Date.between(date_starts, date_ends)].copy()  #not need an actual datetime-type column

    # -- reset/reorder index after subselection 
    dataset.reset_index(inplace=True,drop=True)
    print(dataset)
    print(f"number of candles --> {len(dataset)}")

    return dataset


 mirroring candles
def Mirroring_Candles(dataset):
    from datetime import datetime
    from datetime import timedelta

    # https://thispointer.com/how-to-add-minutes-to-datetime-in-python/
    # last date in master candles 
    time_str = dataset['Date'].iloc[-1]                         # date in df is an str
    #print(type(time_str))
    print(f"last master candle df' --> {time_str}")
    
    date_format_str = '%Y-%m-%d %H:%M:%S'                       # standard    date_format_str = '%Y-%m-%d %H:%M:%S.%f'
    given_time = datetime.strptime(time_str, date_format_str)   # create datetime object from timestamp string --> to be able to add interval 
    print(f"last master candle given: ', {given_time}")
    interval = 15                                               # in minutes 


    for i in range((len(dataset)-1),-1,-1):     #(a)len=X but 0..X-1, (b)-1 go till -1 to capture index 0, (c)-1 is the regression
        #print(dataset.iloc[i])
         
        if i != len(dataset)-1:                 # 1st round and for is inversed 
            final_time = given_time + timedelta(minutes = interval)
            print(f"proyecting next candle: processing {i}  candle {final_time}")
            # Convert datetime object to string in specific format for the df
            final_time_str = final_time.strftime('%Y-%m-%d %H:%M:%S')

            # ohlc inversed --> open = close, high = low, low = high, close = open
            #add a new row in last position 
            dataset.loc[len(dataset.index)] = [final_time_str, dataset['Close'].iloc[i], dataset['Low'].iloc[i], dataset['High'].iloc[i], dataset['Open'].iloc[i], 0]
            given_time = final_time
 
    return dataset


 plot candles projection 
def Plot_Candles_Projection(dataset):
   
    # -- make Date index for plotting with mplfinance 
    dataset['Date'] = pd.to_datetime(dataset.Date, infer_datetime_format=True)
    dataset.set_index('Date', inplace=True)  
    #print(dataset)

    #print(dataset.index[-1])
    print()

    # --- plot with matplotlib / mplfinance
    # https://github.com/matplotlib/mplfinance/blob/master/markdown/customization_and_styles.md
    import matplotlib.pyplot as plt
    import mplfinance as mpf                                    # pip install mpl_finance
    print(f"mplfinance version --> {mpf.__version__}")
    #print(f"available styles --> {mpf.available_styles()}")

    # -- First we set the kwargs 
    #kwargs = dict(type='candle',mav=(2,4,6),volume=True,figratio=(11,8),figscale=0.85)
    kwargs = dict(type='candle',volume=True,figratio=(11,8),figscale=0.85)

    # -- Plot
    #print(mpf.plot(dataset,type='candle'))
    print(mpf.plot(dataset, **kwargs, style='yahoo'))
    
    return





'''