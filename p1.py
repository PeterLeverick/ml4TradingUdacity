# Candles mirroring 
# Peter Leverick Sept-Oct 2021 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import google_libs
import kraken_libs

''' refresh ohlcv.csv <-- get params from GSheet & ohlc from kraken '''
''' these are candles and return the last 720 points '''
def Get_OHLCV_720():

    crypto_pair, interval =  google_libs.google_sheet_lib.Import_Export_Files().Get_Params_Mirror_Gsh()
    print(f"return from GSheet lib crypto_pair --> {crypto_pair}")
    print(f"return from GSheet lib interval --> {interval} \n")

    # -- library returns df and creates a csv file (we will use the csv, the lib returns a df anyway))
    asset_df = kraken_libs.kraken_ohlc_lib.main(crypto_pair, interval) 
    print (f"\nlast candle processed --> {asset_df.index[-1]}    Close --> {asset_df['Close'].iloc[-1]}\n")       #last index

    return 


''' Importing the dataset ''' 
def Importing_Dataset():
    dataset = pd.read_csv('./data/ohlcv.csv')
    #print(dataset)
    #g = input("Importing_Dataset  .... Press any key : ")
    return dataset


''' Preprocessing dataset ''' 
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


''' mirroring candles '''  
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


''' plot candles projection '''  
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




'''-----------------------------------------------------------------'''
'''                 Main Function                                   '''
'''-----------------------------------------------------------------'''
def main():
    
    ''' Get OHLC '''
    Get_OHLCV_720()
    #g = input("Get OHLC  .... Press any key : ")

    ''' Import dataset '''
    dataset = Importing_Dataset()
    #g = input("Importing_Dataset  .... Press any key : ")

    ''' Preprocessing '''
    dataset = Preprocessing_Dataset(dataset)
    #g = input("Preprocessing_Dataset  .... Press any key : ")

    ''' Mirroring Candles '''
    dataset = Mirroring_Candles(dataset)
    #g = input("Mirroring_Candles  .... Press any key : ")

    ''' Plot candles projection  '''
    Plot_Candles_Projection(dataset)
    #g = input("Plot_Candles_Projection  .... Press any key : ")

    return 


  
if __name__== "__main__":
  main()
  g = input("End Program  .... Press any key : "); print (g)

