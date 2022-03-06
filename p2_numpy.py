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

# documentation 
# The N-dimensional array --> https://numpy.org/doc/stable/reference/arrays.ndarray.html
# Data types --> http://docs.scipy.org/doc/numpy/user/basics.types.html
# Array creation [more] --> https://numpy.org/doc/stable/user/basics.creation.html
# Indexing [more] --> http://docs.scipy.org/doc/numpy/user/basics.indexing.html
# Broadcasting --> http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
# Random sampling --> http://docs.scipy.org/doc/numpy/reference/routines.random.html
# Mathematical functions --> https://numpy.org/doc/stable/reference/routines.math.html
# Linear algebra --> https://numpy.org/doc/stable/reference/routines.linalg.html


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


''' arithmetic operations   ''' 
def Arithmetic_Operations():

    arr = np.array([(1,2,3,4,5),(10,20,30,40,50)])
    print(f"original Array --> \n{arr}")

    #mulpitly by 2
    #mean = arr.mean()
    print(f"arr * 2 --> \n{arr * 2}")

    #dividing by 2
    print(f"arr * 2 --> \n{arr / 2.}")

    #operations betweens arrays 
    arr1 = np.array([(1,2,3,4,5),(10,20,30,40,50)])
    print(f"Array 1 --> \n{arr1}")
    arr2 = np.array([(100,200,300,400,500),(1,2,3,4,5)])
    print(f"Array 2 --> \n{arr2}")
    
    #adding arrays
    print(f"arr1 + arr2 --> \n{arr1 + arr2}")

    #multiply arrays element wise
    print(f"arr1 * arr2 --> \n{arr1 * arr2}")

    #divide  arrays element wise
    print(f"arr1 / arr2 --> \n{arr1 / arr2}")

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

    ''' arithmetic operations   ''' 
    Arithmetic_Operations()
    g = input("return from Arithmetic_Operations  .... Press any key : ")


    return 



  
if __name__== "__main__":
  main()
  g = input("End Program  .... Press any key : "); print (g)



