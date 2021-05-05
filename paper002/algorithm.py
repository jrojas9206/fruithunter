import numpy as np 

def inserting_sort(LstOfDict):
    """
    Sort the trees by their estimated volume 
    
    :Algorithm: 
        Insertion Sort 
    :INPUT:
        LstOfDict: List of dictionaries, The dictionary must have the "sortBy" key inside {sortBy}
        sortBy: str, Key in the dictionary that is going to be use to sort 
    :OUTPUT:
        Sorted List 
    :NOTE:
        Use with lists of less than 200 positions, after this value it could be little bit slow 
    """
    A = LstOfDict
    for j in range(1,len(LstOfDict), 1):
        key = A[j]
        # Insert  into the sorted sequence 
        i = j-1 
        while( i>-1 and A[i]>key ):
            A[i+1] = A[i]
            i = i-1
        A[i+1] = key
    return A