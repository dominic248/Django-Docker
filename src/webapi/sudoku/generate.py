import random
import numpy as np 

def busyCol(no_row,no_col,S_array):
    for m in range(0,9):
        if m==no_row:
            continue
        if S_array[m][no_col]==S_array[no_row][no_col]:
            return True
    return False

def busyRow(no_row,no_col,S_array):
    for n in range(0,9):
        if n==no_col:
            continue
        if S_array[no_row][n]==S_array[no_row][no_col]:
            return True
    return False

def busyBoxFor(no_row,no_col,S_array,m_start,m_end,n_start,n_end):
    for m in range(m_start,m_end):
        for n in range(n_start,n_end):
            if m==no_row and n==no_col:
                continue
            if S_array[m][n]==S_array[no_row][no_col]:
                return True
    return False

def busyBox(no_row,no_col,S_array):
    if no_row>=0 and no_row<=2 and no_col>=0 and no_col<=2:
        return busyBoxFor(no_row,no_col,S_array,0,3,0,3)
    if no_row>=0 and no_row<=2 and no_col>=3 and no_col<=5:
        return busyBoxFor(no_row,no_col,S_array,0,3,3,6)
    if no_row>=0 and no_row<=2 and no_col>=6 and no_col<=8:
        return busyBoxFor(no_row,no_col,S_array,0,3,6,9)

    if no_row>=3 and no_row<=5 and no_col>=0 and no_col<=2:
        return busyBoxFor(no_row,no_col,S_array,3,6,0,3)
    if no_row>=3 and no_row<=5 and no_col>=3 and no_col<=5:
        return busyBoxFor(no_row,no_col,S_array,3,6,3,6)
    if no_row>=3 and no_row<=5 and no_col>=6 and no_col<=8:
        return busyBoxFor(no_row,no_col,S_array,3,6,6,9)

    if no_row>=6 and no_row<=8 and no_col>=0 and no_col<=2:
        return busyBoxFor(no_row,no_col,S_array,6,9,0,3)
    if no_row>=6 and no_row<=8 and no_col>=3 and no_col<=5:
        return busyBoxFor(no_row,no_col,S_array,6,9,3,6)
    if no_row>=6 and no_row<=8 and no_col>=6 and no_col<=8:
        return busyBoxFor(no_row,no_col,S_array,6,9,6,9)

def sudoku(n):
    S_array=np.zeros([9,9],dtype=int)
    numbers=list(range(1,10))
    for no_row in range(0,9):
        random.shuffle(numbers)
        for no_col in range(0,9):
            for i in numbers:
                S_array[no_row][no_col]=i
                cond1=busyCol(no_row,no_col,S_array)
                cond2=busyRow(no_row,no_col,S_array)
                cond3=busyBox(no_row,no_col,S_array)
                if cond1==False and cond2==False and cond3==False:
                    break
    count=1
    while count<=n:
        i=random.randint(0,8)
        j=random.randint(0,8)
        if S_array[i][j]==0:
            continue
        else:
            S_array[i][j]=0
            count += 1
    return S_array