import numpy as np

""" Creating Arrays and Common Methods """

python_list = [1, 2, 3, 4, 5]
print(python_list)
print(type(python_list))

numpy_array = np.array(python_list)
print(numpy_array)
print(type(numpy_array))

""" Different example """
python_list = [10, 22, 3, 47, 54]
print(python_list)
print(type(python_list))

numpy_array = np.array(python_list)
print(numpy_array)
print(type(numpy_array))



""" Creating 2D array """
python_2d_array=[[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]
print(python_2d_array)
print(type(python_2d_array))

numpy_2d_array=np.array(python_2d_array)
print(numpy_2d_array)
print(type(numpy_2d_array))

""" Different example """
python_2d_array=[[13,2,31,47,52],[6,74,82,97,10],[111,12,133,124,156]]
print(python_2d_array)
print(type(python_2d_array))

numpy_2d_array=np.array(python_2d_array)
print(numpy_2d_array)
print(type(numpy_2d_array))

""" arange(start, stop, step) the default step size is 1, this function will return the array from S -> ST evenly spaced out"""

print (np.arange(0,15))
print (np.arange(0,15,2)) #This will return the same array range from 0-15 but will be in increments of 2 instead of the default one

""" reshape(rows, columns)  this function will return the numpy array in 2d """
print (np.arange(0,15).reshape(3,5))


""" If we want an array in 0 or 1 we can use .zeros(num) .ones(num) """

print(np.zeros(15))

""" By default it returns float, if we want them to be of another type we can use dtype="type" """

print(np.ones(15,dtype=int).reshape(3,5))

""" Different examples """
print(np.arange(0,44))
print(np.arange(0,15,5))
print (np.arange(0,25).reshape(5,5))
print(np.zeros(25))
print(np.ones(25,dtype=int).reshape(5,5))


""" We can generate the identity matrix by using .eye(num) """
print(np.eye(5))

""" We can use rand to return random samples from the uniform distribution from [0 - 1] """
print( np.random.rand(5))

""" randn will return from the standard normal distribution """
print(np.random.randn(5))

""" return a random int .randint(low,high) """
print(np.random.randint(1,200))

""" Other common methods that you're likely to encounter in this class include min, max, argmin, and
 argmax. The only difference between the two arg methods is that they'll instead return the index
 position of the min/max value. For example: """

A = np.random.randint(0, 100, 20).reshape(4,5)
print(A)

print(f'The smallest value in A is {A.min()}, and is a located at position {A.argmin()}')
print(f'The largest value in A is {A.max()}, and is a located at position {A.argmax()}')

""" If we want to check on the size of the array we can use shape """
print(A.shape)
print(np.shape(A))


""" Common Operations """
A=np.arange(1,16)
B=np.arange(1,30,2)
C=np.arange(0,4).reshape(2,2)
D=np.arange(0,4).reshape(2,2)
E=np.arange(1,16).reshape(3,5)
F=np.arange(11)
print(f'A:{A}')
print(f'B:{B}')
print('C:')
print(C)
print('D:')
print(D)
print('E:')
print(E)
print(f'F:{F}')

""" Arithmetic  careful when dividing by zero we will get a nan"""
print(A + B) #all the standard ops are element wise on the arrays
print(A + 5)
print(A - B)
print(A - 5)
print(A * B)
print(A * 5)
print(A / B)
print(A / 2)

""" If we want to do maxtrix mul we have to use matmul """
print(np.matmul(C ,D))

""" Universal Functions """

print(np.sqrt(E))
print(np.log(E))
""" Common commad where(condition,x(ifTrue),y(ifFalse))  loops through array and replaces values w/ X if true, else Y if false """
print(np.where(F%2==0, -1, F*100))

""" Indexing """
#For 1D array
A = np.arange(15)

# We can go to the index directly ex
print(f'A[0]:{A[0]}')
print(f'A[5]:{A[5]}')
print(f'A[14]:{A[14]}')

#Or we can access a section A[start_indx, stop_indx]
print(A[0:10])
A[0:10] = 500
print(A)

#Indexing for a 2D arr
B = np.arange(50).reshape(5,10)
print(B)

#Direct access by index B[row,col]
print(f'0th row and 2nd col in B:{B[0,2]}')
print(f'3rd row and 3rd col in B:{B[3,3]}')
print(f'4th row and 2nd col in B:{B[4,2]}')

#Same can be done with sections for the 2D arr
print(B[0:2, 3:5])
B[0:2, 3:5] = -1
print(B)

""" Boolean Array Indexing """
C = np.arange(50).reshape(10,5)
print(C)

#We can also index array using comparisons based on met conditions ex
print(C%2 == 0) 

#Even can get the sum of all the true values filtering out the data

print(C[C%2 == 0])
