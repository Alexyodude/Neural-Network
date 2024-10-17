import numpy as np
test = [1, 2, 3, 4, 5]
test2 = np.array([1, 2, 3, 4, 5])
test1 = 0

#function that does a if x is an array and another if it's a number
def testFunction(x):
    if isinstance(x, list):
        print(len(x))
    else:
        print(x)
# print(test)
# print(test2)
# print(isinstance(test2, (list, np.ndarray)))
# print(isinstance(test, (list, np.ndarray)))
# print(isinstance(test1, list))

test3 = np.random.rand(3, 2)

print(test3)
print(test3[0])