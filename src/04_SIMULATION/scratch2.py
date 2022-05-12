import numpy as np

lst = [1, 3, 4]
lst2 = [1, 3, 4, 5]
print(all(elem in lst2 for elem in lst))
