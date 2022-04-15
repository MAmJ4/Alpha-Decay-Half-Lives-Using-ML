import numpy as np
import joblib

numbers = np.arange (15)
joblib.dump(numbers, "test1")
numbers2 = joblib.load ("test1")

print (numbers)
print(numbers2)