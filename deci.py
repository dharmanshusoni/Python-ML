import numpy as np 
import matplotlib.pyplot as plt     
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor

dataset = np.array( 
[['Asset Flip', 100, 1000], 
['Text Based', 500, 3000], 
['Visual Novel', 1500, 5000], 
['2D Pixel Art', 3500, 8000], 
['2D Vector Art', 5000, 6500], 
['Strategy', 6000, 7000], 
['First Person Shooter', 8000, 15000], 
['Simulator', 9500, 20000], 
['Racing', 12000, 21000], 
['RPG', 14000, 25000], 
['Sandbox', 15500, 27000], 
['Open-World', 16500, 30000], 
['MMOFPS', 25000, 52000], 
['MMORPG', 30000, 80000] 
]) 

X = dataset[:,1:2].astype(int)
Y = dataset[:,2].astype(int)

regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(X,Y)

X_grid = np.arange(min(X),max(X),0.01)

X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,Y,color='red')

plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')

plt.title('Plot Graph')

plt.xlabel('Production cost')

plt.ylabel('Profit')

plt.show()






