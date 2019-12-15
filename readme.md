```python
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

%matplotlib inline

data = pd.read_csv('./housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))
```

    Boston housing dataset has 489 data points with 4 variables each.
    


```python
minimum_price = np.min(prices)
maximum_price = np.max(prices)
mean_price = np.mean(prices)
median_price = np.median(prices)
std_price = np.std(prices)

print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price)) 
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))
```

    Statistics for Boston housing dataset:
    
    Minimum price: $105,000.00
    Maximum price: $1,024,800.00
    Mean price: $454,342.94
    Median price $438,900.00
    Standard deviation of prices: $165,171.13
    


```python
from sklearn.metrics import r2_score
def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    score = r2_score(y_true, y_predict)
    
    return score
```


```python
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))
```

    Model has a coefficient of determination, R^2, of 0.923.
    


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split( features, prices, test_size=0.20, random_state=10)

print("Training and testing split was successful.")
```

    Training and testing split was successful.
    


```python
vs.ModelLearning(features, prices)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-6-244cbbba60d8> in <module>
    ----> 1 vs.ModelLearning(features, prices)
    

    NameError: name 'vs' is not defined



```python
vs.ModelComplexity(X_train, y_train)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-7-b95fd17c7021> in <module>
    ----> 1 vs.ModelComplexity(X_train, y_train)
    

    NameError: name 'vs' is not defined



```python
import sklearn as skl
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    regressor = DecisionTreeRegressor()
    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(estimator = regressor,param_grid = params,scoring = scoring_fnc,cv = cv_sets)
    grid = grid.fit(X, y)
    return grid.best_estimator_
```


```python
reg = fit_model(X_train, y_train)
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))
```

    Parameter 'max_depth' is 4 for the optimal model.
    


```python
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))
```

    Predicted selling price for Client 1's home: $406,933.33
    Predicted selling price for Client 2's home: $232,200.00
    Predicted selling price for Client 3's home: $938,053.85
    


```python
import seaborn as sns
import matplotlib.pyplot as plt 
def draw_heatmap(data_x):
            # Correlation thermogram

    D = pd.DataFrame(data_x)
    a = np.corrcoef(D.T)
    fig, ax = plt.subplots(figsize = (10,8)) 
    sns.heatmap(pd.DataFrame(a,columns = data.columns,index=data.columns), annot=True,
    vmax=1,
    vmin = 0,
    xticklabels=True,
    yticklabels=True,
    square=True, cmap="YlGnBu")
    plt.show() 
```


```python
vs.PredictTrials(features, prices, fit_model, client_data)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-12-df235a0b12ac> in <module>
    ----> 1 vs.PredictTrials(features, prices, fit_model, client_data)
    

    NameError: name 'vs' is not defined



```python

```
