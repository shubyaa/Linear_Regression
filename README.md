# Linear_Regression
This is a simple prediction graph in which we predict the profit of the movie if we lauch a movie at certain budget.

## Process
1. First we take our data set. We have taken a set of movies from period of time, you can see it in [cost_revenue_clean.csv](https://github.com/shubyaa/Linear_Regression/blob/main/Movie/res/cost_revenue_clean.csv) and take the **World wide budget** and **Production budget** in variables 'y' and 'x' respectively using [Pytho Dataframes](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) from [pandas](https://pandas.pydata.org/)

```python
import pandas
from pandas import DataFrame

data = pandas.read_csv('res/cost_revenue_clean.csv')    # To read csv files

x = DataFrame(data, columns=['production_budget_usd'])  # x axis data
Y = DataFrame(data, columns=['worldwide_gross_usd'])    # y axis data
```

2. Then, we try to plot a *scatter graph* of the given data with 'X' as Production Budget and 'Y' as Worldwide Budget i.e. the Actual profit. This work is done by using [matplotlib](https://matplotlib.org/) library.

```python
import matplotlib.pyplot as plt

plt.scatter(x, Y, alpha=0.3)      # plotting scatter graph, alpha is nothing but transperency of the dots
plt.title('Movie Budget')         # set title
plt.xlabel('Production Budget')
plt.ylabel('Worldwide Budget')
plt.xlim(0, 500000000)
plt.ylim(0, 3000000000)
plt.show()                        # to display the graph
```

The graph at this stage will look like this:-

  ![data_of_movies](https://github.com/shubyaa/Linear_Regression/blob/main/Figure_2.jpeg "Scatter Graph")

3. Now it's time to implement **Linear Regression** in our graph!

So, to perform linear regression, we will use **scikit learn**, a machine learning library for python. You can see more about this [here](https://scikit-learn.org/stable/)

This is the code snippet of how to use it:-

```python
from sklearn.linear_model import LinearRegression   # As we need only Linear Regression from entire module.
  
reg = LinearRegression()  # Calling the method and storing it in a variable
reg.fit(x, Y)             # fitting the data in regression
```

Now, before performing regression, it is necessary to check how your data is a best fit to perform a regression. In **Linear Regression** we need to calculate the *score* so that we can decide whether our regression would be accurate or not. More the value, more the regression is accurate.
```python
score = reg.score(x, Y)   # Score is predefined method to check accuracy of the regression
print(score)
```

The value is 0.5496485356985727 which is more than enough to perform linear regression.

4. Plotting regression in our graph

Add this line
```python
plt.plot(x, reg.predict(x), color='red', linewidth=4)     # plotting regression with color and thickness.
plt.show()                                                # important to call this, else plotting will not be shown.
```

It will look like this!

  ![Regression](https://github.com/shubyaa/Linear_Regression/blob/main/Figure_1.jpeg)

One thing to observe is that at higher values, the prediction might be inaccurate, but on lower values it predicts the worldwide budget perfectly.

Hope you guys get th idea!!

Any twichings and twisting are most welcome as I'm too a noobie in this.
