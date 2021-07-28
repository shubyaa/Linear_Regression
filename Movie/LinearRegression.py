import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pandas.read_csv('res/cost_revenue_clean.csv')

x = DataFrame(data, columns=['production_budget_usd'])
Y = DataFrame(data, columns=['worldwide_gross_usd'])

reg = LinearRegression()
reg.fit(x, Y)
slope = reg.coef_
intercept = reg.intercept_

score = reg.score(x, Y)
print(score)

plt.scatter(x, Y, alpha=0.3)
plt.title('Movie Budget')
plt.xlabel('Production Budget')
plt.ylabel('Worldwide Budget')
plt.xlim(0, 500000000)
plt.ylim(0, 3000000000)
plt.plot(x, reg.predict(x), color='res', linewidth=4)
plt.show()

print(slope)
print(intercept)
