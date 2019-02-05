import pandas
data = 'pima_indians.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'Outcome']
dataset = pandas.read_csv(data, names = names)
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
print(dataset.groupby('Glucose').size())


import pandas
import matplotlib.pyplot as plt
data = 'iris_df.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(data, names=names)
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#histograms
dataset.hist()
plt.show()

from pandas.plotting import scatter_matrix
scatter_matrix(dataset)
plt.show()
