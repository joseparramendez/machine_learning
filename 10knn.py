#Importing Libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
df = pd.read_csv('iris_df.csv')
df.columns = ['X1', 'X2', 'X3', 'X4', 'Y']
df = df.drop(['X4', 'X3'], 1)
df.head()
import seaborn as sns
sns.set_context('notebook', font_scale=1.1)
sns.set_style('ticks')
sns.lmplot('X1','X2', scatter=True, fit_reg=False, data=df, hue='Y')
import matplotlib.pyplot as plt
plt.ylabel('X2')
plt.xlabel('X1')
from sklearn.cross_validation import train_test_split
# Create KNeighbors classifier object model
neighbors = KNeighborsClassifier(n_neighbors=6)
# default value for n_neighbors is 5
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
X = df.values[:, 0:2]
Y = df.values[:, 2]
# Train the model using the training sets and check score
trainX, testX, trainY, testY = train_test_split( X, Y, test_size = 0.3)
neighbors.fit(trainX, trainY)
print('Accuracy: \n', neighbors.score(testX, testY))
pred = neighbors.predict(testX)
#Predict Output
plt.show()
