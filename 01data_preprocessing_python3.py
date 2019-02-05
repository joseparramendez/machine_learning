import numpy as np

from sklearn import preprocessing

#We imported a couple of packages. Let's create some sample data and add the line to this file:

input_data = np.array([[3, -1.5, 3, -6.4], [0, 3, -1.3, 4.1], [1, 2.3, -2.9, -4.3]])
#Los siguientes comandos estandarizan, es dedir, calcula la media y la desviacion estandar. Le resta la media a cada dato y lo divide entre la desviacion estandar
data_standardized = preprocessing.scale(input_data)
print ("\nMean = ", data_standardized.mean(axis = 0))
print ("Std deviation = ", data_standardized.std(axis = 0))

print ("----------------------------------------------------------------------")

#Los siguientes comandos buscan el maximo y minimo del las componentes de cada entrada y de acuerdo con estos rescalan los demas datos en las componentes entre 0 y 1.
data_scaler = preprocessing.MinMaxScaler(feature_range = (0, 1))
data_scaled = data_scaler.fit_transform(input_data)
print ("\nMin max scaled data = ", data_scaled)

print("-----------------------------------------------------------------------")
#Los siguientes comandos normalizan los vectores
data_normalized = preprocessing.normalize(input_data, norm  = 'l1')
print ("\nL1 normalized data = ", data_normalized)
print("-----------------------------------------------------------------------")
#Los siguientes comandos ponen un condicional sobre cada componente de los vectores. En este caso mayor a 1.4
data_binarized = preprocessing.Binarizer(threshold=1.4).transform(input_data)
print ("\nBinarized data =", data_binarized)

print("-----------------------------------------------------------------------")

#Los siguientes comandos cuentan el numero de datos diferentes en cada componente, de estos valores distintos ve cuales coinciden con un valor otorgado y regresa un areglo de ceros igual al numero de valores distintos exepto un 1 en el elemento cuya componente conincida con el numero otorgado.

encoder = preprocessing.OneHotEncoder()
encoder.fit([  [0, 2, 1, 12], 
               [1, 3, 5, 3], 
               [2, 3, 2, 12], 
               [1, 2, 4, 3]
])
encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
print ("\nEncoded vector =", encoded_vector)

print("-------------------------------------Label Encoding-----------------------------------")

#Los siguientes comandos toman una seria de categorias y les asignan una etiqueta numerica
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
input_classes = ['suzuki', 'ford', 'suzuki', 'toyota', 'ford', 'bmw']
label_encoder.fit(input_classes)
print ("\nClass mapping:")
for i, item in enumerate(label_encoder.classes_):
    print(item, '-->', i)
#En esta parte se comprueba si una serie de etiquetas proporcionadas coinciden con las asignaciones previas. Y en caso poitivo te regresa la etiqueta numerica
labels = ['toyota', 'ford', 'suzuki']
encoded_labels = label_encoder.transform(labels)
print ("\nLabels =", labels)
print ("Encoded labels =", list(encoded_labels))

encoded_labels = [3, 2, 0, 2, 1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print ("\nEncoded labels =", encoded_labels)
print ("Decoded labels =", list(decoded_labels))
