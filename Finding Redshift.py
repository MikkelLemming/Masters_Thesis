import numpy as np
from sklearn.model_selection import train_test_split
import time as t
from main import data_list, df, generalize_datapoints
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


N = len(data_list)
#N = 10



t_start = t.time()

X = []
Y = []
WAVE_list, _ = generalize_datapoints(data_list[0], plot = False)
n = 0
for i in range(N):
    try:
        #y = data_list[i][1].data.FLUX[0][np.where(data_list[i][1].data.WAVE[0] == round(1216 * (1 + df.REDSHIFT[i])))[0][0]]
        y = (1216 * (1 + df.REDSHIFT[i]) - WAVE_list[0])#/(1216 * (1 + max(df.REDSHIFT)) - WAVE_list[0])
        Y.append(y)
        #Y.append(df.REDSHIFT[i])
        _, x = generalize_datapoints(data_list[i], plot = False)
        x = [X/max(x) for X in x]
        X.append(x)
    except:
        n += 1
        #print('Mistake at i=',i)

print('Middle time check:', t.time()-t_start,'s')

X = np.array(X)
Y = np.array(Y)

# X_train, X_test, Y_train,  Y_test = train_test_split(X, Y, test_size=0.2)
#
# model = MLPRegressor()
# model.fit(X_train, Y_train)
#
# pred = model.predict(X_test)
#
# diff = 0
# for i in range(len(pred)):
#     pred_flux = min(X_test[-i], key=lambda x: abs(pred[-i] - x))
#     pred_flux_index = np.where( X_test[-i] == pred_flux )
#     wave_pred = WAVE_list[pred_flux_index]
#     diff += abs( wave_pred - 1216*(1+df.REDSHIFT[len(df.REDSHIFT) - 2 - i]) )
#
# print('The avarage distance from true Lyman Alpha is:', diff[0]/len(pred))
#
#


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your quasar dataset and extract features, labels

# Data preprocessing
# ...

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=15, batch_size=32, validation_split=0.2)
#model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
mse = model.evaluate(X_test_scaled, y_test)
print(f'Mean Squared Error on Test Set: {mse}')

# Make predictions
predictions = model.predict(X_test_scaled)
#predictions = model.predict(X_test)

diff = 0
for i in range(len(predictions)):
    diff += abs(predictions[i] - y_test[i])#*(1216 * (1 + max(df.REDSHIFT)) - WAVE_list[0])
print('Avg Diff:', diff/len(predictions))





print('Took ', t.time()-t_start, 's')



