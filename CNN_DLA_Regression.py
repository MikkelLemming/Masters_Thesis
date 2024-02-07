globals().clear()
# imports:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from astropy.io import fits
import time as t
import random as r
import winsound
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from kerastuner.tuners import RandomSearch

t_0 = t.time()

#################################################################################################################################
#################################################################################################################################
############################ CHANGE THE FOLLOWING CODE TO IMPORT THE SPLIT DATA INSTEAD OF MAKING IT ############################
#################################################################################################################################
#################################################################################################################################

# Imports the data and sets the data type needed:
directory_path = 'C:\\Users\\mikke\\Desktop\\Kandidat_Stuff\\simpaqs-main\\simpaqs-main\\output\\Processed_Data'
DLA = pd.read_csv(directory_path + '\\DLA')
Abs_Info = pd.read_csv(directory_path + '\\Abs_Info')
DF = pd.read_csv(directory_path + '\\Extra_Info')
Abs = pd.read_csv(directory_path + '\\Absorbers')
temp_file_list = os.listdir(directory_path)
temp_file_list = [file for file in temp_file_list if 'Quasar' in file]
sorted_files = sorted(temp_file_list, key=lambda x: int(x.split('Quasar')[-1].split('.')[0]))

data_generator = (fits.open(directory_path + '\\' + file_name) for file_name in sorted_files if 'Quasar' in file_name)
#data_set_flux, data_set_wave, scale = 'Flux_gen_15', 'Wave_gen_15', 4/15
data_set_flux, data_set_wave, scale = 'Flux_imp', 'Wave_imp', 4
# Scale is how many datapoints are needed for 1Å

X = []
Y = []
Z = []
# Create Flux data
for data, z in zip(data_generator, DF.REDSHIFT):
    X.append(data[1].data[data_set_wave])
    Y.append(data[1].data[data_set_flux])
    Z.append(z)
    data.close()



# Creating DLA_Z data
#         DLA's only:
Ns, file = 'N_DLA', DLA
        # Include Sub-DLA's:
# Ns, file = 'N_ABS', Abs

# Creates the listes contining things like DLA_Z:
n = 0
DLA_Z = []
Index = []
LOG_NHI = []
DELTA = []
DV90 = []
for i in range(len(Abs_Info['N_DLA'])):
    dla_z = []
    log_nhi = []
    delta = []
    dv90 = []
    for j in range(Abs_Info[Ns][i]):
        if file['LOG_NHI'][n] >= 0:
        # if file['LOG_NHI'][n] >= 19:
            dla_z.append(file['Z_ABS'][n])
            log_nhi.append(file['LOG_NHI'][n])
            delta.append(file['DELTA'][n])
            dv90.append(file['DV_90'][n])
        n += 1
    DLA_Z.append(dla_z)
    LOG_NHI.append(log_nhi)
    DELTA.append(delta)
    DV90.append(dv90)
    Index.append(i)


# plt.plot(X[11485],Y[11485])
# plt.xlim(912*(1+Z[11485]), 1215*(1+Z[11485]))
# for TEST in DLA_Z[11485]:
#     plt.plot( [(1+TEST)*1215, (1+TEST)*1215], [min(Y[8397]), max(Y[8397])] )
# plt.show()


# Splits data into strips of sertain width and then moves the fram some percent of the given width, Datapoints
# past the last fram is not included, so all frames are the same length
def Data_Splitting(X=X, Y=Y, Z=Z, DLA_Z=DLA_Z, DELTA=DELTA, LOG_NHI=LOG_NHI, DV90=DV90, I=Index, Band_size = 200, Min_DLA_width = 10, DLA_place_lim = 1, Band_move_pecent = 0.1):
# Function creating banded data
    Flux = []
    Wave = []
    DLA_Reg = []
    Index = []
    REDSHIFT = []
    DELTA_new = []
    LOG_NHI_new = []
    DV90_new = []
    n = 0
    for i in range(len(X)):
        if DLA_Z != []:
            z = Z[i]
            Ly_lim = (1+z)*912
            Lya = (1+z)*1215
            if Lya >= 3674 + Band_size:
                band_start = max(3674, Ly_lim)
                j = 0
                while band_start + j + Band_size < Lya:
                    start, end = round((band_start+j-3674)*scale),round((band_start+j-3674+Band_size)*scale)
                    flux = Y[i][start:end]
                    wave = X[i][start:end]
                    n_DLAs = 0
                    viable_DLAs = 0
                    for II in range(len(DLA_Z[i])):
                        z = DLA_Z[i][II]
                        if band_start + j < (z + 1) * 1215 < band_start + j + Band_size:
                            n_DLAs += 1
                            if band_start + j + (1 - DLA_place_lim) / 2 * Band_size < (z + 1) * 1215 < band_start + j + Band_size - (1 - DLA_place_lim) / 2 * Band_size:
                                Z_index = wave.tolist().index(min(wave, key=lambda x: abs(x - (z + 1) * 1215)))
                                if all(x <= max(flux) / 2 for x in flux[round(Z_index - scale * 0.5 * Min_DLA_width):round(Z_index - scale * 0.5 * Min_DLA_width)]):
                                    viable_DLAs += 1
                                    redshift = z
                                    delta = DELTA[i][II]
                                    log_nhi = LOG_NHI[i][II]
                                    dv90 = DV90[i][II]

                    if n_DLAs == 1 and viable_DLAs == 1:
                        Flux.append(flux)
                        Wave.append(wave)
                        Index.append(I[i])
                        DLA_Reg.append(Z_index)
                        REDSHIFT.append(redshift)
                        DELTA_new.append(delta)
                        LOG_NHI_new.append(log_nhi)
                        DV90_new.append(dv90)

                    j += Band_size * Band_move_pecent
    return Flux, Wave, DLA_Reg, Index, REDSHIFT, DELTA_new, LOG_NHI_new, DV90_new

#Flux, Wave, DLA_Reg, Index, REDSHIFT, DELTA, LOG_NHI, DV90 = Data_Splitting(Band_size=200, Min_DLA_width=1, DLA_place_lim=0.8, Band_move_pecent=1)

t_0 = t.time()

if True:
    import itertools
    combs = [[200],[1],[0.8],[0.05]]
    #combs = [[],[],[],[]]
    zipped = itertools.product(*combs)
    Accs = []

    for Band_size, Min_DLA_width, DLA_place_lim, Band_move_pecent in zipped:
        # Creating data splitting, Generating test and training set and Scale data
        Flux, Wave, DLA_Reg, Index, REDSHIFT, DELTA, LOG_NHI, DV90 = Data_Splitting(Band_size=Band_size, Min_DLA_width=Min_DLA_width, DLA_place_lim=DLA_place_lim, Band_move_pecent=Band_move_pecent)

        try:
            Y = np.array(DLA_Reg)
            X = np.array(Flux)
        except:
            Flux_new = []
            m = max([len(Flux[i]) for i in range(len(Flux))])
            for f in Flux:
                while len(f) < m:
                    f = np.append(f, sum(f) / len(f))
                Flux_new.append(f)
            Y = np.array(DLA_Reg)
            X = np.array(Flux_new)

        # X = np.array(X)
        # Y = np.array([1 if len(y)!=0 else 0 for y in DLA_Z])

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, shuffle=False)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).reshape(len(X_train), len(X_train[0]),1)
        X_test_scaled = scaler.transform(X_test).reshape(len(X_test), len(X_test[0]),1)

        model = tf.keras.Sequential([
            # tf.keras.layers.Conv1D(64, kernel_size=10, activation='relu'),
            # tf.keras.layers.AveragePooling1D(pool_size=5),
            tf.keras.layers.Conv1D(128, kernel_size=10, activation='relu'),
            tf.keras.layers.AveragePooling1D(pool_size=5),
            # tf.keras.layers.Conv1D(256, kernel_size=10, activation='relu'),
            # tf.keras.layers.Conv1D(256, kernel_size=10, activation='relu'),
            tf.keras.layers.Conv1D(256, kernel_size=10, activation='relu'),
            tf.keras.layers.AveragePooling1D(pool_size=5),
            tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')  # , metrics='accuracy')
    #    model.fit(X_train_scaled, y_train, epochs=35, batch_size=32, validation_split=0.2)
        model.fit(X_train_scaled, y_train, epochs=5, batch_size=32, validation_split=0.2)

        # Get predictions and print results
        diff = []
        pred = model.predict(X_test_scaled)
        for i in range(len(pred)):
            diff.append(abs(pred[i]-y_test[i])[0])
        print('Avarage dist:', sum(diff)/len(diff)*0.25, 'Å')

    i = diff.index(max(diff))
    plt.plot(X_test_scaled[i], color = 'C0')
    plt.plot([y_test[i],y_test[i]],[min(X_test_scaled[i]), max(X_test_scaled[i])], color = 'C1')
    plt.plot([pred[i],pred[i]],[min(X_test_scaled[i]), max(X_test_scaled[i])], color = 'C2')
    plt.title('Offset: ' + str(diff[i]))
    plt.legend(['Spectra','True center','Guessed center'])
    plt.show()



            ###############################################
            ########## Hyperparameter Tuning  #############
            ###############################################



if False:
    Band_size = 200
    Min_DLA_width = 1
    DLA_place_lim = 0.8
    Band_move_pecent = 0.05
    def Model(HP):
        model = tf.keras.Sequential()

        hp_unit1 = HP.Int('unit1', min_value=32, max_value=256, step = 32)
        hp_unit12 = HP.Int('unit12', min_value=5, max_value=15, step = 1)
        model.add(tf.keras.layers.Conv1D(hp_unit1, kernel_size=hp_unit12, activation='relu'))

        hp_unit2 = HP.Int('unit2', min_value=2, max_value=6, step=1)
        model.add(tf.keras.layers.AveragePooling1D(pool_size=hp_unit2))

        hp_unit3 = HP.Int('unit3', min_value=32, max_value=256, step = 32)
        hp_unit32 = HP.Int('unit32', min_value=5, max_value=15, step=1)
        model.add(tf.keras.layers.Conv1D(hp_unit3, kernel_size=hp_unit32, activation='relu'))

        hp_unit4 = HP.Int('unit4', min_value=2, max_value=6, step=1)
        model.add(tf.keras.layers.AveragePooling1D(pool_size=hp_unit4))

        hp_unit5 = HP.Int('unit5', min_value=32, max_value=512, step=32)
        hp_unit52 = HP.Int('unit52', min_value=5, max_value=15, step=1)
        model.add(tf.keras.layers.Conv1D(hp_unit5, kernel_size=hp_unit52, activation='relu'))

        hp_unit6 = HP.Int('unit6', min_value=32, max_value=512, step=32)
        hp_unit62 = HP.Int('unit62', min_value=5, max_value=15, step=1)
        model.add(tf.keras.layers.Conv1D(hp_unit6, kernel_size=hp_unit62, activation='relu'))

        hp_unit7 = HP.Int('unit7', min_value=2, max_value=6, step=1)
        model.add(tf.keras.layers.AveragePooling1D(pool_size=hp_unit7))

        model.add(tf.keras.layers.Flatten())

        hp_unit8 = HP.Int('unit8', min_value=128, max_value=512, step = 32)
        model.add(tf.keras.layers.Dense(units=hp_unit8, activation='relu'))

        hp_learning_rate = HP.Choice('learning_rate', values = [0.01, 0.001, 0.0001])

        model.add(tf.keras.layers.Dense(1, activation='linear'))

        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='mean_absolute_error', metrics=['mean_absolute_error'])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='mean_squared_error', metrics=['mean_squared_error'])

        return model

    tuner = RandomSearch(
        Model,
        # objective='val_mean_absolute_error',
        objective='val_mean_squared_error',
        max_trials=25,
        executions_per_trial=1,
        directory='C:\\Users\\mikke\\Desktop\\Kandidat_Stuff\\simpaqs-main\\simpaqs-main\\HyperParameterTuning',
        project_name='HPTuning_Regression')

    Flux, Wave, DLA_Reg, Index, REDSHIFT, DELTA, LOG_NHI, DV90 = Data_Splitting(Band_size=Band_size,Min_DLA_width=Min_DLA_width,DLA_place_lim=DLA_place_lim,Band_move_pecent=Band_move_pecent)

    try:
        Y = np.array(DLA_Reg)
        X = np.array(Flux)
    except:
        Flux_new = []
        m = max([len(Flux[i]) for i in range(len(Flux))])
        for f in Flux:
            while len(f) < m:
                f = np.append(f, sum(f) / len(f))
            Flux_new.append(f)
        Y = np.array(DLA_Reg)
        X = np.array(Flux_new)

    X_train, X_test, y_train, y_test = train_test_split(X[:5000], Y[:5000], test_size=0.1, random_state=42)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).reshape(len(X_train), len(X_train[0]), 1)
    X_val_scaled = scaler.transform(X_val).reshape(len(X_val), len(X_val[0]), 1)
    X_test_scaled = scaler.transform(X_test).reshape(len(X_test), len(X_test[0]), 1)

    tuner.search(X_train_scaled, y_train, epochs=5, batch_size=10, validation_data=(X_val_scaled, y_val))
    best_hps = tuner.oracle.get_best_trials(1)[0].hyperparameters
    model = tuner.hypermodel.build(best_hps)
    model.fit(X_train_scaled, y_train, epochs=5, validation_data=(X_val_scaled, y_val))



        ###############################################
        ########## ERROR-CORRELATION PART #############
        ###############################################

# plt.hist(diff, bins=50)
# plt.show()
#
# Name, Feature = 'Redshift', REDSHIFT
# # Name, Feature = 'Delta', DELTA
# # Name, Feature = 'Log(NHI)', LOG_NHI
# # Name, Feature = 'DV_90', DV90
# plt.plot([Feature[len(y_train):][i] for i in range(len(pred))], diff, '*')
# plt.xlabel(Name)
# plt.ylabel('Error')
# plt.title('Error correlation '+ Name)
# plt.show()
#
# for name in ['absMag', 'smcDustEBV', 'LOG_MBH', 'LOG_REDD', 'SEEING', 'AIRMASS', 'MAG']:
#     plt.plot([DF[name][Index[len(y_train):][i]] for i in range(len(pred))], diff, '*')
#     plt.title('Error correlation ' + name)
#     plt.xlabel(name)
#     plt.ylabel('Error')
#     plt.show()


plt.show()
print('Took:', t.time()-t_0, 's')
winsound.Beep(440,500)
winsound.Beep(440,750)
winsound.Beep(440,500)




