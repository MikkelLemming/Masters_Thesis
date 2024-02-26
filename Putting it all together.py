import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

N = 50

file_path = 'SSH\\Data\\200_1_1_005'
X = np.load(file_path + '/Flux.npy')
scaler = StandardScaler()
_ = scaler.fit_transform(X).reshape(len(X), len(X[0]), 1)

directory_path = 'C:\\Users\\mikke\\Desktop\\Kandidat_Stuff\\simpaqs-main\\simpaqs-main\\output\\Processed_Data'
DLA = pd.read_csv(directory_path + '\\DLA')
Abs_Info = pd.read_csv(directory_path + '\\Abs_Info')
DF = pd.read_csv(directory_path + '\\Extra_Info')
Abs = pd.read_csv(directory_path + '\\Absorbers')
temp_file_list = os.listdir(directory_path)
temp_file_list = [file for file in temp_file_list if 'Quasar' in file]
sorted_files = sorted(temp_file_list, key=lambda x: int(x.split('Quasar')[-1].split('.')[0]))

data_generator = (fits.open(directory_path + '\\' + file_name) for file_name in sorted_files[-N:-1] if 'Quasar' in file_name)
#data_set_flux, data_set_wave, scale = 'Flux_gen_15', 'Wave_gen_15', 4/15
data_set_flux, data_set_wave, scale = 'Flux_imp', 'Wave_imp', 4
# Scale is how many datapoints are needed for 1Å

X = []
Y = []
Z = []
# Create Flux data
for data, z in zip(data_generator, [z for z in DF.REDSHIFT][-N:-1]):
    X.append(np.array(list(data[1].data[data_set_wave])[:-20]))
    Y.append(np.array(list(data[1].data[data_set_flux])[:-20]))
    Z.append(z)
    data.close()

n = 0
DLA_Z = []
for i in range(len(Abs_Info['N_DLA'])):
    dla_z = []
    for j in range(Abs_Info['N_DLA'][i]):
        dla_z.append(DLA['Z_ABS'][n])
        n += 1
    DLA_Z.append(dla_z)
DLA_Z = DLA_Z[-N:-1]

def Data_Splitting(X, Y, Z, DLA_Z, Band_size = 200, Band_move_pecent = 0.05):
# Function creating banded data
    Flux = []
    Wave = []
    DLA_Reg = []
    DLA_Class = []
    REDSHIFT = []
    Index = []

    for i in range(len(X)):
        z = Z[i]
        Ly_lim = (1 + z) * 912
        Lya = (1 + z) * 1215.67
        if Lya >= 3674 + Band_size:
            band_start = max(3674, Ly_lim)
            j = 0
            while band_start + j + Band_size < Lya:
                start, end = round((band_start + j - 3674) * scale), round((band_start + j - 3674 + Band_size) * scale)
                flux = Y[i][start:end]
                wave = X[i][start:end]
                n_DLAs = 0
                Z_indexs = []
                redshifts = []
                for II in range(len(DLA_Z[i])):
                    z = DLA_Z[i][II]
                    if band_start + j < (z + 1) * 1215.67 < band_start + j + Band_size:
                        n_DLAs += 1
                        Z_indexs.append(wave.tolist().index(min(wave, key=lambda x: abs(x - (z + 1) * 1215.67))))
                        redshifts.append(z)

                Flux.append(flux)
                Wave.append(wave)
                DLA_Reg.append(Z_indexs)
                if n_DLAs > 0:
                    DLA_Class.append(1)
                else:
                    DLA_Class.append(0)
                REDSHIFT.append(redshifts)
                Index.append(i)

                j += Band_size * Band_move_pecent



    return Flux, Wave, DLA_Reg, DLA_Class, REDSHIFT, Index


Flux, Wave, DLA_Reg, DLA_Class, REDSHIFT, Index = Data_Splitting(X, Y, Z, DLA_Z, Band_size = 200, Band_move_pecent = 0.05)

def Stitch(flux, wave, dla_reg, guess):
    Flux_new = list(flux[0])
    Wave_new = list(wave[0])
    DLA_Redshift = []
    Pred = []
    for j in dla_reg[0]:
        DLA_Redshift.append(wave[0][j])
    for j in guess[0]:
        try:
            Pred.append(wave[0][int(round(j))])
        except:
            _ = 0
            # if int(round(j)) > len(wave[0])-1:
            #     Pred.append(wave[0][len(wave[0])-1])
            # elif int(round(j)) < 0:
            #     Pred.append(wave[0][0])

    for i in range(1, len(flux)):
        Flux_new += list(flux[i])[int(-800 * 0.05):]
        Wave_new += list(wave[i])[int(-800 * 0.05):]
        for j in dla_reg[i]:
            DLA_Redshift.append(wave[i][j])
        for j in guess[i]:
            try:
                Pred.append(wave[i][int(round(j))])
            except:
                _ = 0
                # if int(round(j)) > len(wave[i]) - 1:
                #     Pred.append(wave[i][len(wave[i]) - 1])
                # elif int(round(j)) < 0:
                #     Pred.append(wave[i][0])

    plt.plot(Wave_new, Flux_new)
    for i in DLA_Redshift:
        plt.plot([i, i], [min(Flux_new), max(Flux_new)*1.1], color = 'green')
    for i in Pred:
        plt.plot([i, i], [min(Flux_new), max(Flux_new)],  color = 'orange')
    plt.title('Full Lyman Alpha Forest')
    plt.xlabel('Wavelength [Å]')
    plt.ylabel('Flux')
    plt.show()

    return Flux_new, Wave_new, DLA_Redshift, Pred


def total_model(num):
    list_of_index = list(np.where(np.array(Index) == num)[0])
    if list_of_index == []:
        return
    flux, wave, dla_reg, dla_class = [],[],[],[]
    for i in list_of_index:
        flux.append(Flux[i])
        wave.append(Wave[i])
        dla_reg.append(DLA_Reg[i])
        dla_class.append(DLA_Class[i])

    flux_scaled = scaler.transform(flux).reshape(len(flux), len(flux[0]),1)

    model_Class = tf.keras.models.load_model('C:\\Users\\mikke\\PycharmProjects\\pythonProject3\\Models\\CNN_Class.keras')
    pred_proba = model_Class.predict(flux_scaled)
    pred_Class = [int(np.round(p)[0]) for p in pred_proba]
    accuracy = accuracy_score(dla_class, pred_Class)
    print('Accuracy: ', accuracy)


    model_Reg = tf.keras.models.load_model('C:\\Users\\mikke\\PycharmProjects\\pythonProject3\\Models\\CNN_Reg.h5')
    pred_Reg = []
    PREDS = model_Reg.predict(flux_scaled)

    for i in range(len(flux_scaled)):
        if pred_Class[i] == 0:
            pred_Reg.append([])
        elif pred_Class[i] == 1:
            pred_Reg.append(PREDS[i])

    _,_,_,PREDS = Stitch(flux,wave,dla_reg,pred_Reg)

    return PREDS


# for i in range(10):
#     total_model(i)


def total_model_test(num):
    list_of_index = list(np.where(np.array(Index) == num)[0])
    if list_of_index == []:
        return
    flux, wave, dla_reg, dla_class = [],[],[],[]
    for i in list_of_index:
        flux.append(Flux[i])
        wave.append(Wave[i])
        dla_reg.append(DLA_Reg[i])
        dla_class.append(DLA_Class[i])

    flux_scaled = scaler.transform(flux).reshape(len(flux), len(flux[0]),1)

    model_Class = tf.keras.models.load_model('C:\\Users\\mikke\\PycharmProjects\\pythonProject3\\Models\\CNN_Class.keras')
    pred_proba = model_Class.predict(flux_scaled)
    pred_Class = [int(np.round(p)[0]) for p in pred_proba]
    accuracy = accuracy_score(dla_class, pred_Class)
    print('Accuracy: ', accuracy)

    pred_Reg = []

    for i in range(len(flux_scaled)):
        if pred_Class[i] == 0:
            pred_Reg.append([])
        elif pred_Class[i] == 1:
            pred_Reg.append([400])

    Stitch(flux,wave,dla_reg,pred_Reg)

# for i in range(10):
#     total_model_test(i)


for n in range(40,50):
#for n in [15,N-2]:
    if any([True for i in list(np.where(np.array(Index) == n)[0]) if DLA_Reg[i] != []]):
        _ = 0
    else:
        try:
            PREDS = total_model(n)
            # plt.hist(PREDS, bins=len(PREDS))
            # plt.show()
        except:
            print('No Guessed DLAs')

