import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import numpy as np
import math as m
from sklearn.metrics import mean_squared_error

Scale = 2

directory_path = 'C:\\Users\\mikke\\Desktop\\Kandidat_Stuff\\simpaqs-main\\simpaqs-main\\output\\l1_data'
file_path = os.path.abspath(os.path.join(directory_path, 'observations.csv'))
df = pd.read_csv(file_path)

temp_file_list = os.listdir(directory_path)
data_list = [fits.open(directory_path + '\\' + file_name) for file_name in
             temp_file_list if 'LJ1' in file_name]

def plot_improved_data(hdu, N=10, Triple_plot=False, redshift=0, plot = True):
    data = hdu[1].data

    Wave_Length = data.WAVE[0]
    Flux = data.FLUX[0]

    Flux_new = []
    WL_new = []

    for i in range(N, len(Flux) - N):
        Avg = (sum(Flux[i - N:i - 3]) + sum(Flux[i + 3:i + N - 3])) / (
                    len(Flux[i - N:i - 3]) + len(Flux[i + 3:i + N - 3]))

        if abs(Flux[i] - Avg) <= Scale * Avg:
            Flux_new.append(Flux[i])
            WL_new.append(Wave_Length[i])
        else:
            Flux_new.append(Avg/2)
            WL_new.append(Wave_Length[i])

    if Triple_plot:
        plt.figure()
        plt.plot(Wave_Length, Flux)
        plt.ylim(0, 0.5 * 10 ** -15)
        plt.title('Not imrpoved data')
        plt.show()

        plt.figure()
        plt.plot(WL_new, Flux_new)
        #plt.ylim(0, 0.5 * 10 ** -15)
        plt.title('Imrpoved data')
        plt.show()

    if plot:
        plt.figure()
        plt.plot(Wave_Length, Flux)
        if redshift == 0:
            plt.title('Combined data')
        else:
            plt.title('Combined data z = {}'.format(round(redshift, 2)))
        plt.plot(WL_new, Flux_new)
        plt.ylim(0, 0.5 * 10 ** -15)
        plt.show()

    return WL_new, Flux_new, Wave_Length, Flux

def generalize_datapoints(data, n=15, improved=True, x_lim = False, plot = True):
    WAVE1, FLUX1, WAVE2, FLUX2 = plot_improved_data(data, plot = False)

    if improved:
        FLUX = FLUX1
        WAVE = WAVE1
    else:
        FLUX = FLUX2
        WAVE = WAVE2

    FLUX_new = []
    WAVE_new = []

    for i in range(len(FLUX)):
        if i > m.floor(n / 2) and i < len(FLUX) - m.floor(n / 2) and i % n == 0:
            FLUX_new.append(sum(FLUX[i - m.floor(n / 2):i + m.floor(n / 2)]) / n)
            WAVE_new.append(WAVE[i])

    if plot:
        plt.figure()
        plt.plot(WAVE_new, FLUX_new)
        plt.title('Generalized Spectra')
        if x_lim != False:
            plt.xlim(x_lim)


    return WAVE_new, FLUX_new

def est_z(hdu, redshift, n=10):
    WL_new, Flux_new, Wave_Length, Flux = plot_improved_data(hdu, N=n, redshift=redshift)

    Obs_wl = WL_new[np.array(Flux_new).argmax()]

    return (Obs_wl - 1216) / 1216

def norm_data(data, plot = False):
    Wave, Flux = generalize_datapoints(data)
    max_value = max(Flux)

    for i in range(len(Flux)):
        Flux[i] = Flux[i]/max_value

    if plot == True:
        plt.figure()
        plt.plot(Wave, Flux)
        plt.show()

    return Wave, Flux


count = 0

if False:
    n = 0
    for i in data_list[10:15]:
        # plot_improved_data(i, redshift = df.REDSHIFT[n])
        # print("Estimate:", round(est_z(i, redshift = df.REDSHIFT[n]),2), "True value:", round(df.REDSHIFT[n],2))
        #        if abs(est_z(i, redshift = df.REDSHIFT[n]) - df.REDSHIFT[n]) <= 0.015:
        #            count += 1
        z = est_z(i, redshift=df.REDSHIFT[n])
        generalize_datapoints(i)
        if abs(z - df.REDSHIFT[n]) <= 0.015:
            count += 1
        n += 1
    print("In the end, this process was good enougth in", count / n * 100,
          "%, based on a +/- 0.015 tolorance")

if False:
    for i in [2, 5, 10, 15, 20, 25]:
        generalize_datapoints(data_list[10], n=i)

if False:
    #Matches https://ned.ipac.caltech.edu/level5/Glossary/Figures/figure2_2.gif
    n, n_slut = 80, 99
    for i in data_list[n:n_slut]:
        z = df.REDSHIFT[n]
        Wave_gen, Flux_gen = generalize_datapoints(i)
        plt.plot([1216*(1+z), 1216*(1+z)], [0, max(Flux_gen)])
        plt.plot([1549*(1+z), 1549*(1+z)], [0, max(Flux_gen)])
        plt.plot([1909*(1+z), 1909*(1+z)], [0, max(Flux_gen)])
        plt.plot([1400*(1+z), 1400*(1+z)], [0, max(Flux_gen)])
        plt.title("Number {}".format(n) + "   z = {}".format(round(z, 2)))
        plt.legend(["data","Lyman alpha", "C-IV", "C-III", "Si-IV"])
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Relative Flux")
        n += 1
        plt.show()



def extract_wave_flux(Data):
    return Data[1].data.WAVE[0], Data[1].data.FLUX[0]


if False:
    X = []
    for i in range(len(data_list)):
        data_points = []
        WAVE, FLUX = extract_wave_flux(data_list[i])
        WAVE = np.array(WAVE)
        FLUX = np.array(FLUX)
        for j in range(len(WAVE)):
            data_points.append([WAVE[j], FLUX[j]])
        X.append(np.array(data_points))

    X = np.array([x.flatten() for x in X])

    y = []
    for i in range(len(df.REDSHIFT)):
        y.append(df.REDSHIFT[i])
    y = np.array(y)

if False:
#    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    model = XGBRegressor()
#    model = RandomForestRegressor()
    model.fit(X[0:150], y[0:150])

    Y = model.predict(X[150:200])

    print(mean_squared_error(Y, y[150:200], squared=False))

    diff = []
    for i in range(len(Y)):
        diff.append(abs(Y[i]-y[150:200][i]))


#X_1, _, _, _ = plot_improved_data(data_list[98], redshift=df.REDSHIFT[98])



#generalize_datapoints(data_list[14], x_lim=(4250, 6000), n=2)
_, X_1 = generalize_datapoints(data_list[66])
_, X_2 = generalize_datapoints(data_list[96])
_, X_3 = generalize_datapoints(data_list[166])


# for i in range(30):
#    generalize_datapoints(data_list[i])

plt.show()
#plt.close()
