import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import math as m

# It would seem, that the absorptions are stored in the abs_templates folder.
# It shows deviation from "perfect" quasar spectrum

directory_path = 'C:\\Users\\mikke\\Desktop\\Kandidat_Stuff\\simpaqs-main\\simpaqs-main\\output'


temp_file_list = os.listdir(directory_path + '\\abs_templates')
data_list = [fits.open(directory_path + '\\abs_templates' + '\\' + file_name) for file_name in temp_file_list]

data_abs = pd.read_csv(directory_path + '\\list_absorbers.csv')
data_dlas = pd.read_csv(directory_path + '\\list_dlas.csv')
data_templ = pd.read_csv(directory_path + '\\list_templates.csv')

model_param = fits.open(directory_path + '\\' + 'quasar_models\\model_parameters.fits')[1].data
model_input = pd.read_csv(directory_path + '\\quasar_models\\model_input.csv')
Quasar_0 = fits.open(directory_path + '\\' + 'quasar_models\\PAQS_quasar_000001.fits')

def plot_abs_temp(n, range=None):
    data_point = data_list[n][1].data

    x = []
    y = []

    for i in data_point:
        x.append(i[0])
        y.append(i[1])

    plt.plot(x,y)
    if range != None:
        plt.xlim(range)
    plt.show()

#for i in range(5):
#    plot_abs_temp(i)

#plot_abs_temp(0)

#plt.hist(data_abs['Z_TOT'])

#plt.plot(Quasar_0[1].data['LAMBDA'], [Quasar_0[1].data['FLUX_DENSITY'][i]*data_list[0][1].data[i][1] for i in range(len(Quasar_0[1].data['LAMBDA']))])

plt.show()
