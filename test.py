# Visulaise all spectra in a folder
import pandas as pd
import os
from parse import *

path = "LSC Spectra for AI proj including calibration certs/"

# files = []
# for file in os.listdir(dir):
#     if file.endswith(".001"):
#         files.append(os.path.join(dir, file))
# print(files)

# for file in files:
#     spect_list,first_num,isotope = parse_file(file)
#     quick_plot(spect_list,file,first_num,isotope)

df = data_from_files(path)
print(df)


df.to_excel('test.xlsx')