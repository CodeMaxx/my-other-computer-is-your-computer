import os
import sys
import pandas as pd
from random import shuffle

if len(sys.argv) < 2:
    print("Usage: %s <num_samples_to_extract>")
    sys.exit(1)

p = pd.read_csv("trainLabels.csv")
filenames = list(p[p.columns[0]])
shuffle(filenames)

num_samples_to_extract = int(sys.argv[1])

os.system("rm -rf trainingData; mkdir trainingData")

for i in range(len(filenames)):
    if i < num_samples_to_extract:
        os.system("7z x train.7z train/" + filenames[i] + ".asm -o trainingData/" + filenames[i] + ".asm")
        os.system("7z x train.7z train/" + filenames[i] + ".bytes -o trainingData/" + filenames[i] + ".bytes")
    else:
        break
