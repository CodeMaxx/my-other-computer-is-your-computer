import os
import sys
import pandas as pd
from random import shuffle
import threading

if len(sys.argv) < 2:
    print("Usage: %s <num_samples_to_extract>")
    sys.exit(1)

p = pd.read_csv("trainLabels.csv")
filenames = list(p[p.columns[0]])
shuffle(filenames)

NUM_THREADS = 20

num_samples_to_extract = int(sys.argv[1])

os.system("rm -rf trainingData; mkdir trainingData")


class Extract(threading.Thread):
    def __init__(self, start_index, num):
        threading.Thread.__init__(self)
        self.start_index = start_index
        self.end_index = start_index + num

    def run(self):
        for i in range(self.start_index, self.end_index):
            os.system("7z x train.7z train/" +
                      filenames[i] + ".* -otrainingData/ > /dev/null")


def main():
    threads = []
    num = int(num_samples_to_extract/NUM_THREADS)
    for i in range(NUM_THREADS):
        t = Extract(num*i, num)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

if __name__ == '__main__':
    main()
