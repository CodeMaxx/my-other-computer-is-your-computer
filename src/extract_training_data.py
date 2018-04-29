#! /usr/bin/env python3

#  Copyright(c) Akash Trehan
#  All rights reserved.

#  This code is licensed under the MIT License.

#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files(the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:

#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.

#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

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
