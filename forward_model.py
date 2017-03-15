import numpy as np
import matplotlib.pyplot as plt


def make_elev():
    e = np.linspace(0,90,181)
    return e

if __name__ == '__main__':
    e = make_elev()
    print(e)
