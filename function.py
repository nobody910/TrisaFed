# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 19:44:15 2021

@author: liush
"""
import matplotlib.pyplot as plt

def plot_result(data,plot_label,ylabel):
    plt.plot(data, label=plot_label)
    plt.xlabel('round')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()