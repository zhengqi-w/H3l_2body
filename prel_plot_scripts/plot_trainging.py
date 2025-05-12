import ROOT

import numpy as np
import uproot
import matplotlib.pyplot as plt
import pandas as pd
from hipe4ml.tree_handler import TreeHandler
from scipy.optimize import curve_fit
from scipy import integrate

import pdb 

efficiency_arr = np.load('/Users/zhengqingwang/alice/run3task/HyperRoutine/results/pass4full/training_test/efficiency_arr.npy')
score_efficiency_arr = np.load('/Users/zhengqingwang/alice/run3task/HyperRoutine/results/pass4full/training_test/score_efficiency_arr.npy')
plt.figure()
plt.scatter(score_efficiency_arr, efficiency_arr, c='red', alpha=0.7)
plt.xlabel('BDT Score')
plt.ylabel('Efficiency')
plt.title('Efficiency vs BDT Score')
plt.savefig("/Users/zhengqingwang/alice/run3task/HyperRoutine/results/pass4full/training_test/efficiency.png", bbox_inches='tight')
plt.clf()

hdl_output = TreeHandler("/Users/zhengqingwang/alice/run3task/HyperRoutine/results/pass4full/training_test/dataH.parquet.parquet.gzip")
hdl_output.print_summary()
print("candidate_num:",len(hdl_output))

filtered_data = hdl_output.apply_preselections("model_output > 3.7 and fMassH3L<3.03 and fMassH3L>2.96", inplace=False)["fMassH3L"]
plt.figure()
plt.hist(filtered_data, bins=80, label='model_output > 3.7', alpha=0.7)
plt.xlabel('fMassH3L')
plt.ylabel('Counts')
plt.title('Invariant Mass Distribution after Selections')
plt.legend()
plt.savefig("/Users/zhengqingwang/alice/run3task/HyperRoutine/results/pass4full/training_test/invmass_after.png", bbox_inches='tight')
plt.clf()

fit_range = [2.96, 3.03]
counts, bins = np.histogram(filtered_data, bins=80, range=fit_range)
def gaus(x,N,mu,sigma):
    return N*np.exp(-(x-mu)**2/(2*sigma**2))
def pol2(x,a,b,c):
    return a*x**2 + b*x + c
def fit_func(x, a, b, c, N, mu, sigma):
    return gaus(x, N, mu, sigma) + pol2(x, a, b, c)

x_point = 0.5*(bins[1:]+bins[:-1])
r = np.arange(fit_range[0], fit_range[1], 0.00001)

popt, pcov = curve_fit(fit_func, x_point, counts, p0=[100, -3, 0.1, 100, 2.99, 0.001])
plt.errorbar(x_point, counts, yerr=np.sqrt(counts), fmt='.', ecolor='k', color='k', elinewidth=1., label='Data_model_output > 3.7')
plt.plot(r, gaus(r,N=popt[3],mu=popt[4],sigma=popt[5]), label='Gaussian', color='red')
plt.plot(r, pol2(r,a=popt[0],b=popt[1],c=popt[2]), label='Polynomial', color='green', linestyle='--')
plt.plot(r, fit_func(r, *popt), label='pol2+Gaus', color='blue')
signal = integrate.quad(gaus, fit_range[0], fit_range[1], args=(popt[3], popt[4], popt[5]))[0] / ((fit_range[1]-fit_range[0]) / bins)
background = integrate.quad(pol2, fit_range[0], fit_range[1], args=(popt[0], popt[1], popt[2]))[0] / ((fit_range[1]-fit_range[0]) / bins) 
significance = signal / np.sqrt(signal + background)
print("",popt)
print("Significance:",significance)
print("Signal counts:",signal)
print("Background counts:",background)

plt.xlabel('$M_{^{3}_{}He\pi}$ $(\mathrm{GeV/}\it{c}^2)$')
plt.ylabel('Counts')
plt.legend()
plt.savefig("/Users/zhengqingwang/alice/run3task/HyperRoutine/results/pass4full/training_test/invmass_fit.png", bbox_inches='tight')
plt.clf()
