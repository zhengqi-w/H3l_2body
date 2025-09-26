import argparse
import yaml
import uproot
import ROOT
import numpy as np
import os

import sys
sys.path.append('../utils')
import utils as utils
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def exp_func(x, a, tau):
    return a * np.exp(-x / tau)

parser = argparse.ArgumentParser(description='Configure the parameters of the script.')
parser.add_argument('--config-file', dest='config_file', help="path to the YAML file with configuration.", default='')
args = parser.parse_args()
if args.config_file == "":
    print('** No config file provided. Exiting. **')
    exit()

config_file = open(args.config_file, 'r')
config = yaml.full_load(config_file)

pt_bins = config['pt_bins']
n_pt_bins = len(pt_bins) - 1
ct_bins = config['ct_bins']
mc_file = config['mc_file']
org_tao = config['org_tao']
save_path = config['save_path']
name_suffix = config['name_suffix']
fig_path = save_path + '/' + name_suffix
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
tree = uproot.open(mc_file)['he3candidates'].arrays(library="pd")
HE_3_MASS = 2.809230089
# definations for all histograms
h_he_ct_list = []
h_he_ct_abso_list = []
h_he_ct_ratio_list = []
for i in range(n_pt_bins):
    h_he_ct = {"mat": ROOT.TH1F(f'h_he_ct_mat_pt{i}', f'He3 ct (cm) pt bin {i}', len(ct_bins[i])-1, np.array(ct_bins[i], dtype=np.float32)),
                "antimat": ROOT.TH1F(f'h_he_ct_antimat_pt{i}', f'Absorption ct (cm) pt bin {i}', len(ct_bins[i])-1, np.array(ct_bins[i], dtype=np.float32)),
                "both": ROOT.TH1F(f'h_he_ct_both_pt{i}', f'He3 ct (cm) pt bin {i}', len(ct_bins[i])-1, np.array(ct_bins[i], dtype=np.float32))}
    h_he_ct_list.append(h_he_ct)
    h_he_ct_abso = {"mat": ROOT.TH1F(f'h_he_ct_abso_mat_pt{i}', f'He3 ct (cm) pt bin {i}', len(ct_bins[i])-1, np.array(ct_bins[i], dtype=np.float32)),
                    "antimat": ROOT.TH1F(f'h_he_ct_abso_antimat_pt{i}', f'Absorption ct (cm) pt bin {i}', len(ct_bins[i])-1, np.array(ct_bins[i], dtype=np.float32)),
                    "both": ROOT.TH1F(f'h_he_ct_abso_both_pt{i}', f'He3 ct (cm) pt bin {i}', len(ct_bins[i])-1, np.array(ct_bins[i], dtype=np.float32))}
    h_he_ct_abso_list.append(h_he_ct_abso)
    h_he_ct_ratio  = {"mat": ROOT.TH1F(f'h_he_ct_ratio_mat_pt{i}', f'He3 ct (cm) pt bin {i}', len(ct_bins[i])-1, np.array(ct_bins[i], dtype=np.float32)),
                    "antimat": ROOT.TH1F(f'h_he_ct_ratio_antimat_pt{i}', f'Absorption ct (cm) pt bin {i}', len(ct_bins[i])-1, np.array(ct_bins[i], dtype=np.float32)),
                    "both": ROOT.TH1F(f'h_he_ct_ratio_both_pt{i}', f'He3 ct (cm) pt bin {i}', len(ct_bins[i])-1, np.array(ct_bins[i], dtype=np.float32))}
    h_he_ct_ratio_list.append(h_he_ct_ratio)
h_tao_pt = ROOT.TH1F('h_tao_pt', 'He3 p_{T} (GeV/c)', len(pt_bins)-1, np.array(pt_bins, dtype=np.float32))

# for ct efficiency analysis we don't reweight MC pT spectrum
# spectra_file = ROOT.TFile.Open('../utils/H3L_BWFit.root')
# he3_spectrum = spectra_file.Get('BlastWave_H3L_10_30')
# spectra_file.Close()
# utils.reweight_pt_spectrum(tree, 'pt', he3_spectrum)
# tree = tree.query('rej>0')
# Split the DataFrame into a list of DataFrames according to pt bins

tree_pt_bins = []
for i in range(n_pt_bins):
    pt_min = pt_bins[i]
    pt_max = pt_bins[i+1]
    tree_bin = tree[(tree['pt'] >= pt_min) & (tree['pt'] < pt_max)].copy()
    tree_pt_bins.append(tree_bin)

for i, tree_bin in enumerate(tree_pt_bins):
    absorpted_list = []
    gentct_list = []
    absoct_list = []
    for (pt, eta, phi, absoX, absoY, absoZ, pdg) in zip(tree_bin['pt'], tree_bin['eta'], tree_bin['phi'], tree_bin['absoX'], tree_bin['absoY'], tree_bin['absoZ'], tree_bin['pdg']):
        mat_type = 'mat' if pdg > 0 else 'antimat'
        he3TLV = ROOT.TLorentzVector()
        he3TLV.SetPtEtaPhiM(pt, eta, phi, HE_3_MASS)
        he3p = he3TLV.P()
        absoR = np.sqrt(absoX**2 + absoY**2)
        absoL = np.sqrt(absoX**2 + absoY**2 + absoZ**2)
        absoCt = absoL * HE_3_MASS / he3p
        decCt = ROOT.gRandom.Exp(org_tao)
        absorpted_list.append(absoCt < decCt)
        gentct_list.append(decCt)
        absoct_list.append(absoCt)
        h_he_ct_list[i][mat_type].Fill(decCt)
        h_he_ct_list[i]['both'].Fill(decCt)
        if absoCt > decCt:
            h_he_ct_abso_list[i][mat_type].Fill(decCt)
            h_he_ct_abso_list[i]['both'].Fill(decCt)
    tree_bin['absorpted'] = absorpted_list
    tree_bin['gentct'] = gentct_list
    tree_bin['absoct'] = absoct_list

for i in range(n_pt_bins):
    for mat_type in ['mat', 'antimat', 'both']:
        h_he_ct_ratio_list[i][mat_type].Divide(h_he_ct_abso_list[i][mat_type], h_he_ct_list[i][mat_type], 1, 1, "B")

for i, tree_bin in enumerate(tree_pt_bins):
    fig, ax = plt.subplots()
    # 全部gentct分布
    gentct_all = np.array(tree_bin['gentct'])
    bins = np.linspace(0, 40, 21)
    counts_all, bin_edges = np.histogram(gentct_all, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.hist(gentct_all, bins=bins, alpha=0.5, label='All', color='blue')
    # 拟合全部gentct
    try:
        popt_all, pcov_all = curve_fit(exp_func, bin_centers, counts_all, p0=(counts_all[0], 7.6), maxfev=10000)
        ax.plot(bin_centers, exp_func(bin_centers, *popt_all), 'b--', label=f'All fit: tau={popt_all[1]:.2f}±{np.sqrt(np.diag(pcov_all))[1]:.2f}')
    except Exception as e:
        print(f"Fit failed for all gentct in pt bin {i}: {e}")
        popt_all = [0, 0]
        pcov_all = np.zeros((2,2))
    # absorpted==0的gentct分布
    mask = np.array(tree_bin['absorpted']) == 0
    gentct_unabsorbed = gentct_all[mask]
    counts_unabs, _ = np.histogram(gentct_unabsorbed, bins=bins)
    ax.hist(gentct_unabsorbed, bins=bins, alpha=0.5, label='Unabsorbed', color='red')
    # 拟合absorpted==0
    try:
        popt_unabs, pcov_unabs = curve_fit(exp_func, bin_centers, counts_unabs, p0=(counts_unabs[0], 7.6), maxfev=10000)
        ax.plot(bin_centers, exp_func(bin_centers, *popt_unabs), 'r--', label=f'Unabs fit: tau={popt_unabs[1]:.2f}±{np.sqrt(np.diag(pcov_unabs))[1]:.2f}')
        # 填入h_tao_pt
        h_tao_pt.SetBinContent(i+1, popt_unabs[1])
        h_tao_pt.SetBinError(i+1, np.sqrt(np.diag(pcov_unabs))[1])
    except Exception as e:
        print(f"Fit failed for unabsorbed gentct in pt bin {i}: {e}")
        h_tao_pt.SetBinContent(i+1, 0)
        h_tao_pt.SetBinError(i+1, 0)
    ax.set_xlabel('gentct')
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    ax.set_title(f'gentct distribution pt: {pt_bins[i]:.1f}-{pt_bins[i+1]:.1f} GeV/c')
    ax.legend()
    plt.savefig(f'{fig_path}/gentct_distribution_pt{i}.pdf')
    plt.close()

out_file = ROOT.TFile(f'{save_path}/absorption_ct_histos_{name_suffix}.root', 'RECREATE')
out_file.cd()
h_tao_pt.Write()
for i in range(n_pt_bins):
    out_file.mkdir(f'pt_{pt_bins[i]:.1f}_{pt_bins[i+1]:.1f}').cd()
    for mat_type in ['mat', 'antimat', 'both']:
        h_he_ct_list[i][mat_type].Write()
        h_he_ct_abso_list[i][mat_type].Write()
        h_he_ct_ratio_list[i][mat_type].Write()
    out_file.cd()
out_file.Close()
