import ROOT

import numpy as np
import uproot
import matplotlib.pyplot as plt
import pandas as pd
from hipe4ml.tree_handler import TreeHandler
import os
import sys
sys.path.append('../utils')
import utils as utils

###Pre settings parameters###
mc_file_path = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass4/AO2D.root"
# mc_file_path = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/AO2D.root"
#mc_file_path = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11/AO2D.root"
#mc_file_path = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11_G4list/AO2D.root"
output_dir = "../../results/ep4/MC_QC"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#suffix = 'LHC25g11'
#suffix = 'LHC25g11_G4list'
suffix = 'LHC24i5_latest'

pt_bins = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8]
ct_bins = [0, 2, 4, 6, 8, 10, 14, 18, 23, 35]

mc_hdl = TreeHandler(mc_file_path,'O2mchypcands', folder_name='DF*')
utils.correct_and_convert_df(mc_hdl, calibrate_he3_pt=False, isMC=True)
mc_hdl_evsel = mc_hdl.apply_preselections('fIsSurvEvSel==True', inplace=False)
mc_reco_hdl = mc_hdl.apply_preselections('fIsReco == 1', inplace=False)
###GentPt distribution
gent_pt = mc_hdl.get_data_frame()['fGenPt']
plt.figure(figsize=(8,6))
plt.hist(gent_pt, bins=200, histtype='step', color='blue')
plt.xlabel('GentPt (GeV/c)')
plt.ylabel('Entries')
plt.title('GentPt Distribution' + '_' + suffix)
plt.grid(True)
plt.savefig(f'{output_dir}/GentPt_{suffix}.pdf')
plt.close()
### GentCt distribution
gent_ct = mc_hdl.get_data_frame()['fGenCt']
plt.figure(figsize=(8,6))
plt.hist(gent_ct, bins=200, histtype='step', color='blue')
plt.xlabel('GentCt (cm)')
plt.ylabel('Entries')
plt.title('GentCt Distribution' + '_' + suffix)
plt.grid(True)
plt.savefig(f'{output_dir}/GentCt_{suffix}.pdf')
plt.close()
###
###ct Resolution = (GentCt - Ct)/ GentCt vs GentCt
gent_ct_reco = mc_reco_hdl.get_data_frame()['fGenCt']
ct_reco = mc_reco_hdl.get_data_frame()['fCt']
ct_res = (gent_ct_reco - ct_reco)/gent_ct_reco
plt.figure(figsize=(8,6))
plt.hist2d(gent_ct_reco, ct_res, bins=[100,100], range=[[0,40],[-2,2]], cmap='viridis')
plt.colorbar(label='Entries')
plt.xlabel('GentCt (cm)')
plt.ylabel('(GentCt - Ct)/GentCt')
plt.title('Ct Resolution vs GentCt' + '_' + suffix)
plt.grid(True)
plt.savefig(f'{output_dir}/Ct_Resolution_vs_GentCt_{suffix}.pdf')
plt.close()

###pt acc
acc_arr = []
acc_err_arr = []
for i_pt in range(len(pt_bins)-1):
    pt_min = pt_bins[i_pt]
    pt_max = pt_bins[i_pt+1]
    pt_sel = f'fAbsGenPt>{pt_min} & fAbsGenPt<{pt_max}'
    bin_mc_evsel = mc_hdl_evsel.apply_preselections(pt_sel, inplace=False)
    bin_mc_reco_hdl = mc_reco_hdl.apply_preselections(pt_sel, inplace=False)
    acc = len(bin_mc_reco_hdl)/len(bin_mc_evsel)
    acc_err = np.sqrt(acc*(1-acc)/len(bin_mc_evsel)) if len(bin_mc_evsel)>0 else 0
    acc_arr.append(acc)
    acc_err_arr.append(acc_err)
plt.figure(figsize=(8,6))
plt.errorbar([(pt_bins[i]+pt_bins[i+1])/2 for i in range(len(pt_bins)-1)], acc_arr, yerr=acc_err_arr, fmt='o', color='blue', ecolor='red', capsize=5)
plt.xlabel('pT (GeV/c)')
plt.ylabel('Acceptance')
plt.title('Acceptance vs pT' + '_' + suffix)
plt.grid(True)
plt.savefig(f'{output_dir}/Acceptance_vs_pT_{suffix}.pdf')
plt.close()
###ct acc
acc_arr = []
acc_err_arr = []
for i_ct in range(len(ct_bins)-1):
    ct_min = ct_bins[i_ct]
    ct_max = ct_bins[i_ct+1]
    ct_sel = f'fGenCt>{ct_min} & fGenCt<{ct_max}'
    bin_mc_evsel = mc_hdl_evsel.apply_preselections(ct_sel, inplace=False)
    bin_mc_reco_hdl = mc_reco_hdl.apply_preselections(ct_sel, inplace=False)
    acc = len(bin_mc_reco_hdl)/len(bin_mc_evsel)
    acc_err = np.sqrt(acc*(1-acc)/len(bin_mc_evsel)) if len(bin_mc_evsel)>0 else 0
    acc_arr.append(acc)
    acc_err_arr.append(acc_err)
plt.figure(figsize=(8,6))
plt.errorbar([(ct_bins[i]+ct_bins[i+1])/2 for i in range(len(ct_bins)-1)], acc_arr, yerr=acc_err_arr, fmt='o', color='blue', ecolor='red', capsize=5)
plt.xlabel('ct (cm)')
plt.ylabel('Acceptance')
plt.title('Acceptance vs ct' + '_' + suffix)
plt.grid(True)
plt.savefig(f'{output_dir}/Acceptance_vs_ct_{suffix}.pdf')
plt.close()

###H3l mass distribution

