import ROOT

import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
from hipe4ml.tree_handler import TreeHandler
import os
import sys
sys.path.append('../utils')
import utils as utils

###Pre settings parameters###
# mc_file_path = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass4/AO2D.root"
# mc_file_path = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/AO2D.root"
# mc_file_path = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11/AO2D.root"
# mc_file_path = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11_G4list/AO2D.root"
# mc_file_path = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11/AO2D_CustomV0s.root"
mc_file_path = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11_G4list/AO2D_CustomV0s.root"
# mc_file_path = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11/AO2D_V0s_full.root"
# mc_file_path = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11_G4list/AO2D_V0s_full.root"
output_dir = "../../results/ep5/MC_QC"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
suffix = 'LHC25g11_G4list_CustomV0s'
#suffix = 'LHC25g11_G4list'
#suffix = 'LHC24i5_latest'
data_file_path = "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/apass5/AO2D_HadronPID.root"

pt_bins = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8]
ct_bins = [0, 2, 4, 6, 8, 10, 14, 18, 23, 35]
# pt_bins_for_ct_acc = [2, 3.5, 5, 8]
# ct_bins_for_ct_acc = [[1, 2, 4, 6, 8, 10, 14, 18, 35],
#                       [1, 2, 4, 6, 8, 10, 14, 18, 35],
#                       [1, 2, 4, 6, 8, 10, 14, 23]]
pt_bins_for_ct_acc = [2, 3, 4, 5.5, 8]
ct_bins_for_ct_acc = [[1, 3, 6, 9, 12, 18, 30],
                      [1, 3, 6, 9, 12, 18, 25],
                      [1, 3, 6, 9, 15, 25],
                      [1, 3, 6, 10, 23]]

mc_hdl = TreeHandler(mc_file_path,'O2mchypcands', folder_name='DF*')
utils.correct_and_convert_df(mc_hdl, calibrate_he3_pt=False, isMC=True)
mc_hdl_evsel = mc_hdl.apply_preselections('fIsSurvEvSel==True', inplace=False)
mc_reco_hdl = mc_hdl.apply_preselections('fIsReco == 1', inplace=False)
# ###GentPt distribution
# gent_pt = mc_hdl.get_data_frame()['fGenPt']
# plt.figure(figsize=(8,6))
# plt.hist(gent_pt, bins=200, histtype='step', color='blue')
# plt.xlabel('GentPt (GeV/c)')
# plt.ylabel('Entries')
# plt.title('GentPt Distribution' + '_' + suffix)
# plt.grid(True)
# plt.savefig(f'{output_dir}/GentPt_{suffix}.pdf')
# plt.close()
# ### GentCt distribution
# gent_ct = mc_hdl.get_data_frame()['fGenCt']
# plt.figure(figsize=(8,6))
# plt.hist(gent_ct, bins=200, histtype='step', color='blue')
# plt.xlabel('GentCt (cm)')
# plt.ylabel('Entries')
# plt.title('GentCt Distribution' + '_' + suffix)
# plt.grid(True)
# plt.savefig(f'{output_dir}/GentCt_{suffix}.pdf')
# plt.close()
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
x = np.array([(pt_bins[i]+pt_bins[i+1])/2 for i in range(len(pt_bins)-1)])
y = np.array(acc_arr)
err = np.array(acc_err_arr)
lower = np.clip(y - err, 0.0, 1.0)
upper = np.clip(y + err, 0.0, 1.0)
plt.fill_between(x, lower, upper, color='#cfe8ff', alpha=0.6, label='1σ band')
plt.plot(x, y, '-o', color='#0b3d91', markeredgecolor='#0b3d91', markerfacecolor='white', markersize=7, linewidth=2)
plt.scatter(x, y, color='#0b3d91', s=30)
plt.xlabel('pT (GeV/c)')
plt.ylabel('Acceptance')
plt.title('Acceptance vs pT' + '_' + suffix)
plt.ylim(-0.02, 1.02)
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
plt.grid(which='major', linestyle='--', alpha=0.5)
plt.grid(which='minor', linestyle=':', alpha=0.2)
plt.minorticks_on()
plt.legend()
plt.tight_layout()
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
x = np.array([(ct_bins[i]+ct_bins[i+1])/2 for i in range(len(ct_bins)-1)])
y = np.array(acc_arr)
err = np.array(acc_err_arr)
lower = np.clip(y - err, 0.0, 1.0)
upper = np.clip(y + err, 0.0, 1.0)
plt.fill_between(x, lower, upper, color='#fdebd0', alpha=0.6, label='1σ band')
plt.plot(x, y, '-s', color='#b03a2e', markeredgecolor='#b03a2e', markerfacecolor='white', markersize=7, linewidth=2)
plt.scatter(x, y, color='#b03a2e', s=30)
plt.xlabel('ct (cm)')
plt.ylabel('Acceptance')
plt.title('Acceptance vs ct' + '_' + suffix)
plt.ylim(-0.02, 1.02)
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
plt.grid(which='major', linestyle='--', alpha=0.5)
plt.grid(which='minor', linestyle=':', alpha=0.2)
plt.minorticks_on()
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/Acceptance_vs_ct_{suffix}.pdf')
plt.close()

###ct acc in pt bins
for i_pt in range(len(pt_bins_for_ct_acc)-1):
    pt_min = pt_bins_for_ct_acc[i_pt]
    pt_max = pt_bins_for_ct_acc[i_pt+1]
    pt_sel = f'fAbsGenPt>{pt_min} & fAbsGenPt<{pt_max}'
    acc_arr = []
    acc_err_arr = []
    for i_ct in range(len(ct_bins_for_ct_acc[i_pt])-1):
        ct_min = ct_bins_for_ct_acc[i_pt][i_ct]
        ct_max = ct_bins_for_ct_acc[i_pt][i_ct+1]
        ct_sel = f'fGenCt>{ct_min} & fGenCt<{ct_max}'
        bin_mc_evsel = mc_hdl_evsel.apply_preselections(pt_sel + ' & ' + ct_sel, inplace=False)
        bin_mc_reco_hdl = mc_reco_hdl.apply_preselections(pt_sel + ' & ' + ct_sel, inplace=False)
        acc = len(bin_mc_reco_hdl)/len(bin_mc_evsel) if len(bin_mc_evsel)>0 else 0
        acc_err = np.sqrt(acc*(1-acc)/len(bin_mc_evsel)) if len(bin_mc_evsel)>0 else 0
        acc_arr.append(acc)
        acc_err_arr.append(acc_err)
    plt.figure(figsize=(8,6))
    x = np.array([(ct_bins_for_ct_acc[i_pt][i]+ct_bins_for_ct_acc[i_pt][i+1])/2 for i in range(len(ct_bins_for_ct_acc[i_pt])-1)])
    y = np.array(acc_arr)
    err = np.array(acc_err_arr)
    lower = np.clip(y - err, 0.0, 1.0)
    upper = np.clip(y + err, 0.0, 1.0)
    plt.fill_between(x, lower, upper, color='#e8f8f5', alpha=0.6, label='1σ band')
    plt.plot(x, y, '-^', color='#117a65', markeredgecolor='#117a65', markerfacecolor='white', markersize=7, linewidth=2)
    plt.scatter(x, y, color='#117a65', s=30)
    plt.xlabel('ct (cm)')
    plt.ylabel('Acceptance')
    plt.title(f'Acceptance vs ct in pT bin {pt_min}-{pt_max} GeV/c' + '_' + suffix)
    plt.ylim(-0.02, 1.02)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.grid(which='major', linestyle='--', alpha=0.5)
    plt.grid(which='minor', linestyle=':', alpha=0.2)
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Acceptance_vs_ct_pTbin{pt_min}_{pt_max}_{suffix}.pdf')
    plt.close()
    try:
        combined_ct_x_list
    except NameError:
        combined_ct_x_list = []
        combined_acc_y_list = []
        combined_err_list = []
        combined_labels = []

    combined_ct_x_list.append(x)
    combined_acc_y_list.append(y)
    combined_err_list.append(err)
    combined_labels.append(f'{pt_min}-{pt_max}')

    if i_pt == len(pt_bins_for_ct_acc) - 2:
        plt.figure(figsize=(10,7))
        cmap = plt.get_cmap('tab10')
        for idx, (xx, yy, ee, lab) in enumerate(zip(combined_ct_x_list, combined_acc_y_list, combined_err_list, combined_labels)):
            color = cmap(idx % 10)
            lower_all = np.clip(yy - ee, 0.0, 1.0)
            upper_all = np.clip(yy + ee, 0.0, 1.0)
            plt.fill_between(xx, lower_all, upper_all, color=color, alpha=0.15)
            plt.plot(xx, yy, marker='o', linestyle='-', color=color, label=f'pT {lab} GeV/c', linewidth=2)
            plt.scatter(xx, yy, color=color, s=30)
        plt.xlabel('ct (cm)')
        plt.ylabel('Acceptance')
        plt.title('Acceptance vs ct for all pT bins' + '_' + suffix)
        plt.ylim(-0.02, 1.02)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
        plt.grid(which='major', linestyle='--', alpha=0.5)
        plt.grid(which='minor', linestyle=':', alpha=0.2)
        plt.minorticks_on()
        plt.legend(title='pT bins', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/Acceptance_vs_ct_all_pTbins_{suffix}.pdf')
        plt.close()


###H3l radious reconstructed 
rad_reco = mc_reco_hdl.get_data_frame()['fDecRad']
plt.figure(figsize=(8,6))
plt.hist(rad_reco, bins=200, histtype='step', color='blue')
plt.xlabel('Decay Radious (cm)')
plt.ylabel('Entries')
plt.title('Decay Radious Distribution' + '_' + suffix)
plt.grid(True)
plt.savefig(f'{output_dir}/Decay_Radious_{suffix}.pdf')
plt.close()
###H3l radious data
# data_hdl = TreeHandler(data_file_path,'O2hypcands', folder_name='DF*')
# utils.correct_and_convert_df(data_hdl, calibrate_he3_pt=False, isMC=False)
# rad_reco_data = data_hdl.get_data_frame()['fDecRad']
# plt.figure(figsize=(8,6))
# plt.hist(rad_reco_data, bins=200, histtype='step', color='blue')
# plt.xlabel('Decay Radious (cm)')
# plt.ylabel('Entries')
# plt.title('Decay Radious Data Distribution')
# plt.grid(True)
# plt.savefig(f'{output_dir}/Decay_Radious_data.pdf')
# plt.close()


