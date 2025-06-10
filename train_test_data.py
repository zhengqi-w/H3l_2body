import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from signal_extraction import SignalExtraction
import ROOT
import pandas as pd

from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
import hipe4ml.analysis_utils as au
import hipe4ml.plot_utils as pu

import mplhep as mpl
import xgboost as xgb

import sys
sys.path.append('utils')
import utils as utils

import yaml
import pdb

matplotlib.use('pdf')
plt.style.use(mpl.style.ALICE)


###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--config-file', dest='config_file', help="path to the YAML file with configuration.", default='')
args = parser.parse_args()

config_file = args.config_file

with open(config_file, 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


do_training = params['do_training']
do_application = params['do_application']

cen_pt_bins_dic = params['cen_pt_bins_dic']

input_data_path = params['input_data_path']
input_mc_path = params['input_mc_path']
output_base_dir = params['output_base_dir']

training_preselections = params['training_preselections']
training_variables = params['training_variables']
test_set_size = params['test_set_size']
bkg_fraction_max = params["background_over_signal_max"]
random_state = params["random_state"]
hyperparams = params["hyperparams"]
opean_NSigmaH3_signal_shift = params["opean_NSigmaH3_signal_shift"]

input_AnalysisResults_file_path = params['input_AnalysisResults_file_path']
input_H3l_BWFit_file_path = params['input_H3l_BWFit_file_path']

calibrate_he_momentum = params['calibrate_he_momentum']
BDT_values_test = params['BDT_values_test']
input_absorption_file_path= params['input_absorption_file_path']

def gauss_pol3(x, A, mu, sigma, B, C, D, E):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + B * x**3 + C * x**2 + D * x + E


if not cen_pt_bins_dic:
    print("Fatal::: not knowing which centrality and pt bin to run on exit!")
    exit()
### create output directory(list) if it does not exist
output_dir_list = []
output_figurs_dir_list = []
for cen_bin, pt_bins in cen_pt_bins_dic.items():
    output_dir_sublist = []
    output_figurs_dir_sublist = []
    cen_min, cen_max = map(int, cen_bin.split("_"))
    for i in range(len(pt_bins)-1):
        output_premium_prase = f'/cen_{cen_min}_{cen_max}_pt_{pt_bins[i]}_{pt_bins[i+1]}'
        output_dir = output_base_dir + output_premium_prase
        output_figurs_dir = output_dir + "/figures_ML"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(output_figurs_dir):
            os.makedirs(output_figurs_dir)
        output_dir_sublist.append(output_dir)
        output_figurs_dir_sublist.append(output_figurs_dir)
    output_dir_list.append(output_dir_sublist)
    output_figurs_dir_list.append(output_figurs_dir_sublist)

print('**********************************')
print('    Running train_test_data.py    ')
print('**********************************')

if do_training:

        signalH = TreeHandler(input_mc_path, "O2mchypcands", folder_name='DF*')
        bkgH = TreeHandler(input_data_path, "O2hypcands", folder_name='DF*')
        mc_hdl = []

        utils.correct_and_convert_df(signalH, calibrate_he3_pt=calibrate_he_momentum, isMC=True)
        utils.correct_and_convert_df(bkgH, calibrate_he3_pt=calibrate_he_momentum, isMC=False)

        # reweight MC pT spectrum and plot GentCt - Ct in the same time
        df_signal = signalH.get_data_frame() # or just signalH._full_data_frame
        print(df_signal)
        # column_GentCt = df_signal["fGenCt"]
        # column_Ct = df_signal["fCt"]
        # column_difference = (column_GentCt - column_Ct)
        # plt.figure(figsize=(10, 6))
        # plt.hist(column_difference, bins=200, color='blue', alpha=0.7, label='GentCt - Ct')
        # #plt.axvline(0, color='red', linestyle='--', label='Zero Line')  # 可选，标记零点
        # plt.xlabel('GentCt - Ct (cm)')
        # plt.ylabel('Counts')
        # plt.ylim(0, 150000)
        # plt.title('')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(output_base_dir + '/GentCt_Ct.pdf')
        # plt.close("all")
        # plt.plot(column_GentCt, column_difference, label='(Gentct-ct)/Gentct vs Gentct', marker='o')
        # plt.xlabel('GentCt (cm)')
        # plt.ylabel('(GentCt - Ct)/GentCt')
        # plt.title('')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(output_base_dir + '/Gentct_ct_div_vs_Gentct.pdf')
        # plt.close("all")
        spectra_file = ROOT.TFile.Open('utils/H3L_BWFit.root')
        reweighted_df = pd.DataFrame()
        prev_pt_bins = None     
        for cen_range, pt_bins in cen_pt_bins_dic.items():
            reweighted_df_cen = pd.DataFrame()
            cen_min, cen_max = map(float, cen_range.split('_'))
            if cen_range == "0_5" or cen_range == "5_10":
                H3l_spectrum = spectra_file.Get(f'BlastWave_H3L_0_10') #use h3lBW_0_10 for 0-5 and 5-10
                df_cen_bin =  df_signal[(df_signal['fCentralityFT0C'] >= cen_min) & (df_signal['fCentralityFT0C'] < cen_max)].copy()
            elif cen_range == "50_80":
                H3l_spectrum = spectra_file.Get(f'BlastWave_H3L_30_50') #use h3lBW_30_50 for 50_80
                df_cen_bin =  df_signal[(df_signal['fCentralityFT0C'] >= 50) & (df_signal['fCentralityFT0C'] < 110)].copy()
            else:
                H3l_spectrum = spectra_file.Get(f'BlastWave_H3L_{cen_range}')
                df_cen_bin =  df_signal[(df_signal['fCentralityFT0C'] >= cen_min) & (df_signal['fCentralityFT0C'] < cen_max)].copy()
            column_GentPt = df_cen_bin["fAbsGenPt"]
            plt.figure(figsize=(10, 6))
            plt.hist(column_GentPt, bins=200, color='blue', alpha=0.7, label=f'AbsGenPt unshifted cen: {cen_min}-{cen_max}')
            plt.xlabel('AbsGenPt (GeV/c)')
            plt.ylabel('Counts')
            #plt.ylim(0, 500)
            plt.title('')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_base_dir + f'/Pt_Gent_unshifted_cen_{cen_min}_{cen_max}.pdf')
            plt.close("all")
            column_Pt = df_cen_bin["fPt"]
            plt.figure(figsize=(10, 6))
            plt.hist(column_Pt, bins=200, color='blue', alpha=0.7, label=f'Pt unshifted cen: {cen_min}-{cen_max}')
            plt.xlabel('Pt (GeV/c)')
            plt.ylabel('Counts')
            plt.ylim(0, 500)
            plt.title('')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_base_dir + f'/Pt_unshifted_cen_{cen_min}_{cen_max}.pdf')
            plt.close("all")
            if pt_bins == []:
                pt_bins = prev_pt_bins
            else: 
                prev_pt_bins = pt_bins
            for pt_idx in range(len(pt_bins)-1):
                print(f'idx_pt{pt_idx} {pt_bins[pt_idx]}-{pt_bins[pt_idx+1]}')
                # new_H3l_spectrum = H3l_spectrum.Clone()
                pt_min, pt_max = pt_bins[pt_idx], pt_bins[pt_idx+1]
                H3l_spectrum.SetRange(pt_min, pt_max)
                df_cen_pt_bin = df_cen_bin.query("fAbsGenPt >= @pt_min and fAbsGenPt < @pt_max").copy()
                # column_GentPt = df_cen_pt_bin["fAbsGenPt"]
                # plt.figure(figsize=(10, 6))
                # plt.hist(column_GentPt, bins=200, color='blue', alpha=0.7, label=f'AbsGenPt unshifted cen: {cen_min}-{cen_max}')
                # plt.xlabel('AbsGenPt (GeV/c)')
                # plt.ylabel('Counts')
                # #plt.ylim(0, 500)
                # plt.title('')
                # plt.legend()
                # plt.tight_layout()
                # plt.savefig(output_base_dir + f'/Pt_Gent_unshifted_cen_{cen_min}_{cen_max}_pt_{pt_min}_{pt_max}.pdf')
                # plt.close("all")
                utils.reweight_pt_spectrum(df_cen_pt_bin, 'fAbsGenPt', H3l_spectrum)
                df_cen_pt_bin = df_cen_pt_bin.query("rej == 1")
                # print(df_cen_pt_bin)
                reweighted_df_cen = pd.concat([reweighted_df_cen, df_cen_pt_bin])
                # column_GentPt = df_cen_pt_bin["fAbsGenPt"]
                # plt.figure(figsize=(10, 6))
                # plt.hist(column_GentPt, bins=200, color='red', alpha=0.7, label=f'AbsGenPt shifted cen: {cen_min}-{cen_max}')
                # plt.xlabel('GentPt (GeV/c)')
                # plt.ylabel('Counts')
                # #plt.ylim(0, 500)
                # plt.title('')
                # plt.legend()
                # plt.tight_layout()
                # plt.savefig(output_base_dir + f'/Pt_Gent_shifted_cen_{cen_min}_{cen_max}_pt_{pt_min}_{pt_max}.pdf')
                # plt.close("all")
            column_GentPt = reweighted_df_cen['fAbsGenPt']
            plt.figure(figsize=(10, 6))
            plt.hist(column_GentPt, bins=200, color='red', alpha=0.7, label=f'AbsGenPt shifted cen: {cen_min}-{cen_max}')
            plt.xlabel('GentPt (GeV/c)')
            plt.ylabel('Counts')
            #plt.ylim(0, 500)
            plt.title('')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_base_dir + f'/Pt_Gent_shifted_cen_{cen_min}_{cen_max}.pdf')
            plt.close("all")
            column_Pt = reweighted_df_cen["fPt"]
            plt.figure(figsize=(10, 6))
            plt.hist(column_Pt, bins=200, color='red', alpha=0.7, label=f'Pt shifted cen: {cen_min}-{cen_max}')
            plt.xlabel('Pt (GeV/c)')
            plt.ylabel('Counts')
            plt.ylim(0, 500)
            plt.title('')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_base_dir + f'/Pt_shifted_cen_{cen_min}_{cen_max}.pdf')
            plt.close("all")
            reweighted_df = pd.concat([reweighted_df,reweighted_df_cen])
        spectra_file.Close()
        #df_signal = pd.concat(df_split).sort_index()
        signalH.set_data_frame(reweighted_df)
        print(reweighted_df)
        # selection for one pt & centrality bin
        for cen_idx, (cen_range, pt_bins) in enumerate(cen_pt_bins_dic.items()):
            cen_min, cen_max = map(float, cen_range.split('_'))
            mc_hdl_sub = []
            for i in range(len(pt_bins)-1):
                bin_sel_bkg = f'fCentralityFT0C >= {cen_min} & fCentralityFT0C < {cen_max} & fPt > {pt_bins[i]} & fPt < {pt_bins[i+1]}'
                bin_sel_signal = f'fPt > {pt_bins[i]} & fPt < {pt_bins[i+1]}'
                bin_bkgH = bkgH.apply_preselections(bin_sel_bkg,inplace=False)
                bin_signalH = signalH.apply_preselections(bin_sel_signal,inplace=False)
                ## select background by taking the sidebands of the mass distribution
                if training_preselections != '':
                        bin_signalH.apply_preselections(f'{training_preselections}')
                        bin_bkgH.apply_preselections(f"(fMassH3L<2.95 or fMassH3L>3.02) and {training_preselections}")
                else:
                        bin_bkgH.apply_preselections(f"(fMassH3L<2.95 or fMassH3L>3.02)")
                mc_hdl_sub.append(bin_signalH)
                ## shift nSigmaH3 distribution 
                if opean_NSigmaH3_signal_shift:
                    df_signalH = bin_signalH.get_data_frame()
                    df_signalH['fNSigmaHe'] = df_signalH['fNSigmaHe'] - df_signalH['fNSigmaHe'].mean()
                    df_bkgH = bin_bkgH.get_data_frame()
                    x_bkgH = df_bkgH['fNSigmaHe'].values
                    y_bkgH = np.histogram(x_bkgH, bins=100, density=True)[0]
                    x_bkgH_hist = np.histogram(x_bkgH, bins=100, density=True)[1][:-1]
                    init_guess = [max(y_bkgH), 0, 1, 0, 0, 0, 0]
                    popt, _ = curve_fit(gauss_pol3, x_bkgH_hist, y_bkgH, p0=init_guess, bounds=([-np.inf, -1, 0, -np.inf, -np.inf, -np.inf, -np.inf], [np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf]))
                    A, mu, sigma, B, C, D, E = popt
                    plt.plot(x_bkgH_hist, y_bkgH, label='Background Data')
                    plt.plot(x_bkgH_hist, gauss_pol3(x_bkgH_hist, *popt), label='Gaussian+Poly Fit', linestyle='--')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(output_figurs_dir_list[cen_idx][i] + "/Gauss_pol3_fit_bkg_df.pdf")
                    plt.close("all")
                    df_bkgH['fNSigmaHe'] = df_bkgH['fNSigmaHe'] - mu
                    # df_bkgH = bin_bkgH.get_data_frame()
                    # df_bkgH['fNSigmaHe'] = df_bkgH['fNSigmaHe'] - df_bkgH['fNSigmaHe'].mean()
                    # mean, std = norm.fit(df_signalH['fNSigmaHe'])
                    # n_sigma = 3
                    # filtered_data = df_signalH[np.abs(df_signalH['fNSigmaHe']) <= n_sigma * std]
                    # while len(filtered_data) / len(df_signalH) > 0.95:  
                    #     n_sigma += 0.5
                    #     filtered_data = df_signalH[np.abs(df_signalH['fNSigmaHe']) <= n_sigma * std]
                    # df_signalH = filtered_data
                    bin_signalH.set_data_frame(df_signalH)
                    bin_bkgH.set_data_frame(df_bkgH)
                utils.cut_elements_to_same_range(bin_signalH,bin_bkgH,['fDcaHe','fDcaPi'])
                print("------------------------------------------")
                print("Origin Signal events: ", len(bin_signalH))
                print("Origin Background events: ", len(bin_bkgH))
                if bkg_fraction_max != None:
                        if(len(bin_bkgH) > bkg_fraction_max * len(bin_signalH)):
                            bin_bkgH.shuffle_data_frame(size=bkg_fraction_max*len(bin_signalH), inplace=True, random_state=random_state)
                print("------------------------------------------")
                print("Final Signal events: ", len(bin_signalH))
                print("Final Background events: ", len(bin_bkgH))
                infofile = f'{output_dir_list[cen_idx][i]}/info.txt'
                with open(infofile, 'w') as f:
                    pass
                with open(infofile, 'a') as f:
                    f.write(f'======Training centrality: {cen_min}_{cen_max} pt: {pt_bins[i]}_{pt_bins[i+1]}======\n')
                    f.write(f'------------------------------------------\n')
                    f.write(f'Origin Signal events: {len(bin_signalH)}\n')
                    f.write(f'Origin Background events: {len(bin_bkgH)}\n')
                    f.write(f'------------------------------------------\n')
                    f.write(f'Final Signal events: {len(bin_signalH)}\n')
                    f.write(f'Final Background events: {len(bin_bkgH)}\n')
                train_test_data = au.train_test_generator([bin_signalH, bin_bkgH], [1,0], test_size=test_set_size, random_state=random_state)
                train_features = train_test_data[0]
                train_labels = train_test_data[1]
                test_features = train_test_data[2]
                test_labels = train_test_data[3]
        
                distr = pu.plot_distr([bin_signalH, bin_bkgH], training_variables + ["fMassH3L"], bins=100, labels=['Signal',"Background"],colors=["blue","red"], log=True, density=True, figsize=(18, 13), alpha=0.5, grid=False)
                plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
                plt.savefig(output_figurs_dir_list[cen_idx][i] + "/features_distributions.pdf", bbox_inches='tight')
                plt.close("all")
                
                corr = pu.plot_corr([bin_signalH,bin_bkgH], training_variables + ["fMassH3L"], ['Signal',"Background"])
                corr[0].savefig(output_figurs_dir_list[cen_idx][i] + "/correlations_signal.pdf",bbox_inches='tight')
                corr[1].savefig(output_figurs_dir_list[cen_idx][i] + "/correlations_bkg.pdf",bbox_inches='tight')
                plt.close("all")
        
                print("---------------------------------------------")
                print("Data loaded. Training and testing ....")
        
                model_hdl = ModelHandler(xgb.XGBClassifier(), training_variables)
                model_hdl.set_model_params(hyperparams)
                model_hdl.train_test_model(train_test_data, False, True)
                y_pred_test = model_hdl.predict(test_features)
                y_pred_train = model_hdl.predict(train_features)
        
                print("Model trained and tested. Saving results ...")
        
                bdt_out_plot = pu.plot_output_train_test(model_hdl, train_test_data, 100, True, ["Signal", "Background"], True, density=True)
                bdt_out_plot.savefig(output_figurs_dir_list[cen_idx][i] + "/bdt_output.pdf")
                plt.close("all")
                feature_importance_plot = pu.plot_feature_imp(test_features, test_labels, model_hdl, ["Signal", "Background"])
                feature_importance_plot[0].savefig(output_figurs_dir_list[cen_idx][i] + "/feature_importance_1.pdf")
                feature_importance_plot[1].savefig(output_figurs_dir_list[cen_idx][i] + "/feature_importance_2.pdf") 
                plt.close("all")
                
    
                #plot score distirbutions
                plt.hist(y_pred_test, bins=100, label='test set score_full sample', alpha=0.5, density=True)
                plt.xlabel("test BDT_score")
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_figurs_dir_list[cen_idx][i] + "/testset_score_distribution_full.pdf")
                plt.close("all")
                plt.hist(y_pred_test[test_labels==0], bins=100, label='background', alpha=0.5, density=True)
                plt.hist(y_pred_test[test_labels==1], bins=100, label='signal', alpha=0.5, density=True)
                plt.xlabel("test BDT_score")
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_figurs_dir_list[cen_idx][i] + "/testset_score_distribution_split.pdf")
                plt.close("all")
    
                #plot roc
                roc_plot = pu.plot_roc_train_test(test_labels, y_pred_test, train_labels, y_pred_train)
                roc_plot.savefig(output_figurs_dir_list[cen_idx][i] + "/roc_test_vs_train.pdf")
                plt.close("all")
                
                
                ## dump model handler and efficiencies vs score
                ### for fixed efficiency array
                hdl_saving_name = "/model_hndl"
                model_hdl.dump_model_handler(output_dir_list[cen_idx][i] + hdl_saving_name + ".pkl")
                eff_arr = np.round(np.arange(0.5,0.99,0.005),3) # 1 means save with 1 digits after point
                score_eff_arr = au.score_from_efficiency_array(test_labels, y_pred_test, eff_arr)
                plt.plot(score_eff_arr, eff_arr, label='BDT_efficency_fixed_efficencyarray', marker='o')
                plt.xlabel('BDT Score')
                plt.ylabel('Efficiency')
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_figurs_dir_list[cen_idx][i] + "/efficency_vs_model_output_fixedeff.pdf")
                plt.close("all")
                np.save(output_dir_list[cen_idx][i] + "/efficiency_arr_fixedeff.npy", eff_arr)
                np.save(output_dir_list[cen_idx][i] + "/score_efficiency_arr_fixedeff.npy",score_eff_arr)
                ### for fixed score array
                score_arr = np.round(np.arange(-3,6,0.1),1) # 1 means save with 1 digits after point
                eff_arr = []
                test_features['BDT_training_score'] = y_pred_test
                for score in score_arr:
                    selection = f'BDT_training_score > {score}'
                    sel_df = test_features.query(selection)
                    eff_arr.append(len(sel_df)/len(test_features))
                plt.plot(score_arr, eff_arr, label='BDT_efficency_fixed_scorearray', marker='o')
                plt.xlabel('BDT Score')
                plt.ylabel('Efficiency')
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_figurs_dir_list[cen_idx][i] + "/efficency_vs_model_output_fixedsco.pdf")
                plt.close("all")
                np.save(output_dir_list[cen_idx][i] + "/efficiency_arr_fixedsco.npy", eff_arr)
                np.save(output_dir_list[cen_idx][i] + "/score_efficiency_arr_fixedsco.npy",score_eff_arr)
                print("Training done")
                print("---------------------------------------------")
            mc_hdl.append(mc_hdl_sub)
        del signalH, bkgH
        if not do_application:
            del mc_hdl


if do_application:
        if not do_training:
            print(":::Fatal you should do training first exit")
            exit()
        print("---------------------------------------------")
        print("Starting application: ..")
        
        dataH = TreeHandler(input_data_path, "O2hypcands", folder_name='DF*')
        utils.correct_and_convert_df(dataH,calibrate_he3_pt=calibrate_he_momentum, isMC=False)
        mcH = TreeHandler(input_mc_path, "O2mchypcands", folder_name='DF*')
        utils.correct_and_convert_df(mcH,calibrate_he3_pt=calibrate_he_momentum, isMC=True)
        mcH_generated = mcH.apply_preselections('fIsSurvEvSel==True', inplace=False)
        mcH_reconstructed = mcH.apply_preselections('fIsReco == 1', inplace=False)

        for cen_idx, (cen_range, pt_bins) in enumerate(cen_pt_bins_dic.items()):
            cen_min, cen_max = map(float, cen_range.split('_'))
            cen_bin_n_env = utils.getNEvents(input_AnalysisResults_file_path,False,cen_min,cen_max)
            ### absorption calculations here 
            absorption_histo = None
            if input_absorption_file_path:
                cen_min_int = int(cen_min)
                cen_max_int = int(cen_max)
                absorption_file = ROOT.TFile.Open(input_absorption_file_path + f'/absorption_histos_x1.5_pt_cen_{cen_min_int}_{cen_max_int}.root')
                absorption_histo_mat = absorption_file.Get('h_abso_frac_pt_mat')
                absorption_histo_anti = absorption_file.Get('h_abso_frac_pt_antimat')
                absorption_histo_mat.SetDirectory(0)
                absorption_histo_anti.SetDirectory(0)
                absorption_histo = absorption_histo_mat.Clone('h_abso_frac_pt')
                absorption_histo.Add(absorption_histo_anti)# gte average between matter and antimatter
                absorption_histo.Scale(0.5)
            ### BW_fit func for h3l
            BWfile = ROOT.TFile.Open(input_H3l_BWFit_file_path)
            if cen_range == "0_5" or cen_range == "5_10":
                H3l_spectrum = BWfile.Get('BlastWave_H3L_0_10')
            elif cen_range == "50_80":
                H3l_spectrum = BWfile.Get('BlastWave_H3L_30_50')
            else :
                H3l_spectrum = BWfile.Get(f'BlastWave_H3L_{cen_range}')
            for i in range(len(pt_bins)-1):
                infofile = f'{output_dir_list[cen_idx][i]}/info.txt'
                selection_string = f'fCentralityFT0C >= {cen_min} & fCentralityFT0C < {cen_max} & fPt > {pt_bins[i]} & fPt < {pt_bins[i+1]}'
                selection_string_mc = f'fCentralityFT0C >= {cen_min} & fCentralityFT0C < {cen_max} & fAbsGenPt > {pt_bins[i]} & fAbsGenPt < {pt_bins[i+1]}'
                ### efficency for this pt and cen bin
                mcH_generated_bin = mcH_generated.apply_preselections(selection_string_mc, inplace=False)
                mcH_reconstructed_bin = mcH_reconstructed.apply_preselections(selection_string_mc, inplace=False)
                efficency_bin = len(mcH_reconstructed_bin)/len(mcH_generated_bin)
                if absorption_histo:
                    absorption_bin = absorption_histo.GetBinContent(i+1)
                else:
                    absorption_bin = 1
                ### expected Signal counts
                exp_signal_bin = cen_bin_n_env * H3l_spectrum.Integral(pt_bins[i],pt_bins[i+1]) * efficency_bin * absorption_bin * 0.25 * 2 * 1 * 1
                print(f'*** expected_signal_counts in {cen_min} < cen < {cen_max} and {pt_bins[i]} < pt < {pt_bins[i+1]} is {exp_signal_bin} ***')
                with open(infofile, 'a') as f:
                    f.write(f'*** expected_signal_counts in {cen_min} < cen < {cen_max} and {pt_bins[i]} < pt < {pt_bins[i+1]} is {exp_signal_bin} ***\n')
                bin_data_hdl = dataH.apply_preselections(selection_string, inplace = False)
                # if training_preselections != '':
                #     bin_data_hdl.apply_preselections(training_preselections)
                model_hdl = ModelHandler()
                model_hdl.load_model_handler(output_dir_list[cen_idx][i] + hdl_saving_name + ".pkl")
                bin_data_hdl.apply_model_handler(model_hdl, column_name="model_output")
                bin_data_hdl.print_summary()
                #plot the model BDT result distribution
                df = bin_data_hdl.get_data_frame()
                hist = df.hist(column='model_output', bins=100, range=(-15,15), figsize=(12, 7), grid=False, density=False, alpha=0.6, label="BDT_output")
                plt.xlabel('BDT score')
                plt.ylabel('Counts')
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_figurs_dir_list[cen_idx][i] + "/model_output_distribution_applied.pdf")
                plt.close("all")
                #pre_plot some random cut
                if BDT_values_test:
                    for BDT_idx, threshold in enumerate(BDT_values_test):
                        BDT_sel = f"model_output > {threshold}"
                        filtered_df = df.query(BDT_sel).copy()
                        hist = filtered_df.hist(column='fMassH3L', bins=40, range=(2.96, 3.04), figsize=(12, 7), grid=False, density=False, alpha=0.6, label=f'model_output > {threshold}')
                        plt.xlabel(r'$M(\mathrm{\pi ^{3}He})$ (GeV/$c^2$)')
                        plt.ylabel('Counts')
                        plt.legend()
                        plt.title('')
                        plt.tight_layout()
                        plt.savefig(output_figurs_dir_list[cen_idx][i] + f"/invmass_applied_BDTscore_cut_{BDT_idx}.pdf")
                        plt.close("all")
                # plot the fitted raw count vs BDT score 
                bdt_eff_arr = np.load(output_dir_list[cen_idx][i] + "/efficiency_arr_fixedeff.npy")
                score_eff_arr = np.load(output_dir_list[cen_idx][i] + "/score_efficiency_arr_fixedeff.npy")
                raw_counts_arr = []
                raw_counts_arr_err = []
                significance_arr = []
                significance_arr_err = []
                s_b_ratio_arr = []
                s_b_ratio_arr_err = []
                chi2_arr = []
                mc_bin_sel = f'fCentralityFT0C >= {cen_min} & fCentralityFT0C < {cen_max}'
                mc_hdl[cen_idx][i].apply_preselections(mc_bin_sel)
                output_file = ROOT.TFile.Open(f'{output_dir_list[cen_idx][i]}/training_spectrum.root', 'recreate')
                output_dir_std = output_file.mkdir('std')
                for score in score_eff_arr:
                    input_bin_data_hdl = bin_data_hdl.apply_preselections(f"model_output > {score}",inplace = False)
                    score_label = [f'{pt_bins[i]} #leq #it{{p}}_{{T}} < {pt_bins[i+1]} GeV/#it{{c}}', f'Centrality: {cen_min}-{cen_max} %',f'BDT_Score > {score}']
                    signal_extraction = SignalExtraction(input_bin_data_hdl, mc_hdl[cen_idx][i])
                    signal_extraction.bkg_fit_func = "pol2"
                    signal_extraction.signal_fit_func = "dscb"
                    signal_extraction.n_bins_data = 40
                    signal_extraction.n_bins_mc = 80
                    signal_extraction.n_evts = cen_bin_n_env
                    signal_extraction.is_matter = "both"
                    signal_extraction.performance = False
                    signal_extraction.is_3lh = True
                    signal_extraction.out_file = output_dir_std
                    signal_extraction.data_frame_fit_name = f'data_fit_BDT_score_{score}'
                    signal_extraction.mc_frame_fit_name = f'mc_fit_cen_{cen_min}_{cen_max}_pt_{pt_bins[i]}_{pt_bins[i+1]}'
                    signal_extraction.additional_pave_text = score_label
                    signal_extraction.sigma_range_mc_to_data = [1., 1.5]
                    fit_stats = signal_extraction.process_fit()
                    raw_counts_arr.append(fit_stats['signal'][0])
                    raw_counts_arr_err.append(fit_stats['signal'][1])
                    significance_arr.append(fit_stats['significance'][0])
                    significance_arr_err.append(fit_stats['significance'][1])
                    s_b_ratio_arr.append(fit_stats['s_b_ratio'][0])
                    s_b_ratio_arr_err.append(fit_stats['s_b_ratio'][1])
                    chi2_arr.append(fit_stats['chi2'])
                filtered_raw_counts = raw_counts_arr.copy()
                for rc_idx in range(len(raw_counts_arr)):
                    if rc_idx == 0:
                        mean_adjacent = (filtered_raw_counts[rc_idx + 1] + filtered_raw_counts[rc_idx + 2]) / 2
                        if filtered_raw_counts[rc_idx] > 5 * mean_adjacent or filtered_raw_counts[rc_idx] < mean_adjacent / 5:
                            #filtered_raw_counts[rc_idx] = mean_adjacent
                            print(f':::======NOTICE!!! raw_counts_arr id: {rc_idx}; BDTefficency: {bdt_eff_arr[rc_idx]}; BDT_score: {score_eff_arr[rc_idx]}; fit status is bad!!!======:::')
                            with open(infofile, 'a') as f:
                                f.write(f':::======NOTICE!!! raw_counts_arr id: {rc_idx}; BDTefficency: {bdt_eff_arr[rc_idx]}; BDT_score: {score_eff_arr[rc_idx]}; fit status is bad!!!======:::\n')
                    elif rc_idx == 1: 
                        mean_adjacent = (filtered_raw_counts[rc_idx + 1] + filtered_raw_counts[rc_idx + 2]) / 2
                        if filtered_raw_counts[rc_idx] > 5 * mean_adjacent or filtered_raw_counts[rc_idx] < mean_adjacent / 5:
                            #filtered_raw_counts[rc_idx] = mean_adjacent
                            print(f':::======NOTICE!!! raw_counts_arr id: {rc_idx}; BDTefficency: {bdt_eff_arr[rc_idx]}; BDT_score: {score_eff_arr[rc_idx]}; fit status is bad!!!======:::')
                            with open(infofile, 'a') as f:
                                f.write(f':::======NOTICE!!! raw_counts_arr id: {rc_idx}; BDTefficency: {bdt_eff_arr[rc_idx]}; BDT_score: {score_eff_arr[rc_idx]}; fit status is bad!!!======:::\n')
                    else: 
                        mean_adjacent = (filtered_raw_counts[rc_idx - 1] + filtered_raw_counts[rc_idx - 2]) / 2
                        if filtered_raw_counts[rc_idx] > 5 * mean_adjacent or filtered_raw_counts[rc_idx] < mean_adjacent / 5:
                            #filtered_raw_counts[rc_idx] = mean_adjacent
                            print(f':::======NOTICE!!! raw_counts_arr id: {rc_idx}; BDTefficency: {bdt_eff_arr[rc_idx]}; BDT_score: {score_eff_arr[rc_idx]}; fit status is bad!!!======:::')
                            with open(infofile, 'a') as f:
                                f.write(f':::======NOTICE!!! raw_counts_arr id: {rc_idx}; BDTefficency: {bdt_eff_arr[rc_idx]}; BDT_score: {score_eff_arr[rc_idx]}; fit status is bad!!!======:::\n')
                smoothed_raw_counts_arr = medfilt(raw_counts_arr, kernel_size=5)
                exp_significance_array = [exp_signal_bin / np.sqrt(exp_signal_bin + (a / b)) for a, b in zip(raw_counts_arr, s_b_ratio_arr)]
                bkg_3sigma_array = [a/b for a,b in zip(raw_counts_arr, s_b_ratio_arr)]
                df_working_point = pd.DataFrame({
                    'exp_significance': exp_significance_array,
                    'raw_counts': raw_counts_arr,
                    'raw_counts_err': raw_counts_arr_err,
                    'bkg_3sigma': bkg_3sigma_array,
                    'BDT_efficiency': bdt_eff_arr,
                    'BDT_score': score_eff_arr,
                    'chi2': chi2_arr
                })
                df_working_point = df_working_point[df_working_point['raw_counts'] != 0]
                df_working_point = df_working_point.query('chi2 < 1.4 & raw_counts_err / raw_counts < 1')
                df_working_point.to_csv(output_dir_list[cen_idx][i] + "/df_working_point.csv")
                # smoothed_raw_counts_arr = gaussian_filter1d(raw_counts_arr, sigma=2)
                df_working_point['product'] = df_working_point['BDT_efficiency'] * df_working_point['exp_significance']
                max_row = df_working_point.loc[df_working_point['product'].idxmax()]
                max_product = max_row['product']
                max_efficiency = max_row['BDT_efficiency']
                max_score = max_row['BDT_score']
                plt.figure(figsize=(10, 6))
                plt.scatter(
                    df_working_point['BDT_score'],
                    df_working_point['product'],
                    label='All Points'
                )
                plt.scatter(
                    max_score,
                    max_product,
                    color='red',
                    s=100,  # 点大小
                    label=f'Max: BDT_eff={max_efficiency:.2f}, Score={max_score:.2f}'
                )
                plt.annotate(
                    f'Max: ({max_score:.2f}, {max_product:.2f})',
                    xy=(max_score, max_product),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->')
                )
                plt.title('BDT_efficiency * exp_significance vs BDT_score')
                plt.xlabel('BDT_score')
                plt.ylabel('BDT_efficiency * exp_significance')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                #plt.savefig('max_product_vs_score.png', dpi=300, bbox_inches='tight')
                plt.savefig(output_figurs_dir_list[cen_idx][i] + "/exp_significance_vs_BDT_score.pdf")
                plt.close("all")
                print(f'Application for centrality:{cen_min}-{cen_max} pt:{pt_bins[i]}-{pt_bins[i+1]} done. Saving results ...')
                bin_data_hdl.write_df_to_parquet_files("dataH_BDTapplied", output_dir_list[cen_idx][i])