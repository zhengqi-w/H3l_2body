# this code is designed for extractiong the lifetime of hypertriton from ct spectra, using BDT training method
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setSilentMode(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(0)
kOrangeC  = ROOT.TColor.GetColor('#ff7f00')

import os
import numpy as np
import uproot
import argparse
import yaml
from itertools import product
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
import hipe4ml.analysis_utils as au
import hipe4ml.plot_utils as pu
import matplotlib.pyplot as plt
import xgboost as xgb
from iminuit import Minuit

import copy
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar
import pandas as pd
from pathlib import Path
import xarray as xr
from scipy.integrate import quad

import sys
sys.path.append('utils')
import utils as utils
from spectra import SpectraMaker


if __name__ == '__main__':

    def gauss_pol3(x, A, mu, sigma, B, C, D, E):
        return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + B * x**3 + C * x**2 + D * x + E
    
    def gauss_pol2(x, a, mu, sigma, c0, c1, c2):
        gaussian = a * np.exp(-(x - mu)**2 / (2 * sigma**2))
        polynomial = c0 + c1*x + c2*x**2
        return gaussian + polynomial
    
    def pol2(x, a, b, c):
        return a * x**2 + b * x + c
    
    def pol3(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d
    
    def gauss(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    def combined_fit(x, a, b, c, A, mu, sigma):
        return pol2(x, a, b, c) + gauss(x, A, mu, sigma)
    
    def normalized_expon(ct, tau, ct_range=None):
        """
        归一化的指数衰减函数（用于拟合粒子衰变时间 ct）
        
        参数:
            ct       : 衰变时间观测值（可以是标量或数组）
            tau      : 平均寿命（衰变常数 τ = 1/λ）
            ct_range : 归一化范围 (ct_min, ct_max)，若为None则使用理论归一化
        
        返回:
            归一化的概率密度值
        """
        # 定义指数分布 (loc=0表示从ct=0开始衰减)
        dist = expon(scale=tau, loc=0)
        
        # 如果指定范围，计算归一化因子
        if ct_range is not None:
            ct_min, ct_max = ct_range
            norm_factor = dist.cdf(ct_max) - dist.cdf(ct_min)
        else:
            norm_factor = 1.0  # 理论归一化
        
        # 计算概率密度
        pdf = dist.pdf(ct) / norm_factor
        
        # 处理范围外的点（可选）
        if ct_range is not None:
            pdf = np.where((ct >= ct_min) & (ct <= ct_max), pdf, 0.0)
        
        return pdf 

    parser = argparse.ArgumentParser(description='Configure the parameters of the script.')
    parser.add_argument('--config-file', dest='config_file', help="path to the YAML file with configuration.", default='')
    args = parser.parse_args()
    if args.config_file == "":
        print('** No config file provided. Exiting. **')
        exit()

    config_file = open(args.config_file, 'r')
    config = yaml.full_load(config_file)
    input_file_name_data = config['input_files_data']
    input_file_name_mc = config['input_files_mc']
    input_analysis_results_file = config['input_analysis_results_file']
    output_dir_name = config['output_dir']
    output_file_name = config['output_file']
    pt_bins = config['pt_bins']
    ct_bins = config['ct_bins']
    selections_std = config['selection']
    is_matter = config['is_matter']
    output_dir_name = output_dir_name + f'/{is_matter}'
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)
    use_BDT = config['use_BDT']
    new_training = config['new_training']
    new_WP_process = config['new_WP_process']
    ##BDT parameters
    training_variables = config['training_variables']
    training_preselections = config['training_preselections']
    random_state = config['random_state']
    test_set_size = config['test_set_size']
    hyperparams = config['hyperparams']
    bkg_fraction_max = config['bkg_fraction_max']
    save_working_points_fit = config['save_working_points_fit']
    npoints_for_WP = config['npoints_for_WP']

    mass_range = config['mass_range']
    signal_fit_func = config['signal_fit_func']
    bkg_fit_func = config['bkg_fit_func']
    sigma_range_mc_to_data = config['sigma_range_mc_to_data']
    do_syst = config['do_syst']
    n_trials = config['n_trials']
    n_bins_mass_data = config['n_bins_mass_data']
    n_bins_mass_mc = config['n_bins_mass_mc']

    matter_options = ['matter', 'antimatter', 'both']
    if is_matter not in matter_options:
        raise ValueError(f'Invalid is-matter option. Expected one of: {matter_options}')

    print('**********************************')
    print('    Running ct_analysis.py')
    print('**********************************\n')
    print("----------------------------------")
    print("** Loading data and apply preselections **")

    tree_names = ['O2datahypcands','O2hypcands', 'O2hypcandsflow']
    tree_keys = uproot.open(input_file_name_data[0]).keys()
    for tree in tree_names:
        for key in tree_keys:
            if tree in key:
                tree_name = tree
                break
    print(f"Data tree found: {tree_name}")
    data_hdl = TreeHandler(input_file_name_data, tree_name, folder_name='DF*')
    mc_hdl = TreeHandler(input_file_name_mc, 'O2mchypcands', folder_name='DF*')

    # lifetime_dist = ROOT.TH1D('syst_lifetime', ';#tau ps ;Counts', 40, 120, 380)
    # lifetime_prob = ROOT.TH1D('prob_lifetime', ';prob. ;Counts', 100, 0, 1)


    # Add columns to the handlers
    utils.correct_and_convert_df(data_hdl, calibrate_he3_pt=False)
    utils.correct_and_convert_df(mc_hdl, calibrate_he3_pt=False, isMC=True)

    # apply preselections
    # matter_sel = ''
    # mc_matter_sel = ''
    # if is_matter == 'matter':
    #     matter_sel = 'fIsMatter == True'
    #     mc_matter_sel = 'fGenPt > 0'

    # elif is_matter == 'antimatter':
    #     matter_sel = 'fIsMatter == False'
    #     mc_matter_sel = 'fGenPt < 0'

    # if matter_sel != '':
    #     data_hdl.apply_preselections(matter_sel)
    #     mc_hdl.apply_preselections(mc_matter_sel)
    
    # reweight MC pT spectrum
    spectra_file = ROOT.TFile.Open('utils/H3L_BWFit.root')
    he3_spectrum = spectra_file.Get('BlastWave_H3L_10_30') #here we use only one certain centrality func
    spectra_file.Close()
    utils.reweight_pt_spectrum(mc_hdl, 'fAbsGenPt', he3_spectrum)

    inv_mass_string = '#it{M}_{^{3}He+#pi^{-}}' if is_matter == 'matter' else '#it{M}_{^{3}#bar{He}+#pi^{+}}' if is_matter == 'antimatter' else '#it{M}_{^{3}He+#pi}' if is_matter == 'both' else None
    decay_string = '{}^{3}_{#Lambda}H #rightarrow ^{3}He+#pi^{-}' if is_matter == 'matter' else '{}^{3}_{#bar{#Lambda}}#bar{H} #rightarrow ^{3}#bar{He}+#pi^{+}' if is_matter == 'antimatter' else '{}^{3}_{#Lambda}H #rightarrow ^{3}He+#pi' if is_matter == 'both' else None
    n_events = utils.getNEvents(input_analysis_results_file,False)
    x_label = r'#it{ct} (cm)'
    y_raw_label = r'#it{N}_{raw}'
    y_eff_label = r'#epsilon #times acc.'
    y_eff_BDT_label = r'#epsilon_{BDT}'
    y_corr_label = r'#frac{d#it{N}}{d(#it{ct})} (cm^{-1})'



    mc_hdl.apply_preselections('rej==True')
    mc_reco_hdl = mc_hdl.apply_preselections('fIsReco == 1', inplace=False)

    print("** Data loaded. ** \n")
    print("----------------------------------")
    print("** Starting ct analysis **")

    # declare output file
    output_file = ROOT.TFile.Open(f'{output_dir_name}/{output_file_name}.root', 'recreate')

    output_dir_std = output_file.mkdir('std')
    if use_BDT:
        BDT_QA_dir = output_dir_name + '/BDT_QA'
        if not os.path.exists(BDT_QA_dir):
            os.makedirs(BDT_QA_dir)
        if save_working_points_fit:
            working_point_fit_dir = BDT_QA_dir + '/WPDetail'
            if not os.path.exists(working_point_fit_dir):
                os.makedirs(working_point_fit_dir)
    corrected_counts_arr = []
    corrected_counts_err_arr = []
    BDT_efficiency_arr = []
    acceptance_arr = []
    raw_counts_arr = []
    raw_counts_err_arr = []
    for i_pt in range(len(pt_bins)-1):
        corrected_counts_arr_pt = []
        corrected_counts_err_arr_pt = []
        BDT_efficiency_arr_pt = []
        acceptance_arr_pt = []
        raw_counts_arr_pt = []
        raw_counts_err_arr_pt = []
        pt_min = pt_bins[i_pt]
        pt_max = pt_bins[i_pt+1]
        pt_sel = f'fPt > {pt_min} & fPt < {pt_max}'
        pt_sel_MC = f'fAbsGenPt > {pt_min} & fAbsGenPt < {pt_max}'
        bin_pt_data_hdl = data_hdl.apply_preselections(pt_sel, inplace=False)
        bin_pt_mc_hdl = mc_hdl.apply_preselections(pt_sel_MC, inplace=False)
        bin_pt_mc_reco_hdl = mc_reco_hdl.apply_preselections(pt_sel_MC, inplace=False)
        i_ct_bins = ct_bins[i_pt]
        bin_output_dir = output_dir_std.mkdir(f'pt_{pt_min}_{pt_max}')
        h_raw_counts = ROOT.TH1D(f'h_raw_counts_pt_{pt_min}_{pt_max}', f';{x_label};{y_raw_label}', len(i_ct_bins) - 1, np.array(i_ct_bins, dtype=np.float64))
        h_raw_counts.SetTitle(f'Raw counts in pT bin {pt_min} < #it{{p}}_{{T}} < {pt_max} GeV/c')
        h_acc_eff = ROOT.TH1D(f'h_acc_eff_pt_{pt_min}_{pt_max}', f';{x_label};{y_eff_label}', len(i_ct_bins) - 1, np.array(i_ct_bins, dtype=np.float64))
        h_acc_eff.SetTitle(f'Acceptance x Efficiency in pT bin {pt_min} < #it{{p}}_{{T}} < {pt_max} GeV/c')
        h_spectrum = ROOT.TH1D(f'h_spectrum_pt_{pt_min}_{pt_max}', f';{x_label};{y_corr_label}', len(i_ct_bins) - 1, np.array(i_ct_bins, dtype=np.float64))
        h_spectrum.SetTitle(f'Corrected ct spectrum in pT bin {pt_min} < #it{{p}}_{{T}} < {pt_max} GeV/c')
        h_spectrum.GetXaxis().SetTitleSize(0.5)
        h_spectrum.GetYaxis().SetTitleSize(0.5)
        if use_BDT:
            h_BDT_efficiency = ROOT.TH1D(f'h_BDT_efficiency_pt_{pt_min}_{pt_max}', f';{x_label};{y_eff_BDT_label}', len(i_ct_bins) - 1, np.array(i_ct_bins, dtype=np.float64))
            h_BDT_efficiency.SetTitle(f'BDT Efficiency in pT bin {pt_min} < #it{{p}}_{{T}} < {pt_max} GeV/c')
            for i_ct in range(len(i_ct_bins)-1):
                ct_min = i_ct_bins[i_ct]
                ct_max = i_ct_bins[i_ct+1]
                ct_sel = f'fCt > {ct_min} & fCt < {ct_max}'
                ct_sel_MC = f'fGenCt > {ct_min} & fGenCt < {ct_max}'
                bin_data_hdl = bin_pt_data_hdl.apply_preselections(ct_sel, inplace=False)
                bin_mc_hdl = bin_pt_mc_hdl.apply_preselections(ct_sel_MC, inplace=False)
                bin_mc_evnsel = bin_mc_hdl.apply_preselections('fIsSurvEvSel==True', inplace=False)
                bin_mc_reco_hdl = bin_pt_mc_reco_hdl.apply_preselections(ct_sel_MC, inplace=False)
                accptance_bin = len(bin_mc_reco_hdl) / len(bin_mc_evnsel)
                exp_signal_bin = n_events * 3 * he3_spectrum.Integral(pt_min,pt_max) * accptance_bin * 1 * 0.25 * 2 * 1 * 1 / (len(i_ct_bins)-1)
                if new_training:
                    print("**Using Mechine Learning for H3l pre-selection**")
                    print(f'** Applying BDT to data for pt: {pt_min}-{pt_max} ct: {ct_min}-{ct_max}**')
                    if training_preselections != '':
                        bin_mc_hdl_ML = bin_mc_hdl.apply_preselections(training_preselections, inplace=False)
                        bin_data_hdl_ML = bin_data_hdl.apply_preselections(f"(fMassH3L<2.95 or fMassH3L>3.02) and {training_preselections}", inplace=False)
                    else:
                        bin_data_hdl_ML.apply_preselections(f"(fMassH3L<2.95 or fMassH3L>3.02)", inplace=True)
                    #Shift He3 nSigma
                    df_mcH = bin_mc_hdl_ML.get_data_frame()
                    df_mcH['fNSigmaHe'] = df_mcH['fNSigmaHe'] - df_mcH['fNSigmaHe'].mean()
                    bin_mc_hdl_ML.set_data_frame(df_mcH)
                    df_dataH = bin_data_hdl_ML.get_data_frame()
                    x_dataH = df_dataH['fNSigmaHe'].values
                    y_dataH = np.histogram(x_dataH, bins=100, density=True)[0]
                    x_dataH_hist = np.histogram(x_dataH, bins=100, density=True)[1][:-1]
                    init_guess = [max(y_dataH), 0, 1, 0, 0, 0, 0]
                    popt, _ = curve_fit(gauss_pol3, x_dataH_hist, y_dataH, p0=init_guess, bounds=([-np.inf, -1, 0, -np.inf, -np.inf, -np.inf, -np.inf], [np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf]))
                    A, mu, sigma, B, C, D, E = popt
                    #pdf check for the shift
                    plt.plot(x_dataH_hist, y_dataH, label='Background Data')
                    plt.plot(x_dataH_hist, gauss_pol3(x_dataH_hist, *popt), label='Gaussian+Poly Fit', linestyle='--')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"{BDT_QA_dir}/Gauss_pol3_fit_data_df_pt_{pt_min}_{pt_max}_ct_{ct_min}_{ct_max}.pdf")
                    plt.close()
                    df_dataH['fNSigmaHe'] = df_dataH['fNSigmaHe'] - mu
                    bin_data_hdl_ML.set_data_frame(df_dataH)
                    print("Let's do some cutting to balance the var range for data and MC samples ...")
                    print("***---------------Training Info------------------***")
                    print("Origin MC events: ", len(bin_mc_hdl_ML))
                    print("Origin Data events: ", len(bin_data_hdl_ML))
                    utils.cut_elements_to_same_range(bin_mc_hdl_ML,bin_data_hdl_ML,['fDcaHe','fDcaPi'])
                    if bkg_fraction_max != None:
                        if(len(bin_data_hdl_ML) > bkg_fraction_max * len(bin_mc_hdl_ML)):
                            bin_data_hdl_ML.shuffle_data_frame(size=bkg_fraction_max*len(bin_mc_hdl_ML), inplace=True, random_state=random_state)
                    print("------------------------------------------------")
                    print("Final MC events: ", len(bin_mc_hdl_ML))
                    print("Final Data events: ", len(bin_data_hdl_ML))
                    print("***---------------Training Info------------------***\n")
                    #Start training
                    train_test_data = au.train_test_generator([bin_mc_hdl_ML, bin_data_hdl_ML], [1,0], test_size=test_set_size, random_state=random_state)
                    train_features = train_test_data[0]
                    train_labels = train_test_data[1]
                    test_features = train_test_data[2]
                    test_labels = train_test_data[3]
                    distr = pu.plot_distr([bin_mc_hdl_ML, bin_data_hdl_ML], training_variables + ["fMassH3L"], bins=100, labels=['Signal',"Background"],colors=["blue","red"], log=True, density=True, figsize=(18, 13), alpha=0.5, grid=False)
                    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
                    plt.savefig(f"{BDT_QA_dir}/features_distributions_{pt_min}_{pt_max}_ct_{ct_min}_{ct_max}.pdf", bbox_inches='tight')
                    plt.close()
                    corr = pu.plot_corr([bin_mc_hdl_ML,bin_data_hdl_ML], training_variables + ["fMassH3L"], ['Signal',"Background"])
                    corr[0].savefig(f"{BDT_QA_dir}/correlations_mc_{pt_min}_{pt_max}_ct_{ct_min}_{ct_max}.pdf", bbox_inches='tight')
                    corr[1].savefig(f"{BDT_QA_dir}/correlations_data_{pt_min}_{pt_max}_ct_{ct_min}_{ct_max}.pdf", bbox_inches='tight')
                    plt.close("all")
                    model_hdl = ModelHandler(xgb.XGBClassifier(), training_variables)
                    model_hdl.set_model_params(hyperparams)
                    model_hdl.train_test_model(train_test_data, False, output_margin = True) ###output_margin == True return the unnormalized score
                    y_pred_test = model_hdl.predict(test_features, output_margin = True)
                    y_pred_train = model_hdl.predict(train_features, output_margin = True)
                    bdt_out_plot = pu.plot_output_train_test(model_hdl, train_test_data, 100, True, ["Signal", "Background"], True, density=True)
                    bdt_out_plot.savefig(f"{output_dir_name}/bdt_output_{pt_min}_{pt_max}_ct_{ct_min}_{ct_max}.pdf")
                    plt.close("all")
                    ### Applying the model to real data set
                    print("** Applying BDT model to data ...**")
                    bin_data_hdl.apply_model_handler(model_hdl, column_name="model_output")
                    bin_data_hdl.print_summary()
                    bin_data_hdl.write_df_to_parquet_files(f'dataH_BDTapplied_pt_{pt_min}_{pt_max}_ct_{ct_min}_{ct_max}', output_dir_name)
                    # Save test_labels and y_pred_test for later use
                    np.savez_compressed(
                        f"{output_dir_name}/bdt_test_labels_preds_pt_{pt_min}_{pt_max}_ct_{ct_min}_{ct_max}.npz",
                        test_labels=test_labels,
                        y_pred_test=y_pred_test
                    )
                else: 
                    bdt_file = Path(f'{output_dir_name}/dataH_BDTapplied_pt_{pt_min}_{pt_max}_ct_{ct_min}_{ct_max}.parquet.gzip')
                    bin_data_hdl = TreeHandler(bdt_file) if bdt_file.exists() else None
                if bin_data_hdl is None or len(bin_data_hdl) == 0:
                    raise ValueError("No data available after applying preselections and BDT model.\n please check the input data or the preselections. ")
                    exit()
                if new_WP_process:
                    ###calculate the significance x BDT efficiency
                    ##efficiencies vs score
                    #eff_arr, score_eff_arr = au.bdt_efficiency_array(test_labels, y_pred_test, n_points=npoints_for_WP)
                    if not new_training:
                        bdt_test_file = f"{output_dir_name}/bdt_test_labels_preds_pt_{pt_min}_{pt_max}_ct_{ct_min}_{ct_max}.npz"
                        bdt_test_data = np.load(bdt_test_file)
                        test_labels = bdt_test_data['test_labels']
                        y_pred_test = bdt_test_data['y_pred_test']
                    eff_arr = np.round(np.linspace(0.5, 0.99, npoints_for_WP), 3)
                    score_eff_arr = au.score_from_efficiency_array(test_labels, y_pred_test, eff_arr)
                    ##============just simple functions curvefit + Minuit==================
                    WP_BDT_score = []
                    WP_BDT_eff = []
                    WP_BKG_bkgcounts = []
                    WP_BKG_bkgcounts_up = []
                    WP_BKG_bkgcounts_down = []
                    WP_exp_significance = []
                    WP_exp_significance_up = []
                    WP_exp_significance_down = []
                    WP_chi2 = []        
                    for i_score in range(len(score_eff_arr)):
                        score = score_eff_arr[i_score]
                        efficency = eff_arr[i_score]
                        input_bin_data_hdl = bin_data_hdl.apply_preselections(f"model_output > {score}",inplace = False)
                        score_bin_df = input_bin_data_hdl.get_data_frame()
                        ### for side band
                        signal_region = (score_bin_df['fMassH3L'] >= 2.98) & (score_bin_df['fMassH3L'] <= 3.005)
                        sideband_low = score_bin_df['fMassH3L'] < 2.98
                        sideband_high = score_bin_df['fMassH3L'] > 3.005
                        sideband_data = score_bin_df[sideband_low | sideband_high]
                        x_side = sideband_data['fMassH3L'].values
                        y_side, bins = np.histogram(x_side, bins=n_bins_mass_data, range=(mass_range[0], mass_range[1]))
                        x_side_centers = (bins[:-1] + bins[1:]) / 2
                        try: 
                            popt_side, pcov_side = curve_fit(pol2, x_side_centers, y_side)
                            x_all = score_bin_df['fMassH3L'].values
                            y_all, bins = np.histogram(x_all, bins=n_bins_mass_data, range=(mass_range[0], mass_range[1]))
                            x_all_centers = (bins[:-1] + bins[1:]) / 2
                            initial_guess = list(popt_side) + [(min(y_all) + max(y_all))/2, 2.991, 0.01]  # A, mu, sigma
                            def chi2(a, b, c, A, mu, sigma):
                                y_pred = combined_fit(x_all_centers, a, b, c, A, mu, sigma)
                                return np.sum((y_all - y_pred)**2)
                            m = Minuit(chi2, *initial_guess)
                            m.limits['sigma'] = (0.001, 0.1)  # 限制sigma范围
                            m.limits['mu'] = (2.99, 3.01)     # 限制mu范围
                            m.migrad()  # 执行最小化
                            params = m.values
                            errors = m.errors
                            chi2_value = m.fval
                            if save_working_points_fit:
                                x_plot = np.linspace(mass_range[0], mass_range[1], 1000)
                                plt.plot(x_all_centers, y_all, label='Data', marker='o', linestyle='None')
                                plt.plot(x_plot, combined_fit(x_plot, *params), label='Gauss + pol2', linestyle='--', color='orange')
                                plt.plot(x_plot, pol2(x_plot, *params[:3]), label='pol2', linestyle='--', color='green')
                                plt.plot(x_plot, gauss(x_plot, *params[3:]), label='Gauss', linestyle='--', color='red')
                                plt.xlabel(inv_mass_string)
                                plt.title(f'Combined Fit (BDT cut: {score:.2f}, Eff: {efficency:.2f})')
                                plt.ylabel('Counts')
                                plt.legend()
                                plt.tight_layout()
                                plt.savefig(f"{working_point_fit_dir}/combined_fit_score_{score:.2f}_pt_{pt_min}_{pt_max}_ct_{ct_min}_{ct_max}.pdf")
                                plt.close()
                            a, b, c, sigma = params['a'], params['b'], params['c'], params['sigma']
                            background_signal, count_err = quad(lambda x: pol2(x, a, b, c), 2.98, 3.005)
                            background_signal_1sigmadown, count_err_1sigmadown = quad(lambda x: pol2(x, a, b, c), 2.98+sigma, 3.005-sigma)
                            background_signal_1sigmaup, count_err_1sigmaup = quad(lambda x: pol2(x, a, b, c), 2.98-sigma, 3.005+sigma)
                            binwidth = bins[1] - bins[0]
                            total_background = background_signal / binwidth
                            total_background_1sigmaup = background_signal_1sigmaup / binwidth
                            total_background_1sigmadown = background_signal_1sigmadown / binwidth
                            exp_significance = exp_signal_bin / np.sqrt(exp_signal_bin + total_background)
                            exp_significance_1sigmaup = exp_signal_bin / np.sqrt(exp_signal_bin + total_background_1sigmaup)
                            exp_significance_1sigmadown = exp_signal_bin / np.sqrt(exp_signal_bin + total_background_1sigmadown)
                            WP_BDT_score.append(score)
                            WP_BDT_eff.append(efficency)
                            WP_BKG_bkgcounts.append(total_background)
                            WP_BKG_bkgcounts_up.append(total_background_1sigmaup)
                            WP_BKG_bkgcounts_down.append(total_background_1sigmadown)
                            WP_exp_significance.append(exp_significance)
                            WP_exp_significance_up.append(exp_significance_1sigmaup)
                            WP_exp_significance_down.append(exp_significance_1sigmadown)
                            WP_chi2.append(chi2_value)
                        except RuntimeWarning:
                            print("Fit failed for score:", score)
                            continue
                    df_working_point = pd.DataFrame({
                        'BDT_score': WP_BDT_score,
                        'BDT_efficiency': WP_BDT_eff,
                        'BKG_bkgcounts': WP_BKG_bkgcounts,
                        'BKG_bkgcounts_up': WP_BKG_bkgcounts_up,
                        'BKG_bkgcounts_down': WP_BKG_bkgcounts_down,
                        'exp_significance': WP_exp_significance,
                        'exp_significance_up': WP_exp_significance_up,
                        'exp_significance_down': WP_exp_significance_down,
                        'chi2': WP_chi2
                    })
                    df_working_point['product'] = df_working_point['BDT_efficiency'] * df_working_point['exp_significance']
                    df_working_point['product_up'] = df_working_point['BDT_efficiency'] * df_working_point['exp_significance_down']
                    df_working_point['product_down'] = df_working_point['BDT_efficiency'] * df_working_point['exp_significance_up']
                    df_working_point.to_csv(f"{output_dir_name}/working_point_data_frame_pt_{pt_min}_{pt_max}_ct_{ct_min}_{ct_max}.csv")
                else:
                    wp_file = Path(f"{output_dir_name}/working_point_data_frame_pt_{pt_min}_{pt_max}_ct_{ct_min}_{ct_max}.csv")
                    df_working_point = pd.read_csv(wp_file) if wp_file.exists() else None
                if df_working_point is None or len(df_working_point) == 0:
                    raise ValueError("No working point data available after applying preselections and BDT model.\n please check the input data or the preselections. ")
                    exit()
                #df_working_point = df_working_point[df_working_point['BKG_bkgcounts'] != 0]
                df_working_point = df_working_point.query('BKG_bkgcounts > 0 & BKG_bkgcounts_down > 0 & BKG_bkgcounts_up > 0 & BKG_bkgcounts_down / BKG_bkgcounts < 1 & BKG_bkgcounts_up / BKG_bkgcounts > 1')
                max_row = df_working_point.loc[df_working_point['product'].idxmax()]
                max_product = max_row['product']
                max_efficiency = max_row['BDT_efficiency']
                max_score = max_row['BDT_score']
                ds = xr.Dataset(
                    {
                        "center": ("x", df_working_point['product']),
                        "upper": ("x", df_working_point['product_up']),
                        "lower": ("x", df_working_point['product_down']),
                    },
                    coords={"x": df_working_point['BDT_score']},
                )
                plt.figure(figsize=(10, 6))
                plt.plot(ds.x, ds.center, color="black", label="Center", linewidth=2)
                plt.scatter(ds.x, ds.upper, color="red", s=10, label="Upper 1σ", alpha=0.5)
                plt.scatter(ds.x, ds.lower, color="blue", s=10, label="Lower 1σ", alpha=0.5)
                plt.fill_between(ds.x, ds.lower, ds.upper, color="gray", alpha=0.2, label="1σ Region")
                plt.xlabel("BDT_Score")
                plt.ylabel("Expected Significance x BDT_Efficiency")
                plt.title(f"pt:{pt_min}-{pt_max} ct:{ct_min}-{ct_max} WP: {max_score:.2f} (Eff: {max_efficiency:.2f})")
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.5)
                plt.tight_layout()
                plt.text(
                    x=0.5,  # x 坐标（0~1 是相对坐标，也可以直接用数据坐标）
                    y=0.2,  # y 坐标（0~1 是相对坐标）
                    s=f"Working Point: BDT_Score={max_score:.2f}, BDT_eff={max_efficiency:.2f}",  # 文本内容
                    transform=plt.gca().transAxes,  # 使用相对坐标
                    bbox=dict(  # 设置文本框样式
                        facecolor="white",  # 背景色
                        edgecolor="white",  # 边框颜色
                        boxstyle="round,pad=0.5",  # 圆角边框
                        alpha=0.8,  # 透明度
                    ),
                    fontsize=12,
                    ha="center",  # 水平对齐
                    va="center",  # 垂直对齐
                )
                plt.savefig(f"{output_dir_name}/exp_significance_vs_BDT_score_pt_{pt_min}_{pt_max}_ct_{ct_min}_{ct_max}.pdf")
                plt.close()
                matter_sel = 'fIsMatter == True' if is_matter == 'matter' else 'fIsMatter == False' if is_matter == 'antimatter' else ''
                matter_sel_MC = 'fGenPt > 0' if is_matter == 'matter' else 'fGenPt < 0' if is_matter == 'antimatter' else ''
                bin_data_hdl.apply_preselections(f"model_output > {max_score}", inplace=True)
                if matter_sel != '':
                    bin_data_hdl.apply_preselections(matter_sel, inplace=True)
                    bin_mc_hdl.apply_preselections(matter_sel_MC, inplace=True)
                    bin_mc_reco_hdl.apply_preselections(matter_sel_MC, inplace=True)
                    bin_mc_evnsel.apply_preselections(matter_sel_MC, inplace=True)
                #ROOFIT DSCB + pol2 for signal extraction
                mass = ROOT.RooRealVar('m', inv_mass_string, 2.96, 3.04, 'GeV/c^{2}')
                mu = ROOT.RooRealVar('mu', 'hypernucl mass', 2.991 , 2.985, 2.992, 'GeV/c^{2}') # 2.985, 2.992 (big range for systematics)
                sigma = ROOT.RooRealVar('sigma', 'hypernucl width', 0.002, 0.001, 0.003, 'GeV/c^{2}')# 0.001, 0.0024
                a1 = ROOT.RooRealVar('a1', 'a1', 0.7, 5.)
                a2 = ROOT.RooRealVar('a2', 'a2', 0.7, 5.)
                n1 = ROOT.RooRealVar('n1', 'n1', 0., 5.)
                n2 = ROOT.RooRealVar('n2', 'n2', 0., 5.)
                c0 = ROOT.RooRealVar('c0', 'constant c0', -1., 1)
                c1 = ROOT.RooRealVar('c1', 'constant c1', -1., 1)
                signal = ROOT.RooCrystalBall('signal', 'dscb', mass, mu, sigma, a1, n1, a2, n2)
                background = ROOT.RooChebychev('bkg', 'pol2 bkg', mass, ROOT.RooArgList(c0, c1))
                n_signal = ROOT.RooRealVar('n_signal', 'n_signal', 0., 1e4)#5e2, 1e3
                n_background = ROOT.RooRealVar('n_background', 'n_background', 0., 1e6)#1e4, 1e6
                # fit mc first and fix tails parameter
                mass_roo_mc = utils.ndarray2roo(np.array(bin_mc_hdl['fMassH3L'].values, dtype=np.float64), mass, 'histo_mc')
                fit_results_mc = signal.fitTo(mass_roo_mc, ROOT.RooFit.Range(2.97, 3.01), ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-1))
                a1.setConstant()
                a2.setConstant()
                n1.setConstant()
                n2.setConstant()
                sigma.setRange(sigma_range_mc_to_data[i_ct][0]*sigma.getVal(), sigma_range_mc_to_data[i_ct][1]*sigma.getVal())
                print("sigma MC: ", sigma.getVal())
                print("sigma range set to: ", sigma_range_mc_to_data[i_ct][0]*sigma.getVal(), sigma_range_mc_to_data[i_ct][1]*sigma.getVal())
                mc_frame_fit = mass.frame(n_bins_mass_mc)
                mc_frame_fit.SetName(f"MC Mass fit for Pt_{pt_min}_{pt_max}_Ct_{ct_min}_{ct_max}")
                mass_roo_mc.plotOn(mc_frame_fit, ROOT.RooFit.Name('mc'), ROOT.RooFit.DrawOption('p'))
                signal.plotOn(mc_frame_fit, ROOT.RooFit.Name('signal'), ROOT.RooFit.DrawOption('p'))
                fit_param = ROOT.TPaveText(0.6, 0.43, 0.9, 0.85, 'NDC')
                fit_param.SetBorderSize(0)
                fit_param.SetFillStyle(0)
                fit_param.SetTextAlign(12)
                fit_param.AddText(r'#mu = ' + f'{mu.getVal()*1e3:.2f} #pm {mu.getError()*1e3:.2f}' + ' MeV/#it{c}^{2}')
                fit_param.AddText(r'#sigma = ' + f'{sigma.getVal()*1e3:.2f} #pm {sigma.getError()*1e3:.2f}' + ' MeV/#it{c}^{2}')
                fit_param.AddText(r'alpha_{L} = ' + f'{a1.getVal():.2f} #pm {a1.getError():.2f}')
                fit_param.AddText(r'alpha_{R} = ' + f'{a2.getVal():.2f} #pm {a2.getError():.2f}')
                fit_param.AddText(r'n_{L} = ' + f'{n1.getVal():.2f} #pm {n1.getError():.2f}')
                fit_param.AddText(r'n_{R} = ' + f'{n2.getVal():.2f} #pm {n2.getError():.2f}')
                mc_frame_fit.addObject(fit_param)
                chi2_mc = mc_frame_fit.chiSquare('signal', 'mc')
                ndf_mc = n_bins_mass_mc - fit_results_mc.floatParsFinal().getSize()
                fit_param.AddText('#chi^{2} / NDF = ' + f'{chi2_mc:.3f} (NDF: {ndf_mc})')
                pdf = ROOT.RooAddPdf('total_pdf', 'signal + background', ROOT.RooArgList(signal, background), ROOT.RooArgList(n_signal, n_background))
                mass_roo_data = utils.ndarray2roo(np.array(bin_data_hdl['fMassH3L'].values, dtype=np.float64), mass, 'histo_data')
                fit_results_data = pdf.fitTo(mass_roo_data, ROOT.RooFit.Extended(True), ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-1))
                fit_pars = pdf.getParameters(mass_roo_data)
                sigma_val = fit_pars.find('sigma').getVal()
                sigma_val_error = fit_pars.find('sigma').getError()
                print("Data fit sigma: ", sigma_val, "+/-", sigma_val_error)
                mu_val = fit_pars.find('mu').getVal()
                mu_val_error = fit_pars.find('mu').getError()
                signal_counts = n_signal.getVal()
                signal_counts_error = n_signal.getError()
                background_counts = n_background.getVal()
                background_counts_error = n_background.getError()
                data_frame_fit = mass.frame(n_bins_mass_data)
                data_frame_fit.SetName(f"Data Mass fit for Pt_{pt_min}_{pt_max}_Ct_{ct_min}_{ct_max}")
                mass_roo_data.plotOn(data_frame_fit, ROOT.RooFit.Name('data'), ROOT.RooFit.DrawOption('p'))
                pdf.plotOn(data_frame_fit, ROOT.RooFit.Components('bkg'), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(kOrangeC))
                pdf.plotOn(data_frame_fit, ROOT.RooFit.Components('signal'), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kGreen + 2 ))
                pdf.plotOn(data_frame_fit, ROOT.RooFit.LineColor(ROOT.kAzure + 2 ), ROOT.RooFit.Name('fit_func'))
                chi2_data = data_frame_fit.chiSquare('fit_func', 'data')
                ndf_data = n_bins_mass_data - fit_results_data.floatParsFinal().getSize()
                data_frame_fit.GetYaxis().SetTitleSize(0.06)
                data_frame_fit.GetYaxis().SetTitleOffset(0.9)
                data_frame_fit.GetYaxis().SetMaxDigits(2)
                data_frame_fit.GetXaxis().SetTitleOffset(1.1)
                # signal within 3 sigma
                mass.setRange('signal', mu_val-3*sigma_val, mu_val+3*sigma_val)
                signal_int = signal.createIntegral(ROOT.RooArgSet(mass), ROOT.RooArgSet(mass), 'signal')
                signal_int_val_3s = signal_int.getVal()*signal_counts
                signal_int_val_3s_error = signal_int_val_3s*signal_counts_error/signal_counts
                # background within 3 sigma
                mass.setRange('bkg', mu_val-3*sigma_val, mu_val+3*sigma_val)
                bkg_int = background.createIntegral(ROOT.RooArgSet(mass), ROOT.RooArgSet(mass), 'bkg')
                bkg_int_val_3s = bkg_int.getVal()*background_counts
                bkg_int_val_3s_error = bkg_int_val_3s*background_counts_error/background_counts
                significance = signal_int_val_3s / np.sqrt(signal_int_val_3s + bkg_int_val_3s)
                significance_err = utils.significance_error(signal_int_val_3s, bkg_int_val_3s, signal_int_val_3s_error, bkg_int_val_3s_error)
                s_b_ratio_err = np.sqrt((signal_int_val_3s_error/signal_int_val_3s)**2 + (bkg_int_val_3s_error/bkg_int_val_3s)**2)*signal_int_val_3s/bkg_int_val_3s
                # add pave for stats
                pinfo_vals = ROOT.TPaveText(0.632, 0.5, 0.932, 0.85, 'NDC')
                pinfo_vals.SetBorderSize(0)
                pinfo_vals.SetFillStyle(0)
                pinfo_vals.SetTextAlign(11)
                pinfo_vals.SetTextFont(42)
                pinfo_vals.AddText(f'BDT cut: {max_score:.2f} (Eff: {max_efficiency:.2f})')
                pinfo_vals.AddText(f'Signal (S): {signal_counts:.0f} #pm {signal_counts_error:.0f}')
                pinfo_vals.AddText(f'S/B (3 #sigma): {signal_int_val_3s/bkg_int_val_3s:.1f} #pm {s_b_ratio_err:.1f}')
                pinfo_vals.AddText('S/#sqrt{S+B} (3 #sigma): ' + f'{significance:.1f} #pm {significance_err:.1f}')
                pinfo_vals.AddText('#mu = ' + f'{mu_val*1e3:.2f} #pm {mu.getError()*1e3:.2f}' + ' MeV/#it{c}^{2}')
                pinfo_vals.AddText('#sigma = ' + f'{sigma_val*1e3:.2f} #pm {sigma.getError()*1e3:.2f}' + ' MeV/#it{c}^{2}')
                pinfo_vals.AddText('#chi^{2} / NDF = ' + f'{chi2_data:.3f} (NDF: {ndf_data})')
                ## add pave for ALICE performance
                pinfo_alice = ROOT.TPaveText(0.14, 0.6, 0.42, 0.85, 'NDC')
                pinfo_alice.SetBorderSize(0)
                pinfo_alice.SetFillStyle(0)
                pinfo_alice.SetTextAlign(11)
                pinfo_alice.SetTextFont(42)
                pinfo_alice.AddText('ALICE Performance')
                sqrtsnn = "#sqrt{#it{s_{NN}}}"
                pinfo_alice.AddText(f'Run 3, Pb--Pb @ {sqrtsnn} = 5.36 TeV')
                exponent = np.floor(np.log10(n_events))
                n_events_scaled = n_events / 10**(exponent)
                pinfo_alice.AddText('N_{ev} = ' + f'{n_events_scaled:.1f} ' + '#times 10^{' + f'{exponent:.0f}' + '}') 
                pinfo_alice.AddText(decay_string)
                data_frame_fit.addObject(pinfo_vals)
                data_frame_fit.addObject(pinfo_alice)
                bin_output_dir.cd()
                mc_frame_fit.Write()
                data_frame_fit.Write()
                canvas_saving = ROOT.TCanvas(f'c_mass_fit_pt_{pt_min}_{pt_max}_ct_{ct_min}_{ct_max}', f'c_mass_fit_pt_{pt_min}_{pt_max}_ct_{ct_min}_{ct_max}', 800, 600)
                mc_frame_fit.Draw()
                canvas_saving.SaveAs(f"{output_dir_name}/mc_mass_fit_pt_{pt_min}_{pt_max}_ct_{ct_min}_{ct_max}.pdf")
                canvas_saving.Clear()
                data_frame_fit.Draw()
                canvas_saving.SaveAs(f"{output_dir_name}/data_mass_fit_pt_{pt_min}_{pt_max}_ct_{ct_min}_{ct_max}.pdf")
                ###calculate the ct_spectrum

                acc = len(bin_mc_reco_hdl)/len(bin_mc_evnsel)
                bin_width = i_ct_bins[i_ct+1] - i_ct_bins[i_ct]
                corrected_counts = signal_int_val_3s/(acc * max_efficiency * bin_width)
                corrected_counts_err = signal_int_val_3s_error/(acc * max_efficiency * bin_width)
                h_raw_counts.SetBinContent(i_ct+1, signal_int_val_3s)
                h_raw_counts.SetBinError(i_ct+1, signal_int_val_3s_error)
                h_acc_eff.SetBinContent(i_ct+1, acc * max_efficiency)
                h_BDT_efficiency.SetBinContent(i_ct+1, max_efficiency)
                h_spectrum.SetBinContent(i_ct+1, corrected_counts)
                h_spectrum.SetBinError(i_ct+1, corrected_counts_err)
                corrected_counts_arr_pt.append(corrected_counts)
                corrected_counts_err_arr_pt.append(corrected_counts_err)
                BDT_efficiency_arr_pt.append(max_efficiency)
                acceptance_arr_pt.append(acc)
                raw_counts_arr_pt.append(signal_int_val_3s)
                raw_counts_err_arr_pt.append(signal_int_val_3s_error)
        corrected_counts_arr.append(corrected_counts_arr_pt)
        corrected_counts_err_arr.append(corrected_counts_err_arr_pt)
        BDT_efficiency_arr.append(BDT_efficiency_arr_pt)
        acceptance_arr.append(acceptance_arr_pt)
        raw_counts_arr.append(raw_counts_arr_pt)
        raw_counts_err_arr.append(raw_counts_err_arr_pt)
    # --- Fit corrected ct spectrum with exponential function for each pt bin ---
    tau_hist = ROOT.TH1D('tau_per_ptbin', ';#tau (ps);Counts', len(pt_bins)-1, np.array(pt_bins, dtype=np.float64))
    tau_err_hist = ROOT.TH1D('tau_err_per_ptbin', ';#tau error (ps);Counts', len(pt_bins)-1, np.array(pt_bins, dtype=np.float64))
    for i_pt_draw in range(len(pt_bins)-1):
        ct_centers = []
        ct_values = []
        ct_errors = []
        for i_ct in range(len(ct_bins[i_pt_draw])-1):
            ct_center = 0.5 * (ct_bins[i_pt_draw][i_ct] + ct_bins[i_pt_draw][i_ct+1])
            ct_centers.append(ct_center)
            ct_values.append(corrected_counts_arr[i_pt_draw][i_ct])
            ct_errors.append(corrected_counts_err_arr[i_pt_draw][i_ct])
        ct_centers = np.array(ct_centers)
        ct_values = np.array(ct_values)
        ct_errors = np.array(ct_errors)
        # Exponential fit: f(ct) = N0 * exp(-ct/tau)
        def expo_func(x, N0, tau):
            return N0 * np.exp(-x / tau)
        # Initial guess
        N0_guess = ct_values[0] if len(ct_values) > 0 else 1
        tau_guess = 8
        # Use Minuit for fit
        def chi2(N0, tau):
            model = expo_func(ct_centers, N0, tau)
            return np.sum(((ct_values - model) / ct_errors) ** 2)
        m = Minuit(chi2, N0=N0_guess, tau=tau_guess)
        m.limits['N0'] = (0, None)
        m.limits['tau'] = (0, 20)
        m.migrad()
        tau_val = m.values['tau']
        tau_err = m.errors['tau']
        tau_hist.SetBinContent(i_pt_draw+1, tau_val)
        tau_hist.SetBinError(i_pt_draw+1, tau_err)
        tau_err_hist.SetBinContent(i_pt_draw+1, tau_err)
        tau_err_hist.SetBinError(i_pt_draw+1, 0)
        # Draw fit QA plot
        plt.errorbar(ct_centers, ct_values, yerr=ct_errors, fmt='o', label='Corrected counts')
        plt.plot(ct_centers, expo_func(ct_centers, m.values['N0'], tau_val), label=f'Exp Fit: tau={tau_val:.1f}±{tau_err:.1f} ps')
        plt.xlabel('ct (cm)')
        plt.ylabel('Corrected counts')
        plt.ylim(1, 1e4)
        plt.title(f'Pt bin {pt_bins[i_pt_draw]}-{pt_bins[i_pt_draw+1]} GeV/c')
        plt.text(
            0.05, 0.95,
            f'Fit ct: {tau_val:.2f} ± {tau_err:.2f} ps',
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir_name}/ct_spectrum_fit_pt_{pt_bins[i_pt_draw]}_{pt_bins[i_pt_draw+1]}.pdf")
        plt.close()
    # Save tau histograms to ROOT file
    output_dir_std.cd()
    tau_hist.Write()
    tau_err_hist.Write()

    # --- QA plots for raw_counts, acceptance, BDT_efficiency ---
    for i_dx, (arr, arr_err, name, ylabel) in enumerate([
        (raw_counts_arr, raw_counts_err_arr, 'raw_counts', 'Raw counts'),
        (acceptance_arr, None, 'acceptance', 'Acceptance'),
        (BDT_efficiency_arr, None, 'BDT_efficiency', 'BDT efficiency')
    ]):
        for i_pt_qa in range(len(pt_bins)-1):
            plt.figure(figsize=(8,6))
            ct_centers = [0.5*(ct_bins[i_pt_qa][i]+ct_bins[i_pt_qa][i+1]) for i in range(len(ct_bins[i_pt_qa])-1)]
            values = arr[i_pt_qa]
            errors = arr_err[i_pt_qa] if arr_err is not None else None
            label = f'Pt {pt_bins[i_pt_qa]}-{pt_bins[i_pt_qa+1]}'
            # print("ct centers:", ct_centers)
            # print("values:", values)
            if errors is not None:
                plt.errorbar(ct_centers, values, yerr=errors, fmt='o', label=label)
            else:
                plt.plot(ct_centers, values, marker='o', label=label)
            plt.xlabel('ct (cm)')
            plt.ylabel(ylabel)
            plt.title(f'{ylabel} vs ct')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(f"{output_dir_name}/QA_fig_{name}_pt_{pt_bins[i_pt_qa]}_{pt_bins[i_pt_qa+1]}.pdf")
            plt.close()
    print("** ct analysis done. ** \n")


    if do_syst:
        n_trials = config['n_trials']
        output_dir_syst = output_file.mkdir('trials')
        ## list of trial strings to be printed to a text file
        trial_strings = []
        print("----------------------------------")
        print("** Starting systematics analysis **")
        print(f'** {n_trials} trials will be tested **')


        cut_dict_syst = config['cut_dict_syst']
        signal_fit_func_syst = config['signal_fit_func_syst']
        bkg_fit_func_syst = config['bkg_fit_func_syst']
        ### create a dictionary with the same keys
        cut_string_dict = {}
        for var in cut_dict_syst:
            var_dict = cut_dict_syst[var]
            cut_greater = var_dict['cut_greater']
            cut_greater_string = " > " if cut_greater else " < "

            cut_list = var_dict['cut_list']
            cut_arr = np.linspace(cut_list[0], cut_list[1], cut_list[2])
            cut_string_dict[var] = []
            for cut in cut_arr:
                cut_string_dict[var].append(var + cut_greater_string + str(cut))

        cut_string_dict['signal_fit_func'] = signal_fit_func_syst
        cut_string_dict['bkg_fit_func'] = bkg_fit_func_syst
        combos = list(product(*list(cut_string_dict.values())))

        if n_trials < len(combos):
            combo_random_indices = np.random.randint(len(combos), size=(n_trials, len(ct_bins) - 1))
        else:
            print(f"** Warning: n_trials > n_combinations ({n_trials}, {len(combos)}), taking all the possible combinations **")
            indices = np.arange(len(combos))
            ## create a (len(combos), len(ct_bins) - 1) array with the indices repeated for each ct bin
            combo_random_indices = np.repeat(indices[:, np.newaxis], len(ct_bins) - 1, axis=1)
            ## now shuffle each column of the array
            for i in range(combo_random_indices.shape[1]):
                np.random.shuffle(combo_random_indices[:, i])

        combo_check_map = {}

        for i_combo, combo_indices in enumerate(combo_random_indices):
            trial_strings.append("----------------------------------")
            trial_num_string = f'Trial: {i_combo} / {len(combo_random_indices)}'
            trial_strings.append(trial_num_string)
            print(trial_num_string)

            cut_selection_list = []
            bkg_fit_func_list = []
            signal_fit_func_list = []

            for ict in range(len(ct_bins) - 1):
                combo = combos[combo_indices[ict]]
                ct_bin = [ct_bins[ict], ct_bins[ict + 1]]
                full_combo_string = f'ct {ct_bin[0]}_{ct_bin[1]} | '
                full_combo_string += " & ".join(combo)

                ### extract a signal and a background fit function
                sel_string = " & ".join(combo[: -2])
                signal_fit_func = combo[-2]
                bkg_fit_func = combo[-1]

                if full_combo_string in combo_check_map:
                    break

                combo_check_map[full_combo_string] = True

                cut_selection_list.append(sel_string)
                bkg_fit_func_list.append(bkg_fit_func)
                signal_fit_func_list.append(signal_fit_func)

            if len(cut_selection_list) != len(ct_bins) - 1:
                continue

            trial_strings.append(str(cut_selection_list))
            trial_strings.append(str(bkg_fit_func_list))
            trial_strings.append(str(signal_fit_func_list))

            ### make_spectra
            spectra_maker.selection_string = cut_selection_list
            spectra_maker.inv_mass_signal_func = signal_fit_func_list
            spectra_maker.inv_mass_bkg_func = bkg_fit_func_list

            spectra_maker.make_spectra()
            spectra_maker.make_histos()

            ## prepare for exponential fit
            start_bin = spectra_maker.h_corrected_counts.FindBin(fit_range[0])
            end_bin = spectra_maker.h_corrected_counts.FindBin(fit_range[1])
            expo.FixParameter(0, spectra_maker.h_corrected_counts.Integral(start_bin, end_bin, "width"))
            spectra_maker.fit()

            res_string = "Lifetime: " + str(spectra_maker.fit_func.GetParameter(1)) + " +- " + str(spectra_maker.fit_func.GetParError(1)) + " Prob: " + str(spectra_maker.fit_func.GetProb())
            trial_strings.append(res_string)

            if spectra_maker.fit_func.GetProb() > 0.15:
                trial_dir = output_dir_syst.mkdir(f'trial_{i_combo}')
                spectra_maker.output_dir = trial_dir
                spectra_maker.dump_to_output_dir()
                lifetime_dist.Fill(spectra_maker.fit_func.GetParameter(1))
                lifetime_prob.Fill(spectra_maker.fit_func.GetProb())

            spectra_maker.del_dyn_members()
    
    ## write trial strings to a text file
    if do_syst:
        if os.path.exists(f'{output_dir_name}/{output_file_name}.txt'):
            os.remove(f'{output_dir_name}/{output_file_name}.txt')
        with open(f'{output_dir_name}/{output_file_name}.txt', 'w') as f:
            for trial_string in trial_strings:
                if isinstance(trial_string, list):
                    for line in trial_string:
                        f.write("%s\n" % line)
                else:
                    f.write("%s\n" % trial_string)
