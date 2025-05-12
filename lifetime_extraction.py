import ROOT
import numpy as np
import uproot
import pandas as pd
import argparse
import os
import yaml
import matplotlib.pyplot as plt
from math import exp, sqrt, pi, gamma

from scipy.stats import norm, expon, crystalball
from scipy.special import erf, eval_chebyt
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
import hipe4ml.analysis_utils as au
import hipe4ml.plot_utils as pu

import mplhep as mpl
import xgboost as xgb

### As we use dscb function we choose ROOFit instate of iminuit
### tutorial using iminuit for SWeights and COWs: see https://sweights.github.io/sweights/notebooks/basic.html
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL
from iminuit.cost import UnbinnedNLL

# sweights imports
from sweights import SWeight # for classic sweights
from sweights import Cow     # for custom orthogonal weight functions
from sweights import cov_correct, approx_cov_correct # for covariance corrections
from sweights.testing import make_classic_toy # to generate a toy dataset
from sweights.util import plot_binned, make_weighted_negative_log_likelihood, convert_rf_pdf

import sys
sys.path.append('utils')
import utils as utils
kOrangeC  = ROOT.TColor.GetColor('#ff7f00')


### def of the python signal and bkg pdf
def double_crystalball(x, mu, sigma, alpha1, n1, alpha2, n2, xmin=None, xmax=None):
    """
    改进的双侧 Crystal Ball PDF，支持标量和数组输入
    
    参数:
        x      : 观测值（可以是标量或数组）
        mu     : 均值
        sigma  : 标准差
        alpha1 : 左尾阈值
        n1     : 左尾幂次
        alpha2 : 右尾阈值
        n2     : 右尾幂次
        xmin   : 归一化范围下限（可选）
        xmax   : 归一化范围上限（可选）
    """
    # 确保输入为 numpy 数组（兼容标量输入）
    x = np.asarray(x)
    is_scalar = x.ndim == 0  # 检查是否为标量
    if is_scalar:
        x = np.array([x])  # 转换为单元素数组
    
    # 计算归一化常数
    def norm(a, n):
        return (n / np.abs(a)) ** n * np.exp(-0.5 * a**2)
    
    A1 = norm(alpha1, n1)
    B1 = n1 / np.abs(alpha1) - np.abs(alpha1)
    A2 = norm(alpha2, n2)
    B2 = n2 / np.abs(alpha2) - np.abs(alpha2)
    
    # 理论归一化因子
    theoretical_norm = 1.0 / (sigma * (
        np.sqrt(np.pi/2) * (erf(np.abs(alpha1)/np.sqrt(2)) + erf(np.abs(alpha2)/np.sqrt(2))) +
        A1 / (n1 - 1) + A2 / (n2 - 1)
    ))
    
    # 计算未归一化的PDF
    z = (x - mu) / sigma
    pdf = np.zeros_like(x)
    
    # 分别处理三个区域
    mask_left = z < -alpha1
    mask_right = z > alpha2
    mask_center = ~mask_left & ~mask_right
    
    pdf[mask_center] = np.exp(-0.5 * z[mask_center]**2)
    pdf[mask_left] = A1 * (B1 - z[mask_left]) ** (-n1)
    pdf[mask_right] = A2 * (B2 + z[mask_right]) ** (-n2)
    pdf *= theoretical_norm
    
    # 如果指定了范围，进行重新归一化
    if xmin is not None or xmax is not None:
        # 定义被积函数（处理标量输入）
        def integrand(x_val):
            z_val = (x_val - mu)/sigma
            if z_val < -alpha1:
                return A1 * (B1 - z_val) ** (-n1) * theoretical_norm
            elif z_val > alpha2:
                return A2 * (B2 + z_val) ** (-n2) * theoretical_norm
            else:
                return np.exp(-0.5 * z_val**2) * theoretical_norm
        
        # 计算数值积分
        int_min = xmin if xmin is not None else mu - 10*sigma
        int_max = xmax if xmax is not None else mu + 10*sigma
        area, _ = quad(integrand, int_min, int_max)
        pdf /= area
    
    # 如果是标量输入，返回标量结果
    return pdf[0] if is_scalar else pdf

def chebyshev_poly(x, c0, c1, c2, xmin=-1, xmax=1):
    """仅计算Chebyshev多项式部分(未归一化)"""
    x_scaled = 2 * (x - xmin) / (xmax - xmin) - 1  # 映射到[-1,1]
    return c0 * eval_chebyt(0, x_scaled) + c1 * eval_chebyt(1, x_scaled) + c2 * eval_chebyt(2, x_scaled)

def chebyshev_pdf(x, c0, c1, c2, xmin=-1, xmax=1): #c0 = 1 for ROOFit
    """归一化的PDF"""
    poly = chebyshev_poly(x, c0, c1, c2, xmin, xmax)
    norm, _ = quad(lambda x: chebyshev_poly(x, c0, c1, c2, xmin, xmax), xmin, xmax)  # 调用poly而非pdf
    return poly / norm

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure the parameters of the script.')
    parser.add_argument('--config-file', dest='config_file', help="path to the YAML file with configuration.", default='')
    args = parser.parse_args()
    if args.config_file == '':
        print('** No config file provided. Exiting. **')
        exit()

    config_file = open(args.config_file, 'r')
    config = yaml.full_load(config_file)
    
    input_file_name_data = config['input_file_name_data']
    input_file_name_mc = config['input_file_name_mc']
    output_dir_name = config['output_dir_name']
    pt_bins = config['pt_bins']
    is_matter = config['is_matter']
    method = config['method']
    pre_selection_method = config['pre_selection_method']
    signal_fit_func = config['signal_fit_func']
    bkg_fit_func = config['bkg_fit_func']
    mass_range = config['mass_range']
    ct_range = config['ct_range']
    sigma_range_mc_to_data = config['sigma_range_mc_to_data']
    nbins_plot_mc = config['nbins_plot_mc']
    nbins_plot_data = config['nbins_plot_data']
    save_fit_results = config['save_fit_results']
    selection = config['selection']
    
    print('**********************************')
    print('    Running lifetime_extraction.py')
    print('**********************************\n')
    print("----------------------------------")
    print("** Loading data and apply preselections **")
    
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)
    out_file = ROOT.TFile.Open(f'{output_dir_name}/H3l_lifetime_extraction.root', 'RECREATE')
    stddir = out_file.mkdir('std')
    
    tree_names = ['O2datahypcands','O2hypcands', 'O2hypcandsflow']
    tree_keys = uproot.open(input_file_name_data).keys()
    for tree in tree_names:
        for key in tree_keys:
            if tree in key:
                tree_name = tree
                break
    print(f"Data tree found: {tree_name}")
    data_hdl = TreeHandler(input_file_name_data, tree_name, folder_name='DF*')
    mc_hdl = TreeHandler(input_file_name_mc, "O2mchypcands", folder_name='DF*')
    utils.correct_and_convert_df(data_hdl, calibrate_he3_pt=False, isMC=False)
    utils.correct_and_convert_df(mc_hdl, calibrate_he3_pt=False, isMC=True)
    
    ## mc reweightting sprctrum 
    spectra_file = ROOT.TFile.Open('utils/H3L_BWFit.root')
    ###string infomation
    inv_mass_string = '#it{M}_{^{3}He+#pi^{-}}' if is_matter == 'matter' else '#it{M}_{^{3}#bar{He}+#pi^{+}}' if is_matter == 'antimatter' else '#it{M}_{^{3}He+#pi}' if is_matter == 'both' else None
    decay_string = '{}^{3}_{#Lambda}H #rightarrow ^{3}He+#pi^{-}' if is_matter == 'matter' else '{}^{3}_{#bar{#Lambda}}#bar{H} #rightarrow ^{3}#bar{He}+#pi^{+}' if is_matter == 'antimatter' else '{}^{3}_{#Lambda}H #rightarrow ^{3}He+#pi' if is_matter == 'both' else None
    ###Histograms for ct extracted
    ct_hist_sweights = ROOT.TH1D('ct_hist_sweights', 'sWeights;#it{P_{t}};#it{c}_{t} (cm)', len(pt_bins)-1, np.array(pt_bins ,dtype=np.float64))
    ct_hist_cows = ROOT.TH1D('ct_hist_cows', 'COWs;#it{P_{t}};#it{c}_{t} (cm)', len(pt_bins)-1, np.array(pt_bins ,dtype=np.float64))
    ## for each pt bins
    for i in range(len(pt_bins)-1):
        pt_min = pt_bins[i]
        pt_max = pt_bins[i+1]
        bin_sel_data = f"fPt > {pt_min} & fPt < {pt_max}" + " and " + f"fMassH3L > {mass_range[0]} & fMassH3L < {mass_range[1]}"
        bin_sel_mc = f"fAbsGenPt > {pt_min} & fAbsGenPt < {pt_max}"
        if is_matter == 'matter':
            bin_sel_data += ' and fIsMatter == True'
            bin_sel_mc += 'and fGenPt>0'
        elif is_matter == 'antimatter':
            bin_sel_data += 'and fIsMatter == False'
        bin_data_hdl = data_hdl.apply_preselections(bin_sel_data, inplace = False)
        bin_mc_hdl = mc_hdl.apply_preselections(bin_sel_mc, inplace = False)
        ### reweight mc spectrum 
        H3l_spectrum = spectra_file.Get(f'BlastWave_H3L_0_10') # use 0-10 for calculation since this analysis is independent of centrality
        H3l_spectrum.SetRange(pt_min, pt_max)
        df_bin_mc = bin_mc_hdl.get_data_frame()
        utils.reweight_pt_spectrum(df_bin_mc, 'fAbsGenPt', H3l_spectrum)
        df_bin_mc = df_bin_mc.query("rej == 1")
        bin_mc_hdl.set_data_frame(df_bin_mc)
        ### end of reweighting
        if pre_selection_method == 'BDT':
            print("**Using Mechine Learning for H3l pre-selection**")
            print(f'** Applying BDT to data for pt: {pt_min}-{pt_max}**')
        elif pre_selection_method == 'topology':
            print("**Using Topology Cuts for H3l pre-selection**")
            print(f'** Applying Cuts to data for pt: {pt_min}-{pt_max}**')
            topology_cuts = utils.convert_sel_to_string(selection[i])
            bin_data_hdl.apply_preselections(topology_cuts, inplace = True)
        ###Get data and mc related var arrays
        mass = ROOT.RooRealVar('m', inv_mass_string, mass_range[0], mass_range[1], 'GeV/c^{2}')
        ct = ROOT.RooRealVar('ct', 'deacy length', ct_range[0], ct_range[1], 'cm')
        mass_array = np.array(bin_data_hdl['fMassH3L'].values, dtype=np.float64)
        mass_array_mc = np.array(bin_mc_hdl['fMassH3L'].values, dtype=np.float64)
        ct_array = np.array(bin_data_hdl['fCt'].values, dtype=np.float64)
        # mass_array = mass_array[np.logical_and(mass_array > mass_range[0], mass_array < mass_range[1])]
        # mass_array_mc = mass_array_mc[np.logical_and(mass_array_mc > mass_range[0], mass_array_mc < mass_range[1])]
        mass_dataset = utils.ndarray2roo(mass_array, mass, "mass_data")
        mass_dataset_mc = utils.ndarray2roo(mass_array_mc, mass, "mass_mc")
        ct_dataset = utils.ndarray2roo(ct_array, ct, "ct_data")
        ###ROOFit process
        ##ROOT vars
        mu = ROOT.RooRealVar('mu', 'hypernucl mass', 2.991 , 2.985, 2.992, 'GeV/c^{2}')
        sigma = ROOT.RooRealVar('sigma', 'sigma', 0.002, 0.001, 0.003, 'GeV/c^{2}')
        al = ROOT.RooRealVar('al', 'dscb alpha left', 0.7, 5.)
        ar = ROOT.RooRealVar('ar', 'dscb alpha right', 0.7, 5.)
        nl = ROOT.RooRealVar('nl', 'dscb n left', 0., 5.)
        nr = ROOT.RooRealVar('nr', 'dscb n right', 0., 5.)
        c0 = ROOT.RooRealVar('c0', 'constant c0', -1., 1)
        c1 = ROOT.RooRealVar('c1', 'constant c1', -1., 1)
        ## signal pdf bkg pdf and total pdf
        if signal_fit_func == 'dscb':
            signal_pdf = ROOT.RooCrystalBall('signal_pdf', 'signal pdf', mass, mu, sigma, al, ar, nl, nr)
        elif signal_fit_func == 'gauss':
            signal_pdf = ROOT.RooGaussian('signal_pdf', 'signal pdf', mass, mu, sigma)
        else:
            raise ValueError(f'Invalid signal fit function. Expected one of: dscb, gaus')
        if bkg_fit_func == 'pol1':
            bkg_pdf = ROOT.RooChebychev('bkg_pdf', 'pol1 bkg', mass, ROOT.RooArgList(c0))
        elif bkg_fit_func == 'pol2':
            bkg_pdf = ROOT.RooChebychev('bkg_pdf', 'pol2 bkg', mass, ROOT.RooArgList(c0, c1))
        elif bkg_fit_func == 'expo':
            bkg_pdf = ROOT.RooExponential('bkg_pdf', 'expo bkg', mass, c0)
        else:
            raise ValueError(f'Invalid background fit function. Expected one of: pol1, pol2, expo')
        n_signal = ROOT.RooRealVar('n_signal', 'n_signal', 0., 1e4)#1e4
        n_bkg = ROOT.RooRealVar('n_bkg', 'n_bkg', 0., 1e6)#1e6
        total_pdf = ROOT.RooAddPdf('total_pdf', 'signal + bkg extended', ROOT.RooArgList(signal_pdf, bkg_pdf), ROOT.RooArgList(n_signal, n_bkg))
        ### fix the mc dscb parameters (al, ar, nl, nr)
        if signal_fit_func == 'dscb':
            fit_res_mc = signal_pdf.fitTo(mass_dataset_mc, ROOT.RooFit.Range(2.97, 3.01), ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.NumCPU(8))
            al.setConstant()
            ar.setConstant()
            nl.setConstant()
            nr.setConstant()
            print("MC dscb parameters fixed to:", al.getVal(), ar.getVal(), nl.getVal(), nr.getVal())
            sigma.setRange(sigma_range_mc_to_data[i][0]*sigma.getVal(), sigma_range_mc_to_data[i][1]*sigma.getVal())
            mass_frame_mc = mass.frame(nbins_plot_mc)
            mass_frame_mc.SetName(f"mc_frame_fit_pt_{pt_min}_{pt_max}")
            mass_frame_mc.SetTitle(f"MC mass fit for pt: {pt_min}-{pt_max}")
            mass_dataset_mc.plotOn(mass_frame_mc, ROOT.RooFit.Name("mc"), ROOT.RooFit.MarkerColor(ROOT.kBlack), ROOT.RooFit.MarkerStyle(20), ROOT.RooFit.MarkerSize(1.5), ROOT.RooFit.DrawOption('P'))
            signal_pdf.plotOn(mass_frame_mc, ROOT.RooFit.Name("mc_signal"), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.LineWidth(2), ROOT.RooFit.DrawOption('L'))
            fit_param = ROOT.TPaveText(0.6, 0.43, 0.9, 0.85, 'NDC')
            fit_param.SetBorderSize(0)
            fit_param.SetFillStyle(0)
            fit_param.SetTextAlign(12)
            fit_param.AddText(r'#mu = ' + f'{mu.getVal()*1e3:.2f} #pm {mu.getError()*1e3:.2f}' + ' MeV/#it{c}^{2}')
            fit_param.AddText(r'#sigma = ' + f'{sigma.getVal()*1e3:.2f} #pm {sigma.getError()*1e3:.2f}' + ' MeV/#it{c}^{2}')
            fit_param.AddText(r'alpha_{L} = ' + f'{al.getVal():.2f} #pm {al.getError():.2f}')
            fit_param.AddText(r'alpha_{R} = ' + f'{ar.getVal():.2f} #pm {ar.getError():.2f}')
            fit_param.AddText(r'n_{L} = ' + f'{nl.getVal():.2f} #pm {nl.getError():.2f}')
            fit_param.AddText(r'n_{R} = ' + f'{nr.getVal():.2f} #pm {nr.getError():.2f}')
            mass_frame_mc.addObject(fit_param)
            chi2_mc = mass_frame_mc.chiSquare('mc_signal', 'mc')
            ndf_mc = nbins_plot_mc - fit_res_mc.floatParsFinal().getSize()
            fit_param.AddText('#chi^{2} / NDF = ' + f'{chi2_mc:.3f} (NDF: {ndf_mc})')
            stddir.cd()
            mass_frame_mc.Write()
            if save_fit_results:
                x = np.linspace(mass_range[0], mass_range[1], 1000)
                signal_pdf_mc = convert_rf_pdf(signal_pdf, mass)
                plt.hist(mass_array_mc, bins=nbins_plot_mc, range=(mass_range[0], mass_range[1]), density=True, alpha=0.7, color='blue', edgecolor='black',label='MC mass(Normalized)')
                plt.plot(x, signal_pdf_mc(x), "r--", label=f'MC fit pdf({signal_fit_func})')
                plt.xlabel(inv_mass_string)
                plt.ylabel('Normalized counts')
                plt.legend()
                plt.title(f"MC mass fit for pt: {pt_min}-{pt_max}")
                plt.tight_layout()
                #plt.show()
                plt.savefig(f"{output_dir_name}/MC_mass_fit_pt_{pt_min}_{pt_max}.pdf")
                plt.close()
        ### fit the data 
        fit_res_data = total_pdf.fitTo(mass_dataset,ROOT.RooFit.Extended(True), ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-1),ROOT.RooFit.NumCPU(8))
        mass_frame_data = mass.frame(nbins_plot_data)
        mass_frame_data.SetName(f"data_frame_fit_pt_{pt_min}_{pt_max}")
        mass_frame_data.SetTitle(f"Data mass fit for pt: {pt_min}-{pt_max}")
        mass_dataset.plotOn(mass_frame_data, ROOT.RooFit.Name("data"), ROOT.RooFit.MarkerColor(ROOT.kBlack), ROOT.RooFit.MarkerStyle(20), ROOT.RooFit.MarkerSize(1.5), ROOT.RooFit.DrawOption('P'))
        total_pdf.plotOn(mass_frame_data, ROOT.RooFit.Name("total"), ROOT.RooFit.LineColor(ROOT.kAzure + 2), ROOT.RooFit.LineWidth(2), ROOT.RooFit.DrawOption('L'))
        total_pdf.plotOn(mass_frame_data, ROOT.RooFit.Name("signal"), ROOT.RooFit.Components("signal_pdf"), ROOT.RooFit.LineColor(ROOT.kPink - 2), ROOT.RooFit.LineWidth(2), ROOT.RooFit.DrawOption('L'))
        total_pdf.plotOn(mass_frame_data, ROOT.RooFit.Name("background"), ROOT.RooFit.Components("bkg_pdf"), ROOT.RooFit.LineColor(kOrangeC), ROOT.RooFit.LineWidth(2), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.DrawOption('L'))
        ##Get the parameters
        fit_pars_data = total_pdf.getParameters(mass_dataset)
        signal_counts = fit_pars_data.find('n_signal').getVal()
        signal_counts_error = fit_pars_data.find('n_signal').getError()
        background_counts = fit_pars_data.find('n_bkg').getVal()
        background_counts_error = fit_pars_data.find('n_bkg').getError()
        mu_value = fit_pars_data.find('mu').getVal()
        mu_error = fit_pars_data.find('mu').getError()
        sigma_value = fit_pars_data.find('sigma').getVal()
        sigma_error = fit_pars_data.find('sigma').getError()
        al_value = fit_pars_data.find('al').getVal()
        al_error = fit_pars_data.find('al').getError()
        ar_value = fit_pars_data.find('ar').getVal()
        ar_error = fit_pars_data.find('ar').getError()
        nl_value = fit_pars_data.find('nl').getVal()
        nl_error = fit_pars_data.find('nl').getError()
        nr_value = fit_pars_data.find('nr').getVal()
        nr_error = fit_pars_data.find('nr').getError()
        c0_value = fit_pars_data.find('c0').getVal()
        c0_error = fit_pars_data.find('c0').getError()
        c1_value = fit_pars_data.find('c1').getVal()
        c1_error = fit_pars_data.find('c1').getError()
        chi2_data = mass_frame_data.chiSquare('total', 'data')
        ndf_data = nbins_plot_data - fit_res_data.floatParsFinal().getSize()
        ##signal and bkg yields within 3 sigma
        mass.setRange('signal', mu_value - 3*sigma_value, mu_value + 3*sigma_value)
        signal_int = signal_pdf.createIntegral(ROOT.RooArgSet(mass), ROOT.RooArgSet(mass), 'signal')
        signal_yield = signal_int.getVal() * signal_counts
        signal_yield_error = signal_int.getVal() * signal_counts_error
        mass.setRange('bkg', mu_value - 3*sigma_value, mu_value + 3*sigma_value)
        bkg_int = bkg_pdf.createIntegral(ROOT.RooArgSet(mass), ROOT.RooArgSet(mass), 'bkg')
        bkg_yield = bkg_int.getVal() * background_counts
        bkg_yield_error = bkg_int.getVal() * background_counts_error
        significance = signal_yield / np.sqrt(signal_yield + bkg_yield)
        significance_error = utils.significance_error(signal_yield, bkg_yield, signal_yield_error, bkg_yield_error)
        s_b_ratio_err = np.sqrt((signal_yield_error/signal_yield)**2 + (bkg_yield_error/bkg_yield)**2)*signal_yield/bkg_yield
        pave_text = ROOT.TPaveText(0.632, 0.4, 0.98, 0.9, 'NDC')
        pave_text.SetTextFont(42)
        pave_text.SetBorderSize(0)
        pave_text.SetTextAlign(11)
        pave_text.SetFillStyle(0)
        pave_text.AddText(r'#mu = ' + f'{mu_value*1e3:.2f} #pm {mu_error*1e3:.2f}' + ' MeV/#it{c}^{2}')
        pave_text.AddText(r'#sigma = ' + f'{sigma_value*1e3:.2f} #pm {sigma_error*1e3:.2f}' + ' MeV/#it{c}^{2}')
        pave_text.AddText(r'a_{L} = ' + f'{al_value:.2f} #pm {al_error:.2f}')
        pave_text.AddText(r'a_{R} = ' + f'{ar_value:.2f} #pm {ar_error:.2f}')
        pave_text.AddText(r'n_{L} = ' + f'{nl_value:.2f} #pm {nl_error:.2f}')
        pave_text.AddText(r'n_{R} = ' + f'{nr_value:.2f} #pm {nr_error:.2f}')
        pave_text.AddText(r'c_{0} = ' + f'{c0_value:.2f} #pm {c0_error:.2f}')
        pave_text.AddText(r'c_{1} = ' + f'{c1_value:.2f} #pm {c1_error:.2f}')
        pave_text.AddText(f"Signal counts: {signal_counts:.0f} #pm {signal_counts_error:.0f}")
        pave_text.AddText(f"Background counts: {background_counts:.0f} #pm {background_counts_error:.0f}")
        pave_text.AddText(f"Signal fraction: {signal_yield/(signal_yield + bkg_yield):.2f}")
        pave_text.AddText("S/#sqrt{S+B} (3 #sigma): " + f"{significance:.2f} #pm {significance_error:.2f}")
        pave_text.AddText(f"S/B ratio: {signal_yield/bkg_yield:.2f} #pm {s_b_ratio_err:.2f}")
        pave_text.AddText(f"#chi^2/NDF: {chi2_data:.2f}/{ndf_data:.0f}")
        pinfo_alice = ROOT.TPaveText(0.1, 0.65, 0.5, 0.85, 'NDC')
        pinfo_alice.SetTextFont(42)
        pinfo_alice.SetBorderSize(0)
        pinfo_alice.SetTextAlign(11)
        pinfo_alice.SetFillStyle(0)
        pinfo_alice.AddText("Run3, PbPb @ #sqrt{#it{s_{NN}}} = 5.36 TeV")
        pinfo_alice.AddText(decay_string)
        mass_frame_data.addObject(pave_text)
        mass_frame_data.addObject(pinfo_alice)
        stddir.cd()
        mass_frame_data.Write()
        ###ROOFit fuction to python function
        # mass_pdf = ROOT.RooRealVar('m_pdf', inv_mass_string, mass_range[0], mass_range[1], 'GeV/c^{2}')
        total_pdf_data = convert_rf_pdf(total_pdf, mass) # add npoints=100 for saving running time
        signal_pdf_data = convert_rf_pdf(signal_pdf, mass)
        bkg_pdf_data = convert_rf_pdf(bkg_pdf, mass)
        def signal_pdf_data_py(x):
            return double_crystalball(x, mu_value, sigma_value, al_value, ar_value, nl_value, nr_value, mass_range[0], mass_range[1])
        def bkg_pdf_data_py(x):
            return chebyshev_pdf(x, 1, c0_value, c1_value,  mass_range[0], mass_range[1])
        if save_fit_results:
            x = np.linspace(mass_range[0], mass_range[1], 1000)
            signal_fraction = signal_counts / (signal_counts + background_counts)
            bkg_fraction = background_counts / (signal_counts + background_counts)
            plt.hist(mass_array, bins=nbins_plot_data, range=(mass_range[0], mass_range[1]), density=True, alpha=0.7, color='blue', edgecolor='black',label='Data mass(Normalized)')
            plt.plot(x, total_pdf_data(x), "b-", label=f'Data fit pdf({signal_fit_func} + {bkg_fit_func})')
            plt.plot(x, signal_fraction * signal_pdf_data(x), "r-", label=f'Signal pdf({signal_fit_func})')
            plt.plot(x, bkg_fraction * bkg_pdf_data(x), "g-", label=f'Background pdf({bkg_fit_func})')
            plt.plot(x, signal_fraction * signal_pdf_data_py(x), "k--", label=f'Signal pdf({signal_fit_func}) python')
            plt.plot(x, bkg_fraction * bkg_pdf_data_py(x), "y--", label=f'Background pdf({bkg_fit_func}) python')
            plt.xlabel(inv_mass_string)
            plt.ylabel('Normalized counts')
            plt.legend()
            plt.title(f"Data mass fit for pt: {pt_min}-{pt_max}")
            plt.tight_layout()
            #plt.show()
            plt.savefig(f"{output_dir_name}/Data_mass_fit_pt_{pt_min}_{pt_max}.pdf")
            plt.close()
        ### def of the ct pdf
        def ct_pdf_data_py(ct, tau):
            return normalized_expon(ct, tau, (ct_range[0], ct_range[1]))
        ### construct the sWeights
        if method == "sweights" or method == "both":
            sweight = SWeight(
                mass_array,
                [signal_pdf_data_py, bkg_pdf_data_py],
                [signal_counts, background_counts],
                [(mass_range[0], mass_range[1])],
                method="summation",
                compnames=("sig", "bkg"),
                verbose=True,
                checks=True,
            ) ## run quicker verbose = False and checks = False
            ### plot the weights distributions
            x = np.linspace(mass_range[0], mass_range[1], 500)
            swp = sweight.get_weight(0, x)
            bwp = sweight.get_weight(1, x)
            plt.figure()
            plt.plot(x, swp, "C0--", label="signal")
            plt.plot(x, bwp, "C1:", label="background")
            plt.plot(x, swp + bwp, "k-", label="sum")
            plt.xlabel("h3lmass")
            plt.ylabel("weight")
            plt.legend()
            plt.title("sWeights")
            plt.tight_layout()
            plt.savefig(f"{output_dir_name}/weight_sWeights_pt_{pt_min}_{pt_max}.pdf")
            plt.close()
            ### Fit weighted ct distribution
            sws = sweight(mass_array)
            tmi_sw = Minuit(
                make_weighted_negative_log_likelihood(ct_array, sws, ct_pdf_data_py),
                tau=8,
            )
            tmi_sw.limits["tau"] = (0, 10)
            tmi_sw.migrad()
            tmi_sw.hesse()
            ## corrections
            ncov = approx_cov_correct(
                ct_pdf_data_py, ct_array, sws, tmi_sw.values, tmi_sw.covariance, verbose=False
            )
            # second order correction
            hs = ct_pdf_data_py
            ws = sweight
            W = sweight.Wkl
            # these derivatives can be done numerically but for the sweights / COW case
            # it's straightfoward to compute them
            ws = lambda Wss, Wsb, Wbb, gs, gb: (Wbb * gs - Wsb * gb) / (
                (Wbb - Wsb) * gs + (Wss - Wsb) * gb
            )
            dws_Wss = (
                lambda Wss, Wsb, Wbb, gs, gb: gb
                * (Wsb * gb - Wbb * gs)
                / (-Wss * gb + Wsb * gs + Wsb * gb - Wbb * gs) ** 2
            )
            dws_Wsb = (
                lambda Wss, Wsb, Wbb, gs, gb: (Wbb * gs**2 - Wss * gb**2)
                / (Wss * gb - Wsb * gs - Wsb * gb + Wbb * gs) ** 2
            )
            dws_Wbb = (
                lambda Wss, Wsb, Wbb, gs, gb: gs
                * (Wss * gb - Wsb * gs)
                / (-Wss * gb + Wsb * gs + Wsb * gb - Wbb * gs) ** 2
            )
            tcov = cov_correct(
                hs,
                [signal_pdf_data_py, bkg_pdf_data_py],
                mass_array,
                ct_array,
                sws,
                [signal_counts, background_counts],
                tmi_sw.values,
                tmi_sw.covariance,
                [dws_Wss, dws_Wsb, dws_Wbb],
                [W[0, 0], W[0, 1], W[1, 1]],
                verbose=False,
            )
            tau_value = tmi_sw.values[0]
            tau_error = tmi_sw.errors[0]
            tau_corrected = tcov[0, 0] ** 0.5
            print(
                f"sWeights: "
                f"naive {tau_value:.3f} +/- {tau_error:.3f}, "
                f"corrected {tau_value:.3f} +/- {tau_corrected:.3f}"
            )
            ct_hist_sweights.SetBinContent(i + 1, tau_value)
            ct_hist_sweights.SetBinError(i + 1, tau_corrected)
            ### plot the ct distribution
            bins = 50
            x_ct = np.linspace(ct_range[0], ct_range[1], 500)
            plot_binned(ct_array, bins=bins, range=(ct_range[0], ct_range[1]), color='k', label='Data ct(signal + bkg)')
            plot_binned(ct_array, bins=bins, range=(ct_range[0], ct_range[1]), weights=sws, color='C0', label='ct sWeights extracted')
            tnorm = np.sum(sws) * (ct_range[1] - ct_range[0]) / bins
            plt.plot(x_ct, tnorm * ct_pdf_data_py(x_ct, tau_value), "C0--", label="ct distribution (sWeights)")
            plt.xlabel("ct (cm)")
            plt.ylabel("Events")
            plt.legend()
            plt.title(f"ct distribution (sWeights) pt {pt_min} - {pt_max}")
            plt.tight_layout()
            plt.savefig(f"{output_dir_name}/ct_sWeights_pt_{pt_min}_{pt_max}.pdf")
            plt.close()       
        ### construct the COWs
        if method == "cow" or method == "both":
            # unity: for simple variance function I(m)
            Im = None
            
            # sweight equivalent:
            # def Im(m):
            #     return m_density(m, *mi.values) / (mi.values['s'] + mi.values['b'] )
            
            # histogram:
            # Im = np.histogram(mass_array, range=(mass_range[0], mass_range[1]))
            
            # make the cow
            cow = Cow((mass_range[0], mass_range[1]), signal_pdf_data_py, bkg_pdf_data_py, Im, verbose=True)
            ### plot the weights distributions
            x = np.linspace(mass_range[0], mass_range[1], 500)
            swp = cow.get_weight(0, x)
            bwp = cow.get_weight(1, x)
            plt.figure()
            plt.plot(x, swp, "C0--", label="signal")
            plt.plot(x, bwp, "C1:", label="background")
            plt.plot(x, swp + bwp, "k-", label="sum")
            plt.xlabel("h3lmass")
            plt.ylabel("weight")
            plt.legend()
            plt.title("COWs")
            plt.tight_layout()
            plt.savefig(f"{output_dir_name}/weight_COWs_pt_{pt_min}_{pt_max}.pdf")
            plt.close()
            ### Fit weighted ct distribution
            scow = cow(mass_array)
            tmi_cow = Minuit(
                make_weighted_negative_log_likelihood(ct_array, scow, ct_pdf_data_py),
                tau=8,
            )
            tmi_cow.limits["tau"] = (0, 10)
            tmi_cow.migrad()
            tmi_cow.hesse()
            ## corrections
            ncov = approx_cov_correct(
                ct_pdf_data_py, ct_array, scow, tmi_cow.values, tmi_cow.covariance, verbose=False
            )
            # second order correction
            hs = ct_pdf_data_py
            ws = cow
            W = cow.Wkl
            # these derivatives can be done numerically but for the sweights / COW case
            # it's straightfoward to compute them
            ws = lambda Wss, Wsb, Wbb, gs, gb: (Wbb * gs - Wsb * gb) / (
                (Wbb - Wsb) * gs + (Wss - Wsb) * gb
            )
            dws_Wss = (
                lambda Wss, Wsb, Wbb, gs, gb: gb
                * (Wsb * gb - Wbb * gs)
                / (-Wss * gb + Wsb * gs + Wsb * gb - Wbb * gs) ** 2
            )
            dws_Wsb = (
                lambda Wss, Wsb, Wbb, gs, gb: (Wbb * gs**2 - Wss * gb**2)
                / (Wss * gb - Wsb * gs - Wsb * gb + Wbb * gs) ** 2
            )
            dws_Wbb = (
                lambda Wss, Wsb, Wbb, gs, gb: gs
                * (Wss * gb - Wsb * gs)
                / (-Wss * gb + Wsb * gs + Wsb * gb - Wbb * gs) ** 2
            )
            tcov = cov_correct(
                hs,
                [signal_pdf_data_py, bkg_pdf_data_py],
                mass_array,
                ct_array,
                scow,
                [signal_counts, background_counts],
                tmi_cow.values,
                tmi_cow.covariance,
                [dws_Wss, dws_Wsb, dws_Wbb],
                [W[0, 0], W[0, 1], W[1, 1]],
                verbose=False,
            )
            tau_value = tmi_cow.values[0]
            tau_error = tmi_cow.errors[0]
            tau_corrected = tcov[0, 0] ** 0.5
            print(
                f"COWs: "
                f"naive {tau_value:.3f} +/- {tau_error:.3f}, "
                f"corrected {tau_value:.3f} +/- {tau_corrected:.3f}"
            )
            ct_hist_cows.SetBinContent(i + 1, tau_value)
            ct_hist_cows.SetBinError(i + 1, tau_corrected)
            ### plot the ct distribution
            bins = 50
            x_ct = np.linspace(ct_range[0], ct_range[1], 500)
            plot_binned(ct_array, bins=bins, range=(ct_range[0], ct_range[1]), color='k', label='Data ct(signal + bkg)')
            plot_binned(ct_array, bins=bins, range=(ct_range[0], ct_range[1]), weights=scow, color='C0', label='ct COWs extracted')
            tnorm = np.sum(scow) * (ct_range[1] - ct_range[0]) / bins
            plt.plot(x_ct, tnorm * ct_pdf_data_py(x_ct, tau_value), "C0--", label="ct distribution (COWs)")
            plt.xlabel("ct (cm)")
            plt.ylabel("Events")
            plt.legend()
            plt.title(f"ct distribution (COWs) pt {pt_min} - {pt_max}")
            plt.tight_layout()
            plt.savefig(f"{output_dir_name}/ct_COWs_pt_{pt_min}_{pt_max}.pdf")
            plt.close()
    stddir.cd()
    ct_hist_sweights.Write()
    ct_hist_cows.Write()
    stddir.Close()
    spectra_file.Close()













        

        
            

             


        



