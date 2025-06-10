import ROOT
import numpy as np
import uproot
import pandas as pd
import argparse
import os
import yaml
import matplotlib.pyplot as plt
from math import exp, sqrt, pi, gamma
from pathlib import Path

from signal_extraction import SignalExtraction

from scipy.stats import norm, expon, crystalball
from scipy.special import erf, eval_chebyt
from scipy.integrate import quad
from scipy.interpolate import interp1d
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

def gauss_pol3(x, A, mu, sigma, B, C, D, E):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + B * x**3 + C * x**2 + D * x + E
    
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

def continuous_efficiency_corrected_expon(
    ct_bins: np.ndarray,
    efficiency: np.ndarray,
    tau: float,
    ct_range: tuple = None,
    kind: str = "linear",  # 插值方法，可选 'linear', 'nearest', 'cubic' 等
) -> callable:
    """
    生成一个连续修正后的指数 PDF 函数（适用于任意 ct）
    
    参数:
        ct_bins     : bin 边界数组（shape=(n_bins+1,)）
        efficiency  : 每个 bin 的效率修正（shape=(n_bins,)）
        tau         : 平均寿命
        ct_range    : 归一化范围 (ct_min, ct_max)，若为None则使用 ct_bins 的范围
        kind        : 插值方法（默认 'linear'）
    
    返回:
        一个可调用函数 corrected_pdf_func(ct)，返回修正后的 PDF 值
    """
    # 检查输入
    assert len(ct_bins) == len(efficiency) + 1, "efficiency 的长度必须比 ct_bins 少 1"
    
    # 计算 bin 中心（用于插值）
    bin_centers = 0.5 * (ct_bins[:-1] + ct_bins[1:])
    
    # 插值 efficiency（生成连续函数）
    efficiency_interp = interp1d(
        bin_centers,
        efficiency,
        kind=kind,
        bounds_error=False,  # 允许 ct 超出范围
        fill_value="extrapolate",  # 外推
    )
    
    # 定义修正后的 PDF 函数
    def corrected_pdf_func(ct):
        # 计算原始 PDF
        original_pdf = normalized_expon(ct, tau, ct_range)
        
        # 应用插值后的效率修正
        corrected_pdf = original_pdf * efficiency_interp(ct)
        
        return corrected_pdf
    
    # 重新归一化（使积分=1）
    if ct_range is None:
        ct_range = (ct_bins[0], ct_bins[-1])
    
    # 计算归一化因子（数值积分）
    norm_factor, _ = quad(corrected_pdf_func, ct_range[0], ct_range[1], limit=100)
    
    # 返回归一化后的函数
    def normalized_corrected_pdf(ct):
        return corrected_pdf_func(ct) / norm_factor
    
    return normalized_corrected_pdf

def continuous_efficiency_corrected_expon_with_ct(
    ct: float,
    ct_bins: np.ndarray,
    efficiency: np.ndarray,
    tau: float,
    ct_range: tuple = None,
    kind: str = "linear",
) -> float:
    pdf_func = continuous_efficiency_corrected_expon(ct_bins, efficiency, tau, ct_range, kind)
    return pdf_func(ct)


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
    input_AnalysisResults_file_path = config['input_AnalysisResults_file_path']
    output_dir_name = config['output_dir_name']
    pt_bins = config['pt_bins']
    is_matter = config['is_matter']
    method = config['method']
    pre_selection_method = config['pre_selection_method']
    signal_fit_func = config['signal_fit_func']
    bkg_fit_func = config['bkg_fit_func']
    mass_range = config['mass_range']
    ct_range = config['ct_range']
    nctbins_acceptance = config['nctbins_acceptance']
    sigma_range_mc_to_data = config['sigma_range_mc_to_data']
    nbins_plot_mc = config['nbins_plot_mc']
    nbins_plot_data = config['nbins_plot_data']
    save_fit_results = config['save_fit_results']
    selection = config['selection']
    #BDT config
    new_training = config['new_training']
    opean_NSigmaH3_mc_shift = config['opean_NSigmaH3_mc_shift']
    opean_NSigmaH3_data_shift = config['opean_NSigmaH3_data_shift']
    training_variables = config['training_variables']
    training_preselections = config['training_preselections']
    random_state = config['random_state']
    test_set_size = config['test_set_size']
    hyperparams = config['hyperparams']
    bkg_fraction_max = config['bkg_fraction_max']
    
    print('**********************************')
    print('    Running lifetime_extraction.py')
    print('**********************************\n')
    print("----------------------------------")
    print("** Loading data and apply preselections **")
    
    output_dir_name += f'/{pre_selection_method}'
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
        bin_sel_data = f"fPt > {pt_min} & fPt < {pt_max}" # + " and " + f"fMassH3L > {mass_range[0]} & fMassH3L < {mass_range[1]}"
        bin_sel_mc = f"fAbsGenPt > {pt_min} & fAbsGenPt < {pt_max}"
        if is_matter == 'matter':
            bin_sel_data += ' and fIsMatter == True'
            bin_sel_mc += 'and fGenPt>0'
        elif is_matter == 'antimatter':
            bin_sel_data += 'and fIsMatter == False'
        bin_data_hdl = data_hdl.apply_preselections(bin_sel_data, inplace = False)
        bin_mc_hdl = mc_hdl.apply_preselections(bin_sel_mc, inplace = False)
        ### ct resulution
        df_bin_mc = bin_mc_hdl._full_data_frame
        df_bin_mc['Delta_Ct'] = df_bin_mc['fGenCt'] - df_bin_mc['fCt']
        bin_mc_hdl.set_data_frame(df_bin_mc)
        fCt_array = bin_mc_hdl['fCt'].values
        delta_ct_array = bin_mc_hdl['Delta_Ct'].values
        plt.figure(figsize=(10, 8))
        hb = plt.hexbin(
            fCt_array,
            delta_ct_array,
            gridsize=200,           # 六边形网格密度
            cmap='plasma',        # 颜色映射 'viridis', 'plasma', 'jet'
            mincnt=1              # 忽略空 bin
        )
        plt.colorbar(hb, label='Counts')
        plt.xlim(0, 40)
        plt.xlabel('fCt')
        plt.ylabel('fGenCt - fCt')
        plt.title(f'Hexbin: fGenCt - fCt vs fCt pt range: {pt_min}-{pt_max}')
        plt.grid(alpha=0.3)
        plt.savefig(f'{output_dir_name}/ct_resulution_{pt_min}_{pt_max}.pdf')
        plt.close()
        ### reweight mc spectrum 
        H3l_spectrum = spectra_file.Get(f'BlastWave_H3L_0_10') # use 0-10 for calculation since this analysis is independent of centrality
        H3l_spectrum.SetRange(pt_min, pt_max)
        df_bin_mc = bin_mc_hdl.get_data_frame()
        utils.reweight_pt_spectrum(df_bin_mc, 'fAbsGenPt', H3l_spectrum)
        df_bin_mc = df_bin_mc.query("rej == 1")
        bin_mc_hdl.set_data_frame(df_bin_mc)
        ### end of reweighting
        ### calculate the mc ct acceptance
        bin_mc_reco_hdl = bin_mc_hdl.apply_preselections('fIsReco == 1', inplace=False)
        bin_mc_hdl_gen_evsel = bin_mc_hdl.apply_preselections('fIsSurvEvSel==True', inplace=False)
        acceptance_pt_bin = len(bin_mc_reco_hdl) / len(bin_mc_hdl_gen_evsel)
        ct_bins = np.linspace(ct_range[i][0], ct_range[i][1], nctbins_acceptance + 1)
        acceptance = np.zeros(len(ct_bins) - 1)
        for i_ct in range(len(ct_bins) - 1):
            lower = ct_bins[i_ct]
            upper = ct_bins[i_ct+1]
            bin_sel_ct = f"fGenCt > {lower} & fGenCt < {upper}"
            eff_bin_mc_reco_hdl = bin_mc_reco_hdl.apply_preselections(bin_sel_ct, inplace=False)
            eff_bin_mc_gen_evsel_hdl = bin_mc_hdl_gen_evsel.apply_preselections(bin_sel_ct, inplace=False)
            bin_acceptance = len(eff_bin_mc_reco_hdl) / len(eff_bin_mc_gen_evsel_hdl) if len(eff_bin_mc_gen_evsel_hdl) != 0 else 0
            acceptance[i_ct] = bin_acceptance
        ctbin_centers = 0.5 * (ct_bins[:-1] + ct_bins[1:])
        acceptance_interp = interp1d(
        ctbin_centers,
        acceptance,
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate")
        x_interp = np.linspace(min(ctbin_centers), max(ctbin_centers), 100)  # 100 个点
        y_interp = acceptance_interp(x_interp)  # 插值结果
        plt.figure(figsize=(8, 5))
        plt.scatter(ctbin_centers, acceptance, color='red', label='Data ct acceptance', zorder=3)
        plt.plot(x_interp, y_interp, color='blue', label='Interpolated acceptance', linewidth=2)
        plt.xlabel("Bin Centers")
        plt.ylabel("Acceptance($\epsilon$)")
        plt.title(f"Original vs Interpolated ct acceptance pt: {pt_min}-{pt_max}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f"{output_dir_name}/ct_acceptance_{pt_min}_{pt_max}.pdf")
        plt.close()
        ### preselection
        #### ct selection
        bin_data_hdl.apply_preselections(f"fCt > {ct_range[i][0]} & fCt < {ct_range[i][1]}", inplace=True)
        if pre_selection_method == 'BDT':
            ###BDT method code here
            bdt_file = Path(f'{output_dir_name}/dataH_BDTapplied_{pt_min}_{pt_max}.parquet.gzip')
            wp_file = Path(f"{output_dir_name}/working_point_data_frame_{pt_min}_{pt_max}.csv")
            bin_data_hdl_pretrained = TreeHandler(bdt_file) if bdt_file.exists() else None
            df_working_point_pretrained = pd.read_csv(wp_file) if wp_file.exists() else None
            if bin_data_hdl_pretrained is not None and df_working_point_pretrained is not None and not new_training:
                bin_data_hdl = bin_data_hdl_pretrained
                df_working_point = df_working_point_pretrained
            else:
                print("**Using Mechine Learning for H3l pre-selection**")
                print(f'** Applying BDT to data for pt: {pt_min}-{pt_max}**')
                n_env = utils.getNEvents(input_AnalysisResults_file_path,False,0,99)
                exp_signal_bin = n_env * H3l_spectrum.Integral(pt_min,pt_max) * acceptance_pt_bin * 1 * 0.25 * 2 * 1 * 1
                df_bin_mc = bin_mc_hdl.get_data_frame()
                df_bin_mc_train = df_bin_mc.copy()
                bin_mc_hdl_train = TreeHandler()
                bin_mc_hdl_train.set_data_frame(df_bin_mc_train)
                df_bin_data = bin_data_hdl.get_data_frame()
                df_bin_data_train = df_bin_data.copy()
                bin_data_hdl_train = TreeHandler()
                bin_data_hdl_train.set_data_frame(df_bin_data_train)
                if training_preselections != '':
                    bin_mc_hdl_train.apply_preselections(training_preselections, inplace=True)
                    bin_data_hdl_train.apply_preselections(f"(fMassH3L<2.95 or fMassH3L>3.02) and {training_preselections}", inplace=True)
                else:
                    bin_data_hdl_train.apply_preselections(f"(fMassH3L<2.95 or fMassH3L>3.02)", inplace=True)
                if opean_NSigmaH3_mc_shift:
                    df_mcH = bin_mc_hdl_train.get_data_frame()
                    df_mcH['fNSigmaHe'] = df_mcH['fNSigmaHe'] - df_mcH['fNSigmaHe'].mean()
                    bin_mc_hdl_train.set_data_frame(df_mcH)
                if opean_NSigmaH3_data_shift:
                    df_dataH = bin_data_hdl_train.get_data_frame()
                    x_dataH = df_dataH['fNSigmaHe'].values
                    y_dataH = np.histogram(x_dataH, bins=100, density=True)[0]
                    x_dataH_hist = np.histogram(x_dataH, bins=100, density=True)[1][:-1]
                    init_guess = [max(y_dataH), 0, 1, 0, 0, 0, 0]
                    popt, _ = curve_fit(gauss_pol3, x_dataH_hist, y_dataH, p0=init_guess, bounds=([-np.inf, -1, 0, -np.inf, -np.inf, -np.inf, -np.inf], [np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf]))
                    A, mu, sigma, B, C, D, E = popt
                    plt.plot(x_dataH_hist, y_dataH, label='Background Data')
                    plt.plot(x_dataH_hist, gauss_pol3(x_dataH_hist, *popt), label='Gaussian+Poly Fit', linestyle='--')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"{output_dir_name}/Gauss_pol3_fit_data_df_{pt_min}_{pt_max}.pdf")
                    plt.close()
                    df_dataH['fNSigmaHe'] = df_dataH['fNSigmaHe'] - mu
                    bin_data_hdl_train.set_data_frame(df_dataH)
                utils.cut_elements_to_same_range(bin_mc_hdl_train,bin_data_hdl_train,['fDcaHe','fDcaPi'])
                print("***---------------Training Info------------------***")
                print("Origin MC events: ", len(bin_mc_hdl_train))
                print("Origin Data events: ", len(bin_data_hdl_train))
                if bkg_fraction_max != None:
                    if(len(bin_data_hdl_train) > bkg_fraction_max * len(bin_mc_hdl_train)):
                        bin_data_hdl_train.shuffle_data_frame(size=bkg_fraction_max*len(bin_mc_hdl_train), inplace=True, random_state=random_state)
                print("------------------------------------------------")
                print("Final MC events: ", len(bin_mc_hdl_train))
                print("Final Data events: ", len(bin_data_hdl_train))
                print("***---------------Training Info------------------***\n")
                ###Start training process
                train_test_data = au.train_test_generator([bin_mc_hdl_train, bin_data_hdl_train], [1,0], test_size=test_set_size, random_state=random_state)
                train_features = train_test_data[0]
                train_labels = train_test_data[1]
                test_features = train_test_data[2]
                test_labels = train_test_data[3]
                ####Plot distributions and correlations
                distr = pu.plot_distr([bin_mc_hdl_train, bin_data_hdl_train], training_variables + ["fMassH3L"], bins=100, labels=['Signal',"Background"],colors=["blue","red"], log=True, density=True, figsize=(18, 13), alpha=0.5, grid=False)
                plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
                plt.savefig(f"{output_dir_name}/features_distributions_{pt_min}_{pt_max}.pdf", bbox_inches='tight')
                plt.close()
                corr = pu.plot_corr([bin_mc_hdl_train,bin_data_hdl_train], training_variables + ["fMassH3L"], ['Signal',"Background"])
                corr[0].savefig(f"{output_dir_name}/correlations_mc_{pt_min}_{pt_max}.pdf", bbox_inches='tight')
                corr[1].savefig(f"{output_dir_name}/correlations_data_{pt_min}_{pt_max}.pdf", bbox_inches='tight')
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
                bdt_out_plot.savefig(f"{output_dir_name}/bdt_output_{pt_min}_{pt_max}.pdf")
                plt.close("all")
                feature_importance_plot = pu.plot_feature_imp(test_features, test_labels, model_hdl, ["Signal", "Background"])
                feature_importance_plot[0].savefig(f"{output_dir_name}/feature_importance_1_{pt_min}_{pt_max}.pdf")
                feature_importance_plot[1].savefig(f"{output_dir_name}/feature_importance_2_{pt_min}_{pt_max}.pdf") 
                plt.close("all")
                ####plot score distirbutions
                plt.hist(y_pred_test, bins=100, label='test set score_full sample', alpha=0.5, density=True)
                plt.xlabel("test BDT_score")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{output_dir_name}/testset_score_distribution_full_{pt_min}_{pt_max}.pdf")
                plt.close()
                plt.hist(y_pred_test[test_labels==0], bins=100, label='background', alpha=0.5, density=True)
                plt.hist(y_pred_test[test_labels==1], bins=100, label='signal', alpha=0.5, density=True)
                plt.xlabel("test BDT_score")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{output_dir_name}/testset_score_distribution_split_{pt_min}_{pt_max}.pdf")
                plt.close()
                ####plot roc
                roc_plot = pu.plot_roc_train_test(test_labels, y_pred_test, train_labels, y_pred_train)
                roc_plot.savefig(f"{output_dir_name}/roc_test_vs_train_{pt_min}_{pt_max}.pdf")
                plt.close("all")
                ##efficiencies vs score
                eff_arr = np.round(np.arange(0.5,0.99,0.005),3) # 1 means save with 1 digits after point
                score_eff_arr = au.score_from_efficiency_array(test_labels, y_pred_test, eff_arr)
                plt.plot(score_eff_arr, eff_arr, label='BDT_efficency_fixed_efficencyarray', marker='o')
                plt.xlabel('BDT Score')
                plt.ylabel('Efficiency')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{output_dir_name}/efficency_vs_model_output_{pt_min}_{pt_max}.pdf")
                plt.close()
                ### Applying the model to real data set
                print("** Applying BDT model to data ...**")
                bin_data_hdl.apply_model_handler(model_hdl, column_name="BDT_value")
                bin_data_hdl.print_summary()
                bin_data_hdl.write_df_to_parquet_files(f'dataH_BDTapplied_{pt_min}_{pt_max}', output_dir_name)
                ####plot the model BDT result distribution
                df = bin_data_hdl.get_data_frame()
                hist = df.hist(column='BDT_value', bins=100, range=(-15,15), figsize=(12, 7), grid=False, density=False, alpha=0.6, label="BDT_value")
                plt.xlabel('BDT score')
                plt.ylabel('Counts')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{output_dir_name}/BDT_value_distribution_applied_{pt_min}_{pt_max}.pdf")
                plt.close()
                ###calculate the significance x BDT efficiency
                raw_counts_arr = []
                raw_counts_arr_err = []
                significance_arr = []
                significance_arr_err = []
                s_b_ratio_arr = []
                s_b_ratio_arr_err = []
                chi2_arr = []
                output_file = ROOT.TFile.Open(f"{output_dir_name}/training_spectrum_{pt_min}_{pt_max}.root", 'recreate')
                output_dir_std = output_file.mkdir('std')
                for i_eff in range(len(score_eff_arr)):
                    score = score_eff_arr[i_eff]
                    efficency = eff_arr[i_eff]
                    input_bin_data_hdl = bin_data_hdl.apply_preselections(f"BDT_value > {score}",inplace = False)
                    score_label = [f'{pt_min} #leq #it{{p}}_{{T}} < {pt_max} GeV/#it{{c}}',f'BDT_Score > {score}', f'BDT Efficiency: {efficency:.3f}']
                    signal_extraction = SignalExtraction(input_bin_data_hdl, bin_mc_hdl)
                    signal_extraction.bkg_fit_func = "pol2"
                    signal_extraction.signal_fit_func = "dscb"
                    signal_extraction.n_bins_data = 40
                    signal_extraction.n_bins_mc = 80
                    signal_extraction.n_evts = n_env
                    signal_extraction.is_matter = is_matter
                    signal_extraction.performance = False
                    signal_extraction.is_3lh = True
                    signal_extraction.out_file = output_dir_std
                    signal_extraction.data_frame_fit_name = f'data_fit_BDT_score_{score}'
                    signal_extraction.mc_frame_fit_name = f'mc_fit_pt_{pt_min}_{pt_max}'
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
                exp_significance_array = [exp_signal_bin / np.sqrt(exp_signal_bin + (a / b)) for a, b in zip(raw_counts_arr, s_b_ratio_arr)]
                bkg_3sigma_array = [a/b for a,b in zip(raw_counts_arr, s_b_ratio_arr)]
                df_working_point = pd.DataFrame({
                    'exp_significance': exp_significance_array,
                    'raw_counts': raw_counts_arr,
                    'raw_counts_err': raw_counts_arr_err,
                    'bkg_3sigma': bkg_3sigma_array,
                    'BDT_efficiency': eff_arr,
                    'BDT_score': score_eff_arr,
                    'chi2': chi2_arr
                })
                df_working_point.to_csv(f"{output_dir_name}/working_point_data_frame_{pt_min}_{pt_max}.csv")
            df_working_point = df_working_point[df_working_point['raw_counts'] != 0]
            df_working_point = df_working_point.query('chi2 < 1.4 & raw_counts_err / raw_counts < 1')
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
            plt.title(f'BDT_efficiency * exp_significance vs BDT_score pt: {pt_min}-{pt_max}')
            plt.xlabel('BDT_score')
            plt.ylabel('BDT_efficiency * exp_significance')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir_name}/exp_significance_vs_BDT_score_{pt_min}_{pt_max}.pdf")
            plt.close()
            bin_data_hdl.apply_preselections(f"BDT_value > {max_score + 3}", inplace=True)
        ###*****The End of BDT********
        elif pre_selection_method == 'topology':
            print("**Using Topology Cuts for H3l pre-selection**")
            print(f'** Applying Cuts to data for pt: {pt_min}-{pt_max}**')
            topology_cuts = utils.convert_sel_to_string(selection[i])
            bin_data_hdl.apply_preselections(topology_cuts, inplace = True)
        ###Get data and mc related var arrays
        mass = ROOT.RooRealVar('m', inv_mass_string, mass_range[0], mass_range[1], 'GeV/c^{2}')
        ct = ROOT.RooRealVar('ct', 'deacy length', ct_range[i][0], ct_range[i][1], 'cm')
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
        low_3sigma = mu_value - 3*sigma_value
        up_3sigma = mu_value + 3*sigma_value
        ##signal and bkg yields within 3 sigma
        mass.setRange('signal', low_3sigma, up_3sigma)
        signal_int = signal_pdf.createIntegral(ROOT.RooArgSet(mass), ROOT.RooArgSet(mass), 'signal')
        signal_yield = signal_int.getVal() * signal_counts
        signal_yield_error = signal_int.getVal() * signal_counts_error
        mass.setRange('bkg', low_3sigma, up_3sigma)
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
            if pre_selection_method == 'topology':
                plt.axvline(x=low_3sigma, color='m', linestyle='--', linewidth=1, label="3σ boundary")
                plt.axvline(x=up_3sigma, color='m', linestyle='--', linewidth=1)
            plt.xlabel(inv_mass_string)
            plt.ylabel('Normalized counts')
            plt.legend()
            plt.title(f"Data mass fit for pt: {pt_min}-{pt_max}")
            plt.tight_layout()
            #plt.show()
            plt.savefig(f"{output_dir_name}/Data_mass_fit_pt_{pt_min}_{pt_max}.pdf")
            plt.close()
        ### def of the ct pdf
        # def ct_pdf_data_py(ct, tau):
        #     return continuous_efficiency_corrected_expon_with_ct(ct, ct_bins, efficiency, tau, ct_range = (ct_range[i][0], ct_range[i][1]), kind = 'cubic')
        def ct_pdf_data_py(ct, tau):
            return normalized_expon(ct, tau, ct_range = (ct_range[i][0], ct_range[i][1]))
        ### only use 3sigma range for topology selection
        if pre_selection_method == 'topology':
            # mass_max = up_3sigma
            # mass_min = low_3sigma
            mass_max = mass_range[1]
            mass_min = mass_range[0]
        else:
            mass_max = mass_range[1]
            mass_min = mass_range[0]
        bin_data_hdl.apply_preselections(f"fMassH3L > {mass_min} & fMassH3L < {mass_max}", inplace=True)
        mass_array = np.array(bin_data_hdl['fMassH3L'].values, dtype=np.float64)
        ct_array = np.array(bin_data_hdl['fCt'].values, dtype=np.float64)
        ### for ct acceptance correction
        ct_bin_indices = np.digitize(ct_array, bins=ct_bins) - 1
        ct_bin_indices = np.clip(ct_bin_indices, 0, len(acceptance)-1) ##deling with the data out of ct poltting range 
        ct_acceptance_array = acceptance[ct_bin_indices]
        if len(ct_acceptance_array) != len(ct_array):
            raise ValueError("Length of ct acceptance array does not match length of ct array")
        ### construct the sWeights
        if method == "sweights" or method == "both":
            sweight = SWeight(
                mass_array,
                [signal_pdf_data_py, bkg_pdf_data_py],
                [signal_counts, background_counts],
                [(mass_min, mass_max)],
                method="summation",
                compnames=("sig", "bkg"),
                verbose=True,
                checks=True,
            ) ## run quicker verbose = False and checks = False
            ### plot the weights distributions
            x = np.linspace(mass_min, mass_max, 500)
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
            sws_accptance = np.where(ct_acceptance_array != 0, sws / ct_acceptance_array, sws)
            tmi_sw = Minuit(
                make_weighted_negative_log_likelihood(ct_array, sws_accptance, ct_pdf_data_py),
                tau=8,
            )
            tmi_sw.limits["tau"] = (0, 10)
            tmi_sw.migrad()
            tmi_sw.hesse()
            ## corrections
            ncov = approx_cov_correct(
                ct_pdf_data_py, ct_array, sws_accptance, tmi_sw.values, tmi_sw.covariance, verbose=False
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
                sws_accptance,
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
            x_ct = np.linspace(ct_range[i][0], ct_range[i][1], 500)
            plot_binned(ct_array, bins=bins, range=(ct_range[i][0], ct_range[i][1]), color='k', label='Data ct(signal + bkg)')
            plot_binned(ct_array, bins=bins, range=(ct_range[i][0], ct_range[i][1]), weights=sws, color='C1', label='ct sWeights without accptance correction')
            plot_binned(ct_array, bins=bins, range=(ct_range[i][0], ct_range[i][1]), weights=sws_accptance, color='C0', label='ct sWeights accptance corrected')
            tnorm = np.sum(sws_accptance) * (ct_range[i][1] - ct_range[i][0]) / bins
            plt.plot(x_ct, tnorm * ct_pdf_data_py(x_ct, tau_value), "C0--", label="ct distribution (sWeights) accptance weighted")
            #plt.plot(x_ct, tnorm * normalized_expon(x_ct, tau_value, ct_range[i]), "C1:", label="ct distribution (sWeights) unweighted")
            plt.ylim(5e-1, 2e3)
            plt.xlabel("ct (cm)")
            plt.ylabel("Events")
            plt.yscale("log")
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
            # Im = np.histogram(mass_array, range=(mass_min, mass_max))
            
            # make the cow
            cow = Cow((mass_min, mass_max), signal_pdf_data_py, bkg_pdf_data_py, Im, verbose=True)
            ### plot the weights distributions
            x = np.linspace(mass_min, mass_max, 500)
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
            scow_accptance = np.where(ct_acceptance_array != 0, scow / ct_acceptance_array, scow)
            tmi_cow = Minuit(
                make_weighted_negative_log_likelihood(ct_array, scow_accptance, ct_pdf_data_py),
                tau=8,
            )
            tmi_cow.limits["tau"] = (0, 10)
            tmi_cow.migrad()
            tmi_cow.hesse()
            ## corrections
            ncov = approx_cov_correct(
                ct_pdf_data_py, ct_array, scow_accptance, tmi_cow.values, tmi_cow.covariance, verbose=False
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
                scow_accptance,
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
            x_ct = np.linspace(ct_range[i][0], ct_range[i][1], 500)
            plot_binned(ct_array, bins=bins, range=(ct_range[i][0], ct_range[i][1]), color='k', label='Data ct(signal + bkg)')
            plot_binned(ct_array, bins=bins, range=(ct_range[i][0], ct_range[i][1]), weights=scow, color='C1', label='ct COWs without accptance correction')
            plot_binned(ct_array, bins=bins, range=(ct_range[i][0], ct_range[i][1]), weights=scow_accptance, color='C0', label='ct COWs accptance corrected')
            tnorm = np.sum(scow_accptance) * (ct_range[i][1] - ct_range[i][0]) / bins
            plt.plot(x_ct, tnorm * ct_pdf_data_py(x_ct, tau_value), "C0--", label="ct distribution (COWs) accptance weighted")
            #plt.plot(x_ct, tnorm * normalized_expon(x_ct, tau_value, ct_range[i]), "C1:", label="ct distribution (COWs) unweighted")
            plt.ylim(5e-1, 2e3)
            plt.xlabel("ct (cm)")
            plt.ylabel("Events")
            plt.yscale("log")
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













        

        
            

             


        



