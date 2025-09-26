import yaml
import argparse
import uproot
import numpy as np
import os
import ROOT
ROOT.gStyle.SetOptStat(0)
import math
import array
import matplotlib.pyplot as plt

import sys
sys.path.append('../utils')
import utils as utils

# Color Reference
# kPink-2 kRed-10
# kAzure+2 kAzure-9
# kOrange+7 kOrange-9
# kViolet+2 kViolet-9
# kCyan+2 kCyan-8
# kGray+3 kGray

def find_graph_by_name(root_file, target_name):
    keys = root_file.GetListOfKeys()
    
    for key in keys:
        obj = root_file.Get(key.GetName())  

        if obj.GetName() == target_name:
            print(f"target found: {obj.GetName()}")
            return obj  

        elif isinstance(obj, ROOT.TDirectory):
            print(f"Entering directory: {obj.GetName()}")
            result = find_graph_by_name(obj, target_name)  
            if result:
                return result  
    print(f'can not find target: {target_name}')            
    return None

def shift_hist_x(hist, offset, new_name=None):
    """给 TH1 的每个 bin 横坐标添加偏移量，避免绘图重合
    Args:
        hist (TH1): 输入的 ROOT 直方图
        offset (float): X 方向的偏移量
        new_name (str, optional): 新直方图的名字（默认在原名字后加 "_shifted"）
    Returns:
        TH1: 偏移后的新直方图
    """
    if new_name is None:
        new_name = f"{hist.GetName()}_shifted"
    
    # 创建新直方图（保持相同的 bin 数量和范围）
    nbins = hist.GetNbinsX()
    x_min = hist.GetXaxis().GetXmin() + offset
    x_max = hist.GetXaxis().GetXmax() + offset
    new_hist = ROOT.TH1D(new_name, hist.GetTitle(), nbins, x_min, x_max)
    
    # 复制原直方图的数据（加上偏移量）
    for i in range(1, nbins + 1):
        x = hist.GetBinCenter(i) + offset
        y = hist.GetBinContent(i)
        new_hist.Fill(x, y)
    
    return new_hist

def get_ratio_hist(h_num, h_den, name="ratio"):
    # 检查输入类型
    if not (isinstance(h_num, (ROOT.TH1, ROOT.TF1))) or not (isinstance(h_den, (ROOT.TH1, ROOT.TF1))):
        raise ValueError("输入必须是 TH1 或 TF1 类型！")

    # 处理分母：如果是 TF1，转换为与分子相同 binning 的 TH1
    if isinstance(h_den, ROOT.TF1):
        if isinstance(h_num, ROOT.TH1):
            h_den_tmp = h_num.Clone("h_den_tmp")
            h_den_tmp.Reset()
            for i in range(1, h_den_tmp.GetNbinsX() + 1):
                x = h_den_tmp.GetBinCenter(i)
                h_den_tmp.SetBinContent(i, h_den.Eval(x))
        else:
            raise ValueError("当分母为 TF1 时，分子必须是 TH1 以确定 binning！")
    else:
        h_den_tmp = h_den.Clone("h_den_tmp")

    # 处理分子：如果是 TF1，转换为与分母相同 binning 的 TH1
    if isinstance(h_num, ROOT.TF1):
        h_num_tmp = h_den_tmp.Clone("h_num_tmp")
        h_num_tmp.Reset()
        for i in range(1, h_num_tmp.GetNbinsX() + 1):
            x = h_num_tmp.GetBinCenter(i)
            h_num_tmp.SetBinContent(i, h_num.Eval(x))
    else:
        h_num_tmp = h_num.Clone("h_num_tmp")

    # 检查 binning 是否一致
    if h_num_tmp.GetNbinsX() != h_den_tmp.GetNbinsX():
        raise ValueError("分子和分母的 bin 数量不一致！")

    # 计算比值
    h_ratio = h_num_tmp.Clone(name)
    h_ratio.Divide(h_num_tmp, h_den_tmp, 1.0, 1.0, "B")  # "B" 使用误差传播公式

    return h_ratio
###python plot func
def extract_info_TH1(hist):
    n_bins = hist.GetNbinsX()
    bin_values = np.array([hist.GetBinContent(i) for i in range(1, n_bins + 1)])
    bin_errors = np.array([hist.GetBinError(i) for i in range(1, n_bins + 1)])
    bin_edges = np.array([hist.GetBinLowEdge(i) for i in range(1, n_bins + 2)])
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) 
    return bin_centers, bin_values, bin_errors, bin_edges

def extract_info_TGraphAsymmErrors(graph):
    n_points = graph.GetN()
    x = np.ndarray(n_points, 'd', graph.GetX())
    y = np.ndarray(n_points, 'd', graph.GetY())
    x_err_low = np.ndarray(n_points, 'd', graph.GetEXlow())
    x_err_high = np.ndarray(n_points, 'd', graph.GetEXhigh())
    y_err_low = np.ndarray(n_points, 'd', graph.GetEYlow())
    y_err_high = np.ndarray(n_points, 'd', graph.GetEYhigh())
    return x, y, x_err_low, x_err_high, y_err_low, y_err_high

def calculate_ratio_with_tf1(bin_centers, bin_values, bin_errors, tf1):
    """
    计算每个 bin 的比值: bin_value / TF1_value
    
    参数:
        bin_centers : array-like, bin 中心值
        bin_values  : array-like, bin 的内容值
        bin_errors  : array-like, bin 的误差
        tf1         : ROOT.TF1, 用于计算理论值的函数
    
    返回:
        ratios      : np.array, 比值数组
        ratio_errs  : np.array, 比值误差数组
    """
    # 转换为 numpy 数组
    bin_centers = np.asarray(bin_centers)
    bin_values = np.asarray(bin_values)
    bin_errors = np.asarray(bin_errors)
    
    # 计算 TF1 在每个 bin 中心的值
    tf1_values = np.array([tf1.Eval(x) for x in bin_centers])
    
    # 计算比值 (避免除以零)
    ratios = np.divide(
        bin_values, 
        tf1_values, 
        out=np.zeros_like(bin_values), 
        where=(tf1_values != 0)
    )
    
    # 计算比值误差 (误差传播公式: σ_ratio = ratio * sqrt((σ_bin/bin_value)^2 + (σ_tf1/tf1_value)^2)
    # 假设 σ_tf1 = 0（如果 TF1 无误差）
    ratio_errs = np.abs(ratios) * np.sqrt(
        (bin_errors / np.where(bin_values != 0, bin_values, 1))**2
    )
    
    return ratios, ratio_errs
###global var
markersize = 10
###

output_path_base = '../../results/run3_vs_run2'
if not os.path.exists(output_path_base):
    os.makedirs(output_path_base)
text_placeholder = ROOT.TObject()

h3l_0_10_run2_file = ROOT.TFile.Open("/Users/zhengqingwang/alice/data/h3l_spec_run2/h3l_0_10.root")
h3l_10_30_run2_file = ROOT.TFile.Open("/Users/zhengqingwang/alice/data/h3l_spec_run2/h3l_10_30.root")
h3l_30_50_run2_file = ROOT.TFile.Open("/Users/zhengqingwang/alice/data/h3l_spec_run2/h3l_30_50.root")

BW_file = ROOT.TFile.Open('../utils/H3L_BWFit.root')

h3l_0_5_run3_file = ROOT.TFile.Open("../../results/ep4/both/cen0-5/pt_analysis_pbpb.root")
h3l_5_10_run3_file = ROOT.TFile.Open("../../results/ep4/both/cen5-10/pt_analysis_pbpb.root")
h3l_10_30_run3_file = ROOT.TFile.Open("../../results/ep4/both/cen10-30/pt_analysis_pbpb.root")
h3l_30_50_run3_file = ROOT.TFile.Open("../../results/ep4/both/cen30-50/pt_analysis_pbpb.root")
h3l_50_80_run3_file = ROOT.TFile.Open("../../results/ep4/both/cen50-80/pt_analysis_pbpb.root")
h3l_0_5_run3_file_BDT = ROOT.TFile.Open("../../results/ep4/spec_trainboth/cen0-5/pt_analysis_pbpb.root")
h3l_5_10_run3_file_BDT = ROOT.TFile.Open("../../results/ep4/spec_trainboth/cen5-10/pt_analysis_pbpb.root")
h3l_10_30_run3_file_BDT = ROOT.TFile.Open("../../results/ep4/spec_trainboth/cen10-30/pt_analysis_pbpb.root")
h3l_30_50_run3_file_BDT = ROOT.TFile.Open("../../results/ep4/spec_trainboth/cen30-50/pt_analysis_pbpb.root")
h3l_50_80_run3_file_BDT = ROOT.TFile.Open("../../results/ep4/spec_trainboth/cen50-80/pt_analysis_pbpb.root")
h3l_0_5_run3_file_BDT_apass5 = ROOT.TFile.Open("../../results/ep5/spec_trainboth/cen0-5/pt_analysis_pbpb.root")
h3l_5_10_run3_file_BDT_apass5 = ROOT.TFile.Open("../../results/ep5/spec_trainboth/cen5-10/pt_analysis_pbpb.root")
h3l_10_30_run3_file_BDT_apass5 = ROOT.TFile.Open("../../results/ep5/spec_trainboth/cen10-30/pt_analysis_pbpb.root")
h3l_30_50_run3_file_BDT_apass5 = ROOT.TFile.Open("../../results/ep5/spec_trainboth/cen30-50/pt_analysis_pbpb.root")
h3l_50_80_run3_file_BDT_apass5 = ROOT.TFile.Open("../../results/ep5/spec_trainboth/cen50-80/pt_analysis_pbpb.root")


graph_h3l_0_10_run2 = find_graph_by_name(h3l_0_10_run2_file,'Graph1D_y1')
graph_h3l_10_30_run2 = find_graph_by_name(h3l_10_30_run2_file,'Graph1D_y1')
graph_h3l_30_50_run2 = find_graph_by_name(h3l_30_50_run2_file,'Graph1D_y1')


graph_h3l_0_5_run3 = find_graph_by_name(h3l_0_5_run3_file,'h_corrected_counts')
graph_h3l_5_10_run3 = find_graph_by_name(h3l_5_10_run3_file,'h_corrected_counts')
graph_h3l_10_30_run3 = find_graph_by_name(h3l_10_30_run3_file,'h_corrected_counts')
graph_h3l_30_50_run3 = find_graph_by_name(h3l_30_50_run3_file,'h_corrected_counts')
graph_h3l_50_80_run3 = find_graph_by_name(h3l_50_80_run3_file,'h_corrected_counts')
graph_h3l_0_5_run3_BDT = find_graph_by_name(h3l_0_5_run3_file_BDT,'h_corrected_counts')
graph_h3l_5_10_run3_BDT = find_graph_by_name(h3l_5_10_run3_file_BDT,'h_corrected_counts')
graph_h3l_10_30_run3_BDT = find_graph_by_name(h3l_10_30_run3_file_BDT,'h_corrected_counts')
graph_h3l_30_50_run3_BDT = find_graph_by_name(h3l_30_50_run3_file_BDT,'h_corrected_counts')
graph_h3l_50_80_run3_BDT = find_graph_by_name(h3l_50_80_run3_file_BDT,'h_corrected_counts')
graph_h3l_0_5_run3_BDT_apass5 = find_graph_by_name(h3l_0_5_run3_file_BDT_apass5,'h_corrected_counts')
graph_h3l_5_10_run3_BDT_apass5 = find_graph_by_name(h3l_5_10_run3_file_BDT_apass5,'h_corrected_counts')
graph_h3l_10_30_run3_BDT_apass5 = find_graph_by_name(h3l_10_30_run3_file_BDT_apass5,'h_corrected_counts')
graph_h3l_30_50_run3_BDT_apass5 = find_graph_by_name(h3l_30_50_run3_file_BDT_apass5,'h_corrected_counts')
graph_h3l_50_80_run3_BDT_apass5 = find_graph_by_name(h3l_50_80_run3_file_BDT_apass5,'h_corrected_counts')



func_bw_0_10 = find_graph_by_name(BW_file,'BlastWave_H3L_0_10')
func_bw_10_30 = find_graph_by_name(BW_file,'BlastWave_H3L_10_30')
func_bw_30_50 = find_graph_by_name(BW_file,'BlastWave_H3L_30_50')
npoints_TF1 = 1000
x_TF1_0_10 = np.linspace(func_bw_0_10.GetXmin(), func_bw_0_10.GetXmax(), npoints_TF1)
y_TF1_0_10 = np.array([func_bw_0_10.Eval(xi) for xi in x_TF1_0_10])
x_TF1_10_30 = np.linspace(func_bw_10_30.GetXmin(), func_bw_10_30.GetXmax(), npoints_TF1)
y_TF1_10_30 = np.array([func_bw_10_30.Eval(xi) for xi in x_TF1_10_30])
x_TF1_30_50 = np.linspace(func_bw_30_50.GetXmin(), func_bw_30_50.GetXmax(), npoints_TF1)
y_TF1_30_50 = np.array([func_bw_30_50.Eval(xi) for xi in x_TF1_30_50])
ratio_base_0_10 = np.ones_like(x_TF1_0_10)
ratio_base_10_30 = np.ones_like(x_TF1_10_30)
ratio_base_30_50 = np.ones_like(x_TF1_30_50)


###Run2 
x_run2_0_10, y_run2_0_10, x_err_run2_0_10_low, x_err_run2_0_10_high, y_err_run2_0_10_low, y_err_run2_0_10_high = extract_info_TGraphAsymmErrors(graph_h3l_0_10_run2)
x_run2_10_30, y_run2_10_30, x_err_run2_10_30_low, x_err_run2_10_30_high, y_err_run2_10_30_low, y_err_run2_10_30_high = extract_info_TGraphAsymmErrors(graph_h3l_10_30_run2)
x_run2_30_50, y_run2_30_50, x_err_run2_30_50_low, x_err_run2_30_50_high, y_err_run2_30_50_low, y_err_run2_30_50_high = extract_info_TGraphAsymmErrors(graph_h3l_30_50_run2)

###0-5%
x_run3_0_5, y_run3_0_5, y_err_run3_0_5, x_edge_run3_0_5  = extract_info_TH1(graph_h3l_0_5_run3)
x_run3_0_5_BDT, y_run3_0_5_BDT, y_err_run3_0_5_BDT, x_edge_run3_0_5_BDT = extract_info_TH1(graph_h3l_0_5_run3_BDT)
x_run3_0_5_BDT_apass5, y_run3_0_5_BDT_apass5, y_err_run3_0_5_BDT_apass5, x_edge_run3_0_5_BDT_apass5 = extract_info_TH1(graph_h3l_0_5_run3_BDT_apass5)
y_ratio_run3_BDT_Topo = y_run3_0_5_BDT/y_run3_0_5
y_ratio_run3_BDT_Topo_err = np.sqrt((y_err_run3_0_5_BDT/y_run3_0_5)**2 + (y_err_run3_0_5/y_run3_0_5_BDT)**2)
y_ratio_run2, y_ratio_run2_err = calculate_ratio_with_tf1(x_run3_0_5, y_run3_0_5, y_err_run3_0_5, func_bw_0_10)
y_ratio_run2_BDT, y_ratio_run2_BDT_err = calculate_ratio_with_tf1(x_run3_0_5_BDT, y_run3_0_5_BDT, y_err_run3_0_5_BDT, func_bw_0_10)
y_ratio_run2_BDT_apass5, y_ratio_run2_BDT_apass5_err = calculate_ratio_with_tf1(x_run3_0_5_BDT_apass5, y_run3_0_5_BDT_apass5, y_err_run3_0_5_BDT_apass5, func_bw_0_10)
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8))
fig.subplots_adjust(hspace=0.3)  # 调整水平间距
ax_top.set_position([0.1, 0.3, 0.8, 0.6])  # [左, 下, 宽, 高]
ax_bottom.set_position([0.1, 0.1, 0.8, 0.25])  # 下轴高度更大
ax_top.errorbar(
    x = x_run2_0_10,
    y = y_run2_0_10,
    #xerr = [0, 0],  # 非对称 x 误差
    yerr = [y_err_run2_0_10_low, y_err_run2_0_10_high],  # 非对称 y 误差
    fmt = "s",                       # 方形标记
    markersize = markersize,
    color = "blue",
    label = "Run2 Reference(0-10%)",
)
ax_top.errorbar(
    x = x_run3_0_5,
    y = y_run3_0_5,
    #xerr = 0,  
    yerr = y_err_run3_0_5,  
    fmt = "o",  
    markersize = markersize,                   
    color = "red",
    label = "Topology Spectrum apass4(0-5%)",  
)
ax_top.errorbar(
    x = x_run3_0_5_BDT,
    y = y_run3_0_5_BDT,
    #xerr = 0,  
    yerr = y_err_run3_0_5_BDT,  
    fmt = "D",  
    markersize = markersize,                   
    color = "cyan",
    label = "BDT Spectruma apass4(0-5%)",  
)
ax_top.errorbar(
    x = x_run3_0_5_BDT_apass5,
    y = y_run3_0_5_BDT_apass5,
    #xerr = 0,  
    yerr = y_err_run3_0_5_BDT_apass5,  
    fmt = "*",   
    markersize = markersize,                  
    color = "magenta",
    label = "BDT Spectrum apass5(0-5%)",  
)
ax_top.plot(x_TF1_0_10, y_TF1_0_10, 'g--', linewidth=2, label='BlastWave Fit Run2(0_10%)')
ax_top.set_title('H3L Spectrum Run3(0-5%) vs Run2(0-10%)')
ax_top.set_ylabel(r'$\frac{d\it{N}}{d\it{p}_{\!T}} (\mathrm{GeV}/\it{c})^{-1}$', fontsize=14)
ax_top.set_yscale('log')
ax_top.xaxis.set_visible(False)
ax_top.set_xlim(x_edge_run3_0_5[0],x_edge_run3_0_5[-1])
ax_top.set_ylim(1e-7, 7*1e-5) 
ax_top.legend(loc='upper right')
ax_bottom.errorbar(
    x = x_run3_0_5,
    y = y_ratio_run3_BDT_Topo,
    #xerr = 0,  
    yerr = y_ratio_run3_BDT_Topo_err,  
    fmt = "o",                     
    color = "#FF7F00",
    label = "Ratio apass4 BDT/Topology",  
)
ax_bottom.errorbar(
    x = x_run3_0_5,
    y = y_ratio_run2,
    #xerr = 0,  
    yerr = y_ratio_run2_err,  
    fmt = "o",
    color = "red",
    label = "Ratio apass4 Topology/Run2",  
)
ax_bottom.errorbar(
    x = x_run3_0_5_BDT,
    y = y_ratio_run2_BDT,
    #xerr = 0,  
    yerr = y_ratio_run2_BDT_err,  
    fmt = "o",                     
    color = "cyan",
    label = "Ratio apass4 BDT/Run2",  
)
ax_bottom.errorbar(
    x = x_run3_0_5_BDT_apass5,
    y = y_ratio_run2_BDT_apass5,
    #xerr = 0,  
    yerr = y_ratio_run2_BDT_apass5_err,  
    fmt = "o",                     
    color = "magenta",
    label = "Ratio apass5 BDT/Run2",  
)
ax_bottom.plot(x_TF1_0_10, ratio_base_0_10, '-.', color='0.7', linewidth=1)
ax_bottom.set_ylabel(r'Ratio', fontsize=14)
ax_bottom.set_xlabel(r'$\mathit{p}_{\mathrm{T}}$ (GeV/$\mathit{c}$)', fontsize=14)
ax_bottom.set_xlim(x_edge_run3_0_5[0],x_edge_run3_0_5[-1])
ax_bottom.legend(loc='upper right')
plt.savefig(f'{output_path_base}/H3L_spectrum_0_5.pdf', dpi=300)

###5-10%
x_run3_5_10, y_run3_5_10, y_err_run3_5_10, x_edge_run3_5_10  = extract_info_TH1(graph_h3l_5_10_run3)
x_run3_5_10_BDT, y_run3_5_10_BDT, y_err_run3_5_10_BDT, x_edge_run3_5_10_BDT = extract_info_TH1(graph_h3l_5_10_run3_BDT)
x_run3_5_10_BDT_apass5, y_run3_5_10_BDT_apass5, y_err_run3_5_10_BDT_apass5, x_edge_run3_5_10_BDT_apass5 = extract_info_TH1(graph_h3l_5_10_run3_BDT_apass5)
y_ratio_run3_BDT_Topo = y_run3_5_10_BDT/y_run3_5_10
y_ratio_run3_BDT_Topo_err = np.sqrt((y_err_run3_5_10_BDT/y_run3_5_10)**2 + (y_err_run3_5_10/y_run3_5_10_BDT)**2)
y_ratio_run2, y_ratio_run2_err = calculate_ratio_with_tf1(x_run3_5_10, y_run3_5_10, y_err_run3_5_10, func_bw_0_10)
y_ratio_run2_BDT, y_ratio_run2_BDT_err = calculate_ratio_with_tf1(x_run3_5_10_BDT, y_run3_5_10_BDT, y_err_run3_5_10_BDT, func_bw_0_10)
y_ratio_run2_BDT_apass5, y_ratio_run2_BDT_apass5_err = calculate_ratio_with_tf1(x_run3_5_10_BDT_apass5, y_run3_5_10_BDT_apass5, y_err_run3_5_10_BDT_apass5, func_bw_0_10)
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8))
fig.subplots_adjust(hspace=0.3)  # 调整水平间距
ax_top.set_position([0.1, 0.3, 0.8, 0.6])  # [左, 下, 宽, 高]
ax_bottom.set_position([0.1, 0.1, 0.8, 0.25])  # 下轴高度更大
ax_top.errorbar(
    x = x_run2_0_10,
    y = y_run2_0_10,
    #xerr = [0, 0],  # 非对称 x 误差
    yerr = [y_err_run2_0_10_low, y_err_run2_0_10_high],  # 非对称 y 误差
    fmt = "s",                       # 方形标记
    markersize = markersize,
    color = "blue",
    label = "Run2 Reference(0-10%)",
)
ax_top.errorbar(
    x = x_run3_5_10,
    y = y_run3_5_10,
    #xerr = 0,  
    yerr = y_err_run3_5_10,  
    fmt = "o",  
    markersize = markersize,                   
    color = "red",
    label = "Topology Spectrum apass4(0-5%)",  
)
ax_top.errorbar(
    x = x_run3_5_10_BDT,
    y = y_run3_5_10_BDT,
    #xerr = 0,  
    yerr = y_err_run3_5_10_BDT,  
    fmt = "D",  
    markersize = markersize,                   
    color = "cyan",
    label = "BDT Spectruma apass4(0-5%)",  
)
ax_top.errorbar(
    x = x_run3_5_10_BDT_apass5,
    y = y_run3_5_10_BDT_apass5,
    #xerr = 0,  
    yerr = y_err_run3_5_10_BDT_apass5,  
    fmt = "*",   
    markersize = markersize,                  
    color = "magenta",
    label = "BDT Spectrum apass5(0-5%)",  
)
ax_top.plot(x_TF1_0_10, y_TF1_0_10, 'g--', linewidth=2, label='BlastWave Fit Run2(0_10%)')
ax_top.set_title('H3L Spectrum Run3(5-10%) vs Run2(0-10%)')
ax_top.set_ylabel(r'$\frac{d\it{N}}{d\it{p}_{\!T}} (\mathrm{GeV}/\it{c})^{-1}$', fontsize=14)
ax_top.set_yscale('log')
ax_top.xaxis.set_visible(False)
ax_top.set_xlim(x_edge_run3_5_10[0],x_edge_run3_5_10[-1])
ax_top.set_ylim(1e-7, 7*1e-5) 
ax_top.legend(loc='upper right')
ax_bottom.errorbar(
    x = x_run3_5_10,
    y = y_ratio_run3_BDT_Topo,
    #xerr = 0,  
    yerr = y_ratio_run3_BDT_Topo_err,  
    fmt = "o",                     
    color = "#FF7F00",
    label = "Ratio apass4 BDT/Topology",  
)
ax_bottom.errorbar(
    x = x_run3_5_10,
    y = y_ratio_run2,
    #xerr = 0,  
    yerr = y_ratio_run2_err,  
    fmt = "o",
    color = "red",
    label = "Ratio apass4 Topology/Run2",  
)
ax_bottom.errorbar(
    x = x_run3_5_10_BDT,
    y = y_ratio_run2_BDT,
    #xerr = 0,  
    yerr = y_ratio_run2_BDT_err,  
    fmt = "o",                     
    color = "cyan",
    label = "Ratio apass4 BDT/Run2",  
)
ax_bottom.errorbar(
    x = x_run3_5_10_BDT_apass5,
    y = y_ratio_run2_BDT_apass5,
    #xerr = 0,  
    yerr = y_ratio_run2_BDT_apass5_err,  
    fmt = "o",                     
    color = "magenta",
    label = "Ratio apass5 BDT/Run2",  
)
ax_bottom.plot(x_TF1_0_10, ratio_base_0_10, '-.', color='0.7', linewidth=1)
ax_bottom.set_ylabel(r'Ratio', fontsize=14)
ax_bottom.set_xlabel(r'$\mathit{p}_{\mathrm{T}}$ (GeV/$\mathit{c}$)', fontsize=14)
ax_bottom.set_xlim(x_edge_run3_5_10[0],x_edge_run3_5_10[-1])
ax_bottom.legend(loc='upper right')
plt.savefig(f'{output_path_base}/H3L_spectrum_5_10.pdf', dpi=300)
###10-30%
x_run3_10_30, y_run3_10_30, y_err_run3_10_30, x_edge_run3_10_30  = extract_info_TH1(graph_h3l_10_30_run3)
x_run3_10_30_BDT, y_run3_10_30_BDT, y_err_run3_10_30_BDT, x_edge_run3_10_30_BDT = extract_info_TH1(graph_h3l_10_30_run3_BDT)
x_run3_10_30_BDT_apass5, y_run3_10_30_BDT_apass5, y_err_run3_10_30_BDT_apass5, x_edge_run3_10_30_BDT_apass5 = extract_info_TH1(graph_h3l_10_30_run3_BDT_apass5)
y_ratio_run3_BDT_Topo = y_run3_10_30_BDT/y_run3_10_30
y_ratio_run3_BDT_Topo_err = np.sqrt((y_err_run3_10_30_BDT/y_run3_10_30)**2 + (y_err_run3_10_30/y_run3_10_30_BDT)**2)
y_ratio_run2, y_ratio_run2_err = calculate_ratio_with_tf1(x_run3_10_30, y_run3_10_30, y_err_run3_10_30, func_bw_10_30)
y_ratio_run2_BDT, y_ratio_run2_BDT_err = calculate_ratio_with_tf1(x_run3_10_30_BDT, y_run3_10_30_BDT, y_err_run3_10_30_BDT, func_bw_10_30)
y_ratio_run2_BDT_apass5, y_ratio_run2_BDT_apass5_err = calculate_ratio_with_tf1(x_run3_10_30_BDT_apass5, y_run3_10_30_BDT_apass5, y_err_run3_10_30_BDT_apass5, func_bw_10_30)
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8))
fig.subplots_adjust(hspace=0.3)  # 调整水平间距
ax_top.set_position([0.1, 0.3, 0.8, 0.6])  # [左, 下, 宽, 高]
ax_bottom.set_position([0.1, 0.1, 0.8, 0.25])  # 下轴高度更大
ax_top.errorbar(
    x = x_run2_10_30,
    y = y_run2_10_30,
    #xerr = [0, 0],  # 非对称 x 误差
    yerr = [y_err_run2_10_30_low, y_err_run2_10_30_high],  # 非对称 y 误差
    fmt = "s",                       # 方形标记
    markersize = markersize,
    color = "blue",
    label = "Run2 Reference(0-10%)",
)
ax_top.errorbar(
    x = x_run3_10_30,
    y = y_run3_10_30,
    #xerr = 0,  
    yerr = y_err_run3_10_30,  
    fmt = "o",  
    markersize = markersize,                   
    color = "red",
    label = "Topology Spectrum apass4(5-10%)",  
)
ax_top.errorbar(
    x = x_run3_10_30_BDT,
    y = y_run3_10_30_BDT,
    #xerr = 0,  
    yerr = y_err_run3_10_30_BDT,  
    fmt = "D",  
    markersize = markersize,                   
    color = "cyan",
    label = "BDT Spectruma apass4(5-10%)",  
)
ax_top.errorbar(
    x = x_run3_10_30_BDT_apass5,
    y = y_run3_10_30_BDT_apass5,
    #xerr = 0,  
    yerr = y_err_run3_10_30_BDT_apass5,  
    fmt = "*",   
    markersize = markersize,                  
    color = "magenta",
    label = "BDT Spectrum apass5(5-10%)",  
)
ax_top.plot(x_TF1_10_30, y_TF1_10_30, 'g--', linewidth=2, label='BlastWave Fit Run2(10_30%)')
ax_top.set_title('H3L Spectrum Run3(10-30%) vs Run2(10-30%)')
ax_top.set_ylabel(r'$\frac{d\it{N}}{d\it{p}_{\!T}} (\mathrm{GeV}/\it{c})^{-1}$', fontsize=14)
ax_top.set_yscale('log')
ax_top.xaxis.set_visible(False)
ax_top.set_xlim(x_edge_run3_10_30[0],x_edge_run3_10_30[-1])
ax_top.set_ylim(1e-7, 3*1e-5) 
ax_top.legend(loc='upper right')
ax_bottom.errorbar(
    x = x_run3_10_30,
    y = y_ratio_run3_BDT_Topo,
    #xerr = 0,  
    yerr = y_ratio_run3_BDT_Topo_err,  
    fmt = "o",                     
    color = "#FF7F00",
    label = "Ratio apass4 BDT/Topology",  
)
ax_bottom.errorbar(
    x = x_run3_10_30,
    y = y_ratio_run2,
    #xerr = 0,  
    yerr = y_ratio_run2_err,  
    fmt = "o",
    color = "red",
    label = "Ratio apass4 Topology/Run2",  
)
ax_bottom.errorbar(
    x = x_run3_10_30_BDT,
    y = y_ratio_run2_BDT,
    #xerr = 0,  
    yerr = y_ratio_run2_BDT_err,  
    fmt = "o",                     
    color = "cyan",
    label = "Ratio apass4 BDT/Run2",  
)
ax_bottom.errorbar(
    x = x_run3_10_30_BDT_apass5,
    y = y_ratio_run2_BDT_apass5,
    #xerr = 0,  
    yerr = y_ratio_run2_BDT_apass5_err,  
    fmt = "o",                     
    color = "magenta",
    label = "Ratio apass5 BDT/Run2",  
)
ax_bottom.plot(x_TF1_10_30, ratio_base_10_30, '-.', color='0.7', linewidth=1)
ax_bottom.set_ylabel(r'Ratio', fontsize=14)
ax_bottom.set_xlabel(r'$\mathit{p}_{\mathrm{T}}$ (GeV/$\mathit{c}$)', fontsize=14)
ax_bottom.set_xlim(x_edge_run3_10_30[0],x_edge_run3_10_30[-1])
ax_bottom.legend(loc='upper right')
plt.savefig(f'{output_path_base}/H3L_spectrum_10_30.pdf', dpi=300)
###30-50%
x_run3_30_50, y_run3_30_50, y_err_run3_30_50, x_edge_run3_30_50  = extract_info_TH1(graph_h3l_30_50_run3)
x_run3_30_50_BDT, y_run3_30_50_BDT, y_err_run3_30_50_BDT, x_edge_run3_30_50_BDT = extract_info_TH1(graph_h3l_30_50_run3_BDT)
x_run3_30_50_BDT_apass5, y_run3_30_50_BDT_apass5, y_err_run3_30_50_BDT_apass5, x_edge_run3_30_50_BDT_apass5 = extract_info_TH1(graph_h3l_30_50_run3_BDT_apass5)
y_ratio_run3_BDT_Topo = y_run3_30_50_BDT/y_run3_30_50
y_ratio_run3_BDT_Topo_err = np.sqrt((y_err_run3_30_50_BDT/y_run3_30_50)**2 + (y_err_run3_30_50/y_run3_30_50_BDT)**2)
y_ratio_run2, y_ratio_run2_err = calculate_ratio_with_tf1(x_run3_30_50, y_run3_30_50, y_err_run3_30_50, func_bw_30_50)
y_ratio_run2_BDT, y_ratio_run2_BDT_err = calculate_ratio_with_tf1(x_run3_30_50_BDT, y_run3_30_50_BDT, y_err_run3_30_50_BDT, func_bw_30_50)
y_ratio_run2_BDT_apass5, y_ratio_run2_BDT_apass5_err = calculate_ratio_with_tf1(x_run3_30_50_BDT_apass5, y_run3_30_50_BDT_apass5, y_err_run3_30_50_BDT_apass5, func_bw_30_50)
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8))
fig.subplots_adjust(hspace=0.3)  # 调整水平间距
ax_top.set_position([0.1, 0.3, 0.8, 0.6])  # [左, 下, 宽, 高]
ax_bottom.set_position([0.1, 0.1, 0.8, 0.25])  # 下轴高度更大
ax_top.errorbar(
    x = x_run2_30_50,
    y = y_run2_30_50,
    #xerr = [0, 0],  # 非对称 x 误差
    yerr = [y_err_run2_30_50_low, y_err_run2_30_50_high],  # 非对称 y 误差
    fmt = "s",                       # 方形标记
    markersize = markersize,
    color = "blue",
    label = "Run2 Reference(30-50%)",
)
ax_top.errorbar(
    x = x_run3_30_50,
    y = y_run3_30_50,
    #xerr = 0,  
    yerr = y_err_run3_30_50,  
    fmt = "o",  
    markersize = markersize,                   
    color = "red",
    label = "Topology Spectrum apass4(30-50%)",  
)
ax_top.errorbar(
    x = x_run3_30_50_BDT,
    y = y_run3_30_50_BDT,
    #xerr = 0,  
    yerr = y_err_run3_30_50_BDT,  
    fmt = "D",  
    markersize = markersize,                   
    color = "cyan",
    label = "BDT Spectruma apass4(30-50%)",  
)
ax_top.errorbar(
    x = x_run3_30_50_BDT_apass5,
    y = y_run3_30_50_BDT_apass5,
    #xerr = 0,  
    yerr = y_err_run3_30_50_BDT_apass5,  
    fmt = "*",   
    markersize = markersize,                  
    color = "magenta",
    label = "BDT Spectrum apass5(30-50%)",  
)
ax_top.plot(x_TF1_30_50, y_TF1_30_50, 'g--', linewidth=2, label='BlastWave Fit Run2(30_50%)')
ax_top.set_title('H3L Spectrum Run3(30-50%) vs Run2(30-50%)')
ax_top.set_ylabel(r'$\frac{d\it{N}}{d\it{p}_{\!T}} (\mathrm{GeV}/\it{c})^{-1}$', fontsize=14)
ax_top.set_yscale('log')
ax_top.xaxis.set_visible(False)
ax_top.set_xlim(x_edge_run3_30_50[0],x_edge_run3_30_50[-1])
ax_top.set_ylim(5*1e-8, 1e-5) 
ax_top.legend(loc='upper right')
ax_bottom.errorbar(
    x = x_run3_30_50,
    y = y_ratio_run3_BDT_Topo,
    #xerr = 0,  
    yerr = y_ratio_run3_BDT_Topo_err,  
    fmt = "o",                     
    color = "#FF7F00",
    label = "Ratio apass4 BDT/Topology",  
)
ax_bottom.errorbar(
    x = x_run3_30_50,
    y = y_ratio_run2,
    #xerr = 0,  
    yerr = y_ratio_run2_err,  
    fmt = "o",
    color = "red",
    label = "Ratio apass4 Topology/Run2",  
)
ax_bottom.errorbar(
    x = x_run3_30_50_BDT,
    y = y_ratio_run2_BDT,
    #xerr = 0,  
    yerr = y_ratio_run2_BDT_err,  
    fmt = "o",                     
    color = "cyan",
    label = "Ratio apass4 BDT/Run2",  
)
ax_bottom.errorbar(
    x = x_run3_30_50_BDT_apass5,
    y = y_ratio_run2_BDT_apass5,
    #xerr = 0,  
    yerr = y_ratio_run2_BDT_apass5_err,  
    fmt = "o",                     
    color = "magenta",
    label = "Ratio apass5 BDT/Run2",  
)
ax_bottom.plot(x_TF1_30_50, ratio_base_30_50, '-.', color='0.7', linewidth=1)
ax_bottom.set_ylabel(r'Ratio', fontsize=14)
ax_bottom.set_xlabel(r'$\mathit{p}_{\mathrm{T}}$ (GeV/$\mathit{c}$)', fontsize=14)
ax_bottom.set_xlim(x_edge_run3_30_50[0],x_edge_run3_30_50[-1])
ax_bottom.legend(loc='upper right')
plt.savefig(f'{output_path_base}/H3L_spectrum_30_50.pdf', dpi=300)
###50-80%
x_run3_50_80, y_run3_50_80, y_err_run3_50_80, x_edge_run3_50_80  = extract_info_TH1(graph_h3l_50_80_run3)
x_run3_50_80_BDT, y_run3_50_80_BDT, y_err_run3_50_80_BDT, x_edge_run3_50_80_BDT = extract_info_TH1(graph_h3l_50_80_run3_BDT)
x_run3_50_80_BDT_apass5, y_run3_50_80_BDT_apass5, y_err_run3_50_80_BDT_apass5, x_edge_run3_50_80_BDT_apass5 = extract_info_TH1(graph_h3l_50_80_run3_BDT_apass5)
y_ratio_run3_BDT_Topo = y_run3_50_80_BDT/y_run3_50_80
y_ratio_run3_BDT_Topo_err = np.sqrt((y_err_run3_50_80_BDT/y_run3_50_80)**2 + (y_err_run3_50_80/y_run3_50_80_BDT)**2)
y_ratio_run3_BDT_apass5 = y_run3_50_80_BDT_apass5/y_run3_50_80
y_ratio_run3_BDT_apass5_err = np.sqrt((y_err_run3_50_80_BDT_apass5/y_run3_50_80)**2 + (y_err_run3_50_80/y_run3_50_80_BDT_apass5)**2)
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8))
fig.subplots_adjust(hspace=0.3)  # 调整水平间距
ax_top.set_position([0.1, 0.3, 0.8, 0.6])  # [左, 下, 宽, 高]
ax_bottom.set_position([0.1, 0.1, 0.8, 0.25])  # 下轴高度更大
ax_top.errorbar(
    x = x_run3_50_80,
    y = y_run3_50_80,
    #xerr = 0,  
    yerr = y_err_run3_50_80,  
    fmt = "o",  
    markersize = markersize,                   
    color = "red",
    label = "Topology Spectrum apass4(50-80%)",  
)
ax_top.errorbar(
    x = x_run3_50_80_BDT,
    y = y_run3_50_80_BDT,
    #xerr = 0,  
    yerr = y_err_run3_50_80_BDT,  
    fmt = "D",  
    markersize = markersize,                   
    color = "cyan",
    label = "BDT Spectruma apass4(50-80%)",  
)
ax_top.errorbar(
    x = x_run3_50_80_BDT_apass5,
    y = y_run3_50_80_BDT_apass5,
    #xerr = 0,  
    yerr = y_err_run3_50_80_BDT_apass5,  
    fmt = "*",   
    markersize = markersize,                  
    color = "magenta",
    label = "BDT Spectrum apass5(50-80%)",  
)
ax_top.set_title('H3L Spectrum Run3(50-80%) vs Run2(50-80%)')
ax_top.set_ylabel(r'$\frac{d\it{N}}{d\it{p}_{\!T}} (\mathrm{GeV}/\it{c})^{-1}$', fontsize=14)
ax_top.set_yscale('log')
ax_top.xaxis.set_visible(False)
ax_top.set_xlim(x_edge_run3_50_80[0],x_edge_run3_50_80[-1])
ax_top.set_ylim(5*1e-9, 2*1e-6) 
ax_top.legend(loc='upper right')
ax_bottom.errorbar(
    x = x_run3_50_80,
    y = y_ratio_run3_BDT_Topo,
    #xerr = 0,  
    yerr = y_ratio_run3_BDT_Topo_err,  
    fmt = "o",                     
    color = "#FF7F00",
    label = "Ratio apass4 BDT/Topology",  
)
ax_bottom.errorbar(
    x = x_run3_50_80,
    y = y_ratio_run3_BDT_apass5,
    #xerr = 0,  
    yerr = y_ratio_run3_BDT_apass5_err,  
    fmt = "o",                     
    color = "magenta",
    label = "Ratio apass5/apass4 BDT",  
)
ax_bottom.plot(x_TF1_30_50, ratio_base_30_50, '-.', color='0.7', linewidth=1)
ax_bottom.set_ylabel(r'Ratio', fontsize=14)
ax_bottom.set_xlabel(r'$\mathit{p}_{\mathrm{T}}$ (GeV/$\mathit{c}$)', fontsize=14)
ax_bottom.set_xlim(x_edge_run3_50_80[0],x_edge_run3_50_80[-1])
ax_bottom.legend(loc='upper right')
plt.savefig(f'{output_path_base}/H3L_spectrum_50_80.pdf', dpi=300)


# canvas_0_5 = utils.get_canvas(f'spec_h3l_0_5',600,400,0,0,0.1,0.1,0.1,0.1)
# Pad_top = utils.get_pad("pad_top",0.0,0.3,1.0,1.0,0.1,0.3,0.15,0.1)
# Pad_top.Draw()
# Pad_top.cd()
# canvas_0_5.SetLogy()
# canvas_0_5.SetFillColor(10)
# utils.set_title_th1(graph_h3l_0_5_run3, None, 0.04, 1.07, None, 0.04, 1, 0, 0)
# utils.set_axis_th1(graph_h3l_0_5_run3,None,None,None,None,0.03,0.03)
# utils.set_marker_th1(graph_h3l_0_5_run3,"",20,1.5,ROOT.kPink-2,ROOT.kPink-2)
# utils.set_marker_th1(graph_h3l_0_10_run2,"",54,1.5,ROOT.kAzure+2,ROOT.kAzure+2)
# utils.set_marker_th1(graph_h3l_0_5_run3_BDT,"",22,1.8,ROOT.kOrange+7,ROOT.kOrange+7)
# graph_h3l_0_5_run3_BDT.GetListOfFunctions().Clear()
# func_bw_0_10.SetLineStyle(4)
# func_bw_0_10.SetLineColor(ROOT.kGray+3)
# graph_h3l_0_5_run3.Draw("PE0")
# graph_h3l_0_10_run2.Draw("PE1 SAME")
# graph_h3l_0_5_run3_BDT.Draw("PE1 SAME")
# func_bw_0_10.Draw("SAME")
# legend_0_5 = utils.get_legend(0.65,0.6,0.85,0.88,0,0,42,0.02,1)
# legend_0_5.AddEntry(text_placeholder,'ALICE Run2: #sqrt{#it{s_{NN}}} = 5.02 TeV, Pb-Pb','')
# legend_0_5.AddEntry(text_placeholder,'ALICE Run3: #sqrt{#it{s_{NN}}} = 5.36 TeV, Pb-Pb','')
# legend_0_5.AddEntry(graph_h3l_0_5_run3,'Run3 {}^{3}_{#Lambda}H spectrum(topology), 0-5%','lep')
# legend_0_5.AddEntry(graph_h3l_0_10_run2,'Run2 {}^{3}_{#Lambda}H spectrum, 0-10%','lep')
# legend_0_5.AddEntry(graph_h3l_0_5_run3_BDT,'Run3 {}^{3}_{#Lambda}H spectrum(BDT), 0-5% BDT','lep')
# legend_0_5.AddEntry(func_bw_0_10,'Blast Wave Fit for Run2, 0-10%','l')
# legend_0_5.Draw("SAME")
# utils.draw_my_text_ndc(0.2,0.2,0.05,'0-5%',2)

# canvas_0_5.SaveAs(output_path_base + f'/spec_h3l_0_5.pdf')

# canvas_5_10 = utils.get_canvas(f'spec_h3l_5_10',600,400,0,0,0.1,0.1,0.1,0.1)
# canvas_5_10.SetLogy()
# canvas_5_10.SetFillColor(10)
# utils.set_title_th1(graph_h3l_5_10_run3, None, 0.04, 1.07, None, 0.04, 1, 0, 0)
# utils.set_axis_th1(graph_h3l_5_10_run3,None,None,None,None,0.03,0.03)
# utils.set_marker_th1(graph_h3l_5_10_run3,"",20,1.5,ROOT.kPink-2,ROOT.kPink-2)
# utils.set_marker_th1(graph_h3l_0_10_run2,"",54,1.5,ROOT.kAzure+2,ROOT.kAzure+2)
# utils.set_marker_th1(graph_h3l_5_10_run3_BDT,"",22,1.8,ROOT.kOrange+7,ROOT.kOrange+7)
# graph_h3l_5_10_run3_BDT.GetListOfFunctions().Clear()
# func_bw_0_10.SetLineStyle(4)
# func_bw_0_10.SetLineColor(ROOT.kGray+3)
# graph_h3l_5_10_run3.Draw("PE0")
# graph_h3l_0_10_run2.Draw("PE1 SAME")
# graph_h3l_5_10_run3_BDT.Draw("PE1 SAME")
# func_bw_0_10.Draw("SAME")
# legend_5_10 = utils.get_legend(0.65,0.6,0.85,0.88,0,0,42,0.02,1)
# legend_5_10.AddEntry(text_placeholder,'ALICE Run2: #sqrt{#it{s_{NN}}} = 5.02 TeV, Pb-Pb','')
# legend_5_10.AddEntry(text_placeholder,'ALICE Run3: #sqrt{#it{s_{NN}}} = 5.36 TeV, Pb-Pb','')
# legend_5_10.AddEntry(graph_h3l_5_10_run3,'Run3 {}^{3}_{#Lambda}H spectrum(topology), 5-10%','lep')
# legend_5_10.AddEntry(graph_h3l_0_10_run2,'Run2 {}^{3}_{#Lambda}H spectrum, 0-10%','lep')
# legend_5_10.AddEntry(graph_h3l_5_10_run3_BDT,'Run3 {}^{3}_{#Lambda}H spectrum(BDT), 5-10% BDT','lep')
# legend_5_10.AddEntry(func_bw_0_10,'Blast Wave Fit for Run2, 0-10%','l')
# legend_5_10.Draw("SAME")
# utils.draw_my_text_ndc(0.2,0.2,0.05,'5-10%',2)
# canvas_5_10.SaveAs(output_path_base + f'/spec_h3l_5_10.pdf')

# canvas_10_30 = utils.get_canvas(f'spec_h3l_10_30',600,400,0,0,0.1,0.1,0.1,0.1)
# canvas_10_30.SetLogy()
# canvas_10_30.SetFillColor(10)
# utils.set_title_th1(graph_h3l_10_30_run3, None, 0.04, 1.07, None, 0.04, 1, 0, 0)
# utils.set_axis_th1(graph_h3l_10_30_run3,None,None,None,None,0.03,0.03)
# utils.set_marker_th1(graph_h3l_10_30_run3,"",20,1.5,ROOT.kPink-2,ROOT.kPink-2)
# utils.set_marker_th1(graph_h3l_10_30_run2,"",54,1.5,ROOT.kAzure+2,ROOT.kAzure+2)
# utils.set_marker_th1(graph_h3l_10_30_run3_BDT,"",22,1.8,ROOT.kOrange+7,ROOT.kOrange+7)
# graph_h3l_10_30_run3_BDT.GetListOfFunctions().Clear()
# func_bw_10_30.SetLineStyle(4)
# func_bw_10_30.SetLineColor(ROOT.kGray+3)
# graph_h3l_10_30_run3.Draw("PE0")
# graph_h3l_10_30_run2.Draw("PE1 SAME")
# graph_h3l_10_30_run3_BDT.Draw("PE1 SAME")
# func_bw_10_30.Draw("SAME")
# legend_10_30 = utils.get_legend(0.65,0.6,0.85,0.88,0,0,42,0.02,1)
# legend_10_30.AddEntry(text_placeholder,'ALICE Run2: #sqrt{#it{s_{NN}}} = 5.02 TeV, Pb-Pb','')
# legend_10_30.AddEntry(text_placeholder,'ALICE Run3: #sqrt{#it{s_{NN}}} = 5.36 TeV, Pb-Pb','')
# legend_10_30.AddEntry(graph_h3l_10_30_run3,'Run3 {}^{3}_{#Lambda}H spectrum(topology), 10-30%','lep')
# legend_10_30.AddEntry(graph_h3l_10_30_run2,'Run2 {}^{3}_{#Lambda}H spectrum, 10-30%','lep')
# legend_10_30.AddEntry(graph_h3l_10_30_run3_BDT,'Run3 {}^{3}_{#Lambda}H spectrum(BDT), 10-30% BDT','lep')
# legend_10_30.AddEntry(func_bw_10_30,'Blast Wave Fit for Run2, 10-30%','l')
# legend_10_30.Draw("SAME")
# utils.draw_my_text_ndc(0.2,0.2,0.05,'10-30%',2)
# canvas_10_30.SaveAs(output_path_base + f'/spec_h3l_10_30.pdf')


# canvas_30_50 = utils.get_canvas(f'spec_h3l_30_50',600,400,0,0,0.1,0.1,0.1,0.1)
# canvas_30_50.SetLogy()
# canvas_30_50.SetFillColor(10)
# utils.set_title_th1(graph_h3l_30_50_run3, None, 0.04, 1.07, None, 0.04, 1, 0, 0)
# utils.set_axis_th1(graph_h3l_30_50_run3,None,None,None,None,0.03,0.03)
# utils.set_marker_th1(graph_h3l_30_50_run3,"",20,1.5,ROOT.kPink-2,ROOT.kPink-2)
# utils.set_marker_th1(graph_h3l_30_50_run2,"",54,1.5,ROOT.kAzure+2,ROOT.kAzure+2)
# utils.set_marker_th1(graph_h3l_30_50_run3_BDT,"",22,1.8,ROOT.kOrange+7,ROOT.kOrange+7)
# graph_h3l_30_50_run3_BDT.GetListOfFunctions().Clear()
# func_bw_30_50.SetLineStyle(4)
# func_bw_30_50.SetLineColor(ROOT.kGray+3)
# graph_h3l_30_50_run3.Draw("PE0")
# graph_h3l_30_50_run2.Draw("PE1 SAME")
# graph_h3l_30_50_run3_BDT.Draw("PE1 SAME")
# func_bw_30_50.Draw("SAME")
# legend_30_50 = utils.get_legend(0.65,0.6,0.85,0.88,0,0,42,0.02,1)
# legend_30_50.AddEntry(text_placeholder,'ALICE Run2: #sqrt{#it{s_{NN}}} = 5.02 TeV, Pb-Pb','')
# legend_30_50.AddEntry(text_placeholder,'ALICE Run3: #sqrt{#it{s_{NN}}} = 5.36 TeV, Pb-Pb','')
# legend_30_50.AddEntry(graph_h3l_30_50_run3,'Run3 {}^{3}_{#Lambda}H spectrum(topology), 30-50%','lep')
# legend_30_50.AddEntry(graph_h3l_30_50_run2,'Run2 {}^{3}_{#Lambda}H spectrum, 30-50%','lep')
# legend_30_50.AddEntry(graph_h3l_30_50_run3_BDT,'Run3 {}^{3}_{#Lambda}H spectrum(BDT), 30-50% BDT','lep')
# legend_30_50.AddEntry(func_bw_30_50,'Blast Wave Fit for Run2, 30-50%','l')
# legend_30_50.Draw("SAME")
# utils.draw_my_text_ndc(0.2,0.2,0.05,'30-50%',2)
# canvas_30_50.SaveAs(output_path_base + f'/spec_h3l_30_50.pdf')

# #graph_h3l_50_80_run3_BDT_shifted = shift_hist_x(graph_h3l_50_80_run3_BDT, 0.3)
# canvas_50_80 = utils.get_canvas(f'spec_h3l_50_80',600,400,0,0,0.1,0.1,0.1,0.1)
# canvas_50_80.SetLogy()
# canvas_50_80.SetFillColor(10)
# utils.set_title_th1(graph_h3l_50_80_run3, None, 0.04, 1.07, None, 0.04, 1, 0, 0)
# utils.set_axis_th1(graph_h3l_50_80_run3,None,None,None,None,0.03,0.03)
# utils.set_marker_th1(graph_h3l_50_80_run3,"",23,1.8,ROOT.kViolet+2,ROOT.kViolet+2)
# utils.set_marker_th1(graph_h3l_50_80_run3_BDT,"",59,1.8,ROOT.kViolet+2,ROOT.kViolet+2)
# graph_h3l_50_80_run3_BDT.GetListOfFunctions().Clear()
# graph_h3l_50_80_run3.Draw("PE0")
# graph_h3l_50_80_run3_BDT.Draw("PE1 SAME")
# legend_50_80 = utils.get_legend(0.5,0.6,0.85,0.88,0,0,42,0.03,1)
# legend_50_80.AddEntry(text_placeholder,'ALICE Run3: #sqrt{#it{s_{NN}}} = 5.36 TeV, Pb-Pb','')
# legend_50_80.AddEntry(graph_h3l_50_80_run3,'Run3 {}^{3}_{#Lambda}H spectrum(topology), 50-80%','lep')
# legend_50_80.AddEntry(graph_h3l_50_80_run3_BDT,'Run3 {}^{3}_{#Lambda}H spectrum(BDT), 50-80% BDT','lep')
# legend_50_80.Draw("SAME")
# utils.draw_my_text_ndc(0.2,0.2,0.05,'50-80%',2)
# canvas_50_80.SaveAs(output_path_base + f'/spec_h3l_50_80.pdf')
