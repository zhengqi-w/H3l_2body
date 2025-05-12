import yaml
import argparse
import uproot
import numpy as np
import os
import ROOT
ROOT.gStyle.SetOptStat(0)
import math
import array

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
h3l_0_5_run3_file_BDT = ROOT.TFile.Open("../../results/ep4/spec_trainboth/cen0-5/pt_analysis_pbpb.root.root")
h3l_5_10_run3_file_BDT = ROOT.TFile.Open("../../results/ep4/spec_trainboth/cen5-10/pt_analysis_pbpb.root.root")
h3l_10_30_run3_file_BDT = ROOT.TFile.Open("../../results/ep4/spec_trainboth/cen10-30/pt_analysis_pbpb.root.root")
h3l_30_50_run3_file_BDT = ROOT.TFile.Open("../../results/ep4/spec_trainboth/cen30-50/pt_analysis_pbpb.root.root")
h3l_50_80_run3_file_BDT = ROOT.TFile.Open("../../results/ep4/spec_trainboth/cen50-80/pt_analysis_pbpb.root.root")

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

func_bw_0_10 = find_graph_by_name(BW_file,'BlastWave_H3L_0_10')
func_bw_10_30 = find_graph_by_name(BW_file,'BlastWave_H3L_10_30')
func_bw_30_50 = find_graph_by_name(BW_file,'BlastWave_H3L_30_50')

canvas_0_5 = utils.get_canvas(f'spec_h3l_0_5',600,400,0,0,0.1,0.1,0.1,0.1)
canvas_0_5.SetLogy()
canvas_0_5.SetFillColor(10)
utils.set_title_th1(graph_h3l_0_5_run3, None, 0.04, 1.07, None, 0.04, 1, 0, 0)
utils.set_axis_th1(graph_h3l_0_5_run3,None,None,None,None,0.03,0.03)
utils.set_marker_th1(graph_h3l_0_5_run3,"",20,1.5,ROOT.kPink-2,ROOT.kPink-2)
utils.set_marker_th1(graph_h3l_0_10_run2,"",54,1.5,ROOT.kAzure+2,ROOT.kAzure+2)
utils.set_marker_th1(graph_h3l_0_5_run3_BDT,"",22,1.8,ROOT.kOrange+7,ROOT.kOrange+7)
graph_h3l_0_5_run3_BDT.GetListOfFunctions().Clear()
func_bw_0_10.SetLineStyle(4)
func_bw_0_10.SetLineColor(ROOT.kGray+3)
graph_h3l_0_5_run3.Draw("PE0")
graph_h3l_0_10_run2.Draw("PE1 SAME")
graph_h3l_0_5_run3_BDT.Draw("PE1 SAME")
func_bw_0_10.Draw("SAME")
legend_0_5 = utils.get_legend(0.65,0.6,0.85,0.88,0,0,42,0.02,1)
legend_0_5.AddEntry(text_placeholder,'ALICE Run2: #sqrt{#it{s_{NN}}} = 5.02 TeV, Pb-Pb','')
legend_0_5.AddEntry(text_placeholder,'ALICE Run3: #sqrt{#it{s_{NN}}} = 5.36 TeV, Pb-Pb','')
legend_0_5.AddEntry(graph_h3l_0_5_run3,'Run3 {}^{3}_{#Lambda}H spectrum(topology), 0-5%','lep')
legend_0_5.AddEntry(graph_h3l_0_10_run2,'Run2 {}^{3}_{#Lambda}H spectrum, 0-10%','lep')
legend_0_5.AddEntry(graph_h3l_0_5_run3_BDT,'Run3 {}^{3}_{#Lambda}H spectrum(BDT), 0-5% BDT','lep')
legend_0_5.AddEntry(func_bw_0_10,'Blast Wave Fit for Run2, 0-10%','l')
legend_0_5.Draw("SAME")
utils.draw_my_text_ndc(0.2,0.2,0.05,'0-5%',2)
canvas_0_5.SaveAs(output_path_base + f'/spec_h3l_0_5.pdf')

canvas_5_10 = utils.get_canvas(f'spec_h3l_5_10',600,400,0,0,0.1,0.1,0.1,0.1)
canvas_5_10.SetLogy()
canvas_5_10.SetFillColor(10)
utils.set_title_th1(graph_h3l_5_10_run3, None, 0.04, 1.07, None, 0.04, 1, 0, 0)
utils.set_axis_th1(graph_h3l_5_10_run3,None,None,None,None,0.03,0.03)
utils.set_marker_th1(graph_h3l_5_10_run3,"",20,1.5,ROOT.kPink-2,ROOT.kPink-2)
utils.set_marker_th1(graph_h3l_0_10_run2,"",54,1.5,ROOT.kAzure+2,ROOT.kAzure+2)
utils.set_marker_th1(graph_h3l_5_10_run3_BDT,"",22,1.8,ROOT.kOrange+7,ROOT.kOrange+7)
graph_h3l_5_10_run3_BDT.GetListOfFunctions().Clear()
func_bw_0_10.SetLineStyle(4)
func_bw_0_10.SetLineColor(ROOT.kGray+3)
graph_h3l_5_10_run3.Draw("PE0")
graph_h3l_0_10_run2.Draw("PE1 SAME")
graph_h3l_5_10_run3_BDT.Draw("PE1 SAME")
func_bw_0_10.Draw("SAME")
legend_5_10 = utils.get_legend(0.65,0.6,0.85,0.88,0,0,42,0.02,1)
legend_5_10.AddEntry(text_placeholder,'ALICE Run2: #sqrt{#it{s_{NN}}} = 5.02 TeV, Pb-Pb','')
legend_5_10.AddEntry(text_placeholder,'ALICE Run3: #sqrt{#it{s_{NN}}} = 5.36 TeV, Pb-Pb','')
legend_5_10.AddEntry(graph_h3l_5_10_run3,'Run3 {}^{3}_{#Lambda}H spectrum(topology), 5-10%','lep')
legend_5_10.AddEntry(graph_h3l_0_10_run2,'Run2 {}^{3}_{#Lambda}H spectrum, 0-10%','lep')
legend_5_10.AddEntry(graph_h3l_5_10_run3_BDT,'Run3 {}^{3}_{#Lambda}H spectrum(BDT), 5-10% BDT','lep')
legend_5_10.AddEntry(func_bw_0_10,'Blast Wave Fit for Run2, 0-10%','l')
legend_5_10.Draw("SAME")
utils.draw_my_text_ndc(0.2,0.2,0.05,'5-10%',2)
canvas_5_10.SaveAs(output_path_base + f'/spec_h3l_5_10.pdf')

canvas_10_30 = utils.get_canvas(f'spec_h3l_10_30',600,400,0,0,0.1,0.1,0.1,0.1)
canvas_10_30.SetLogy()
canvas_10_30.SetFillColor(10)
utils.set_title_th1(graph_h3l_10_30_run3, None, 0.04, 1.07, None, 0.04, 1, 0, 0)
utils.set_axis_th1(graph_h3l_10_30_run3,None,None,None,None,0.03,0.03)
utils.set_marker_th1(graph_h3l_10_30_run3,"",20,1.5,ROOT.kPink-2,ROOT.kPink-2)
utils.set_marker_th1(graph_h3l_10_30_run2,"",54,1.5,ROOT.kAzure+2,ROOT.kAzure+2)
utils.set_marker_th1(graph_h3l_10_30_run3_BDT,"",22,1.8,ROOT.kOrange+7,ROOT.kOrange+7)
graph_h3l_10_30_run3_BDT.GetListOfFunctions().Clear()
func_bw_10_30.SetLineStyle(4)
func_bw_10_30.SetLineColor(ROOT.kGray+3)
graph_h3l_10_30_run3.Draw("PE0")
graph_h3l_10_30_run2.Draw("PE1 SAME")
graph_h3l_10_30_run3_BDT.Draw("PE1 SAME")
func_bw_10_30.Draw("SAME")
legend_10_30 = utils.get_legend(0.65,0.6,0.85,0.88,0,0,42,0.02,1)
legend_10_30.AddEntry(text_placeholder,'ALICE Run2: #sqrt{#it{s_{NN}}} = 5.02 TeV, Pb-Pb','')
legend_10_30.AddEntry(text_placeholder,'ALICE Run3: #sqrt{#it{s_{NN}}} = 5.36 TeV, Pb-Pb','')
legend_10_30.AddEntry(graph_h3l_10_30_run3,'Run3 {}^{3}_{#Lambda}H spectrum(topology), 10-30%','lep')
legend_10_30.AddEntry(graph_h3l_10_30_run2,'Run2 {}^{3}_{#Lambda}H spectrum, 10-30%','lep')
legend_10_30.AddEntry(graph_h3l_10_30_run3_BDT,'Run3 {}^{3}_{#Lambda}H spectrum(BDT), 10-30% BDT','lep')
legend_10_30.AddEntry(func_bw_10_30,'Blast Wave Fit for Run2, 10-30%','l')
legend_10_30.Draw("SAME")
utils.draw_my_text_ndc(0.2,0.2,0.05,'10-30%',2)
canvas_10_30.SaveAs(output_path_base + f'/spec_h3l_10_30.pdf')


canvas_30_50 = utils.get_canvas(f'spec_h3l_30_50',600,400,0,0,0.1,0.1,0.1,0.1)
canvas_30_50.SetLogy()
canvas_30_50.SetFillColor(10)
utils.set_title_th1(graph_h3l_30_50_run3, None, 0.04, 1.07, None, 0.04, 1, 0, 0)
utils.set_axis_th1(graph_h3l_30_50_run3,None,None,None,None,0.03,0.03)
utils.set_marker_th1(graph_h3l_30_50_run3,"",20,1.5,ROOT.kPink-2,ROOT.kPink-2)
utils.set_marker_th1(graph_h3l_30_50_run2,"",54,1.5,ROOT.kAzure+2,ROOT.kAzure+2)
utils.set_marker_th1(graph_h3l_30_50_run3_BDT,"",22,1.8,ROOT.kOrange+7,ROOT.kOrange+7)
graph_h3l_30_50_run3_BDT.GetListOfFunctions().Clear()
func_bw_30_50.SetLineStyle(4)
func_bw_30_50.SetLineColor(ROOT.kGray+3)
graph_h3l_30_50_run3.Draw("PE0")
graph_h3l_30_50_run2.Draw("PE1 SAME")
graph_h3l_30_50_run3_BDT.Draw("PE1 SAME")
func_bw_30_50.Draw("SAME")
legend_30_50 = utils.get_legend(0.65,0.6,0.85,0.88,0,0,42,0.02,1)
legend_30_50.AddEntry(text_placeholder,'ALICE Run2: #sqrt{#it{s_{NN}}} = 5.02 TeV, Pb-Pb','')
legend_30_50.AddEntry(text_placeholder,'ALICE Run3: #sqrt{#it{s_{NN}}} = 5.36 TeV, Pb-Pb','')
legend_30_50.AddEntry(graph_h3l_30_50_run3,'Run3 {}^{3}_{#Lambda}H spectrum(topology), 30-50%','lep')
legend_30_50.AddEntry(graph_h3l_30_50_run2,'Run2 {}^{3}_{#Lambda}H spectrum, 30-50%','lep')
legend_30_50.AddEntry(graph_h3l_30_50_run3_BDT,'Run3 {}^{3}_{#Lambda}H spectrum(BDT), 30-50% BDT','lep')
legend_30_50.AddEntry(func_bw_30_50,'Blast Wave Fit for Run2, 30-50%','l')
legend_30_50.Draw("SAME")
utils.draw_my_text_ndc(0.2,0.2,0.05,'30-50%',2)
canvas_30_50.SaveAs(output_path_base + f'/spec_h3l_30_50.pdf')

#graph_h3l_50_80_run3_BDT_shifted = shift_hist_x(graph_h3l_50_80_run3_BDT, 0.3)
canvas_50_80 = utils.get_canvas(f'spec_h3l_50_80',600,400,0,0,0.1,0.1,0.1,0.1)
canvas_50_80.SetLogy()
canvas_50_80.SetFillColor(10)
utils.set_title_th1(graph_h3l_50_80_run3, None, 0.04, 1.07, None, 0.04, 1, 0, 0)
utils.set_axis_th1(graph_h3l_50_80_run3,None,None,None,None,0.03,0.03)
utils.set_marker_th1(graph_h3l_50_80_run3,"",23,1.8,ROOT.kViolet+2,ROOT.kViolet+2)
utils.set_marker_th1(graph_h3l_50_80_run3_BDT,"",59,1.8,ROOT.kViolet+2,ROOT.kViolet+2)
graph_h3l_50_80_run3_BDT.GetListOfFunctions().Clear()
graph_h3l_50_80_run3.Draw("PE0")
graph_h3l_50_80_run3_BDT.Draw("PE1 SAME")
legend_50_80 = utils.get_legend(0.5,0.6,0.85,0.88,0,0,42,0.03,1)
legend_50_80.AddEntry(text_placeholder,'ALICE Run3: #sqrt{#it{s_{NN}}} = 5.36 TeV, Pb-Pb','')
legend_50_80.AddEntry(graph_h3l_50_80_run3,'Run3 {}^{3}_{#Lambda}H spectrum(topology), 50-80%','lep')
legend_50_80.AddEntry(graph_h3l_50_80_run3_BDT,'Run3 {}^{3}_{#Lambda}H spectrum(BDT), 50-80% BDT','lep')
legend_50_80.Draw("SAME")
utils.draw_my_text_ndc(0.2,0.2,0.05,'50-80%',2)
canvas_50_80.SaveAs(output_path_base + f'/spec_h3l_50_80.pdf')
