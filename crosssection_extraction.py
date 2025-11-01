import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setSilentMode(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(0)
kOrangeC  = ROOT.TColor.GetColor('#ff7f00')

import os
import sys
sys.path.append('absorption_study')
from absorption_study.draw_absorption import MixedAbsorptionCalculator as mix_abs_calc
import numpy as np
import uproot
import argparse
import yaml
import re
sys.path.append('utils')
from utils import utils
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Configure the parameters of the script.')
parser.add_argument('--config-file', dest='config_file', help="path to the YAML file with configuration.", default='')
args = parser.parse_args()
if args.config_file == "":
    print('** No config file provided. Exiting. **')
    exit()

config_file = open(args.config_file, 'r')
config = yaml.full_load(config_file)

ct_extraction_file = config['ct_extraction_file']
pt_bins = config['pt_bins']
absorption_tree_file_list = config['absorption_tree_file_list']
n_times_list = config['n_times_list']
is_matter = config['is_matter']
out_path = config['out_path']
org_ctao = config['org_ctao']
if not os.path.exists(out_path):
    os.makedirs(out_path)
c_light = 2.99792458e10  # speed of light in cm/s


# Create a ROOT file to store the absorption-corrected histograms
output_file = ROOT.TFile(os.path.join(out_path, "abso_corrected_hists.root"), "RECREATE")

# we read all absorption trees first
trees_n_times_list = []
for path in absorption_tree_file_list:
    tree = uproot.open(path)['he3candidates'].arrays(library="pd")
    trees_n_times_list.append(tree)

#ct_counts hist lists attachment
ct_file = ROOT.TFile(ct_extraction_file, 'READ')
ct_hist_list = []
ct_bins = []
std_dir = ct_file.Get('std')
keys = std_dir.GetListOfKeys()
keys = [key for key in keys if key.IsFolder()]
#pattern = re.compile(r'h_spectrum_pt_([\d\.]+)_([\d]+)')
pattern = re.compile(r'h_spectrum_pt_([\d\.]+)_([\d\.]+)')
for key in keys:
    subdir_name = key.GetName()
    subdir = std_dir.Get(subdir_name)
    subkeys = subdir.GetListOfKeys()
    for subkey in subkeys:
        hist_name = subkey.GetName()
        if pattern.match(hist_name):
            ct_hist = subdir.Get(hist_name)
            ct_hist_list.append(ct_hist)
            ct_centers, spec, spec_error, ct_edges = utils.extract_info_TH1(ct_hist)
            ct_bins.append(ct_edges)

abso_corrected_hist_list = []
abso_eff_hist_list = []
mac = mix_abs_calc(abso_tree=None, pt_bins=pt_bins, ct_bins=ct_bins, org_ctao=org_ctao)
for i_n_times in range(len(n_times_list)):
    mac.abso_tree = trees_n_times_list[i_n_times]
    mac.calculate_absorption()
    for i_pt in range(len(pt_bins)-1):
        cloned_hist = ct_hist_list[i_pt].Clone()
        cloned_hist.SetName(cloned_hist.GetName() + "_abso_corrected_x{}".format(n_times_list[i_n_times]))
        abso_corrected_hist_list.append(cloned_hist)
        abso_corrected_hist_list[-1].Divide(mac.h_he_ratio_absorb_list[i_pt][is_matter])
        eff_cloned_hist = mac.h_he_ratio_absorb_list[i_pt][is_matter].Clone()
        eff_cloned_hist.SetName(eff_cloned_hist.GetName() + "_abso_eff_x{}".format(n_times_list[i_n_times]))
        abso_eff_hist_list.append(eff_cloned_hist)
        # save the efficiency histogram to a PDF for checking
        n_times = n_times_list[i_n_times]
        canvas_name = f"c_eff_pt{i_pt}_ntimes{n_times}"
        c_eff = ROOT.TCanvas(canvas_name, canvas_name, 800, 600)
        mac.h_he_ratio_absorb_list[i_pt][is_matter].SetTitle(f"Efficiency pt {pt_bins[i_pt]}-{pt_bins[i_pt+1]}, n_times {n_times}")
        mac.h_he_ratio_absorb_list[i_pt][is_matter].GetXaxis().SetTitle("ct [cm]")
        mac.h_he_ratio_absorb_list[i_pt][is_matter].GetYaxis().SetTitle("Efficiency")
        mac.h_he_ratio_absorb_list[i_pt][is_matter].Draw("E")
        out_pdf = os.path.join(out_path, f"eff_hist_pt{i_pt}_ntimes{n_times}.pdf")
        c_eff.SaveAs(out_pdf)
        c_eff.Close()
        # x_center, x_values, x_errors, bin_edges = utils.extract_info_TH1(cloned_hist)
        # print(x_values)
        # x_center, x_values, x_errors, bin_edges = utils.extract_info_TH1(mac.h_he_ratio_absorb_list[i_pt][is_matter])
        # print("divider:",x_values)
        # x_center, x_values, x_errors, bin_edges = utils.extract_info_TH1(abso_corrected_hist_list[-1])
        # print("after divide:",x_values)

# Write abso_corrected_hist_list into output_file
output_dir = output_file.mkdir("abso_corrected")
output_dir.cd()
for i_pt in range(len(pt_bins)-1):
    pt_subdir = output_dir.mkdir(f"pt_{pt_bins[i_pt]}_{pt_bins[i_pt+1]}")
    pt_subdir.cd()
    for i_n_times, n_times in enumerate(n_times_list):
        hist_idx = i_n_times * (len(pt_bins)-1) + i_pt
        hist = abso_corrected_hist_list[hist_idx]
        hist.Write()

# extract ctao from abso_corrected_hist_list
tao_vs_n_times = [[] for _ in range(len(pt_bins)-1)]
tao_err_vs_n_times = [[] for _ in range(len(pt_bins)-1)]
for i_n_times, n_times in enumerate(n_times_list):
    for i_pt in range(len(pt_bins)-1):
        hist_idx = i_n_times * (len(pt_bins)-1) + i_pt
        hist = abso_corrected_hist_list[hist_idx]
        fit_result = hist.Fit("expo", "I", "", hist.GetXaxis().GetXmin(), hist.GetXaxis().GetXmax())
        fit_func = hist.GetFunction("expo")
        if fit_func:
            # expo: f(x) = A * exp(B*x), B = -1/ctau
            B = fit_func.GetParameter(1)
            B_error = fit_func.GetParError(1)
            if B != 0:
                tao = -1.0 / B / (c_light * 1e-12)  # c_light in cm/s, convert to ps
                tao_err = abs(B_error / (B * B)) / (c_light * 1e-12)
            else:
                tao = 0
                tao_err = 0
            tao_vs_n_times[i_pt].append(tao)
            tao_err_vs_n_times[i_pt].append(tao_err)
# Draw fit detail plots for each pt_bin and n_times
for i_n_times, n_times in enumerate(n_times_list):
    for i_pt in range(len(pt_bins)-1):
        hist_idx = i_n_times * (len(pt_bins)-1) + i_pt
        hist = abso_corrected_hist_list[hist_idx]
        fit_func = hist.GetFunction("expo")
        
        # Draw refined ROOT canvas with 3 sub-pads and save to ROOT file and PDF
        canvas_name_refined = f"c_refined_pt{i_pt}_ntimes{n_times}"
        c_refined = ROOT.TCanvas(canvas_name_refined, canvas_name_refined, 900, 900)

        # pads: top (integral counts), bottom-left (difference), bottom-right (difference/value)
        p_top = ROOT.TPad("p_top", "p_top", 0.0, 0.35, 1.0, 1.0)
        p_bl  = ROOT.TPad("p_bl",  "p_bl",  0.0, 0.0,  0.5, 0.35)
        p_br  = ROOT.TPad("p_br",  "p_br",  0.5, 0.0,  1.0, 0.35)
        for p in (p_top, p_bl, p_br):
            p.SetLeftMargin(0.12)
            p.SetRightMargin(0.05)
            p.SetTopMargin(0.08)
            p.SetBottomMargin(0.12)
            p.Draw()

        # prepare arrays from histogram
        nbins = hist.GetNbinsX()
        bin_centers = []
        bin_widths = []
        integral_counts = []
        integral_errors = []
        fit_integrals = []
        diffs = []
        rel_diffs = []

        for ib in range(1, nbins+1):
            xlow = hist.GetBinLowEdge(ib)
            xhigh = xlow + hist.GetBinWidth(ib)
            center = 0.5*(xlow + xhigh)
            bw = hist.GetBinWidth(ib)
            content = hist.GetBinContent(ib)
            cerr = hist.GetBinError(ib)

            bin_centers.append(center)
            bin_widths.append(bw)
            integral_counts.append(content * bw)
            integral_errors.append(cerr * bw)

            if fit_func:
                fit_int = fit_func.Integral(xlow, xhigh)
            else:
                fit_int = 0.0
            fit_integrals.append(fit_int)

            diff = content * bw - fit_int
            diffs.append(diff)
            rel = diff / (content * bw) if (content * bw) != 0 else 0.0
            rel_diffs.append(rel)

        # Top pad: integral counts (data points with errors) and fit integral (line)
        p_top.cd()
        g_counts = ROOT.TGraphErrors()
        g_fit_int = ROOT.TGraph()
        for i in range(nbins):
            g_counts.SetPoint(i, bin_centers[i], integral_counts[i])
            g_counts.SetPointError(i, bin_widths[i]/2.0, integral_errors[i])
            g_fit_int.SetPoint(i, bin_centers[i], fit_integrals[i])

        g_counts.SetMarkerStyle(20)
        g_counts.SetMarkerSize(0.9)
        g_counts.SetLineWidth(1)
        g_counts.GetXaxis().SetTitle("ct [cm]")
        g_counts.GetYaxis().SetTitle("Integral counts (bin*width)")
        g_counts.SetTitle(f"Integral counts pt {pt_bins[i_pt]}-{pt_bins[i_pt+1]}, n_times {n_times}")
        g_counts.Draw("AP")
        g_fit_int.SetLineColor(ROOT.kRed)
        g_fit_int.SetLineWidth(2)
        g_fit_int.Draw("L SAME")

        leg = ROOT.TLegend(0.65, 0.7, 0.88, 0.88)
        leg.AddEntry(g_counts, "Data integral (bin*width)", "p")
        leg.AddEntry(g_fit_int, "Fit integral", "l")
        leg.SetFillStyle(0)
        leg.Draw()

        # Bottom-left: absolute difference
        p_bl.cd()
        g_diff = ROOT.TGraphErrors()
        for i in range(nbins):
            g_diff.SetPoint(i, bin_centers[i], diffs[i])
            g_diff.SetPointError(i, 0.0, integral_errors[i])  # use integral error as y-error
        g_diff.SetMarkerStyle(21)
        g_diff.SetMarkerSize(0.8)
        g_diff.SetLineColor(ROOT.kBlue)
        g_diff.SetTitle("Difference: (bin*width) - fit_integral")
        g_diff.GetXaxis().SetTitle("ct [cm]")
        g_diff.GetYaxis().SetTitle("Difference (counts)")
        g_diff.Draw("AP")
        # draw zero line
        zero = ROOT.TLine(hist.GetXaxis().GetXmin(), 0.0, hist.GetXaxis().GetXmax(), 0.0)
        zero.SetLineStyle(2)
        zero.Draw("SAME")

        # Bottom-right: relative difference
        p_br.cd()
        g_reldiff = ROOT.TGraph()
        for i in range(nbins):
            g_reldiff.SetPoint(i, bin_centers[i], rel_diffs[i])
        g_reldiff.SetMarkerStyle(22)
        g_reldiff.SetMarkerSize(0.8)
        g_reldiff.SetLineColor(ROOT.kGreen+2)
        g_reldiff.SetTitle("Relative difference: (bin*width - fit)/ (bin*width)")
        g_reldiff.GetXaxis().SetTitle("ct [cm]")
        g_reldiff.GetYaxis().SetTitle("Relative difference")
        g_reldiff.Draw("AP")
        # horizontal line at 0
        zero_rel = ROOT.TLine(hist.GetXaxis().GetXmin(), 0.0, hist.GetXaxis().GetXmax(), 0.0)
        zero_rel.SetLineStyle(2)
        zero_rel.Draw("SAME")

        # finalize and save
        c_refined.Update()
        pdf_name = os.path.join(out_path, f"refined_fit_pt{i_pt}_ntimes{n_times}.pdf")
        c_refined.SaveAs(pdf_name)

        # Draw histogram with fit overlaid on a single pad
        canvas_name_hist = f"c_histfit_pt{i_pt}_ntimes{n_times}"
        c_hist = ROOT.TCanvas(canvas_name_hist, canvas_name_hist, 800, 500)
        c_hist.SetLeftMargin(0.12)
        c_hist.SetRightMargin(0.05)
        c_hist.SetTopMargin(0.08)
        c_hist.SetBottomMargin(0.12)
        c_hist.SetLogy()
        c_hist.SetGridx()
        c_hist.SetGridy()

        # Ensure we have a fit function; if not, perform the fit
        if not fit_func:
            hist.Fit("expo", "I", "", hist.GetXaxis().GetXmin(), hist.GetXaxis().GetXmax())
            fit_func = hist.GetFunction("expo")

        # Draw histogram and fit on the same pad
        hist.SetTitle(f"Abso-corrected ct spectrum pt {pt_bins[i_pt]}-{pt_bins[i_pt+1]}, n_times {n_times}")
        hist.GetXaxis().SetTitle("ct [cm]")
        hist.GetYaxis().SetTitle("Counts")
        hist.SetStats(0)
        hist.Draw("E")

        if fit_func:
            fit_func.SetLineColor(ROOT.kRed)
            fit_func.SetLineWidth(2)
            fit_func.Draw("SAME")

            # show fit parameters and c*tau
            B = fit_func.GetParameter(1)
            B_err = fit_func.GetParError(1)
            if B != 0:
                tao_ps = -1.0 / B / (c_light * 1e-12)
                tao_err_ps = abs(B_err / (B * B)) / (c_light * 1e-12)
            else:
                tao_ps = 0.0
                tao_err_ps = 0.0
            
            # show c*tau and fit chi2 in a TPaveText
            chi2 = fit_func.GetChisquare() if fit_func else 0.0
            ndf = fit_func.GetNDF() if fit_func else 0
            pave = ROOT.TPaveText(0.15, 0.15, 0.5, 0.35, 'NDC')
            pave.SetFillStyle(0)
            pave.SetBorderSize(0)
            pave.SetTextAlign(12)
            pave.SetTextSize(0.035)
            pave.AddText(f"#tau = {tao_ps:.2f} ± {tao_err_ps:.2f} ps")
            pave.AddText(f"#chi^{{2}}/ndf = {chi2:.2f}/{ndf}")
            pave.Draw()
        leg_hist = ROOT.TLegend(0.70, 0.72, 0.95, 0.9)
        leg_hist.SetBorderSize(0)
        leg_hist.SetFillStyle(0)
        leg_hist.SetTextSize(0.035)
        leg_hist.SetTextAlign(12)
        leg_hist.AddEntry(hist, "Data (counts)", "lep")
        if fit_func:
            leg_hist.AddEntry(fit_func, "Exponential fit", "l")
        leg_hist.Draw()
        c_hist.Update()
        pdf_name_hist = os.path.join(out_path, f"hist_fit_pt{i_pt}_ntimes{n_times}.pdf")
        c_hist.SaveAs(pdf_name_hist)
        # write canvas into open ROOT output file
        try:
            # cd to abso_corrected/pt_{...} inside output_file so canvas is written into the corresponding subdir
            abso_dir = output_file.Get("abso_corrected")
            subdir_name = f"pt_{pt_bins[i_pt]}_{pt_bins[i_pt+1]}"
            pt_dir = abso_dir.Get(subdir_name)
            pt_dir.cd()
            c_refined.Write()
            c_hist.Write()
        except Exception:
            # if writing fails, continue without raising
            pass
        plt.figure()
        # Get bin centers and contents
        bin_edges = [hist.GetBinLowEdge(i+1) for i in range(hist.GetNbinsX()+1)]
        bin_centers = [(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
        bin_contents = [hist.GetBinContent(i+1) for i in range(hist.GetNbinsX())]
        bin_errors = [hist.GetBinError(i+1) for i in range(hist.GetNbinsX())]
        plt.errorbar(bin_centers, bin_contents, yerr=bin_errors, fmt='o', label='Data')
        # Plot fit function if available
        if fit_func:
            x_fit = np.linspace(hist.GetXaxis().GetXmin(), hist.GetXaxis().GetXmax(), 200)
            y_fit = [fit_func.Eval(x) for x in x_fit]
            plt.plot(x_fit, y_fit, 'r-', label='Expo Fit')
            fit_params = [fit_func.GetParameter(i) for i in range(fit_func.GetNpar())]
            fit_errors = [fit_func.GetParError(i) for i in range(fit_func.GetNpar())]
            chi2 = fit_func.GetChisquare()
            ndf = fit_func.GetNDF()
            ctao_ps = tao_vs_n_times[i_pt][i_n_times]  # already in ps
            fit_text = (
                f'$c\\tau$={ctao_ps:.2f} ± {fit_errors[1] / abs(fit_params[1]) * ctao_ps:.2f} ps\n'
                f'$\chi^2$/ndf={chi2:.2f}/{ndf}'
            )
            plt.annotate(fit_text, xy=(0.25, 0.35), xycoords='axes fraction', fontsize=10,
                         verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))
        plt.xlabel('ct [cm]')
        plt.ylabel('Counts')
        plt.yscale('log')
        plt.title(f'Fit for pt: {pt_bins[i_pt]}-{pt_bins[i_pt+1]}, n_times: {n_times}')
        plt.grid(True)
        plt.legend()
        plot_name = f'fit_detail_ptbin{i_pt}_ntimes{n_times}.pdf'
        plt.savefig(os.path.join(out_path, plot_name))
        plt.close()
# Plot tao vs n_times for each pt bin
plt.figure()
# for i_pt in range(len(pt_bins)-1):
#     plt.errorbar(n_times_list, tao_vs_n_times[i_pt], yerr=tao_err_vs_n_times[i_pt], marker='o', label=f'pt: {pt_bins[i_pt]}-{pt_bins[i_pt+1]}')
# Shift each pt points a little bit for better visualization
offset = 0.05
for i_pt in range(len(pt_bins)-1):
    shifted_n_times = [n + (i_pt - (len(pt_bins)-2)/2) * offset for n in n_times_list]
    plt.errorbar(shifted_n_times, tao_vs_n_times[i_pt], yerr=tao_err_vs_n_times[i_pt], marker='o', label=f'pt: {pt_bins[i_pt]}-{pt_bins[i_pt+1]}')
plt.xlabel('n_times')
plt.ylabel('Extracted $tau$ [ps]')
plt.title('Extracted $tau$ vs n_times for all pt bins')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(out_path, 'tao_vs_ntimes_all_ptbins.pdf'))
plt.close()



# Write original ct_hist_list in a separate directory
ct_dir = output_file.mkdir("original_ct_hists")
ct_dir.cd()
for hist in ct_hist_list:
    hist.Write()
# Write h_he_ratio_absorb_list histos in a separate directory
he_ratio_dir = output_file.mkdir("h_he_ratio_absorb")
he_ratio_dir.cd()
for i_n_times in range(len(n_times_list)):
    for i_pt in range(len(pt_bins)-1):
        hist = abso_eff_hist_list[i_n_times * (len(pt_bins)-1) + i_pt]
        hist.Write()
output_file.Close()
    

for hist in ct_hist_list:
    print(hist.GetName())
for hist in abso_corrected_hist_list:
    print(hist.GetName())
