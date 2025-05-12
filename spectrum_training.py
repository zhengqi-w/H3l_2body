import yaml
import argparse
import uproot
import numpy as np
import os
import ROOT
from spectra import SpectraMaker
from hipe4ml.tree_handler import TreeHandler
from itertools import product
import copy
import pandas as pd
ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setSilentMode(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(0)
kOrangeC  = ROOT.TColor.GetColor('#ff7f00')

import sys
sys.path.append('utils')
import utils as utils

parser = argparse.ArgumentParser(description='Configure the parameters of the script.')
parser.add_argument('--config-file', dest='config_file', help="path to the YAML file with configuration.", default='')
args = parser.parse_args()
if args.config_file == "":
    print('** No config file provided. Exiting. **')
    exit()

config_file = open(args.config_file, 'r')
config = yaml.full_load(config_file)

input_training_df_path = config['input_training_df_path']
input_mc_file_path = config['input_mc_file_path']
input_AnalysisResults_file_path = config['input_AnalysisResults_file_path']
input_H3l_BWFit_file_path = config['input_H3l_BWFit_file_path']
absorption_histo_file = config['absorption_histo_file']
output_file_path = config['output_file_path']
output_spectrm_file_name = config['output_spectrm_file_name']

cen_bins = config['cen_bins']
pt_bins = config['pt_bins']
is_matter = config['is_matter']
bdt_numbers_syst = config['bdt_numbers_syst']


bkg_fit_func = config['bkg_fit_func']
signal_fit_func = config['signal_fit_func']
sigma_range_mc_to_data = config['sigma_range_mc_to_data']

do_syst = config['do_syst']
absorption_syst_array = config['absorption_syst']


### create output file path 
output_file_path += f'{is_matter}/'
if cen_bins:
    output_file_path += f'cen{cen_bins[0]}-{cen_bins[1]}'
else:
    output_file_path += 'cenall'
if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

### calculate event numbers for the given centrality
cen_n_env = utils.getNEvents(input_AnalysisResults_file_path,False,cen_bins[0],cen_bins[1])
print(f'Number of events cen {cen_bins[0]}-{cen_bins[1]}: ', cen_n_env)

### absorption 
if absorption_histo_file != '':
    absorption_file = ROOT.TFile.Open(absorption_histo_file)
    absorption_histo_mat = absorption_file.Get('h_abso_frac_pt_mat')
    absorption_histo_anti = absorption_file.Get('h_abso_frac_pt_antimat')
    absorption_histo_mat.SetDirectory(0)
    absorption_histo_anti.SetDirectory(0)
    absorption_histo = absorption_histo_mat.Clone('h_abso_frac_pt')
    absorption_histo.Add(absorption_histo_anti)# gte average between matter and antimatter
    absorption_histo.Scale(0.5)

### determine the BDT working point 
exp_significance_list = []
BDT_cut_list = []
efficency_list = []
BDT_score_syst_list = []
BDT_efficiency_syst_list = []
for i in range(len(pt_bins)-1):
    df_bin = pd.read_csv(f"{input_training_df_path}/cen_{cen_bins[0]}_{cen_bins[1]}_pt_{pt_bins[i]}_{pt_bins[i+1]}/df_working_point.csv")
    df_bin['finder'] = df_bin['exp_significance'] * df_bin['BDT_efficiency']
    max_id = df_bin['finder'].idxmax()
    max_exp_significance = df_bin['exp_significance'].iloc[max_id]
    exp_significance_list.append(max_exp_significance)
    BDT_cut_list.append(df_bin['BDT_score'].iloc[max_id])
    efficency_list.append(df_bin['BDT_efficiency'].iloc[max_id])
    ### calculate the +- 10% efficiency range for systematics
    start_idx = max(0, max_id - bdt_numbers_syst // 2)
    end_idx = start_idx + bdt_numbers_syst
    if end_idx > len(df_bin):
       end_idx = len(df_bin)
       start_idx = end_idx - bdt_numbers_syst
    filtered_df = df_bin[start_idx:end_idx]
    BDT_score_syst_list.append(filtered_df['BDT_score'].tolist())
    BDT_efficiency_syst_list.append(filtered_df['BDT_efficiency'].tolist())
    

### read and merge data hdls after training
data_hdl_list = []
for i in range(len(pt_bins)-1):
    data_hdl_pt_bin =  TreeHandler(f"{input_training_df_path}/cen_{cen_bins[0]}_{cen_bins[1]}_pt_{pt_bins[i]}_{pt_bins[i+1]}/dataH_BDTapplied.parquet.gzip")
    if not data_hdl_pt_bin:
        print(f"** Datahandler file not found for pt bin {pt_bins[i]}-{pt_bins[i+1]} centrality {cen_bins[0]}-{cen_bins[1]}. Exiting. **")
        exit()
    #data_hdl_pt_bin.apply_preselections(f'model_output > {BDT_cut_list[i]}', inplace=True) ### for systematic do bdt cut in the Spectrum Maker
    data_hdl_list.append(data_hdl_pt_bin)
data_hdl = utils.merge_treehandlers(data_hdl_list)

### mc hdl for pt_spectrum 
mc_hdl = TreeHandler(input_mc_file_path, 'O2mchypcands', folder_name='DF*')
utils.correct_and_convert_df(mc_hdl, calibrate_he3_pt = False, isMC=True)
## reweight MC pt spectrum
centrality_bins = [0,10,30,50,80]
df_mc = mc_hdl.get_data_frame()
df_split = []
spectra_file = ROOT.TFile.Open('utils/H3L_BWFit.root')
for i in range(len(centrality_bins)-1):
    cen_min, cen_max = centrality_bins[i], centrality_bins[i+1]
    if cen_min == 50 and cen_max == 80:
        H3l_spectrum = spectra_file.Get(f'BlastWave_H3L_30_50') #use h3lBW_30_50 for 50_80
        df_cen_bin =  df_mc[(df_mc['fCentralityFT0C'] >= 50) & (df_mc['fCentralityFT0C'] < 110)].copy()
    else:
        H3l_spectrum = spectra_file.Get(f'BlastWave_H3L_{cen_min}_{cen_max}')
        df_cen_bin =  df_mc[(df_mc['fCentralityFT0C'] >= cen_min) & (df_mc['fCentralityFT0C'] < cen_max)].copy()
    utils.reweight_pt_spectrum(df_cen_bin, 'fAbsGenPt', H3l_spectrum)
    df_cen_bin = df_cen_bin.query("rej == 1")
    df_split.append(df_cen_bin)
spectra_file.Close()
df_mc = pd.concat(df_split).sort_index()
mc_hdl.set_data_frame(df_mc)
#mc_hdl.apply_preselections('fGenCt < 28.5 or fGenCt > 28.6')
mc_reco_hdl = mc_hdl.apply_preselections('fIsReco == 1', inplace=False)
mc_hdl_evsel = mc_hdl.apply_preselections('fIsSurvEvSel==True', inplace=False)
# print("reco_handler:\n")
# print(mc_reco_hdl.get_data_frame())
# print("evsel_handler:\n")
# print(mc_hdl_evsel.get_data_frame())
# exit()

### start pt analysics make a SpectrumMaker object
output_file = ROOT.TFile.Open(f'{output_file_path}/{output_spectrm_file_name}.root', 'recreate')
output_dir_std = output_file.mkdir('std')
spectra_maker = SpectraMaker()
spectra_maker.data_hdl = data_hdl
spectra_maker.mc_hdl = mc_hdl_evsel
spectra_maker.mc_hdl_sign_extr = mc_hdl
spectra_maker.mc_reco_hdl = mc_reco_hdl
spectra_maker.bins_cen = cen_bins
spectra_maker.n_ev = cen_n_env
spectra_maker.branching_ratio = 0.25
spectra_maker.delta_rap = 2.0
spectra_maker.h_absorption = absorption_histo
spectra_maker.bdt_efficiency = efficency_list
spectra_maker.event_loss = 1 ##PbPb we assume there is no event loss
spectra_maker.signal_loss = 1 ##PbPb we assume there is no signal loss
spectra_maker.var = 'fPt'
spectra_maker.bins = pt_bins
spectra_maker.bdt_cut = BDT_cut_list
spectra_maker.is_matter = is_matter
spectra_maker.inv_mass_signal_func = signal_fit_func
spectra_maker.inv_mass_bkg_func = bkg_fit_func
spectra_maker.sigma_range_mc_to_data = sigma_range_mc_to_data
spectra_maker.output_dir = output_dir_std
fit_range = [pt_bins[0], pt_bins[-1]]
spectra_maker.fit_range = fit_range
spectra_maker.make_spectra()
spectra_maker.make_histos()
# fitfunc for Spectrum maker
h3l_spectrum_func = ROOT.TF1('mtexpo', '[2]*x*exp(-TMath::Sqrt([0]*[0]+x*x)/[1])', 0.1, 8)
h3l_spectrum_func.FixParameter(0, 2.99131)
h3l_spectrum_func.SetParameter(1, 0.5199)
h3l_spectrum_func.SetParameter(2, 1e-06)
h3l_spectrum_func.SetParLimits(2, 1e-08, 5e-04)
h3l_spectrum_func.SetLineColor(kOrangeC)
spectra_maker.fit_func = h3l_spectrum_func
spectra_maker.fit_options = 'MIQ+'
spectra_maker.fit()
spectra_maker.dump_to_output_dir()


### copy the stadard results
std_corrected_counts = copy.deepcopy(spectra_maker.corrected_counts)
std_corrected_counts_err = copy.deepcopy(spectra_maker.corrected_counts_err)
final_stat = copy.deepcopy(spectra_maker.h_corrected_counts)
final_stat.SetName('hStat') #consider the case that final_stat is a vector
utils.setHistStyle(final_stat, ROOT.kAzure + 2)
final_syst = final_stat.Clone('hSyst')
final_syst_rms = final_stat.Clone('hSystRMS')
final_syst_rms.SetLineColor(ROOT.kAzure + 2)
final_syst_rms.SetMarkerColor(ROOT.kAzure + 2)
std_yield = spectra_maker.fit_func.GetParameter(2) # 2 for mT exponential
std_yield_err = spectra_maker.fit_func.GetParError(2)
#delete the dynamic members of SpectrumMaker
spectra_maker.del_dyn_members()
print("** pt analysis done. ** \n")


### systematic uncertainties process
if do_syst:
    # systematic uncertainties
    print("** Starting systematic variations **")
    n_trials = config['n_trials']
    bkg_fit_func_syst = config['bkg_fit_func_syst']
    signal_fit_func_syst = config['signal_fit_func_syst']
    #basic histograms yield variations for each pt bin and final yield distribution
    yield_dist = ROOT.TH1D('hYieldSyst', ';dN/dy ;Counts', 40, 1.e-08, 2.e-08)
    yield_prob = ROOT.TH1D('hYieldProb', ';prob. ;Counts', 100, 0, 1)
    h_pt_syst = []
    for i_bin in range(0, len(pt_bins) - 1):
        bin_label = f'{pt_bins[i_bin]}' + r' #leq #it{p}_{T} < ' f'{pt_bins[i_bin + 1]}' + r' GeV/#it{c}'
        histo = ROOT.TH1D(f'hPtSyst_{i_bin}', f'{bin_label}' + r';d#it{N} / d#it{p}_{T} (GeV/#it{c})^{-1};', 30, 0.5 * std_corrected_counts[i_bin], 2 * std_corrected_counts[i_bin])
        h_pt_syst.append(histo)
    output_dir_syst = output_file.mkdir('trials')
    trial_strings = []
    print("----------------------------------")
    print("** Starting systematics analysis **")
    print(f'** {n_trials} trials will be tested **')
    print("----------------------------------")
    cut_string_dict = {}
    cut_string_dict['signal_fit_func'] = signal_fit_func_syst
    cut_string_dict['bkg_fit_func'] = bkg_fit_func_syst
    combos_list = []
    for pt_idx in range(0, len(pt_bins) - 1):
        cut_string_dict['bdt_cut'] = []
        cut_string_dict['bdt_efficiency'] = []
        for i in range (0, len(BDT_score_syst_list[pt_idx])):
            cut_string_dict['bdt_cut'].append(BDT_score_syst_list[pt_idx][i])
            cut_string_dict['bdt_efficiency'].append(BDT_efficiency_syst_list[pt_idx][i])
        combos = list(product(*list(cut_string_dict.values())))
        combos_list.append(combos)
    ## generate random n_trails x (len(pt_bins) - 1) array of random indices
    if n_trials < len(combos_list[0]):
        combo_random_indices = np.random.randint(len(combos_list[0]), size=(n_trials, len(pt_bins) - 1))
    else:
        print(f"** Warning: n_trials > n_combinations ({n_trials}, {len(combos_list[0])}), taking all the possible combinations **")
        indices = np.arange(len(combos_list[0]))
        # create a (len(combos_list[0]), len(ct_bins) - 1) array with the indices repeated for each ct bin
        combo_random_indices = np.repeat(indices[:, np.newaxis], len(pt_bins) - 1, axis=1)
        # now shuffle each column of the array
        for i in range(combo_random_indices.shape[1]): #combo_random_indices.shape[1] return the number of rows
            np.random.shuffle(combo_random_indices[:, i])
    combo_check_map = {}
    for i_combo, combo_indices in enumerate(combo_random_indices):
        trial_strings.append("----------------------------------")
        trial_num_string = f'Trial: {i_combo} / {len(combo_random_indices)}'
        trial_strings.append(trial_num_string)
        print(trial_num_string)
        print("----------------------------------")
        bdt_cut_list_syst = []
        bdt_efficiency_list_syst = []
        bkg_fit_func_list = []
        signal_fit_func_list = []
        bdt_string_list = []
        bdt_efficiency_string_list = []
        for ipt in range(len(pt_bins) - 1):
            combo = combos_list[ipt][combo_indices[ipt]]
            pt_bin = [pt_bins[ipt], pt_bins[ipt + 1]]
            full_combo_string = f'pt {pt_bin[0]}_{pt_bin[1]} | '
            full_combo_string += " & ".join(map(str, combo))
            bdt_selection_string = f'BDT > {combo[2]}'
            bdt_efficiency_string = f'BDT efficiency: {combo[3]}'
            signal_fit_func = combo[0]
            bkg_fit_func = combo[1]
            if full_combo_string in combo_check_map:
                break
            combo_check_map[full_combo_string] = True
            bdt_cut_list_syst.append(combo[2])
            bdt_efficiency_list_syst.append(combo[3])
            bkg_fit_func_list.append(bkg_fit_func)
            signal_fit_func_list.append(signal_fit_func)
            bdt_string_list.append(bdt_selection_string)
            bdt_efficiency_string_list.append(bdt_efficiency_string)
        if len(bdt_cut_list_syst) != len(pt_bins) - 1:
            continue
        trial_strings.append(str(bdt_string_list))
        trial_strings.append(str(bdt_efficiency_string_list))
        trial_strings.append(str(bkg_fit_func_list))
        trial_strings.append(str(signal_fit_func_list))
        # make Spectrum
        spectra_maker.bdt_cut = bdt_cut_list_syst
        spectra_maker.bdt_efficiency = bdt_efficiency_list_syst
        spectra_maker.inv_mass_signal_func = signal_fit_func_list
        spectra_maker.inv_mass_bkg_func = bkg_fit_func_list
        trial_dir = output_dir_syst.mkdir(f'trial_{i_combo}')
        spectra_maker.output_dir = trial_dir
        spectra_maker.make_spectra()
        spectra_maker.make_histos()
        spectra_maker.fit()
        res_string = "Integral: " + str(spectra_maker.fit_func.Integral(0, 10)) + " Prob: " + str(spectra_maker.fit_func.GetProb())
        trial_strings.append(res_string)
        for i_bin in range(0, len(spectra_maker.bins) - 1):
            h_pt_syst[i_bin].Fill(spectra_maker.corrected_counts[i_bin])
        if spectra_maker.fit_func.GetProb() > 0.05 and spectra_maker.chi2_selection():
            spectra_maker.dump_to_output_dir()
            yield_dist.Fill(spectra_maker.fit_func.Integral(0, 10))
            yield_prob.Fill(spectra_maker.fit_func.GetProb())
        spectra_maker.del_dyn_members()
    # systematic uncetrainty fo each pt bin
    for i_bin in range(0, len(pt_bins) - 1):
        canvas = ROOT.TCanvas(f'cYield_{i_bin}', f'cYield_{i_bin}', 800, 600)
        canvas.SetTopMargin(0.1)
        canvas.SetBottomMargin(0.15)
        canvas.SetLeftMargin(0.08)
        canvas.SetRightMargin(0.08)
        canvas.DrawFrame(0, 0, 2 * std_corrected_counts[i_bin],
                     1.1 * h_pt_syst[i_bin].GetMaximum(), r';d#it{N} / d#it{p}_{T} (GeV/#it{c})^{-1};') ## (xlow, ylow, xup, yup, title)
        # create a line for the standard value of lifetime
        std_line = ROOT.TLine(std_corrected_counts[i_bin], 0, std_corrected_counts[i_bin], 1.05 * h_pt_syst[i_bin].GetMaximum()) ##(x1, y1, x2, y2)
        std_line.SetLineColor(kOrangeC)
        std_line.SetLineWidth(2)
        # create box for statistical uncertainty
        std_errorbox = ROOT.TBox(std_corrected_counts[i_bin] - std_corrected_counts_err[i_bin], 0,
                                 std_corrected_counts[i_bin] + std_corrected_counts_err[i_bin], 1.05 * h_pt_syst[i_bin].GetMaximum())
        std_errorbox.SetFillColorAlpha(kOrangeC, 0.5)
        std_errorbox.SetLineWidth(0)
        # fitting histogram with systematic variations
        fit_func = ROOT.TF1(f'fit_func_{i_bin}', 'gaus', 0.5 *std_corrected_counts[i_bin], 1.5 * std_corrected_counts[i_bin])
        fit_func.SetLineColor(ROOT.kGreen+3)
        h_pt_syst[i_bin].Fit(fit_func, 'Q')
        syst_mu = fit_func.GetParameter(1)
        syst_mu_err = fit_func.GetParError(1)
    
        syst_sigma = fit_func.GetParameter(2)
        ## absorption systematic uncertainty(source from the h3l scattering cross section is unknown standard is 1.5 x he3 cross section)
        if absorption_syst_array != [] and absorption_syst_array is not None:
            syst_sigma = np.sqrt(syst_sigma**2 + (std_corrected_counts[i_bin] * absorption_syst_array[i_bin])**2)
        final_syst.SetBinError(i_bin+1, syst_sigma)
    
        syst_rms = h_pt_syst[i_bin].GetRMS()
        if absorption_syst_array != [] and absorption_syst_array is not None:
            syst_rms = np.sqrt(syst_rms**2 + (std_corrected_counts[i_bin] * absorption_syst_array[i_bin])**2)
        final_syst_rms.SetBinError(i_bin+1, syst_rms)
    
        syst_sigma_err = fit_func.GetParError(2)
        fit_param = ROOT.TPaveText(0.7, 0.6, 0.9, 0.82, 'NDC')
        fit_param.SetBorderSize(0)
        fit_param.SetFillStyle(0)
        fit_param.SetTextAlign(12)
        fit_param.SetTextFont(42)
        fit_param.AddText('#mu = ' + f'{syst_mu:.2e} #pm {syst_mu_err:.2e}' + r' (GeV/#it{c})^{-1}')
        fit_param.AddText('#sigma = ' + f'{syst_sigma:.2e} #pm {syst_sigma_err:.2e}' + r' (GeV/#it{c})^{-1}')
        fit_param.AddText('RMS = ' + f'{syst_rms:.2e} #pm {h_pt_syst[i_bin].GetRMSError():.2e}' + r' (GeV/#it{c})^{-1}')
        fit_param.AddText('standard value = ' + f'{std_corrected_counts[i_bin]:.2e} #pm {std_corrected_counts_err[i_bin]:.2e}' + r' (GeV/#it{c})^{-1}')
        # draw histogram with systematic variations
        canvas.cd()
        h_pt_syst[i_bin].Draw('HISTO SAME')
        fit_func.Draw('SAME')
        std_errorbox.Draw()
        std_line.Draw()
        fit_param.Draw()
        canvas.Write()
        canvas.SaveAs(f'{output_file_path}/cYield_{i_bin}.pdf')
        # write trial strings to a text file
        if os.path.exists(f'{output_file_path}/trailinfo.txt'):
            os.remove(f'{output_file_path}/trailinfo.txt')
        with open(f'{output_file_path}/trailinfo.txt', 'w') as f:
            for trial_string in trial_strings:
                if isinstance(trial_string, list):
                    for line in trial_string:
                        f.write("%s\n" % line)
                else:
                    f.write("%s\n" % trial_string)
    print("----------------------------------")
    if absorption_syst_array != [] and absorption_syst_array is not None:
        print(f'NB: additional systematic uncertainty from absorption added: {absorption_syst_array}')
    print("** Multi trial analysis done ** \n")
### draw the final spectrum
output_dir_std.cd()
cFinalSpectrum = ROOT.TCanvas('cFinalSpectrum', 'cFinalSpectrum', 800, 600)
# define canvas between 0 and 10
cFinalSpectrum.DrawFrame(0.1e-09, 0, 10, 1.2 * final_stat.GetMaximum(), r';#it{p}_{T} (GeV/#it{c});#frac{1}{N_{ev}}#frac{#it{d}N}{#it{d}y#it{d}#it{p}_{T}} (GeV/#it{c})^{-1}')
fit_fun_stat = final_stat.GetFunction(spectra_maker.fit_func.GetName())
fit_fun_stat.SetRange(0, 10)
final_stat.Draw('PE0 SAME') #PEX0 origin
final_syst_rms.Draw('PE2 SAME')

cFinalSpectrum.Write()
cFinalSpectrum.SaveAs(f'{output_file_path}/cFinalSpectrum.pdf')

final_stat.Write()
final_syst.Write()
fit_fun_stat.Write()
if do_syst:
    yield_dist.Write()
    yield_prob.Write()
output_file.Close()

print("Yield for the std selections: ", std_yield, " +- ", std_yield_err)
# print all the fit parameters
print("Final fit parameters: ")
print(fit_fun_stat.GetName())

for i in range(fit_fun_stat.GetNpar()):
    print(f'Parameter {i}: {fit_fun_stat.GetParameter(i)} +- {fit_fun_stat.GetParError(i)}')

print("Fit integral: ", fit_fun_stat.Integral(0, 10), " +- ", fit_fun_stat.IntegralError(0, 10))
print("Chi2/ndf: ", fit_fun_stat.GetChisquare() / fit_fun_stat.GetNDF())
print("Prob: ", fit_fun_stat.GetProb())
