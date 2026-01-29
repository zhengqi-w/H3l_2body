import argparse
import yaml
import uproot
import ROOT
import numpy as np
import os

import sys
sys.path.append('../utils')
import utils as utils
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


HE_3_MASS = 2.809230089
def extend_absorption_info(absorption_tree, org_ctao=7.6, h_he_counts=None, h_he_counts_abso=None):
    absorpted_list = []
    gentct_list = []
    absoct_list = []
    for (pt, eta, phi, absoX, absoY, absoZ, pdg) in zip(absorption_tree['pt'], absorption_tree['eta'], absorption_tree['phi'], absorption_tree['absoX'], absorption_tree['absoY'], absorption_tree['absoZ'], absorption_tree['pdg']):
        mat_type = 'matter' if pdg > 0 else 'antimatter'
        he3TLV = ROOT.TLorentzVector()
        he3TLV.SetPtEtaPhiM(pt, eta, phi, HE_3_MASS)
        he3p = he3TLV.P()
        absoR = np.sqrt(absoX**2 + absoY**2)
        absoL = np.sqrt(absoX**2 + absoY**2 + absoZ**2)
        absoCt = absoL * HE_3_MASS / he3p
        decCt = ROOT.gRandom.Exp(org_ctao)
        absorpted_list.append(absoCt < decCt)
        gentct_list.append(decCt)
        absoct_list.append(absoCt)
        if h_he_counts: 
            h_he_counts[mat_type].Fill(decCt)
            h_he_counts['both'].Fill(decCt)
        if absoCt > decCt and h_he_counts_abso:
            h_he_counts_abso[mat_type].Fill(decCt)
            h_he_counts_abso['both'].Fill(decCt)
    absorption_tree['absorpted'] = absorpted_list
    absorption_tree['gentct'] = gentct_list
    absorption_tree['absoct'] = absoct_list


class MixedAbsorptionCalculator:
    def __init__(self, abso_tree, pt_bins, ct_bins, org_ctao = 7.6):
        self.abso_tree = abso_tree
        self.pt_bins = pt_bins
        self.ct_bins = ct_bins
        self.org_ctao = org_ctao
        
        self.h_he_counts_list = []
        self.h_he_counts_absorb_list = []
        self.h_he_ratio_absorb_list = []

        self.tree_list = []
    
    def _check_members(self):
        assert self.abso_tree is not None, "abso_tree is not set."
        assert self.org_ctao is not None, "org_ctao is not set."
        assert self.pt_bins is not None, "pt_bins is not set."
        assert self.ct_bins is not None, "ct_bins is not set."
        return bool(self.h_he_counts_list and self.h_he_counts_absorb_list and self.h_he_ratio_absorb_list)
    
    def _init_histos(self):
        for i in range(len(self.pt_bins) - 1):
            h_he_counts = {"matter": ROOT.TH1F(f'h_he_counts_pt{i}', f'He3 counts pt bin {i}', len(self.ct_bins[i])-1, np.array(self.ct_bins[i], dtype=np.float32)),
                           "antimatter": ROOT.TH1F(f'h_he_counts_antimat_pt{i}', f'He3 counts pt bin {i}', len(self.ct_bins[i])-1, np.array(self.ct_bins[i], dtype=np.float32)),
                           "both": ROOT.TH1F(f'h_he_counts_both_pt{i}', f'He3 counts pt bin {i}', len(self.ct_bins[i])-1, np.array(self.ct_bins[i], dtype=np.float32))}
            self.h_he_counts_list.append(h_he_counts)
            h_he_counts_absorb = {"matter": ROOT.TH1F(f'h_he_counts_absorb_pt{i}', f'He3 absorbed counts pt bin {i}', len(self.ct_bins[i])-1, np.array(self.ct_bins[i], dtype=np.float32)),
                                  "antimatter": ROOT.TH1F(f'h_he_counts_absorb_antimat_pt{i}', f'He3 absorbed counts pt bin {i}', len(self.ct_bins[i])-1, np.array(self.ct_bins[i], dtype=np.float32)),
                                  "both": ROOT.TH1F(f'h_he_counts_absorb_both_pt{i}', f'He3 absorbed counts pt bin {i}', len(self.ct_bins[i])-1, np.array(self.ct_bins[i], dtype=np.float32))}
            self.h_he_counts_absorb_list.append(h_he_counts_absorb)
            h_he_ratio_absorb = {"matter": ROOT.TH1F(f'h_he_ratio_absorb_pt{i}', f'He3 absorbed ratio pt bin {i}', len(self.ct_bins[i])-1, np.array(self.ct_bins[i], dtype=np.float32)),
                                 "antimatter": ROOT.TH1F(f'h_he_ratio_absorb_antimat_pt{i}', f'He3 absorbed ratio pt bin {i}', len(self.ct_bins[i])-1, np.array(self.ct_bins[i], dtype=np.float32)),
                                 "both": ROOT.TH1F(f'h_he_ratio_absorb_both_pt{i}', f'He3 absorbed ratio pt bin {i}', len(self.ct_bins[i])-1, np.array(self.ct_bins[i], dtype=np.float32))}
            self.h_he_ratio_absorb_list.append(h_he_ratio_absorb)
        print("Mixed Absorption Calculator: Histograms initialized.")

    def _clear_histos(self):
        # Reset contents/errors of all histograms without deleting the objects
        for hist_list in (self.h_he_counts_list, self.h_he_counts_absorb_list, self.h_he_ratio_absorb_list):
            if not hist_list:
                continue
            for hist_dict in hist_list:
                for hist in hist_dict.values():
                    if hist is None:
                        continue
                    try:
                        hist.Reset()  # clears bin contents, errors and statistics but keeps the histogram object/axis
                    except Exception:
                        pass
        print("Mixed Absorption Calculator: Histograms reset.")
    
    def calculate_absorption(self):
        if not self._check_members():
            self._init_histos()
        else:
            self._clear_histos()
        for i in range(len(self.pt_bins) - 1):
            pt_min = self.pt_bins[i]
            pt_max = self.pt_bins[i+1]
            tree_bin = self.abso_tree[(self.abso_tree['pt'] >= pt_min) & (self.abso_tree['pt'] < pt_max)].copy()
            extend_absorption_info(tree_bin, org_ctao=self.org_ctao, 
                                  h_he_counts=self.h_he_counts_list[i], h_he_counts_abso=self.h_he_counts_absorb_list[i])
            self.tree_list.append(tree_bin)
            for mat_type in ['matter', 'antimatter', 'both']:
                self.h_he_ratio_absorb_list[i][mat_type].Divide(self.h_he_counts_absorb_list[i][mat_type], self.h_he_counts_list[i][mat_type], 1, 1, "B")
    
    def delete_dynamic_members(self):
        self.tree_list = []
        self.h_he_counts_list = []
        self.h_he_counts_absorb_list = []
        self.h_he_ratio_absorb_list = []
    
    def get_counts_list(self,matter_type='both'):
        counts_list = []
        counts_error_list = []
        for i in range(len(self.pt_bins) - 1):
            bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_he_counts_list[i][matter_type])
            counts_list.append(bin_values)
            counts_error_list.append(bin_errors)
        return counts_list, counts_error_list

    def get_counts_absorb_list(self,matter_type='both'):
        counts_absorb_list = []
        counts_absorb_error_list = []
        for i in range(len(self.pt_bins) - 1):
            bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_he_counts_absorb_list[i][matter_type])
            counts_absorb_list.append(bin_values)
            counts_absorb_error_list.append(bin_errors)
        return counts_absorb_list, counts_absorb_error_list
    
    def get_absorption_efficiency_list(self,matter_type='both'):
        absorption_efficiency_list = []
        absorption_efficiency_error_list = []
        for i in range(len(self.pt_bins) - 1):
            bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_he_ratio_absorb_list[i][matter_type])
            absorption_efficiency_list.append(bin_values)
            absorption_efficiency_error_list.append(bin_errors)
        return absorption_efficiency_list, absorption_efficiency_error_list

        
        

class PtAbsorptionCalculator:
    def __init__(self, abso_tree, pt_bins, org_ctao = 7.6):
        self.abso_tree = abso_tree
        self.pt_bins = pt_bins
        self.org_ctao = org_ctao
        
        self.h_he_counts = None
        self.h_he_counts_absorb = None
        self.h_he_ratio_absorb = None
    
    def _check_members(self):
        assert self.abso_tree is not None, "abso_tree is not set."
        assert self.org_ctao is not None, "org_ctao is not set."
        assert self.pt_bins is not None, "pt_bins is not set."
        return bool(self.h_he_counts and self.h_he_counts_absorb and self.h_he_ratio_absorb)

    def _init_histos(self):
        h_he_counts = {"matter": ROOT.TH1F('h_he_counts_pt', 'He3 counts', len(self.pt_bins)-1, np.array(self.pt_bins, dtype=np.float32)),
                       "antimatter": ROOT.TH1F('h_he_counts_pt_antimat', 'He3 counts', len(self.pt_bins)-1, np.array(self.pt_bins, dtype=np.float32)),
                       "both": ROOT.TH1F('h_he_counts_pt_both', 'He3 counts', len(self.pt_bins)-1, np.array(self.pt_bins, dtype=np.float32))}
        self.h_he_counts = h_he_counts
        h_he_counts_absorb = {"matter": ROOT.TH1F('h_he_counts_pt_absorb', 'He3 absorbed counts', len(self.pt_bins)-1, np.array(self.pt_bins, dtype=np.float32)),
                              "antimatter": ROOT.TH1F('h_he_counts_pt_absorb_antimat', 'He3 absorbed counts', len(self.pt_bins)-1, np.array(self.pt_bins, dtype=np.float32)),
                              "both": ROOT.TH1F('h_he_counts_pt_absorb_both', 'He3 absorbed counts', len(self.pt_bins)-1, np.array(self.pt_bins, dtype=np.float32))}
        self.h_he_counts_absorb = h_he_counts_absorb
        h_he_ratio_absorb = {"matter": ROOT.TH1F('h_he_ratio_pt_absorb', 'He3 absorbed ratio', len(self.pt_bins)-1, np.array(self.pt_bins, dtype=np.float32)),
                             "antimatter": ROOT.TH1F('h_he_ratio_pt_absorb_antimat', 'He3 absorbed ratio', len(self.pt_bins)-1, np.array(self.pt_bins, dtype=np.float32)),
                             "both": ROOT.TH1F('h_he_ratio_pt_absorb_both', 'He3 absorbed ratio', len(self.pt_bins)-1, np.array(self.pt_bins, dtype=np.float32))}
        self.h_he_ratio_absorb = h_he_ratio_absorb
        print("Pt Absorption Calculator: Histograms initialized.")
    
    def _clear_histos(self):
        # Reset contents/errors of all histograms without deleting the objects
        for hist_dict in [self.h_he_counts, self.h_he_counts_absorb, self.h_he_ratio_absorb]:
            if not hist_dict:
                continue
            for hist in hist_dict.values():
                if hist is None:
                    continue
                try:
                    hist.Reset()  # clears bin contents, errors and statistics but keeps the histogram object/axis
                except Exception:
                    pass
        print("Pt Absorption Calculator: Histograms reset.")
            
    
    def calculate_absorption(self):
        if not self._check_members():
            self._init_histos()
        else:
            self._clear_histos()
        extend_absorption_info(self.abso_tree, org_ctao=self.org_ctao, 
                              h_he_counts=self.h_he_counts, h_he_counts_abso=self.h_he_counts_absorb)
        for mat_type in ['matter', 'antimatter', 'both']:
            self.h_he_ratio_absorb[mat_type].Divide(self.h_he_counts_absorb[mat_type], self.h_he_counts[mat_type], 1, 1, "B")
        
    
    def delete_dynamic_members(self):
        self.abso_tree = None
        self.h_he_counts = None
        self.h_he_counts_absorb = None
        self.h_he_ratio_absorb = None

    def get_counts_list(self,matter_type='both'):
        bin_centers, counts_list, counts_error_list, bin_edges = utils.extract_info_TH1(self.h_he_counts[matter_type])
        return counts_list, counts_error_list

    def get_counts_absorb_list(self,matter_type='both'):
        bin_centers, counts_absorb_list, counts_absorb_error_list, bin_edges = utils.extract_info_TH1(self.h_he_counts_absorb[matter_type])
        return counts_absorb_list, counts_absorb_error_list
    
    def get_absorption_efficiency_list(self,matter_type='both'):
        bin_centers, absorption_efficiency_list, absorption_efficiency_error_list, bin_edges = utils.extract_info_TH1(self.h_he_ratio_absorb[matter_type])
        return absorption_efficiency_list, absorption_efficiency_error_list


class CtAbsorptionCalculator:    
    def __init__(self, abso_tree, ct_bins, org_ctao = 7.6):
        self.abso_tree = abso_tree
        self.ct_bins = ct_bins
        self.org_ctao = org_ctao
        
        self.h_he_counts = None
        self.h_he_counts_absorb = None
        self.h_he_ratio_absorb = None
    
    def _check_members(self):
        assert self.abso_tree is not None, "abso_tree is not set."
        assert self.org_ctao is not None, "org_ctao is not set."
        assert self.ct_bins is not None, "ct_bins is not set."
        return bool(self.h_he_counts and self.h_he_counts_absorb and self.h_he_ratio_absorb)
    
    def _init_histos(self):
        h_he_counts = {"matter": ROOT.TH1F('h_he_counts_ct', 'He3 counts', len(self.ct_bins)-1, np.array(self.ct_bins, dtype=np.float32)),
                       "antimatter": ROOT.TH1F('h_he_counts_ct_antimat', 'He3 counts', len(self.ct_bins)-1, np.array(self.ct_bins, dtype=np.float32)),
                       "both": ROOT.TH1F('h_he_counts_ct_both', 'He3 counts', len(self.ct_bins)-1, np.array(self.ct_bins, dtype=np.float32))}
        self.h_he_counts = h_he_counts
        h_he_counts_absorb = {"matter": ROOT.TH1F('h_he_counts_ct_absorb', 'He3 absorbed counts', len(self.ct_bins)-1, np.array(self.ct_bins, dtype=np.float32)),
                              "antimatter": ROOT.TH1F('h_he_counts_ct_absorb_antimat', 'He3 absorbed counts', len(self.ct_bins)-1, np.array(self.ct_bins, dtype=np.float32)),
                              "both": ROOT.TH1F('h_he_counts_ct_absorb_both', 'He3 absorbed counts', len(self.ct_bins)-1, np.array(self.ct_bins, dtype=np.float32))}
        self.h_he_counts_absorb = h_he_counts_absorb
        h_he_ratio_absorb = {"matter": ROOT.TH1F('h_he_ratio_ct_absorb', 'He3 absorbed ratio', len(self.ct_bins)-1, np.array(self.ct_bins, dtype=np.float32)),
                             "antimatter": ROOT.TH1F('h_he_ratio_ct_absorb_antimat', 'He3 absorbed ratio', len(self.ct_bins)-1, np.array(self.ct_bins, dtype=np.float32)),
                             "both": ROOT.TH1F('h_he_ratio_ct_absorb_both', 'He3 absorbed ratio', len(self.ct_bins)-1, np.array(self.ct_bins, dtype=np.float32))}
        self.h_he_ratio_absorb = h_he_ratio_absorb
        print("Ct Absorption Calculator: Histograms initialized.")
    
    def _clear_histos(self):
        # Reset contents/errors of all histograms without deleting the objects
        for hist_dict in [self.h_he_counts, self.h_he_counts_absorb, self.h_he_ratio_absorb]:
            if not hist_dict:
                continue
            for hist in hist_dict.values():
                if hist is None:
                    continue
                try:
                    hist.Reset()  # clears bin contents, errors and statistics but keeps the histogram object/axis
                except Exception:
                    pass
        print("Ct Absorption Calculator: Histograms reset.")
            
    
    def calculate_absorption(self):
        if not self._check_members():
            self._init_histos()
        else:
            self._clear_histos()
        extend_absorption_info(self.abso_tree, org_ctao=self.org_ctao, 
                              h_he_counts=self.h_he_counts, h_he_counts_abso=self.h_he_counts_absorb)
        for mat_type in ['matter', 'antimatter', 'both']:
            self.h_he_ratio_absorb[mat_type].Divide(self.h_he_counts_absorb[mat_type], self.h_he_counts[mat_type], 1, 1, "B")
    
    def delete_dynamic_members(self):
        self.abso_tree = None
        self.h_he_counts = None
        self.h_he_counts_absorb = None
        self.h_he_ratio_absorb = None

    def get_counts_list(self,matter_type='both'):
        bin_centers, counts_list, counts_error_list, bin_edges = utils.extract_info_TH1(self.h_he_counts[matter_type])
        return counts_list, counts_error_list

    def get_counts_absorb_list(self,matter_type='both'):
        bin_centers, counts_absorb_list, counts_absorb_error_list, bin_edges = utils.extract_info_TH1(self.h_he_counts_absorb[matter_type])
        return counts_absorb_list, counts_absorb_error_list
    
    def get_absorption_efficiency_list(self,matter_type='both'):
        bin_centers, absorption_efficiency_list, absorption_efficiency_error_list, bin_edges = utils.extract_info_TH1(self.h_he_ratio_absorb[matter_type])
        return absorption_efficiency_list, absorption_efficiency_error_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configure the parameters of the script.')
    parser.add_argument('--config-file', dest='config_file', help="path to the YAML file with configuration.", default='')
    args = parser.parse_args()
    if args.config_file == "":
        print('** No config file provided. Exiting. **')
        exit()
    
    config_file = open(args.config_file, 'r')
    config = yaml.full_load(config_file)
    
    pt_bins = config['pt_bins']
    n_pt_bins = len(pt_bins) - 1
    ct_bins = config['ct_bins']
    mc_file = config['mc_file']
    org_ctao = config['org_ctao']
    save_path = config['save_path']
    name_suffix = config['name_suffix']
    fig_path = save_path + '/' + name_suffix
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    tree = uproot.open(mc_file)['he3candidates'].arrays(library="pd")

    # Use MixedAbsorptionCalculator to compute counts / absorbed / ratio per-pt
    mac = MixedAbsorptionCalculator(abso_tree=tree, pt_bins=pt_bins, ct_bins=ct_bins, org_ctao=org_ctao)
    mac.calculate_absorption()
    for i in range(len(mac.h_he_ratio_absorb_list)):
        for mat_type in ['matter', 'antimatter', 'both']:
            h = mac.h_he_ratio_absorb_list[i][mat_type]
            c = ROOT.TCanvas(f'c_ratio_pt{i}_{mat_type}', f'Absorption ratio pt{i} {mat_type}', 800, 600)
            h.SetTitle(f'Absorption ratio pt {pt_bins[i]:.2f}-{pt_bins[i+1]:.2f} ({mat_type})')
            h.GetXaxis().SetTitle('gentct')
            h.GetYaxis().SetTitle('Absorption ratio')
            h.SetStats(0)
            #h.SetMinimum(0.0)
            #h.SetMaximum(1.0)
            h.Draw('E')
            c.SaveAs(f'{fig_path}/h_he_ratio_absorb_pt{i}_{mat_type}.pdf')
            c.Close()

    # prepare output ROOT and h_tao_pt (tau in ps per pt bin)
    n_pt_bins = len(pt_bins) - 1
    h_tao_pt = ROOT.TH1F('h_tao_pt', f'lifetime #tau (ps) per p_{{T}} bin[{name_suffix} #sigma(He3)]; p_{{T}} (GeV/c); #tau', n_pt_bins, np.array(pt_bins, dtype=np.float32))

    # speed-of-light constant for ct->tau conversion
    c_cm_s = 2.99792458e10

    # open ROOT output early so we can write per-pt TH1/TF1
    out_file = ROOT.TFile(f'{save_path}/absorption_ct_histos_{name_suffix}.root', 'RECREATE')

    # For each pt bin, perform All and Unabsorbed gentct fits and save QA plots
    for i, tree_bin in enumerate(mac.tree_list):
        fig, ax = plt.subplots()
        gentct_all = np.array(tree_bin['gentct'])
        # per-pt ct binning from config
        bins = np.array(ct_bins[i], dtype=np.float32)
        counts_all, bin_edges = np.histogram(gentct_all, bins=bins)
        # scale counts by bin width to obtain density (counts per unit ct)
        bin_widths = np.diff(bin_edges)
        counts_all = counts_all.astype(np.float64) / bin_widths
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        nbins = len(bins) - 1
        # ax.hist(gentct_all, bins=bins, alpha=0.5, label='All', color='blue')
        ax.bar(bin_centers, counts_all, width=bin_widths, align='center',
               alpha=0.5, label='All (bin-width corrected)', color='blue', edgecolor='k')
        # Fit 'All' with ROOT TF1 (integral mode)
        func_all = None
        h_all = None
        try:
            h_all = ROOT.TH1F(f'h_all_{i}', f'all gentct pt{i}', nbins, bins)
            for ib in range(nbins):
                c = float(counts_all[ib])
                h_all.SetBinContent(ib+1, c)
                h_all.SetBinError(ib+1, np.sqrt(c) if c > 0 else 1e-6)
            func_all = ROOT.TF1(f'func_all_{i}', '[0]*exp(-x/[1])', float(bins[0]), float(bins[-1]))
            func_all.SetParameter(0, counts_all[0] if len(counts_all)>0 else 1)
            func_all.SetParameter(1, org_ctao)
            fitres_all = h_all.Fit(func_all, 'QIS')
            a_all = func_all.GetParameter(0)
            tau_all = func_all.GetParameter(1)
            tau_all_err = func_all.GetParError(1)
            tau_all_ps = tau_all / c_cm_s * 1e12
            tau_all_err_ps = tau_all_err / c_cm_s * 1e12
            chi2_all = func_all.GetChisquare()
            ndf_all = func_all.GetNDF()
            chi2txt = f'chi2/ndf={chi2_all:.1f}/{ndf_all}' if ndf_all and ndf_all>0 else 'chi2/ndf=N/A'
            ax.plot(bin_centers, a_all * np.exp(-bin_centers / tau_all), 'b--', label=f'All fit: tau={tau_all_ps:.2f}±{tau_all_err_ps:.2f} ps ({chi2txt})')
        except Exception as e:
            print(f"ROOT fit failed for all gentct in pt bin {i}: {e}")

        # Unabsorbed (absorpted == False)
        func_un = None
        h_un = None
        try:
            mask = np.array(tree_bin['absorpted']) == False
            gentct_unabsorbed = gentct_all[mask]
            counts_unabs, bin_edges_un = np.histogram(gentct_unabsorbed, bins=bins)
            bin_widths_un = np.diff(bin_edges_un)
            counts_unabs = counts_unabs.astype(np.float64) / bin_widths_un
            # plot bin-width corrected histogram for unabsorbed
            ax.bar(bin_centers, counts_unabs, width=bin_widths_un, align='center',
                   alpha=0.5, label='Unabsorbed (bin-width corrected)', color='red', edgecolor='k')
            h_un = ROOT.TH1F(f'h_un_{i}', f'unabsorbed gentct pt{i}', nbins, bins)
            for ib in range(nbins):
                c = float(counts_unabs[ib])
                h_un.SetBinContent(ib+1, c)
                h_un.SetBinError(ib+1, np.sqrt(c) if c > 0 else 1e-6)
            func_un = ROOT.TF1(f'func_un_{i}', '[0]*exp(-x/[1])', float(bins[0]), float(bins[-1]))
            func_un.SetParameter(0, counts_unabs[0] if len(counts_unabs)>0 else 1)
            func_un.SetParameter(1, org_ctao)
            fitres_un = h_un.Fit(func_un, 'QIS')
            a_un = func_un.GetParameter(0)
            tau_un = func_un.GetParameter(1)
            tau_un_err = func_un.GetParError(1)
            tau_un_ps = tau_un / c_cm_s * 1e12
            tau_un_err_ps = tau_un_err / c_cm_s * 1e12
            chi2_un = func_un.GetChisquare()
            ndf_un = func_un.GetNDF()
            chi2txt_un = f'chi2/ndf={chi2_un:.1f}/{ndf_un}' if ndf_un and ndf_un>0 else 'chi2/ndf=N/A'
            ax.plot(bin_centers, a_un * np.exp(-bin_centers / tau_un), 'r--', label=f'Unabs fit: tau={tau_un_ps:.2f}±{tau_un_err_ps:.2f} ps ({chi2txt_un})')
            h_tao_pt.SetBinContent(i+1, tau_un_ps)
            h_tao_pt.SetBinError(i+1, tau_un_err_ps)
        except Exception as e:
            print(f"ROOT fit failed for unabsorbed gentct in pt bin {i}: {e}")
            h_tao_pt.SetBinContent(i+1, 0)
            h_tao_pt.SetBinError(i+1, 0)
        finally:
            # write per-pt TH1 and TF1 to ROOT (if they exist)
            out_file.mkdir(f'pt_{pt_bins[i]:.1f}_{pt_bins[i+1]:.1f}').cd()
            if h_all is not None:
                try:
                    h_all.Write()
                except Exception:
                    pass
            if func_all is not None:
                try:
                    func_all.Write()
                except Exception:
                    pass
            if h_un is not None:
                try:
                    h_un.Write()
                except Exception:
                    pass
            if func_un is not None:
                try:
                    func_un.Write()
                except Exception:
                    pass
            out_file.cd()

        ax.set_xlabel('gentct')
        ax.set_ylabel('Counts/bin width')
        ax.set_yscale('log')
        ax.set_title(f'gentct distribution pt: {pt_bins[i]:.1f}-{pt_bins[i+1]:.1f} GeV/c [{name_suffix} σ(He3)]')
        ax.legend()
        plt.savefig(f'{fig_path}/gentct_distribution_pt{i}.pdf')
        plt.close()

    # write remaining histograms (counts/absorb/ratio and h_tao_pt) to ROOT and close
    out_file.cd()
    h_tao_pt.Write()
    for i in range(n_pt_bins):
        dir_name = f'pt_{pt_bins[i]:.1f}_{pt_bins[i+1]:.1f}'
        if out_file.GetDirectory(dir_name) is None:
            out_file.mkdir(dir_name)
        out_file.cd(dir_name)
        for mat_type in ['matter', 'antimatter', 'both']:
            mac.h_he_counts_list[i][mat_type].Write()
            mac.h_he_counts_absorb_list[i][mat_type].Write()
            mac.h_he_ratio_absorb_list[i][mat_type].Write()
        out_file.cd()
    out_file.Close()