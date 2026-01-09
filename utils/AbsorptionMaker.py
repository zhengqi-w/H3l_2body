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
