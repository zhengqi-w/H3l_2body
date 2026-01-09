import uproot
import ROOT
ROOT.ROOT.EnableImplicitMT()
import numpy as np
import os
from hipe4ml.tree_handler import TreeHandler

import sys
sys.path.append('../utils')
import utils as utils
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

class AccptanceCalculator:
    def __init__(self, mc_hdl, pt_bins, ct_bins, reweight_pt=True):
        self.mc_hdl = mc_hdl
        self.pt_bins = pt_bins
        self.ct_bins = ct_bins
        self.reweight_pt = reweight_pt
        
        self.spectrum_func = ROOT.TFile.Open('/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/H3l_2body_spectrum/utils/H3L_BWFit.root').Get('BlastWave_H3L_10_30')
        self.h_evsel_counts_pt = None
        self.h_reco_counts_pt = None
        self.h_accptance_pt = None
        self.h_evsel_counts_ct = None
        self.h_reco_counts_ct = None
        self.h_accptance_ct = None
        self.h_evsel_counts_ct_pt_bins = None
        self.h_reco_counts_ct_pt_bins = None
        self.h_accptance_ct_pt_bins = None
    
    def _check_members(self):
        if isinstance(self.mc_hdl, TreeHandler):
            self.mc_hdl = self.mc_hdl.get_data_frame()
        if not isinstance(self.mc_hdl, pd.DataFrame):
            raise TypeError(f"mc_hdl must be a pandas.DataFrame or a TreeHandler convertible to one, got {type(self.mc_hdl)}")
        if 'fGenCt' not in self.mc_hdl.columns:
            utils.correct_and_convert_df(self.mc_hdl, isMC=True)
        if not self.spectrum_func:
            raise ValueError("spectrum_func is not set.")
        if isinstance(self.ct_bins[0], list):
            if len(self.ct_bins) != len(self.pt_bins) -1:
                raise ValueError("Length of ct_bins list must be equal to length of pt_bins - 1 when use Mixed mode.")
            print("Mixed ct accptance mode activated.")
        elif self.ct_bins is None and self.pt_bins is None:
            raise ValueError("Tar bins not seted.")
        if isinstance(self.ct_bins[0], list):
            if self.h_evsel_counts_ct_pt_bins is not None and \
               self.h_reco_counts_ct_pt_bins is not None and \
               self.h_accptance_ct_pt_bins is not None:
                return True
        else:
            if self.pt_bins is not None:
                if self.h_evsel_counts_pt is not None and \
                   self.h_reco_counts_pt is not None and \
                   self.h_accptance_pt is not None:
                    return True
            if self.ct_bins is not None:
                if self.h_evsel_counts_ct is not None and \
                   self.h_reco_counts_ct is not None and \
                   self.h_accptance_ct is not None:
                    return True
        return False
    
    def _init_histos(self):
        if isinstance(self.ct_bins[0], list):
            self.h_evsel_counts_ct_pt_bins = []
            self.h_reco_counts_ct_pt_bins = []
            self.h_accptance_ct_pt_bins = []
            for i in range(len(self.pt_bins) - 1):
                h_evsel_counts_ct = {"matter": ROOT.TH1F(f'h_evsel_counts_ct_ptbin_{i}', 'He3 evsel counts vs ct', len(self.ct_bins[i])-1, np.array(self.ct_bins[i], dtype=np.float32)),
                                     "antimatter": ROOT.TH1F(f'h_evsel_counts_ct_ptbin_{i}_antimat', 'He3 evsel counts vs ct', len(self.ct_bins[i])-1, np.array(self.ct_bins[i], dtype=np.float32)),
                                     "both": ROOT.TH1F(f'h_evsel_counts_ct_ptbin_{i}_both', 'He3 evsel counts vs ct', len(self.ct_bins[i])-1, np.array(self.ct_bins[i], dtype=np.float32))}
                h_reco_counts_ct = {"matter": ROOT.TH1F(f'h_reco_counts_ct_ptbin_{i}', 'He3 reco counts vs ct', len(self.ct_bins[i])-1, np.array(self.ct_bins[i], dtype=np.float32)),
                                    "antimatter": ROOT.TH1F(f'h_reco_counts_ct_ptbin_{i}_antimat', 'He3 reco counts vs ct', len(self.ct_bins[i])-1, np.array(self.ct_bins[i], dtype=np.float32)),
                                    "both": ROOT.TH1F(f'h_reco_counts_ct_ptbin_{i}_both', 'He3 reco counts vs ct', len(self.ct_bins[i])-1, np.array(self.ct_bins[i], dtype=np.float32))}
                h_accptance_ct = {"matter": ROOT.TH1F(f'h_accptance_ct_ptbin_{i}', 'He3 accptance vs ct', len(self.ct_bins[i])-1, np.array(self.ct_bins[i], dtype=np.float32)),
                                  "antimatter": ROOT.TH1F(f'h_accptance_ct_ptbin_{i}_antimat', 'He3 accptance vs ct', len(self.ct_bins[i])-1, np.array(self.ct_bins[i], dtype=np.float32)),
                                  "both": ROOT.TH1F(f'h_accptance_ct_ptbin_{i}_both', 'He3 accptance vs ct', len(self.ct_bins[i])-1, np.array(self.ct_bins[i], dtype=np.float32))}
                self.h_evsel_counts_ct_pt_bins.append(h_evsel_counts_ct)
                self.h_reco_counts_ct_pt_bins.append(h_reco_counts_ct)
                self.h_accptance_ct_pt_bins.append(h_accptance_ct)
            print("Mixed accptance Calculator: Histograms initialized.")
        else:
            if self.pt_bins is not None:
                h_evsel_counts_pt = {"matter": ROOT.TH1F('h_evsel_counts_pt', 'He3 evsel counts vs pt', len(self.pt_bins)-1, np.array(self.pt_bins, dtype=np.float32)),
                                     "antimatter": ROOT.TH1F('h_evsel_counts_pt_antimat', 'He3 evsel counts vs pt', len(self.pt_bins)-1, np.array(self.pt_bins, dtype=np.float32)),
                                     "both": ROOT.TH1F('h_evsel_counts_pt_both', 'He3 evsel counts vs pt', len(self.pt_bins)-1, np.array(self.pt_bins, dtype=np.float32))}
                h_reco_counts_pt = {"matter": ROOT.TH1F('h_reco_counts_pt', 'He3 reco counts vs pt', len(self.pt_bins)-1, np.array(self.pt_bins, dtype=np.float32)),
                                    "antimatter": ROOT.TH1F('h_reco_counts_pt_antimat', 'He3 reco counts vs pt', len(self.pt_bins)-1, np.array(self.pt_bins, dtype=np.float32)),
                                    "both": ROOT.TH1F('h_reco_counts_pt_both', 'He3 reco counts vs pt', len(self.pt_bins)-1, np.array(self.pt_bins, dtype=np.float32))}
                h_accptance_pt = {"matter": ROOT.TH1F('h_accptance_pt', 'He3 accptance vs pt', len(self.pt_bins)-1, np.array(self.pt_bins, dtype=np.float32)),
                                  "antimatter": ROOT.TH1F('h_accptance_pt_antimat', 'He3 accptance vs pt', len(self.pt_bins)-1, np.array(self.pt_bins, dtype=np.float32)),
                                  "both": ROOT.TH1F('h_accptance_pt_both', 'He3 accptance vs pt', len(self.pt_bins)-1, np.array(self.pt_bins, dtype=np.float32))}
                self.h_evsel_counts_pt = h_evsel_counts_pt
                self.h_reco_counts_pt = h_reco_counts_pt
                self.h_accptance_pt = h_accptance_pt
            if self.ct_bins is not None:
                h_evsel_counts_ct = {"matter": ROOT.TH1F('h_evsel_counts_ct', 'He3 evsel counts vs ct', len(self.ct_bins)-1, np.array(self.ct_bins, dtype=np.float32)),
                                     "antimatter": ROOT.TH1F('h_evsel_counts_ct_antimat', 'He3 evsel counts vs ct', len(self.ct_bins)-1, np.array(self.ct_bins, dtype=np.float32)),
                                     "both": ROOT.TH1F('h_evsel_counts_ct_both', 'He3 evsel counts vs ct', len(self.ct_bins)-1, np.array(self.ct_bins, dtype=np.float32))}
                h_reco_counts_ct = {"matter": ROOT.TH1F('h_reco_counts_ct', 'He3 reco counts vs ct', len(self.ct_bins)-1, np.array(self.ct_bins, dtype=np.float32)),
                                    "antimatter": ROOT.TH1F('h_reco_counts_ct_antimat', 'He3 reco counts vs ct', len(self.ct_bins)-1, np.array(self.ct_bins, dtype=np.float32)),
                                    "both": ROOT.TH1F('h_reco_counts_ct_both', 'He3 reco counts vs ct', len(self.ct_bins)-1, np.array(self.ct_bins, dtype=np.float32))}
                h_accptance_ct = {"matter": ROOT.TH1F('h_accptance_ct', 'He3 accptance vs ct', len(self.ct_bins)-1, np.array(self.ct_bins, dtype=np.float32)),
                                  "antimatter": ROOT.TH1F('h_accptance_ct_antimat', 'He3 accptance vs ct', len(self.ct_bins)-1, np.array(self.ct_bins, dtype=np.float32)),
                                  "both": ROOT.TH1F('h_accptance_ct_both', 'He3 accptance vs ct', len(self.ct_bins)-1, np.array(self.ct_bins, dtype=np.float32))}
                self.h_evsel_counts_ct = h_evsel_counts_ct
                self.h_reco_counts_ct = h_reco_counts_ct
                self.h_accptance_ct = h_accptance_ct
        print("Accptance Calculator: Histograms initialized.")

    def _clear_histos(self):
        # Mixed-mode: list of ct bins per pt bin
        if self.ct_bins and isinstance(self.ct_bins[0], list):
            if self.h_evsel_counts_ct_pt_bins:
                for d in self.h_evsel_counts_ct_pt_bins:
                    for key, hist in d.items():
                        try:
                            hist.Reset()
                        except Exception:
                            pass
            if self.h_reco_counts_ct_pt_bins:
                for d in self.h_reco_counts_ct_pt_bins:
                    for key, hist in d.items():
                        try:
                            hist.Reset()
                        except Exception:
                            pass
            if self.h_accptance_ct_pt_bins:
                for d in self.h_accptance_ct_pt_bins:
                    for key, hist in d.items():
                        try:
                            hist.Reset()
                        except Exception:
                            pass
            return
        # Regular mode: possible pt and ct hist dictionaries
        if getattr(self, 'h_evsel_counts_pt', None) is not None:
            for key, hist in self.h_evsel_counts_pt.items():
                try:
                    hist.Reset()
                except Exception:
                    pass
        if getattr(self, 'h_reco_counts_pt', None) is not None:
            for key, hist in self.h_reco_counts_pt.items():
                try:
                    hist.Reset()
                except Exception:
                    pass
        if getattr(self, 'h_accptance_pt', None) is not None:
            for key, hist in self.h_accptance_pt.items():
                try:
                    hist.Reset()
                except Exception:
                    pass
        if getattr(self, 'h_evsel_counts_ct', None) is not None:
            for key, hist in self.h_evsel_counts_ct.items():
                try:
                    hist.Reset()
                except Exception:
                    pass
        if getattr(self, 'h_reco_counts_ct', None) is not None:
            for key, hist in self.h_reco_counts_ct.items():
                try:
                    hist.Reset()
                except Exception:
                    pass
        if getattr(self, 'h_accptance_ct', None) is not None:
            for key, hist in self.h_accptance_ct.items():
                try:
                    hist.Reset()
                except Exception:
                    pass
    print("Accptance Calculator: Histograms cleared.")
    
            
    
    def calculate_accptance(self):
        if not self._check_members():
            self._init_histos()
        else:
            self._clear_histos()
        if isinstance(self.ct_bins[0], list):
            for i in range(len(self.pt_bins) - 1):
                pt_min = self.pt_bins[i]
                pt_max = self.pt_bins[i+1]
                df_pt_bin = self.mc_hdl[(self.mc_hdl['fAbsGenPt'] > pt_min) & (self.mc_hdl['fAbsGenPt'] <= pt_max)]
                for index, row in df_pt_bin.iterrows():
                    ct_value = row['fGenCt']
                    reco_flag = row['fIsReco']
                    eventsel_flag = row['fIsSurvEvSel']
                    matter_type = 'matter' if row['fGenPt'] > 0 else 'antimatter'
                    if eventsel_flag:
                        self.h_evsel_counts_ct_pt_bins[i][matter_type].Fill(ct_value)
                        self.h_evsel_counts_ct_pt_bins[i]['both'].Fill(ct_value)
                    if reco_flag:
                        self.h_reco_counts_ct_pt_bins[i][matter_type].Fill(ct_value)
                        self.h_reco_counts_ct_pt_bins[i]['both'].Fill(ct_value)
                for matter_type in ['matter', 'antimatter', 'both']:
                    self.h_accptance_ct_pt_bins[i][matter_type].Divide(self.h_reco_counts_ct_pt_bins[i][matter_type], self.h_evsel_counts_ct_pt_bins[i][matter_type], 1, 1, "B")
            print("Mixed accptance Calculator: Accptance calculation completed.")
        else:
            if self.pt_bins is not None:
                for i in range(len(self.pt_bins) - 1):
                    pt_min = self.pt_bins[i]
                    pt_max = self.pt_bins[i+1]
                    df_pt_bin = self.mc_hdl[(self.mc_hdl['fAbsGenPt'] > pt_min) & (self.mc_hdl['fAbsGenPt'] <= pt_max)]
                    for index, row in df_pt_bin.iterrows():
                        reco_flag = row['fIsReco']
                        eventsel_flag = row['fIsSurvEvSel']
                        matter_type = 'matter' if row['fGenPt'] > 0 else 'antimatter'
                        if eventsel_flag:
                            self.h_evsel_counts_pt[matter_type].Fill(row['fAbsGenPt'])
                            self.h_evsel_counts_pt['both'].Fill(row['fAbsGenPt'])
                        if reco_flag:
                            self.h_reco_counts_pt[matter_type].Fill(row['fAbsGenPt'])
                            self.h_reco_counts_pt['both'].Fill(row['fAbsGenPt'])
                for matter_type in ['matter', 'antimatter', 'both']:
                    self.h_accptance_pt[matter_type].Divide(self.h_reco_counts_pt[matter_type], self.h_evsel_counts_pt[matter_type], 1, 1, "B")
                print("pt accptance calculation completed.")
            if self.ct_bins is not None:
                for i in range(len(self.ct_bins) - 1):
                    ct_min = self.ct_bins[i]
                    ct_max = self.ct_bins[i+1]
                    df_ct_bin = self.mc_hdl[(self.mc_hdl['fGenCt'] > ct_min) & (self.mc_hdl['fGenCt'] <= ct_max)]
                    for index, row in df_ct_bin.iterrows():
                        reco_flag = row['fIsReco']
                        eventsel_flag = row['fIsSurvEvSel']
                        matter_type = 'matter' if row['fGenPt'] > 0 else 'antimatter'
                        if eventsel_flag:
                            self.h_evsel_counts_ct[matter_type].Fill(row['fGenCt'])
                            self.h_evsel_counts_ct['both'].Fill(row['fGenCt'])
                        if reco_flag:
                            self.h_reco_counts_ct[matter_type].Fill(row['fGenCt'])
                            self.h_reco_counts_ct['both'].Fill(row['fGenCt'])
                for matter_type in ['matter', 'antimatter', 'both']:
                    self.h_accptance_ct[matter_type].Divide(self.h_reco_counts_ct[matter_type], self.h_evsel_counts_ct[matter_type], 1, 1, "B")
                print("ct accptance calculation completed.")
    
    def delete_dynamic_members(self):
        self.mc_hdl = None
        self.spectrum_func = None
        self._clear_histos()
        self.h_evsel_counts_pt = None
        self.h_reco_counts_pt = None
        self.h_accptance_pt = None
        self.h_evsel_counts_ct = None
        self.h_reco_counts_ct = None
        self.h_accptance_ct = None
        self.h_evsel_counts_ct_pt_bins = None
        self.h_reco_counts_ct_pt_bins = None
        self.h_accptance_ct_pt_bins = None
                
    
    def get_counts_eventsel_list(self,matter_type='both'):
        counts_list = []
        counts_error_list = []
        if isinstance(self.ct_bins[0], list):
            for i in range(len(self.pt_bins) - 1):
                bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_evsel_counts_ct_pt_bins[i][matter_type])
                counts_list.append(bin_values)
                counts_error_list.append(bin_errors)
        else:
            if self.pt_bins is not None:
                bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_evsel_counts_pt[matter_type])
                counts_list.append(bin_values)
                counts_error_list.append(bin_errors)
            if self.ct_bins is not None:
                bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_evsel_counts_ct[matter_type])
                counts_list.append(bin_values)
                counts_error_list.append(bin_errors)
        return counts_list, counts_error_list

    def get_counts_recon_list(self,matter_type='both'):
        counts_list = []
        counts_error_list = []
        if isinstance(self.ct_bins[0], list):
            for i in range(len(self.pt_bins) - 1):
                bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_reco_counts_ct_pt_bins[i][matter_type])
                counts_list.append(bin_values)
                counts_error_list.append(bin_errors)
        else:
            if self.pt_bins is not None:
                bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_reco_counts_pt[matter_type])
                counts_list.append(bin_values)
                counts_error_list.append(bin_errors)
            if self.ct_bins is not None:
                bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_reco_counts_ct[matter_type])
                counts_list.append(bin_values)
                counts_error_list.append(bin_errors)
        return counts_list, counts_error_list
    
    def get_accptance_list(self,matter_type='both'):
        accptance_list = []
        accptance_error_list = []
        if isinstance(self.ct_bins[0], list):
            for i in range(len(self.pt_bins) - 1):
                bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_accptance_ct_pt_bins[i][matter_type])
                accptance_list.append(bin_values)
                accptance_error_list.append(bin_errors)
        else:
            if self.pt_bins is not None:
                bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_accptance_pt[matter_type])
                accptance_list.append(bin_values)
                accptance_error_list.append(bin_errors)
            if self.ct_bins is not None:
                bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_accptance_ct[matter_type])
                accptance_list.append(bin_values)
                accptance_error_list.append(bin_errors)
        return accptance_list, accptance_error_list

class AccptanceCalculatorRDF:
    def __init__(self, mc_rdf, pt_bins, ct_bins, reweight_pt=False):
        self.mc_rdf = mc_rdf
        self.pt_bins = pt_bins
        self.ct_bins = ct_bins
        self.reweight_pt = reweight_pt
        self.mc_rdf_eventsel = self.mc_rdf.Filter("fIsSurvEvSel == 1")
        self.mc_rdf_recon = self.mc_rdf_eventsel.Filter("fIsReco == 1")
        # # create temporary snapshot files for eventsel and recon RDataFrames
        # self.mc_rdf_eventsel.Snapshot("tree", "tmp_eventsel.root", ["fGenCt", "fGenPt", "fAbsGenPt"])
        # self.mc_rdf_eventsel = ROOT.RDataFrame("tree", "tmp_eventsel.root")
        # self.mc_rdf_recon.Snapshot("tree", "tmp_recon.root", ["fGenCt", "fGenPt","fAbsGenPt"])
        # self.mc_rdf_recon = ROOT.RDataFrame("tree", "tmp_recon.root")
        
        self.spectrum_func = ROOT.TFile.Open('/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/H3l_2body_spectrum/utils/H3L_BWFit.root').Get('BlastWave_H3L_10_30')
        self.h_evsel_counts_pt = None
        self.h_reco_counts_pt = None
        self.h_accptance_pt = None
        self.h_evsel_counts_ct = None
        self.h_reco_counts_ct = None
        self.h_accptance_ct = None
        self.h_evsel_counts_ct_pt_bins = []
        self.h_reco_counts_ct_pt_bins = []
        self.h_accptance_ct_pt_bins = []
    
    def _check_members(self):
        if not self.mc_rdf:
            raise ValueError("mc_rdf is not set.")
        if not self.mc_rdf_eventsel or not self.mc_rdf_recon:
            raise ValueError("mc_rdf_eventsel or mc_rdf_recon is not set.")
        if 'fGenCt' not in list(self.mc_rdf.GetColumnNames()):
            self.mc_rdf = utils.correct_and_convert_rdf(self.mc_rdf, isMC=True)
        if self.reweight_pt and not self.spectrum_func:
            raise ValueError("spectrum_func is not set.")
        if isinstance(self.ct_bins[0], list):
            if len(self.ct_bins) != len(self.pt_bins) -1:
                raise ValueError("Length of ct_bins list must be equal to length of pt_bins - 1 when use Mixed mode.")
            print("Mixed ct accptance mode activated.")
        elif self.ct_bins is None and self.pt_bins is None:
            raise ValueError("Tar bins not seted.")
    
    def calculate_accptance(self):
        self._check_members()
        # Mixed-mode: list of ct bins per pt bin
        if isinstance(self.ct_bins[0], list):
            for i_pt, (pt_min, pt_max) in enumerate(zip(self.pt_bins[:-1], self.pt_bins[1:])):
                rdf_reco_pt_bin = self.mc_rdf_recon.Filter(f"(fAbsGenPt > {pt_min}) && (fAbsGenPt <= {pt_max})")
                rdf_eventsel_pt_bin = self.mc_rdf_eventsel.Filter(f"(fAbsGenPt > {pt_min}) && (fAbsGenPt <= {pt_max})")
                ct_bins = self.ct_bins[i_pt]
                ct_edges = np.array(ct_bins, dtype=np.float32)
                nbin = len(ct_edges) - 1
                # event selection histograms (matter / antimatter / both)
                evsel_dict = {}
                evsel_dict['matter'] = rdf_eventsel_pt_bin.Filter("fGenPt > 0").Histo1D(
                    (f'h_evsel_counts_ct_ptbin_{i_pt}_matter', 'Ct counts after Event Selection(Matter);Ct (cm);Counts', nbin, ct_edges), "fGenCt").GetValue()
                evsel_dict['antimatter'] = rdf_eventsel_pt_bin.Filter("fGenPt <= 0").Histo1D(
                    (f'h_evsel_counts_ct_ptbin_{i_pt}_antimat', 'Ct counts after Event Selection(Antimatter);Ct (cm);Counts', nbin, ct_edges), "fGenCt").GetValue()
                evsel_dict['both'] = rdf_eventsel_pt_bin.Histo1D(
                    (f'h_evsel_counts_ct_ptbin_{i_pt}_both', 'Ct counts after Event Selection(Matter+Antimatter);Ct (cm);Counts', nbin, ct_edges), "fGenCt").GetValue()
                # reco histograms
                reco_dict = {}
                reco_dict['matter'] = rdf_reco_pt_bin.Filter("fGenPt > 0").Histo1D(
                    (f'h_reco_counts_ct_ptbin_{i_pt}_matter', 'Ct counts after Reconstruction Selection(Matter);Ct (cm);Counts', nbin, ct_edges), "fGenCt").GetValue()
                reco_dict['antimatter'] = rdf_reco_pt_bin.Filter("fGenPt <= 0").Histo1D(
                    (f'h_reco_counts_ct_ptbin_{i_pt}_antimat', 'Ct counts after Reconstruction Selection(Antimatter);Ct (cm);Counts', nbin, ct_edges), "fGenCt").GetValue()
                reco_dict['both'] = rdf_reco_pt_bin.Histo1D(
                    (f'h_reco_counts_ct_ptbin_{i_pt}_both', 'Ct counts after Reconstruction Selection(Matter+Antimatter);Ct (cm);Counts', nbin, ct_edges), "fGenCt").GetValue()
                # store histograms
                self.h_evsel_counts_ct_pt_bins.append(evsel_dict)
                self.h_reco_counts_ct_pt_bins.append(reco_dict)
                # acceptance histograms (Divide with binomial errors)
                acc_dict = {}
                for key in ['matter', 'antimatter', 'both']:
                    # clone reco to preserve binning and create a target histo for the acceptance
                    acc = reco_dict[key].Clone(f'h_accptance_ct_ptbin_{i_pt}_{key}')
                    acc.SetTitle(f'Accptance vs ct for pt bin {pt_min}-{pt_max} GeV/c ({key})')
                    acc.GetYaxis().SetTitle('#epsilon (ct)')
                    acc.Divide(reco_dict[key], evsel_dict[key], 1.0, 1.0, "B")
                    acc_dict[key] = acc
                self.h_accptance_ct_pt_bins.append(acc_dict)
        else:
            if self.pt_bins is not None:
                nbin = len(self.pt_bins) - 1
                pt_edges = np.array(self.pt_bins, dtype=np.float32)
                # event selection histograms (matter / antimatter / both)
                self.h_evsel_counts_pt = {}
                self.h_evsel_counts_pt['matter'] = self.mc_rdf_eventsel.Filter("fGenPt > 0").Histo1D(
                    ('h_evsel_counts_pt_matter', 'Pt counts after Event Selection(Matter);Pt (GeV/c);Counts', nbin, pt_edges), "fAbsGenPt").GetValue()
                self.h_evsel_counts_pt['antimatter'] = self.mc_rdf_eventsel.Filter("fGenPt <= 0").Histo1D(
                    ('h_evsel_counts_pt_antimat', 'Pt counts after Event Selection(Antimatter);Pt (GeV/c);Counts', nbin, pt_edges), "fAbsGenPt").GetValue()
                self.h_evsel_counts_pt['both'] = self.mc_rdf_eventsel.Histo1D(
                    ('h_evsel_counts_pt_both', 'Pt counts after Event Selection(Matter+Antimatter);Pt (GeV/c);Counts', nbin, pt_edges), "fAbsGenPt").GetValue()
                # reco histograms
                self.h_reco_counts_pt = {}
                self.h_reco_counts_pt['matter'] = self.mc_rdf_recon.Filter("fGenPt > 0").Histo1D(
                    ('h_reco_counts_pt_matter', 'Pt counts after Reconstruction Selection(Matter);Pt (GeV/c);Counts', nbin, pt_edges), "fAbsGenPt").GetValue()
                self.h_reco_counts_pt['antimatter'] = self.mc_rdf_recon.Filter("fGenPt <= 0").Histo1D(
                    ('h_reco_counts_pt_antimat', 'Pt counts after Reconstruction Selection(Antimatter);Pt (GeV/c);Counts', nbin, pt_edges), "fAbsGenPt").GetValue()
                self.h_reco_counts_pt['both'] = self.mc_rdf_recon.Histo1D(
                    ('h_reco_counts_pt_both', 'Pt counts after Reconstruction Selection(Matter+Antimatter);Pt (GeV/c);Counts', nbin, pt_edges), "fAbsGenPt").GetValue()
                # acceptance histograms (Divide with binomial errors)
                self.h_accptance_pt = {}
                for key in ['matter', 'antimatter', 'both']:
                    # clone reco to preserve binning and create a target histo for the acceptance
                    acc = self.h_reco_counts_pt[key].Clone(f'h_accptance_pt_{key}')
                    acc.SetTitle(f'Accptance vs pt ({key})')
                    acc.GetYaxis().SetTitle('#epsilon (pt)')
                    acc.Divide(self.h_reco_counts_pt[key], self.h_evsel_counts_pt[key], 1.0, 1.0, "B")
                    self.h_accptance_pt[key] = acc
            if self.ct_bins is not None:
                nbin = len(self.ct_bins) - 1
                ct_edges = np.array(self.ct_bins, dtype=np.float32)
                # event selection histograms (matter / antimatter / both)
                self.h_evsel_counts_ct = {}
                self.h_evsel_counts_ct['matter'] = self.mc_rdf_eventsel.Filter("fGenPt > 0").Histo1D(
                    ('h_evsel_counts_ct_matter', 'Ct counts after Event Selection(Matter);Ct (cm);Counts', nbin, ct_edges), "fGenCt").GetValue()
                self.h_evsel_counts_ct['antimatter'] = self.mc_rdf_eventsel.Filter("fGenPt <= 0").Histo1D(
                    ('h_evsel_counts_ct_antimat', 'Ct counts after Event Selection(Antimatter);Ct (cm);Counts', nbin, ct_edges), "fGenCt").GetValue()
                self.h_evsel_counts_ct['both'] = self.mc_rdf_eventsel.Histo1D(
                    ('h_evsel_counts_ct_both', 'Ct counts after Event Selection(Matter+Antimatter);Ct (cm);Counts', nbin, ct_edges), "fGenCt").GetValue()
                # reco histograms
                self.h_reco_counts_ct = {}
                self.h_reco_counts_ct['matter'] = self.mc_rdf_recon.Filter("fGenPt > 0").Histo1D(
                    ('h_reco_counts_ct_matter', 'Ct counts after Reconstruction Selection(Matter);Ct (cm);Counts', nbin, ct_edges), "fGenCt").GetValue()
                self.h_reco_counts_ct['antimatter'] = self.mc_rdf_recon.Filter("fGenPt <= 0").Histo1D(
                    ('h_reco_counts_ct_antimat', 'Ct counts after Reconstruction Selection(Antimatter);Ct (cm);Counts', nbin, ct_edges), "fGenCt").GetValue()
                self.h_reco_counts_ct['both'] = self.mc_rdf_recon.Histo1D(
                    ('h_reco_counts_ct_both', 'Ct counts after Reconstruction Selection(Matter+Antimatter);Ct (cm);Counts', nbin, ct_edges), "fGenCt").GetValue()
                # acceptance histograms (Divide with binomial errors)
                self.h_accptance_ct = {}
                for key in ['matter', 'antimatter', 'both']:
                    # clone reco to preserve binning and create a target histo for the acceptance
                    acc = self.h_reco_counts_ct[key].Clone(f'h_accptance_ct_{key}')
                    acc.SetTitle(f'Accptance vs ct ({key})')
                    acc.GetYaxis().SetTitle('#epsilon (ct)')
                    acc.Divide(self.h_reco_counts_ct[key], self.h_evsel_counts_ct[key], 1.0, 1.0, "B")
                    self.h_accptance_ct[key] = acc
    
    def delete_dynamic_members(self):
        self.mc_rdf = None
        self.mc_rdf_eventsel = None
        self.mc_rdf_recon = None
        self.spectrum_func = None
        self.h_evsel_counts_pt = None
        self.h_reco_counts_pt = None
        self.h_accptance_pt = None
        self.h_evsel_counts_ct = None
        self.h_reco_counts_ct = None
        self.h_accptance_ct = None
        self.h_evsel_counts_ct_pt_bins = []
        self.h_reco_counts_ct_pt_bins = []
        self.h_accptance_ct_pt_bins = []
    
    def get_counts_eventsel_list(self,matter_type='both'):
        counts_list = []
        counts_error_list = []
        if isinstance(self.ct_bins[0], list):
            for i in range(len(self.pt_bins) - 1):
                bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_evsel_counts_ct_pt_bins[i][matter_type])
                counts_list.append(bin_values)
                counts_error_list.append(bin_errors)
        else:
            if self.pt_bins is not None:
                bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_evsel_counts_pt[matter_type])
                counts_list.append(bin_values)
                counts_error_list.append(bin_errors)
            if self.ct_bins is not None:
                bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_evsel_counts_ct[matter_type])
                counts_list.append(bin_values)
                counts_error_list.append(bin_errors)
        return counts_list, counts_error_list
    
    def get_counts_recon_list(self,matter_type='both'):
        counts_list = []
        counts_error_list = []
        if isinstance(self.ct_bins[0], list):
            for i in range(len(self.pt_bins) - 1):
                bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_reco_counts_ct_pt_bins[i][matter_type])
                counts_list.append(bin_values)
                counts_error_list.append(bin_errors)
        else:
            if self.pt_bins is not None:
                bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_reco_counts_pt[matter_type])
                counts_list.append(bin_values)
                counts_error_list.append(bin_errors)
            if self.ct_bins is not None:
                bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_reco_counts_ct[matter_type])
                counts_list.append(bin_values)
                counts_error_list.append(bin_errors)
        return counts_list, counts_error_list
    
    def get_accptance_list(self,matter_type='both'):
        accptance_list = []
        accptance_error_list = []
        if isinstance(self.ct_bins[0], list):
            for i in range(len(self.pt_bins) - 1):
                bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_accptance_ct_pt_bins[i][matter_type])
                accptance_list.append(bin_values)
                accptance_error_list.append(bin_errors)
        else:
            if self.pt_bins is not None:
                bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_accptance_pt[matter_type])
                accptance_list.append(bin_values)
                accptance_error_list.append(bin_errors)
            if self.ct_bins is not None:
                bin_centers, bin_values, bin_errors, bin_edges = utils.extract_info_TH1(self.h_accptance_ct[matter_type])
                accptance_list.append(bin_values)
                accptance_error_list.append(bin_errors)
        return accptance_list, accptance_error_list
