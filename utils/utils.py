import ROOT
from ROOT import TCanvas, TFile, TH1, TH2, TLegend, TLatex, TF1, gStyle, TPad, TLine, TGraphErrors
import ctypes
import numpy as np
import pandas as pd
from hipe4ml.tree_handler import TreeHandler
import os as _os
import tempfile
_cc_path = _os.path.join(_os.path.dirname(__file__), 'its_helpers.cc')
# Declare C++ helpers only if they are not already available in the ROOT namespace.
# This prevents duplicate declaration errors when this module and other scripts
# (e.g. test.py) both try to Declare the same file.
if not hasattr(ROOT, 'CountITSHits'):
    if _os.path.exists(_cc_path):
        with open(_cc_path, 'r') as _f:
            ROOT.gInterpreter.Declare(_f.read())
    else:
        # if file not present, raise and let callers handle it
        raise FileNotFoundError(_cc_path)

kBlueC = ROOT.TColor.GetColor('#1f78b4')
kOrangeC = ROOT.TColor.GetColor('#ff7f00')

## set numpy seed
np.random.seed(42)

def computeEfficiency(gen_hist, rec_hist, name, rebin=0):
    if rebin > 1:
        gen_hist.Rebin(rebin)
        rec_hist.Rebin(rebin)
    eff_hist = gen_hist.Clone(name)
    eff_hist.Reset()
    eff_hist.GetYaxis().SetTitle(r'#epsilon #times Acc')
    eff_hist.GetYaxis().SetRangeUser(0., 1.1)
    for iPt in range(1, rec_hist.GetNbinsX() + 1):
        gen_val = gen_hist.GetBinContent(iPt)
        if gen_val < 1e-24:
            continue
        rec_val = rec_hist.GetBinContent(iPt)
        eff_val = rec_val / gen_val
        eff_err = np.sqrt(eff_val * (1 - eff_val) / gen_val)
        # print('iPt: ', iPt, ' eff: ', eff_val, ' +- ', eff_err)
        eff_hist.SetBinContent(iPt, eff_val)
        eff_hist.SetBinError(iPt, eff_err)
    return eff_hist


def setHistStyle(hist, colour, marker=20, fillstyle=0, linewidth=1):
    hist.SetMarkerColor(colour)
    hist.SetLineColor(colour)
    hist.SetMarkerStyle(marker)
    hist.SetFillStyle(fillstyle)
    hist.SetLineWidth(linewidth)

def fill_th1_hist(h, df, var):
    if not type(df) == pd.DataFrame:
        df = df._full_data_frame
    for var_val in df[var]:
        h.Fill(var_val)


def fill_th1_hist_abs(h, df, var):
    if not type(df) == pd.DataFrame:
        df = df._full_data_frame
    for var_val in df[var]:
        h.Fill(abs(var_val))


def fill_th2_hist(h, df, var1, var2):
    if not type(df) == pd.DataFrame:
        df = df._full_data_frame
    for var1_val, var2_val in zip(df[var1], df[var2]):
        h.Fill(var1_val, var2_val)


def fill_th2_hist_abs(h, df, var1, var2):
    if not type(df) == pd.DataFrame:
        df = df._full_data_frame
    for var1_val, var2_val in zip(df[var1], df[var2]):
        h.Fill(abs(var1_val), var2_val)


def fill_res_hist(h, df, var1, var2):
    if not type(df) == pd.DataFrame:
        df = df._full_data_frame
    for var_val1, var_val2 in zip(df[var1], df[var2]):
        h.Fill((var_val1 - var_val2)/var_val1)


def fill_th2_res_hist(h, df, var1, var2):
    if not type(df) == pd.DataFrame:
        df = df._full_data_frame
    for var_val1, var_val2 in zip(df[var1], df[var2]):
        h.Fill(var_val1, (var_val2 - var_val1)/var_val1)

def fill_mass_weighted_hist(h, df, var, weight=[1, 1]):
    if not type(df) == pd.DataFrame:
        df = df._full_data_frame
    for var_val, w in zip(df[var], df['isSignal']):
        if w == 1:
            h.Fill(var_val, weight[0])
        else:
            h.Fill(var_val, weight[1])


def significance_error(signal, background, signal_error, background_error):

    sb = signal + background + 1e-10
    sb_sqrt = np.sqrt(sb)

    s_propag = (sb_sqrt + signal / (2 * sb_sqrt))/sb * signal_error
    b_propag = signal / (2 * sb_sqrt)/sb * background_error

    if signal+background == 0:
        return 0

    return np.sqrt(s_propag * s_propag + b_propag * b_propag)


def scale_hist_content(h, scale):
    # generate poissonian counts
    for i in range(1, h.GetNbinsX()+1):
        pois = ROOT.gRandom.Poisson(scale)
        pois_sqrt = np.sqrt(pois)
        h.SetBinContent(i, h.GetBinContent(i)+pois)
        h.SetBinError(i, np.sqrt(pois_sqrt*pois_sqrt +
                      h.GetBinError(i)*h.GetBinError(i)))


def set_style():
    ROOT.gStyle.SetOptStat(1)
    ROOT.gStyle.SetOptDate(0)
    ROOT.gStyle.SetOptFit(1)
    ROOT.gStyle.SetLabelSize(0.04, 'xyz')
    ROOT.gStyle.SetTitleSize(0.05, 'xyz')
    ROOT.gStyle.SetTitleFont(42, 'xyz')
    ROOT.gStyle.SetLabelFont(42, 'xyz')
    ROOT.gStyle.SetTitleOffset(1.05, 'x')
    ROOT.gStyle.SetTitleOffset(1.1, 'y')
    ROOT.gStyle.SetCanvasDefW(800)
    ROOT.gStyle.SetCanvasDefH(600)
    ROOT.gStyle.SetPadBottomMargin(0.12)
    ROOT.gStyle.SetPadLeftMargin(0.12)
    ROOT.gStyle.SetPadRightMargin(0.05)
    ROOT.gStyle.SetPadGridX(0)
    ROOT.gStyle.SetPadGridY(0)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetFrameBorderMode(0)
    ROOT.gStyle.SetPaperSize(20, 24)
    ROOT.gStyle.SetLegendBorderSize(0)
    ROOT.gStyle.SetLegendFillColor(0)
    ROOT.gStyle.SetEndErrorSize(0.)
    ROOT.gStyle.SetMarkerSize(1)


def ndarray2roo(ndarray, var, name='data'):
    if isinstance(ndarray, ROOT.RooDataSet):
        print('Already a RooDataSet')
        return ndarray

    assert isinstance(ndarray, np.ndarray), 'Did not receive NumPy array'
    assert len(ndarray.shape) == 1, 'Can only handle 1d array'
    x = np.zeros(1, dtype=np.float64)

    tree = ROOT.TTree('tree', 'tree')
    tree.Branch(f'{var.GetName()}', x, f'{var.GetName()}/D')

    for i in ndarray:
        x[0] = i
        tree.Fill()
    array_roo = ROOT.RooDataSet(name, 'dataset from tree', tree, ROOT.RooArgSet(var))
    return array_roo


### reweight a distribution with rejection sampling
def reweight_pt_spectrum(df, var, distribution):
    rej_flag = np.ones(len(df))
    random_arr = np.random.rand(len(df))
    max_bw = distribution.GetMaximum()
    print(f'max_bw:{max_bw}')

    for ind, (val, rand) in enumerate(zip(df[var],random_arr)):
        frac = distribution.Eval(val)/max_bw
        if rand > frac:
            rej_flag[ind] = -1
    ## check if it is a pandas dataframe
    if isinstance(df, pd.DataFrame):
        # df.loc[:, 'rej'] = rej_flag
        df['rej'] = rej_flag
        return
    df._full_data_frame['rej'] = rej_flag

# create histogram for momentum correction

def create_pt_shift_histo(df):
    h2MomResoVsPtHe3 = ROOT.TH2F('h2MomResoVsPtHe3', ';{}^{3}He #it{p}_{T} (GeV/#it{c});{}^{3}He #it{p}_{T}^{gen} - #it{p}_{T}^{reco} (GeV/#it{c})', 30, 1.3, 5, 50, -0.4, 0.4)
    df.eval('PtResHe3 = (fGenPtHe3 - fPtHe3)', inplace=True)
    fill_th2_hist(h2MomResoVsPtHe3, df, 'fPtHe3', 'PtResHe3')
    h2MomResoVsPtHe3.FitSlicesY()
    hShiftVsPtHe3 = ROOT.gDirectory.Get('h2MomResoVsPtHe3_1')
    hShiftVsPtHe3.SetName('hShiftVsPtHe3')
    return h2MomResoVsPtHe3, hShiftVsPtHe3

def heBB(rigidity, mass):
    p1 = -321.34
    p2 = 0.6539
    p3 = 1.591
    p4 = 0.8225
    p5 = 2.363

    betagamma = rigidity * 2 / mass
    beta = betagamma / np.sqrt(1 + betagamma**2)
    aa = beta**p4
    bb = np.log(p3 + (1 / betagamma)**p5)
    return (p2 - aa - bb) * p1 / aa

def computeNSigmaHe4(df):
    expBB = heBB(df['fTPCmomHe'], 3.727)
    nSigma = (df['fTPCsignalHe'] - expBB) / (0.08*df['fTPCsignalHe'])
    return nSigma

def getNEvents(an_files, is_trigger=False, cen_begin = None, cen_end = None):
    n_ev = 0
    if type(an_files) == str:
        an_files = [an_files]

    for an_file in an_files:
        an_file = ROOT.TFile(an_file)
        print(an_file)
        if is_trigger: 
            zorro_summ = an_file.Get('hyper-reco-task').Get('zorroSummary;1')
            n_ev += zorro_summ.getNormalisationFactor(0)
        elif cen_begin is not None and cen_end is not None:
            histo = an_file.Get('hyper-reco-task').Get('hCentFT0C')
            bin_begin = histo.FindBin(cen_begin)  
            bin_end = histo.FindBin(cen_end)      
            n_ev += histo.Integral(bin_begin, bin_end)
        else:
            n_ev += an_file.Get('hyper-reco-task').Get('hZvtx').Integral()
    print(f'centrality_{cen_begin}_{cen_end}: n_ev = {n_ev}')
    return n_ev




def correct_and_convert_df(df, calibrate_he3_pt = False, isMC=False, isH4L=False):

    kDefaultPID = 15
    kPionPID = 2
    kTritonPID = 6
    
    if isinstance(df, ROOT.RDataFrame):
        coloumn_name = list(df.GetColumnNames())
        print("Columns before correction:", coloumn_name)
        if 'fFlags' in coloumn_name:
            df = df.Define("fHePIDHypo", "(int)(fFlags >> 4)") \
                     .Define("fPiPIDHypo", "(int)(fFlags & 0xF)")
        # calibrate 3He pt: per-row conditional.  Use a safe per-row expression:
        if calibrate_he3_pt:
            # if triton hypothesis apply polynomial correction, otherwise apply default correction
            df = df.Define("fPtHe3",
                             "((fHePIDHypo==6) ? (fPtHe3 + (-0.1286 - 0.1269 * fPtHe3 + 0.06 * fPtHe3*fPtHe3)) "
                             ": (fPtHe3 + 2.98019e-02 + 7.66100e-01 * exp(-1.31641e+00 * fPtHe3))))")
        # 3He momentum & energies
        df = df.Define("fPxHe3", "fPtHe3 * cos(fPhiHe3)") \
                 .Define("fPyHe3", "fPtHe3 * sin(fPhiHe3)") \
                 .Define("fPzHe3", "fPtHe3 * sinh(fEtaHe3)") \
                 .Define("fPHe3",  "fPtHe3 * cosh(fEtaHe3)") \
                 .Define("fEnHe3", "sqrt(fPHe3*fPHe3 + 2.8083916*2.8083916)") \
                 .Define("fEnHe4", "sqrt(fPHe3*fPHe3 + 3.7273794*3.7273794)")
        # pion momentum & energy
        df = df.Define("fPxPi", "fPtPi * cos(fPhiPi)") \
                 .Define("fPyPi", "fPtPi * sin(fPhiPi)") \
                 .Define("fPzPi", "fPtPi * sinh(fEtaPi)") \
                 .Define("fPPi",  "fPtPi * cosh(fEtaPi)") \
                 .Define("fEnPi", "sqrt(fPPi*fPPi + 0.139570*0.139570)")
        # hypertriton kinematics
        df = df.Define("fPx", "fPxHe3 + fPxPi") \
                 .Define("fPy", "fPyHe3 + fPyPi") \
                 .Define("fPz", "fPzHe3 + fPzPi") \
                 .Define("fP",  "sqrt(fPx*fPx + fPy*fPy + fPz*fPz)") \
                 .Define("fEn", "fEnHe3 + fEnPi") \
                 .Define("fEn4", "fEnHe4 + fEnPi")
        # derived momentum variables
        df = df.Define("fPt", "sqrt(fPx*fPx + fPy*fPy)") \
                 .Define("fEta", "acosh(fP / fPt)") \
                 .Define("fCosLambda", "fPt / fP") \
                 .Define("fCosLambdaHe", "fPtHe3 / fPHe3")
        # decay lengths, ct, mass
        if not isH4L:
            df = df.Define("fDecLen", "sqrt(fXDecVtx*fXDecVtx + fYDecVtx*fYDecVtx + fZDecVtx*fZDecVtx)") \
                     .Define("fCt", "fDecLen * 2.99131 / fP")
        else:
            df = df.Define("fDecLen", "sqrt(fXDecVtx*fXDecVtx + fYDecVtx*fYDecVtx + fZDecVtx*fZDecVtx)") \
                     .Define("fCt", "fDecLen * 3.922 / fP")
        df = df.Define("fDecRad", "sqrt(fXDecVtx*fXDecVtx + fYDecVtx*fYDecVtx)") \
                 .Define("fCosPA", "(fPx * fXDecVtx + fPy * fYDecVtx + fPz * fZDecVtx) / (fP * fDecLen)") \
                 .Define("fMassH3L", "sqrt(fEn*fEn - fP*fP)") \
                 .Define("fMassH4L", "sqrt(fEn4*fEn4 - fP*fP)")
        # simple signed momenta
        df = df.Define("fTPCSignMomHe3", "fTPCmomHe * (-1 + 2*fIsMatter)") \
                 .Define("fGloSignMomHe3", "fPHe3 / 2. * (-1 + 2*fIsMatter)")
        # if MC add generator-level derived vars
        if isMC:
            df = df.Define("fGenDecLen", "sqrt(fGenXDecVtx*fGenXDecVtx + fGenYDecVtx*fGenYDecVtx + fGenZDecVtx*fGenZDecVtx)") \
                     .Define("fGenPz", "fGenPt * sinh(fGenEta)") \
                     .Define("fGenP", "sqrt(fGenPt*fGenPt + fGenPz*fGenPz)") \
                     .Define("fAbsGenPt", "abs(fGenPt)")
            if not isH4L:
                df = df.Define("fGenCt", "fGenDecLen * 2.99131 / fGenP")
            else:
                df = df.Define("fGenCt", "fGenDecLen * 3.922 / fGenP")
        # Define df columns using the C++ helpers if the bit-packed fields exist
        # Note: these Define calls will silently fail if the input RDataFrame doesn't have those columns
        if 'fITSclusterSizesHe' in coloumn_name and 'fITSclusterSizesPi' in coloumn_name:
            df = df.Define("fAvgClusterSizeHe", "AvgITSClusterSize(fITSclusterSizesHe)") \
                         .Define("nITSHitsHe", "CountITSHits(fITSclusterSizesHe)") \
                         .Define("fAvgClusterSizePi", "AvgITSClusterSize(fITSclusterSizesPi)") \
                         .Define("nITSHitsPi", "CountITSHits(fITSclusterSizesPi)") \
                         .Define("fAvgClSizeCosLambda", "fAvgClusterSizeHe * fCosLambdaHe")
        # # remove temporary momentum/energy columns from the RDataFrame if possible
        # _cols_to_remove = ['fPxHe3', 'fPyHe3', 'fPzHe3', 'fEnHe3',
        #                    'fPxPi', 'fPyPi', 'fPzPi', 'fPPi', 'fEnPi',
        #                    'fPx', 'fPy', 'fPz', 'fP', 'fEn']
        # _tmpf = tempfile.NamedTemporaryFile(suffix='.root', delete=False)
        # _tmpf.close()
        # _tmpname = _tmpf.name
        # _cols_after_define = list(df.GetColumnNames())
        # _keep_cols = [c for c in _cols_after_define if c not in _cols_to_remove]
        # df.Snapshot("tree", _tmpname, _keep_cols)
        # df = ROOT.RDataFrame("tree", _tmpname)
        # try:
        #     _os.remove(_tmpname)
        # except OSError:
        #     pass
        colounm_name_after = list(df.GetColumnNames())
        print("Columns after correction:", colounm_name_after)
        return df
    else:
        if not type(df) == pd.DataFrame:
            df = df._full_data_frame
        
    
        if 'fFlags' in df.columns:
            df['fHePIDHypo'] = np.right_shift(df['fFlags'], 4)
            df['fPiPIDHypo'] = np.bitwise_and(df['fFlags'], 0b1111)
    
        # correct 3He momentum    
    
        if calibrate_he3_pt:
            # print(df.query('fIsReco==True')['fHePIDHypo'])
            no_pid_mask = np.logical_and(df['fHePIDHypo'] != kDefaultPID, df['fHePIDHypo'] != kPionPID)
    
            if (no_pid_mask.sum() == 0):
                print("PID in tracking not detected, using old momentum re-calibration")
                df["fPtHe3"] += 2.98019e-02 + 7.66100e-01 * np.exp(-1.31641e+00 * df["fPtHe3"]) ### functional form given by mpuccio
            else:
                print("PID in tracking detected, using new momentum re-calibration")
                df_Trit_PID = df.query('fHePIDHypo==6')
                df_else = df.query('fHePIDHypo!=6')
                ##pt_new = pt + kp0 + kp1 * pt + kp2 * pt^2 curveParams = {'kp0': -0.200281,'kp1': 0.103039,'kp2': -0.012325}, functional form given by G.A. Lucia
                df_Trit_PID["fPtHe3"] += -0.1286 - 0.1269 * df_Trit_PID["fPtHe3"] + 0.06 * df_Trit_PID["fPtHe3"]**2
                df_new = pd.concat([df_Trit_PID, df_else])
                ## assign the new dataframe to the original one
                df[:] = df_new.values
    
            
        print(df)
        # 3He momentum
        df.eval('fPxHe3 = fPtHe3 * cos(fPhiHe3)', inplace=True)
        df.eval('fPyHe3 = fPtHe3 * sin(fPhiHe3)', inplace=True)
        df.eval('fPzHe3 = fPtHe3 * sinh(fEtaHe3)', inplace=True)
        df.eval('fPHe3 = fPtHe3 * cosh(fEtaHe3)', inplace=True)
        df.eval('fEnHe3 = sqrt(fPHe3**2 + 2.8083916**2)', inplace=True)
        df.eval('fEnHe4 = sqrt(fPHe3**2 + 3.7273794**2)', inplace=True)
        # pi momentum
        df.eval('fPxPi = fPtPi * cos(fPhiPi)', inplace=True)
        df.eval('fPyPi = fPtPi * sin(fPhiPi)', inplace=True)
        df.eval('fPzPi = fPtPi * sinh(fEtaPi)', inplace=True)
        df.eval('fPPi = fPtPi * cosh(fEtaPi)', inplace=True)
        df.eval('fEnPi = sqrt(fPPi**2 + 0.139570**2)', inplace=True)
        # hypertriton momentum
        df.eval('fPx = fPxHe3 + fPxPi', inplace=True)
        df.eval('fPy = fPyHe3 + fPyPi', inplace=True)
        df.eval('fPz = fPzHe3 + fPzPi', inplace=True)
        df.eval('fP = sqrt(fPx**2 + fPy**2 + fPz**2)', inplace=True)
        df.eval('fEn = fEnHe3 + fEnPi', inplace=True)
        df.eval('fEn4 = fEnHe4 + fEnPi', inplace=True)
        # Momentum variables to be stored
        df.eval('fPt = sqrt(fPx**2 + fPy**2)', inplace=True)
        df.eval('fEta = arccosh(fP/fPt)', inplace=True)
        df.eval('fCosLambda = fPt/fP', inplace=True)
        df.eval('fCosLambdaHe = fPtHe3/fPHe3', inplace=True)
    
        df['fNSigmaHe4'] = computeNSigmaHe4(df)
    
        # Variables of interest
        df.eval('fDecLen = sqrt(fXDecVtx**2 + fYDecVtx**2 + fZDecVtx**2)', inplace=True)
        if not isH4L:
            df.eval('fCt = fDecLen * 2.99131 / fP', inplace=True)
        else:
            print('Using H4L decay length')
            df.eval('fCt = fDecLen * 3.922 / fP', inplace=True)
    
        df.eval('fDecRad = sqrt(fXDecVtx**2 + fYDecVtx**2)', inplace=True)
        df.eval('fCosPA = (fPx * fXDecVtx + fPy * fYDecVtx + fPz * fZDecVtx) / (fP * fDecLen)', inplace=True)
        df.eval('fMassH3L = sqrt(fEn**2 - fP**2)', inplace=True)
        df.eval('fMassH4L = sqrt(fEn4**2 - fP**2)', inplace=True)
        print(df.columns)
    
        ## signed TPC mom
        df.eval('fTPCSignMomHe3 = fTPCmomHe * (-1 + 2*fIsMatter)', inplace=True)
        df.eval('fGloSignMomHe3 = fPHe3 / 2 * (-1 + 2*fIsMatter)', inplace=True)
    
        if "fITSclusterSizesHe" in df.columns:
        ## loop over the candidates and compute the average cluster size
            clSizesHe = df['fITSclusterSizesHe'].to_numpy()
            clSizesPi = df['fITSclusterSizesPi'].to_numpy()
            clSizeHeAvg = np.zeros(len(clSizesHe))
            clSizePiAvg = np.zeros(len(clSizesPi))
            nHitsHe = np.zeros(len(clSizesHe))
            nHitsPi = np.zeros(len(clSizesPi))
            for iLayer in range(7):
                clSizeHeAvg += np.right_shift(clSizesHe, 4*iLayer) & 0b1111
                clSizePiAvg += np.right_shift(clSizesPi, 4*iLayer) & 0b1111
                nHitsHe += np.right_shift(clSizesHe, 4*iLayer) & 0b1111 > 0
                nHitsPi += np.right_shift(clSizesPi, 4*iLayer) & 0b1111 > 0
    
            clSizeHeAvg = np.where(nHitsHe > 0, clSizeHeAvg / nHitsHe, clSizeHeAvg)
            clSizePiAvg = np.where(nHitsPi > 0, clSizePiAvg / nHitsPi, clSizePiAvg)
            df['fAvgClusterSizeHe'] = clSizeHeAvg
            df['fAvgClusterSizePi'] = clSizePiAvg
            df['nITSHitsHe'] = nHitsHe
            df['nITSHitsPi'] = nHitsPi
            df.eval('fAvgClSizeCosLambda = fAvgClusterSizeHe * fCosLambdaHe', inplace=True)
    
        if "fPsiFT0C" in df.columns:
            df.eval('fPhi = arctan2(fPy, fPx)', inplace=True)
            df.eval('fV2 = cos(2*(fPhi - fPsiFT0C))', inplace=True)
    
    
        if isMC:
            df.eval('fGenDecLen = sqrt(fGenXDecVtx**2 + fGenYDecVtx**2 + fGenZDecVtx**2)', inplace=True)
            df.eval('fGenPz = fGenPt * sinh(fGenEta)', inplace=True)
            df.eval('fGenP = sqrt(fGenPt**2 + fGenPz**2)', inplace=True)
            df.eval("fAbsGenPt = abs(fGenPt)", inplace=True)
    
            if not isH4L:
                df.eval('fGenCt = fGenDecLen * 2.99131 / fGenP', inplace=True)
            else:
                df.eval('fGenCt = fGenDecLen * 3.922 / fGenP', inplace=True)
    
    
        # remove useless columns
        df.drop(columns=['fPxHe3', 'fPyHe3', 'fPzHe3', 'fEnHe3', 'fPxPi', 'fPyPi', 'fPzPi', 'fPPi', 'fEnPi', 'fPx', 'fPy', 'fPz', 'fP', 'fEn'])


def compute_pvalue_from_sign(significance):
    return ROOT.Math.chisquared_cdf_c(significance**2, 1) / 2

def convert_sel_to_string(selection):
    sel_string = ''
    conj = ' and '
    for _, val in selection.items():
        sel_string = sel_string + val + conj
    return sel_string[:-len(conj)]

def saveCanvasAsPDF(histo, plots_dir, is2D=False):
    histo_name = histo.GetName()
    canvas_name = histo_name.replace('h', 'c', 1)
    canvas = ROOT.TCanvas(canvas_name, canvas_name, 800, 600)
    canvas.SetBottomMargin(0.13)
    canvas.SetLeftMargin(0.13)
    if not is2D:
        histo.Draw('histo')
    else:
        histo.Draw('colz')
    canvas.SaveAs(f'{plots_dir}/{canvas_name}.pdf')

def get_canvas(title, sizeX, sizeY, gridx, gridy, topMgn, botMgn, leftMgn, rightMgn):
    c1 = TCanvas(title, title, sizeX, sizeY)
    c1.SetTopMargin(topMgn)
    c1.SetRightMargin(rightMgn)
    c1.SetLeftMargin(leftMgn)
    c1.SetBottomMargin(botMgn)
    if gridx:
        c1.SetGridx()
    if gridy:
        c1.SetGridy()
    return c1

def get_canvas_std(title, size, margin, gridopt, tickopt, border = None, fillcolor = None):
    c1 = TCanvas(title, "", size[0], size[1])
    c1.SetTopMargin(margin[0])
    c1.SetBottomMargin(margin[1])
    c1.SetLeftMargin(margin[2])
    c1.SetRightMargin(margin[3])
    if gridopt[0]:
        c1.SetGridx()
    if gridopt[1]:
        c1.SetGridy()
    if tickopt[0]:
        c1.SetTickx(1)
    if tickopt[1]:
        c1.SetTicky(1)
    if border:
        c1.SetBorderMode(border[0])
        c1.SetBorderSize(border[1])
        c1.SetFrameBorderMode(border[2])
        c1.SetFrameBorderSize(border[3])
    if fillcolor:
        c1.SetFillColor(fillcolor[0])
        c1.SetFrameFillColor(fillcolor[1])
    return c1

def get_pad(name, xpos1, ypos1, xpos2, ypos2, topMar, botMar, leftMar, rightMar):
    tpad = TPad(name, "", xpos1, ypos1, xpos2, ypos2)
    tpad.SetFillColor(0)
    tpad.SetBorderMode(0)
    tpad.SetBorderSize(2)
    tpad.SetTicks(1, 1)
    tpad.SetFrameBorderMode(0)
    tpad.SetRightMargin(rightMar)
    tpad.SetTopMargin(topMar)
    tpad.SetLeftMargin(leftMar)
    tpad.SetBottomMargin(botMar)
    return tpad

def get_legend(x_begin, y_begin, x_end, y_end, bordersize, fillcolor, textfont, textsize, ncolumns):
    legend = ROOT.TLegend(x_begin,y_begin,x_end,y_end)
    if bordersize != None:
        legend.SetBorderSize(bordersize)
    if fillcolor != None:
        legend.SetFillColor(fillcolor)
    if textfont != None:
        legend.SetTextFont(textfont)
    if textsize != None:
        legend.SetTextSize(textsize)
    if ncolumns:
        legend.SetNColumns(ncolumns)
    return legend
     

def set_marker_th1(h1, hTitle, markStyle, markSize, markColor, lineColor):
    h1.SetTitle(hTitle)
    h1.SetMarkerStyle(markStyle)
    h1.SetMarkerSize(markSize)
    h1.SetMarkerColor(markColor)
    h1.SetLineColor(lineColor)

def set_title_th1(h1, yTitle, yTitleSize, yOffset, xTitle, xTitleSize, xOffset, CenterTitle, stat = None):
    if yTitle is not None:
        h1.GetYaxis().SetTitle(yTitle)
    h1.GetYaxis().SetTitleSize(yTitleSize)
    h1.GetYaxis().SetTitleOffset(yOffset)
    if CenterTitle:
        h1.GetYaxis().CenterTitle(True)
        h1.GetXaxis().CenterTitle(True)
    h1.GetYaxis().SetTitleFont(42)
    if xTitle is not None:
        h1.GetXaxis().SetTitle(xTitle)
    h1.GetXaxis().SetTitleSize(xTitleSize)
    h1.GetXaxis().SetTitleOffset(xOffset)
    h1.GetXaxis().SetTitleFont(42)
    if stat != None:
        h1.SetStats(stat)

def set_axis_th1(h1, yAxisLow, yAxisHigh, xAxisLow, xAxisHigh, yLabelSize, xLabelSize):
    if xAxisLow is not None and  xAxisHigh is not None:
        h1.GetXaxis().SetRangeUser(xAxisLow, xAxisHigh)
    if yAxisLow is not None and  yAxisHigh is not None:
        h1.GetYaxis().SetRangeUser(yAxisLow, yAxisHigh)
    h1.GetXaxis().SetLabelFont(42)
    h1.GetXaxis().SetLabelSize(xLabelSize)
    h1.GetYaxis().SetLabelFont(42)
    h1.GetYaxis().SetLabelSize(yLabelSize)

def draw_my_line(xstart, ystart, xend, yend, style, width, color):
    line = TLine(xstart, ystart, xend, yend)
    line.SetLineStyle(style)
    line.SetLineWidth(width)
    line.SetLineColor(color)
    line.Draw()

def draw_my_text(xPos, yPos, size, text):
    tex = TLatex(xPos, yPos, text)
    tex.SetTextFont(42)
    tex.SetTextSize(size)
    tex.SetLineWidth(2)
    tex.Draw()

def draw_my_text_ndc(xPos, yPos, size, text, color=None):
    tex = TLatex()
    tex.SetTextFont(42)
    tex.SetTextSize(size)
    tex.SetLineWidth(2)
    if color:
        tex.SetTextColor(color)
    tex.DrawLatexNDC(xPos, yPos, text)

def draw_box(xstart, ystart, xstop, ystop, color, style):
    # Bottom edge
    line1 = TLine(xstart, ystart, xstop, ystart)
    line1.SetLineColor(color)
    line1.SetLineStyle(style)
    line1.Draw()
    # Top edge
    line2 = TLine(xstart, ystop, xstop, ystop)
    line2.SetLineColor(color)
    line2.SetLineStyle(style)
    line2.Draw()
    # Left edge
    line3 = TLine(xstart, ystart, xstart, ystop)
    line3.SetLineColor(color)
    line3.SetLineStyle(style)
    line3.Draw()
    # Right edge
    line4 = TLine(xstop, ystart, xstop, ystop)
    line4.SetLineColor(color)
    line4.SetLineStyle(style)
    line4.Draw()

def get_yaxis_range(histograms):
    """
    获取一组 TH1 直方图的 Y 轴全局最大值和最小值。
    
    参数:
        histograms (list of TH1): 包含多个 TH1 对象的列表。
    
    返回:
        tuple: (y_min, y_max) 所有直方图的 Y 轴最小值和最大值。
    """
    if not histograms:
        raise ValueError("The list of histograms is empty.")
    
    global_y_min = float("inf")
    global_y_max = float("-inf")
    
    for hist in histograms:
        if not isinstance(hist, ROOT.TH1):
            raise TypeError("All elements in the list must be instances of ROOT.TH1.")
        
        # 获取当前直方图的 Y 轴范围
        y_min = hist.GetMinimum()
        y_max = hist.GetMaximum()
        
        # 更新全局最小值和最大值
        global_y_min = min(global_y_min, y_min)
        global_y_max = max(global_y_max, y_max)
    
    return global_y_min, global_y_max

def cut_elements_to_same_range(handler1, handler2, element_names):
    """
    对两个 TreeHandler 中的多个元素进行 cut，使它们的分布范围一致。
    
    参数:
    handler1, handler2: 两个 TreeHandler 对象
    element_names: 需要处理的元素名称（可以是单个元素或元素名称的列表）
    
    返回:
    None，直接修改 handler1 和 handler2 中的元素
    """
    if isinstance(element_names, str):
        element_names = [element_names]
    
    df1 = handler1.get_data_frame()  
    df2 = handler2.get_data_frame()  
    
    for element_name in element_names:
        handler1_min = df1[element_name].min()
        handler1_max = df1[element_name].max()
        handler2_min = df2[element_name].min()
        handler2_max = df2[element_name].max()
        
        cut_min = max(handler1_min, handler2_min)
        cut_max = min(handler1_max, handler2_max)
        
        df1 = df1[(df1[element_name] >= cut_min) & (df1[element_name] <= cut_max)]
        df2 = df2[(df2[element_name] >= cut_min) & (df2[element_name] <= cut_max)]
        print(f"Applied cut to {element_name}: range [{cut_min}, {cut_max}]")
    handler1.set_data_frame(df1)
    handler2.set_data_frame(df2)

def merge_treehandlers(handlers):
    """
    合并多个 TreeHandler 对象中的数据到一个新的 TreeHandler 中。
    
    参数:
    handlers: List[TreeHandler]，多个 TreeHandler 对象的列表
    
    返回:
    一个新的 TreeHandler 对象，包含所有合并后的数据
    """
    # 用于存储所有 DataFrame 的列表
    data_frames = []
    
    # 遍历每个 TreeHandler，获取其 DataFrame 并添加到列表
    for handler in handlers:
        df = handler.get_data_frame()  # 获取 TreeHandler 的 DataFrame 数据
        data_frames.append(df)
    
    # 合并所有的 DataFrame
    merged_df = pd.concat(data_frames, ignore_index=True)
    
    # 创建一个新的 TreeHandler 并设置合并后的 DataFrame
    merged_handler = TreeHandler()
    merged_handler.set_data_frame(merged_df)
    
    print(f"Merged data from {len(handlers)} TreeHandlers into one.")
    
    return merged_handler

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
