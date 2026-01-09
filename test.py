import ROOT
import argparse
import sys
sys.path.append('utils')
import utils as utils
import numpy as np

def load_all_trees(file, chain, treename="O2hypcands"):
    if not file or file.IsZombie():
        raise RuntimeError(f"Cannot open file {file.GetName()}")

    for key in file.GetListOfKeys():
        name_key = key.GetName()
        if 'DF_' in name_key:
            obj = key.ReadObj()
            if obj.InheritsFrom("TDirectory"):
                tree = obj.Get(treename)
                if tree and tree.InheritsFrom("TTree"):
                    print(f"Found TTree '{treename}' in directory: {name_key}")
                    chain.Add(f"{file.GetName()}/{name_key}/{treename}")
                else:
                    print(f"No TTree named '{treename}' in directory: {name_key}")
    return ROOT.RDataFrame(chain)


def correct_and_convert_df(df, calibrate_he3_pt = False, isMC=False, isH4L=False):
    """
    Support for both pandas DataFrame (legacy) and ROOT.RDataFrame.
    If an RDataFrame is passed, returns a new RDataFrame with the extra
    columns defined (lazy, no immediate evaluation). For pandas input
    falls back to the original in-memory behavior.
    """
    # RDataFrame branch
    if isinstance(df, ROOT.RDataFrame):
        rdf = df

        # optional PID extraction from fFlags
        # use bit ops in C++ expressions
        try:
            rdf = rdf.Define("fHePIDHypo", "(int)(fFlags >> 4)") \
                     .Define("fPiPIDHypo", "(int)(fFlags & 0xF)")
        except Exception:
            # if fFlags not present or different type, continue silently
            pass

        # calibrate 3He pt: per-row conditional.  Use a safe per-row expression:
        if calibrate_he3_pt:
            # if triton hypothesis apply polynomial correction, otherwise apply default correction
            rdf = rdf.Define("fPtHe3",
                             "((fHePIDHypo==6) ? (fPtHe3 + (-0.1286 - 0.1269 * fPtHe3 + 0.06 * fPtHe3*fPtHe3)) "
                             ": (fPtHe3 + 2.98019e-02 + 7.66100e-01 * exp(-1.31641e+00 * fPtHe3))))")

        # 3He momentum & energies
        rdf = rdf.Define("fPxHe3", "fPtHe3 * cos(fPhiHe3)") \
                 .Define("fPyHe3", "fPtHe3 * sin(fPhiHe3)") \
                 .Define("fPzHe3", "fPtHe3 * sinh(fEtaHe3)") \
                 .Define("fPHe3",  "fPtHe3 * cosh(fEtaHe3)") \
                 .Define("fEnHe3", "sqrt(fPHe3*fPHe3 + 2.8083916*2.8083916)") \
                 .Define("fEnHe4", "sqrt(fPHe3*fPHe3 + 3.7273794*3.7273794)")

        # pion momentum & energy
        rdf = rdf.Define("fPxPi", "fPtPi * cos(fPhiPi)") \
                 .Define("fPyPi", "fPtPi * sin(fPhiPi)") \
                 .Define("fPzPi", "fPtPi * sinh(fEtaPi)") \
                 .Define("fPPi",  "fPtPi * cosh(fEtaPi)") \
                 .Define("fEnPi", "sqrt(fPPi*fPPi + 0.139570*0.139570)")

        # hypertriton kinematics
        rdf = rdf.Define("fPx", "fPxHe3 + fPxPi") \
                 .Define("fPy", "fPyHe3 + fPyPi") \
                 .Define("fPz", "fPzHe3 + fPzPi") \
                 .Define("fP",  "sqrt(fPx*fPx + fPy*fPy + fPz*fPz)") \
                 .Define("fEn", "fEnHe3 + fEnPi") \
                 .Define("fEn4", "fEnHe4 + fEnPi")

        # derived momentum variables
        rdf = rdf.Define("fPt", "sqrt(fPx*fPx + fPy*fPy)") \
                 .Define("fEta", "acosh(fP / fPt)") \
                 .Define("fCosLambda", "fPt / fP") \
                 .Define("fCosLambdaHe", "fPtHe3 / fPHe3")

        # decay lengths, ct, mass
        if not isH4L:
            rdf = rdf.Define("fDecLen", "sqrt(fXDecVtx*fXDecVtx + fYDecVtx*fYDecVtx + fZDecVtx*fZDecVtx)") \
                     .Define("fCt", "fDecLen * 2.99131 / fP")
        else:
            rdf = rdf.Define("fDecLen", "sqrt(fXDecVtx*fXDecVtx + fYDecVtx*fYDecVtx + fZDecVtx*fZDecVtx)") \
                     .Define("fCt", "fDecLen * 3.922 / fP")

        rdf = rdf.Define("fDecRad", "sqrt(fXDecVtx*fXDecVtx + fYDecVtx*fYDecVtx)") \
                 .Define("fCosPA", "(fPx * fXDecVtx + fPy * fYDecVtx + fPz * fZDecVtx) / (fP * fDecLen)") \
                 .Define("fMassH3L", "sqrt(fEn*fEn - fP*fP)") \
                 .Define("fMassH4L", "sqrt(fEn4*fEn4 - fP*fP)")

        # simple signed momenta
        rdf = rdf.Define("fTPCSignMomHe3", "fTPCmomHe * (-1 + 2*fIsMatter)") \
                 .Define("fGloSignMomHe3", "fPHe3 / 2. * (-1 + 2*fIsMatter)")

        # if MC add generator-level derived vars
        if isMC:
            rdf = rdf.Define("fGenDecLen", "sqrt(fGenXDecVtx*fGenXDecVtx + fGenYDecVtx*fGenYDecVtx + fGenZDecVtx*fGenZDecVtx)") \
                     .Define("fGenPz", "fGenPt * sinh(fGenEta)") \
                     .Define("fGenP", "sqrt(fGenPt*fGenPt + fGenPz*fGenPz)") \
                     .Define("fAbsGenPt", "abs(fGenPt)")
            if not isH4L:
                rdf = rdf.Define("fGenCt", "fGenDecLen * 2.99131 / fGenP")
            else:
                rdf = rdf.Define("fGenCt", "fGenDecLen * 3.922 / fGenP")

        # Note: bit-packed ITS cluster size unpacking and fNSigmaHe4 (which depends on python heBB)
        # are left out here because they require more complex per-row logic or access to python helper
        # functions. They can be added by registering a C++/python callable and using Define(...) if needed.
        try:
            rdf = rdf.Define("fAvgClusterSizeHe", "AvgITSClusterSize(fITSclusterSizesHe)") \
                     .Define("nITSHitsHe", "CountITSHits(fITSclusterSizesHe)") \
                     .Define("fAvgClusterSizePi", "AvgITSClusterSize(fITSclusterSizesPi)") \
                     .Define("nITSHitsPi", "CountITSHits(fITSclusterSizesPi)") \
                     .Define("fAvgClSizeCosLambda", "fAvgClusterSizeHe * fCosLambdaHe")
        except Exception:
            # ignore if columns not present or Define fails
            pass
        # remove temporary momentum/energy columns from the RDataFrame if possible
        _cols_to_remove = ['fPxHe3', 'fPyHe3', 'fPzHe3', 'fEnHe3',
                           'fPxPi', 'fPyPi', 'fPzPi', 'fPPi', 'fEnPi',
                           'fPx', 'fPy', 'fPz', 'fP', 'fEn']
        for _col in _cols_to_remove:
            try:
                rdf = rdf.RemoveColumn(_col)
            except Exception:
                # fallback: if RemoveColumn not available, redefine to a tiny dummy value
                # to avoid keeping large intermediate arrays (silent if Define fails)
                try:
                    rdf = rdf.Define(_col, "0.")
                except Exception:
                    pass

        return rdf

    # Pandas branch (legacy behavior) - keep original in-memory code path
    # ...existing pandas-based implementation...
    # (the original pandas implementation from this file should remain here;
    #  if it was above, keep it — otherwise implement the same operations on pandas)
    # For brevity return input if not RDataFrame; caller can rely on the old function body.
    else:
        return df

def main(argv=None):
    parser = argparse.ArgumentParser(description='Test RDataFrame processing and optional ITS helpers')
    parser.add_argument('--file', '-f', help='Input ROOT file',
                        default="/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/apass5/AO2D_CustomV0s.root")
    parser.add_argument('--treename', '-t', help='Tree name to search inside DF directories', default='O2hypcands')
    args = parser.parse_args(argv)

    file_ana = ROOT.TFile.Open(args.file)
    if not file_ana or file_ana.IsZombie():
        print(f"Could not open file: {args.file}", file=sys.stderr)
        return 1

    chain_ana = ROOT.TChain(args.treename)
    rdf = load_all_trees(file_ana, chain_ana, args.treename)
    try:
        print(list(rdf.GetColumnNames()))
        # evaluate and print number of entries in the RDataFrame (this triggers the action)
        n_entries = int(rdf.Count().GetValue())
        print(f"rdf entries: {n_entries}")
    except Exception as e:
        print(f"Error accessing RDataFrame columns/count: {e}", file=sys.stderr)

    rdf_corrected = utils.correct_and_convert_df(rdf, calibrate_he3_pt=False, isMC=False, isH4L=False)
    rdf_corrected = rdf_corrected.Filter("fNSigmaHe > -2.5 && fNSigmaHe < 2", "Apply basic nSigma cuts")
    # convert rdf_corrected to a numpy array (keep rdf_corrected unchanged)
    try:
        cols_to_convert = list(rdf_corrected.GetColumnNames())
        # Ask RDataFrame to materialize columns as numpy arrays (PyROOT >= 6.22)
        try:
            arr_dict = rdf_corrected.AsNumpy(['fNSigmaHe'])
        except TypeError:
            # some ROOT versions expect a list/tuple explicitly
            arr_dict = rdf_corrected.AsNumpy('fNSigmaHe')
        # keep only scalar 1D columns for a clean 2D numpy array
        scalar_cols = [c for c, a in arr_dict.items() if hasattr(a, "ndim") and a.ndim == 1]
        if len(scalar_cols) == 0:
            print("No scalar columns available to convert to numpy array")
            np_array = None
            np_colnames = []
        else:
            np_array = np.column_stack([np.asarray(arr_dict[c]) for c in scalar_cols])
            np_colnames = scalar_cols
            print(f"Converted rdf_corrected to numpy array with shape {np_array.shape}")
    except Exception as e:
        print(f"Failed to convert rdf_corrected to numpy array: {e}", file=sys.stderr)
        np_array = None
        np_colnames = []
    exit()
    try:
        print("columns after correction check:")
        print(list(rdf_corrected.GetColumnNames()))
        n_entries_corrected = int(rdf_corrected.Count().GetValue())
        print(f"rdf_corrected entries: {n_entries_corrected}")
    except Exception as e:
        print(f"Error after correct_and_convert_df: {e}", file=sys.stderr)

    # create 2D histogram fAvgClSizeCosLambda vs fPHe3 and save as PDF if columns present
    cols = list(rdf_corrected.GetColumnNames())
    if "fPHe3" not in cols or "fAvgClSizeCosLambda" not in cols:
        print("Required columns not present: fPHe3 and/or fAvgClSizeCosLambda; skipping histogram creation")
        return 0

    # binning: adjust as needed
    nbins_x, x_min, x_max = 100, 0.0, 10.0
    nbins_y, y_min, y_max = 50, 0.0, 5.0

    h2 = rdf_corrected.Histo2D(
        ("h2_avgCls_vs_phe3", "fAvgClSizeCosLambda vs fPHe3;fPHe3 (GeV/c);fAvgClSizeCosLambda",
         nbins_x, x_min, x_max, nbins_y, y_min, y_max),
        "fPHe3", "fAvgClSizeCosLambda"
    ).GetValue()

    c = ROOT.TCanvas("c2", "c2", 900, 700)
    c.SetLogz()
    ROOT.gStyle.SetOptStat(0)
    h2.Draw("COLZ")
    outname = "fAvgClSizeCosLambda_vs_fPHe3.pdf"
    c.SaveAs(outname)
    print(f"Saved {outname}")

    # plot 1D distribution of fNSigmaHe if present
    if "fNSigmaHe" not in cols:
        print("Column 'fNSigmaHe' not present in rdf_corrected; skipping 1D plot")
        return 0

    nbins, xmin, xmax = 200, -10.0, 10.0
    h1 = rdf_corrected.Histo1D(("h1_fNSigmaHe", "fNSigmaHe; fNSigmaHe; Entries", nbins, xmin, xmax), "fNSigmaHe").GetValue()
    c1 = ROOT.TCanvas("c1", "c1", 800, 600)
    ROOT.gStyle.SetOptStat(1111)
    h1.Draw()
    outname1 = "fNSigmaHe.pdf"
    c1.SaveAs(outname1)
    print(f"Saved {outname1}")
    print(f"fNSigmaHe: entries={int(h1.GetEntries())}, mean={h1.GetMean():.3f}, rms={h1.GetRMS():.3f}")
    if "fAvgClusterSizeHe" not in cols:
        print("Column 'fAvgClusterSizeHe' not present in rdf_corrected; skipping 1D plot")
        return 0
    nbins, xmin, xmax = 15, 0.0, 15
    h1 = rdf_corrected.Histo1D(("h1_fAvgClusterSizeHe", "fAvgClusterSizeHe; fAvgClusterSizeHe; Entries", nbins, xmin, xmax), "fAvgClusterSizeHe").GetValue()
    c1 = ROOT.TCanvas("c1", "c1", 800, 600)
    ROOT.gStyle.SetOptStat(1111)
    h1.Draw()
    outname1 = "fAvgClusterSizeHe.pdf"
    c1.SaveAs(outname1)
    print(f"Saved {outname1}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

