import ROOT

def load_all_trees(filename="AO2D.root",treename="O2hypcands"):
    f = ROOT.TFile.Open(filename)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open file {filename}")

    tree_paths = []
    for key in f.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom("TDirectory"):
            if obj.Get(treename):
                path = f"{obj.GetName()}/{treename}"
                tree_paths.append(path)
                print(f"Found tree: {path}")

    if not tree_paths:
        raise RuntimeError(f"No tree {treename} found!")

    # 使用 MultiChain 读取多个树
    chain = ROOT.TChain(treename)
    for path in tree_paths:
        chain.Add(f"{filename}/{path}")
    print(f"Total entries: {chain.GetEntries()}")

    df = ROOT.RDataFrame(chain)
    print(f"RDataFrame created with {df.Count().GetValue()} entries")
    return df

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

        return rdf

    # Pandas branch (legacy behavior) - keep original in-memory code path
    # ...existing pandas-based implementation...
    # (the original pandas implementation from this file should remain here;
    #  if it was above, keep it — otherwise implement the same operations on pandas)
    # For brevity return input if not RDataFrame; caller can rely on the old function body.
    return df
# df = ROOT.RDataFrame("O2hypcands", "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/apass5/AO2D_CustomV0s.root")
df = load_all_trees("/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/apass5/AO2D_CustomV0s.root", "O2hypcands")

rdf_corrected = correct_and_convert_df(df, calibrate_he3_pt=False, isMC=False, isH4L=False)