// Lightweight ROOT macro to overlay Run3 spectra with Run2 references (no ratio pads)
#include <TGraphAsymmErrors.h>
#include <TGraphErrors.h>
#include <TH1.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TLine.h>
#include <TStyle.h>
#include <TMath.h>
#include <TFile.h>
#include <TF1.h>
#include <TLatex.h>
#include <TSystem.h>
#include <TString.h>

#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <algorithm>

struct LiteSpec {
    std::string label;
    std::string run3File;
    std::string run2File;   // leave empty to skip Run2 (e.g. 50-80)
    std::string run2Graph;
    std::string run3Hist;
    std::string run3Subdir;
    std::string run2Subdir;
    std::string bwName;     // TF1 name in BW file
};

const std::string kBWFitPathLite = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/H3l_2body_spectrum/utils/H3L_BWFit.root";

static TH1* GetHistSimple(const std::string& path, const std::string& histName, const std::string& subdir = "") {
    TFile* file = TFile::Open(path.c_str(), "READ");
    if (!file || file->IsZombie()) {
        std::cerr << "Cannot open file: " << path << std::endl;
        delete file;
        return nullptr;
    }
    TObject* obj = file->Get(histName.c_str());
    if (!obj && !subdir.empty()) {
        TDirectory* dir = file;
        std::string token;
        std::istringstream ss(subdir);
        while (std::getline(ss, token, '/')) {
            if (!dir) break;
            dir = dynamic_cast<TDirectory*>(dir->Get(token.c_str()));
        }
        if (dir) obj = dir->Get(histName.c_str());
    }
    if (!obj && file->Get("std")) obj = file->Get(Form("std/%s", histName.c_str()));
    if (!obj) {
        std::cerr << "Histogram not found: " << histName << " in " << path << std::endl;
        file->Close();
        delete file;
        return nullptr;
    }
    TH1* hist = dynamic_cast<TH1*>(obj);
    if (!hist) {
        std::cerr << histName << " is not a TH1" << std::endl;
        file->Close();
        delete file;
        return nullptr;
    }
    TH1* clone = dynamic_cast<TH1*>(hist->Clone(Form("%s_clone", histName.c_str())));
    if (clone) clone->SetDirectory(nullptr);
    file->Close();
    delete file;
    return clone;
}

static TGraphAsymmErrors* GetGraphSimple(const std::string& path, const std::string& graphName, const std::string& subdir = "") {
    TFile* file = TFile::Open(path.c_str(), "READ");
    if (!file || file->IsZombie()) {
        std::cerr << "Cannot open file: " << path << std::endl;
        delete file;
        return nullptr;
    }
    TObject* obj = file->Get(graphName.c_str());
    if (!obj && !subdir.empty()) {
        TDirectory* dir = file;
        std::string token;
        std::istringstream ss(subdir);
        while (std::getline(ss, token, '/')) {
            if (!dir) break;
            dir = dynamic_cast<TDirectory*>(dir->Get(token.c_str()));
        }
        if (dir) obj = dir->Get(graphName.c_str());
    }
    if (!obj) {
        std::cerr << "Graph not found: " << graphName << " in " << path << std::endl;
        file->Close();
        delete file;
        return nullptr;
    }
    auto* g = dynamic_cast<TGraphAsymmErrors*>(obj);
    auto* clone = g ? dynamic_cast<TGraphAsymmErrors*>(g->Clone(Form("%s_clone", graphName.c_str()))) : nullptr;
    file->Close();
    delete file;
    return clone;
}

static TF1* GetTF1Simple(const std::string& path, const std::string& funcName) {
    if (funcName.empty()) return nullptr;
    TFile* file = TFile::Open(path.c_str(), "READ");
    if (!file || file->IsZombie()) {
        std::cerr << "Cannot open file: " << path << std::endl;
        delete file;
        return nullptr;
    }
    TObject* obj = file->Get(funcName.c_str());
    if (!obj) {
        std::cerr << "TF1 not found: " << funcName << " in " << path << std::endl;
        file->Close();
        delete file;
        return nullptr;
    }
    auto* f = dynamic_cast<TF1*>(obj);
    auto* clone = f ? dynamic_cast<TF1*>(f->Clone(Form("%s_clone", funcName.c_str()))) : nullptr;
    file->Close();
    delete file;
    return clone;
}

static TGraphErrors* HistToGraph(const TH1* hist, const std::string& name) {
    if (!hist) return nullptr;
    auto* g = new TGraphErrors(hist->GetNbinsX());
    g->SetName(name.c_str());
    for (int i = 1; i <= hist->GetNbinsX(); ++i) {
        g->SetPoint(i - 1, hist->GetBinCenter(i), hist->GetBinContent(i));
        g->SetPointError(i - 1, 0.0, hist->GetBinError(i));
    }
    return g;
}

static void ShiftGraphX(TGraph* g, double dx) {
    if (!g) return;
    double x, y;
    for (int i = 0; i < g->GetN(); ++i) {
        g->GetPoint(i, x, y);
        g->SetPoint(i, x + dx, y);
    }
}

static void StyleGraph(TGraphErrors* g, Color_t color, int marker) {
    if (!g) return;
    g->SetMarkerColor(color);
    g->SetLineColor(color);
    g->SetMarkerStyle(marker);
    g->SetMarkerSize(1.0);
    g->SetLineWidth(2);
}

static void StyleGraph(TGraphAsymmErrors* g, Color_t color, int marker) {
    if (!g) return;
    g->SetMarkerColor(color);
    g->SetLineColor(color);
    g->SetMarkerStyle(marker);
    g->SetMarkerSize(1.0);
    g->SetLineWidth(2);
}

static void UpdateRange(TGraph* g, double& xmin, double& xmax, double& ymin, double& ymax) {
    if (!g) return;
    double x, y;
    for (int i = 0; i < g->GetN(); ++i) {
        g->GetPoint(i, x, y);
        xmin = std::min(xmin, x);
        xmax = std::max(xmax, x);
        double ylow = y, yhigh = y;
        if (auto* asym = dynamic_cast<TGraphAsymmErrors*>(g)) {
            ylow = y - asym->GetErrorYlow(i);
            yhigh = y + asym->GetErrorYhigh(i);
        } else if (auto* ge = dynamic_cast<TGraphErrors*>(g)) {
            ylow = y - ge->GetErrorY(i);
            yhigh = y + ge->GetErrorY(i);
        }
        if (yhigh > 0) ymax = std::max(ymax, yhigh);
        if (ylow > 0) ymin = std::min(ymin, ylow);
    }
}

static std::string FormatCentralityRange(const std::string& label) {
    if (label.empty()) return "";
    auto range = label;
    std::replace(range.begin(), range.end(), '_', '-');
    return range;
}

void SpectrumVsRun2Lite() {
    gStyle->SetOptStat(0);
    std::string outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/results/run3_vs_run2";
    gSystem->mkdir(outputDir.c_str(), true);

    std::vector<LiteSpec> specs = {
        {"0_10",
         "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_V0s_HadronPID/BdtSpectrum_LHC25g11_G4list/cen0-10/pt_analysis_pbpb.root",
         "/Users/zhengqingwang/alice/data/h3l_spec_run2/h3l_0_10.root",
         "Graph1D_y1",
         "h_corrected_counts",
         "std",
         "(Anti)hypertriton spectrum in 0-10% V0M centrality class",
         "BlastWave_H3L_0_10"},
        {"10_30",
         "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_V0s_HadronPID/BdtSpectrum_LHC25g11_G4list/cen10-30/pt_analysis_pbpb.root",
         "/Users/zhengqingwang/alice/data/h3l_spec_run2/h3l_10_30.root",
         "Graph1D_y1",
         "h_corrected_counts",
         "std",
         "(Anti)hypertriton spectrum in 10-30% V0M centrality class",
         "BlastWave_H3L_10_30"},
        {"30_50",
         "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_V0s_HadronPID/BdtSpectrum_LHC25g11_G4list/cen30-50/pt_analysis_pbpb.root",
         "/Users/zhengqingwang/alice/data/h3l_spec_run2/h3l_30_50.root",
         "Graph1D_y1",
         "h_corrected_counts",
         "std",
         "(Anti)hypertriton spectrum in 30-50% V0M centrality class",
         "BlastWave_H3L_30_50"},
        {"50_80",
         "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_V0s_HadronPID/BdtSpectrum_LHC25g11_G4list/cen50-80/pt_analysis_pbpb.root",
         "", // skip Run2 spectrum here
         "",
         "h_corrected_counts",
         "std",
         "(Anti)hypertriton spectrum in 50-80% V0M centrality class",
         ""}
    };

    for (const auto& spec : specs) {
        std::unique_ptr<TH1> h3(GetHistSimple(spec.run3File, spec.run3Hist, spec.run3Subdir));
        if (!h3) continue;
        std::unique_ptr<TGraphErrors> g3(HistToGraph(h3.get(), Form("run3_%s", spec.label.c_str())));
        ShiftGraphX(g3.get(), -0.05);
        StyleGraph(g3.get(), kMagenta + 2, 20);

        std::unique_ptr<TGraphAsymmErrors> g2;
        std::unique_ptr<TF1> f2;
        if (!spec.run2File.empty()) {
            g2.reset(GetGraphSimple(spec.run2File, spec.run2Graph, spec.run2Subdir));
            f2.reset(GetTF1Simple(kBWFitPathLite, spec.bwName));
            StyleGraph(g2.get(), kAzure + 2, 21);
            if (f2) {
                f2->SetLineColor(kAzure + 2);
                f2->SetLineStyle(2);
                f2->SetLineWidth(2);
                f2->SetRange(2.0, 8.0);
            }
        }

        double xmin = std::numeric_limits<double>::infinity();
        double xmax = 0.0;
        double ymin = std::numeric_limits<double>::infinity();
        double ymax = 0.0;
        UpdateRange(g3.get(), xmin, xmax, ymin, ymax);
        UpdateRange(g2.get(), xmin, xmax, ymin, ymax);
        if (!std::isfinite(ymin) || ymin <= 0) ymin = 1e-8;
        if (ymax <= 0) ymax = 1.0;
        double yMinDraw = std::max(1e-8, ymin * 0.4);
        double yMaxDraw = ymax * 1.2;

        TCanvas c(Form("c_lite_%s", spec.label.c_str()), "Run3 vs Run2", 900, 700);
        c.SetLeftMargin(0.12);
        c.SetRightMargin(0.05);
        c.SetTopMargin(0.08);
        c.SetBottomMargin(0.12);
        c.SetLogy();
        c.SetGrid();

        auto* frame = gPad->DrawFrame(2.0, yMinDraw, 8.0, yMaxDraw);
        frame->GetXaxis()->SetTitle("p_{T} (GeV/#it{c})");
        frame->GetYaxis()->SetTitle("#frac{1}{N_{ev}} dN/dy dp_{T}");
        frame->GetXaxis()->SetTitleFont(42);
        frame->GetYaxis()->SetTitleFont(42);
        frame->GetXaxis()->SetLabelFont(42);
        frame->GetYaxis()->SetLabelFont(42);
        frame->GetXaxis()->SetTitleSize(0.045);
        frame->GetYaxis()->SetTitleSize(0.045);
        frame->GetXaxis()->SetLabelSize(0.04);
        frame->GetYaxis()->SetLabelSize(0.04);
        frame->GetXaxis()->SetTitleOffset(1.1);
        frame->GetYaxis()->SetTitleOffset(1.2);
        std::string centLabel = FormatCentralityRange(spec.label);
        frame->SetTitle(Form("(Anti)^{3}_{#Lambda}H Spectrum Run 3 vs Run 2 Centrality: %s%%", centLabel.c_str()));
        frame->SetTitleFont(62);
        frame->SetTitleSize(0.052);
        g3->GetXaxis()->SetLimits(2.0, 8.0);
        g3->Draw("P SAME");
        if (g2) g2->Draw("P SAME");
        if (f2) f2->Draw("SAME");

        const bool isPeripheral = (spec.label == "50_80");
        double legX1 = isPeripheral ? 0.60 : 0.16;
        double legY1 = isPeripheral ? 0.70 : 0.25;
        double legX2 = isPeripheral ? 0.92 : 0.46;
        double legY2 = isPeripheral ? 0.85 : 0.55;

        auto legend = std::make_unique<TLegend>(legX1, legY1, legX2, legY2);
        legend->SetFillStyle(0);
        legend->SetBorderSize(0);
        legend->SetTextFont(42);
        legend->AddEntry(g3.get(), "Run 3 (5.36 TeV)", "lep");
        if (g2) legend->AddEntry(g2.get(), "Run 2 (5.02 TeV)", "lep");
        if (f2) legend->AddEntry(f2.get(), "Run 2 Blast-Wave fit", "l");
        legend->Draw();

        TLatex header;
        header.SetTextFont(42);
        header.SetTextSize(0.042);
        header.SetNDC();
        double headerX = isPeripheral ? 0.60 : legX1;
        double headerY = isPeripheral ? 0.87 : std::min(legY2 + 0.025, 0.95);
        header.DrawLatex(headerX, headerY, "LHC23_PbPb_pass5");

        std::string out = outputDir + "/H3l_run3_apass5_vs_run2_lite_" + spec.label + "_cppV0s_LHC25g11_G4list" + ".pdf";
        c.SaveAs(out.c_str());
    }
}
