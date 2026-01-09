// ROOT macro that overlays Run3 apass5 spectra with Run2 measurements for several centrality bins
#include <TGraphAsymmErrors.h>
#include <TGraphErrors.h>
#include <TH1.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TLine.h>
#include <TStyle.h>
#include <TMath.h>
#include <TFile.h>
#include <TF1.h>
#include <TSystem.h>
#include <TKey.h>

#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <utility>
#include <limits>
#include <sstream>

namespace {

struct CentralitySpec {
	std::string label;
	std::string run3File;
	std::string run2File;
	std::string run2Graph;
	std::string run3Hist;
	std::string run3Subdir;
	std::string run2Subdir;
	std::string bwName; // name of TF1 in BW file (if available)
};

const std::string kBWFitPath = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/H3l_2body_spectrum/utils/H3L_BWFit.root";

TH1* CloneHistogram(const std::string& path,
					const std::string& histName,
					const std::string& subdir = "") {
	TFile* file = TFile::Open(path.c_str(), "READ");
	if (!file || file->IsZombie()) {
		std::cerr << "Cannot open Run3 file: " << path << std::endl;
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
	if (!obj) {
		if (auto* dir = dynamic_cast<TDirectory*>(file->Get("std"))) {
			obj = dir->Get(histName.c_str());
		}
	}
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
	if (clone) {
		clone->SetDirectory(nullptr);
	}
	file->Close();
	delete file;
	return clone;
}

std::unique_ptr<TGraphAsymmErrors> CloneRun2Graph(const std::string& path,
												  const std::string& graphName,
												  const std::string& subdir = "") {
	TFile* file = TFile::Open(path.c_str(), "READ");
	if (!file || file->IsZombie()) {
		std::cerr << "Cannot open Run2 file: " << path << std::endl;
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
	auto* graph = dynamic_cast<TGraphAsymmErrors*>(obj);
	if (!graph) {
		std::cerr << graphName << " is not a TGraphAsymmErrors" << std::endl;
		file->Close();
		delete file;
		return nullptr;
	}
	auto* clone = dynamic_cast<TGraphAsymmErrors*>(graph->Clone(Form("%s_clone", graphName.c_str())));
	file->Close();
	delete file;
	return std::unique_ptr<TGraphAsymmErrors>(clone);
}

std::unique_ptr<TF1> CloneRun2TF1(const std::string& path,
								  const std::string& funcName,
								  const std::string& subdir = "") {
	TFile* file = TFile::Open(path.c_str(), "READ");
	if (!file || file->IsZombie()) {
		std::cerr << "Cannot open Run2 file: " << path << std::endl;
		delete file;
		return nullptr;
	}

	// try to locate directory (root or provided subdir)
	TDirectory* dir = file;
	if (!subdir.empty()) {
		std::string token;
		std::istringstream ss(subdir);
		while (std::getline(ss, token, '/')) {
			if (!dir) break;
			dir = dynamic_cast<TDirectory*>(dir->Get(token.c_str()));
		}
	}

	std::string fn = funcName;
	TObject* obj = nullptr;
	if (!fn.empty() && dir) obj = dir->Get(fn.c_str());

	// if no name provided, search for first TF1 in the directory
	if (!obj && dir) {
		TList* keys = dir->GetListOfKeys();
		if (keys) {
			TIter next(keys);
			TKey* key = nullptr;
			while ((key = (TKey*)next())) {
				TObject* tmp = dir->Get(key->GetName());
				if (tmp && tmp->InheritsFrom("TF1")) {
					obj = tmp;
					fn = key->GetName();
					break;
				}
			}
		}
	}

	// fallback: try file root if not found in subdir
	if (!obj && !fn.empty()) obj = file->Get(fn.c_str());

	// final fallback: search entire file for a TF1
	if (!obj) {
		TList* keys = file->GetListOfKeys();
		if (keys) {
			TIter next(keys);
			TKey* key = nullptr;
			while ((key = (TKey*)next())) {
				TObject* tmp = file->Get(key->GetName());
				if (tmp && tmp->InheritsFrom("TF1")) {
					obj = tmp;
					fn = key->GetName();
					break;
				}
			}
		}
	}

	if (!obj) {
		std::cerr << "TF1 not found in " << path << std::endl;
		file->Close();
		delete file;
		return nullptr;
	}

	auto* func = dynamic_cast<TF1*>(obj);
	if (!func) {
		std::cerr << "Found object is not a TF1 in " << path << std::endl;
		file->Close();
		delete file;
		return nullptr;
	}

	std::string cloneName = fn.empty() ? "run2_tf1_clone" : (fn + std::string("_clone"));
	auto* clone = dynamic_cast<TF1*>(func->Clone(cloneName.c_str()));
	file->Close();
	delete file;
	return std::unique_ptr<TF1>(clone);
}

std::unique_ptr<TGraphErrors> GraphFromHist(const TH1* hist, const std::string& name) {
	if (!hist) return nullptr;
	auto graph = std::make_unique<TGraphErrors>(hist->GetNbinsX());
	graph->SetName(name.c_str());
	for (int i = 1; i <= hist->GetNbinsX(); ++i) {
		double x = hist->GetBinCenter(i);
		double y = hist->GetBinContent(i);
		double err = hist->GetBinError(i);
		graph->SetPoint(i - 1, x, y);
		graph->SetPointError(i - 1, 0.0, err);
	}
	return graph;
}

int ClosestPointIndex(TGraphAsymmErrors* graph, double x) {
	if (!graph) return -1;
	const int n = graph->GetN();
	int idx = -1;
	double best = std::numeric_limits<double>::infinity();
	double xi;
	double dummyY;
	for (int i = 0; i < n; ++i) {
		graph->GetPoint(i, xi, dummyY);
		double diff = std::abs(xi - x);
		if (diff < best) {
			best = diff;
			idx = i;
		}
	}
	return idx;
}

std::unique_ptr<TGraphErrors> BuildRatioGraph(TGraphErrors* numer,
											  TGraphAsymmErrors* denom,
											  TF1* denomTF,
											  const std::string& name) {
	if (!numer) return nullptr;
	auto ratio = std::make_unique<TGraphErrors>(numer->GetN());
	ratio->SetName(name.c_str());
	double x, y;
	for (int bin = 0; bin < numer->GetN(); ++bin) {
		numer->GetPoint(bin, x, y);
		double yErr = numer->GetErrorY(bin);
		double denomVal = 0.0;
		double denomErr = 0.0;
		if (denomTF) {
			denomVal = denomTF->Eval(x);
			denomErr = 0.0; // assume negligible
		} else if (denom) {
			denomVal = denom->Eval(x);
			int idx = ClosestPointIndex(denom, x);
			if (idx >= 0) {
				denomErr = 0.5 * (denom->GetErrorYhigh(idx) + denom->GetErrorYlow(idx));
			}
		}
		double value = (denomVal > 0) ? y / denomVal : 0.0;
		double rel = 0.0;
		if (value > 0 && y > 0 && denomVal > 0) {
			rel = std::sqrt(std::pow(yErr / y, 2) + std::pow(denomErr / denomVal, 2));
		}
		ratio->SetPoint(bin, x, value);
		ratio->SetPointError(bin, 0.0, value * rel);
	}
	return ratio;
}

void BeautifyGraph(TGraphErrors* graph) {
    if (!graph) return;
    graph->SetMarkerStyle(24);
    graph->SetMarkerSize(1.1);
    graph->SetLineWidth(2);
    graph->SetLineStyle(1);
    graph->SetFillStyle(0);
    graph->GetXaxis()->SetTitleFont(42);
    graph->GetYaxis()->SetTitleFont(42);
    graph->GetXaxis()->SetLabelFont(42);
    graph->GetYaxis()->SetLabelFont(42);
    graph->GetXaxis()->SetLabelSize(0.04);
    graph->GetYaxis()->SetLabelSize(0.04);
    graph->GetXaxis()->SetTitleSize(0.045);
    graph->GetYaxis()->SetTitleSize(0.045);
    graph->GetXaxis()->SetTitleOffset(1.1);
    graph->GetYaxis()->SetTitleOffset(1.2);
}

void BeautifyGraph(TGraphAsymmErrors* graph) {
	if (!graph) return;
	graph->SetMarkerStyle(21);
	graph->SetMarkerSize(1.1);
	graph->SetLineWidth(2);
	graph->SetLineStyle(1);
	graph->SetFillStyle(0);
	graph->GetXaxis()->SetTitleFont(42);
	graph->GetYaxis()->SetTitleFont(42);
	graph->GetXaxis()->SetLabelFont(42);
	graph->GetYaxis()->SetLabelFont(42);
	graph->GetXaxis()->SetLabelSize(0.04);
	graph->GetYaxis()->SetLabelSize(0.04);
	graph->GetXaxis()->SetTitleSize(0.045);
	graph->GetYaxis()->SetTitleSize(0.045);
	graph->GetXaxis()->SetTitleOffset(1.1);
	graph->GetYaxis()->SetTitleOffset(1.2);
}


void DrawComparison(const CentralitySpec& spec, const std::string& outputDir) {
	auto run3Hist = CloneHistogram(spec.run3File, spec.run3Hist, spec.run3Subdir);
	if (!run3Hist) return;
	auto run3Graph = GraphFromHist(run3Hist, Form("run3_%s", spec.label.c_str()));
	auto run2Graph = CloneRun2Graph(spec.run2File, spec.run2Graph, spec.run2Subdir);
	// load TF1 from BW fit file (separate file) if bwName is provided
	auto run2TF = CloneRun2TF1(kBWFitPath, spec.bwName, "");

	// build ratio using TF1 if available, otherwise use Run2 graph if available
	std::unique_ptr<TGraphErrors> ratioGraph = nullptr;
	if (run2TF || run2Graph) {
		ratioGraph = BuildRatioGraph(run3Graph.get(), run2Graph.get(), run2TF.get(), Form("ratio_%s", spec.label.c_str()));
	}

	run3Graph->SetMarkerColor(kMagenta + 2);
	run3Graph->SetLineColor(kMagenta + 2);
	if (run2Graph) {
		run2Graph->SetMarkerColor(kAzure + 2);
		run2Graph->SetLineColor(kAzure + 2);
		run2Graph->SetMarkerStyle(21);
		run2Graph->SetLineWidth(2);
	}

	BeautifyGraph(run3Graph.get());
	if (run2Graph) BeautifyGraph(run2Graph.get());

	// compute sensible y-range using both Run3 and Run2 graphs (if Run2 exists)
	double ymin = std::numeric_limits<double>::infinity();
	double ymax = 0.0;
	double xmin = std::numeric_limits<double>::infinity();
	double xmax = 0.0;
	double x, y;
	// collect from run3 (TGraphErrors): include point errors in min/max
	int n3 = run3Graph->GetN();
	for (int i = 0; i < n3; ++i) {
		double yerr = run3Graph->GetErrorY(i);
		run3Graph->GetPoint(i+1, x, y);
		xmin = std::min(xmin, x);
		xmax = std::max(xmax, x);
		double ylow = y - yerr;
		double yhigh = y + yerr;
		if (yhigh > 0) ymax = std::max(ymax, yhigh);
		if (ylow > 0) ymin = std::min(ymin, ylow);
	}
	// include run2 graph if available (TGraphAsymmErrors): use asymmetric errors
	if (run2Graph) {
		int n2 = run2Graph->GetN();
		for (int i = 0; i < n2; ++i) {
			double x2, y2;
			run2Graph->GetPoint(i+1, x2, y2);
			xmin = std::min(xmin, x2);
			xmax = std::max(xmax, x2);
			double ylow2 = y2;
			double yhigh2 = y2;
			if (run2Graph->InheritsFrom("TGraphAsymmErrors")) {
				ylow2 = y2 - ((TGraphAsymmErrors*)run2Graph.get())->GetErrorYlow(i);
				yhigh2 = y2 + ((TGraphAsymmErrors*)run2Graph.get())->GetErrorYhigh(i);
			}
			if (yhigh2 > 0) ymax = std::max(ymax, yhigh2);
			if (ylow2 > 0) ymin = std::min(ymin, ylow2);
		}
	}
	if (!std::isfinite(ymin) || ymin <= 0) ymin = 1e-8;
	if (ymax <= 0) ymax = 1.0;
	double yMin = std::max(1e-8, ymin * 0.4);
	double yMax = ymax * 1.2;
	// apply same y-limits to both Run3 and Run2 graphs (if present)
	run3Graph->GetYaxis()->SetLimits(yMin, yMax);
	if (run2Graph) run2Graph->GetYaxis()->SetLimits(yMin, yMax);
    cout << "Determined y-range: [" << yMin << ", " << yMax << "]" << std::endl;
	// if TF1 exists, set its drawing range to match data
	if (run2TF) {
		run2TF->SetRange(2.0, 8.0);
	}
	// set x-range for graphs to 2-8 GeV/c using SetLimits
	run3Graph->GetXaxis()->SetLimits(2.0, 8.0);
	if (run2Graph) run2Graph->GetXaxis()->SetLimits(2.0, 8.0);
	if (ratioGraph) ratioGraph->GetXaxis()->SetLimits(2.0, 8.0);
	

	bool fullscreen = (spec.label == "50_80");

	auto canvas = std::make_unique<TCanvas>(Form("c_compare_%s", spec.label.c_str()), "Run3 apass5 vs Run2", 1200, 900);
	if (fullscreen) {
		canvas->SetLeftMargin(0.12);
		canvas->SetRightMargin(0.05);
		canvas->SetTopMargin(0.08);
		canvas->SetBottomMargin(0.12);
		canvas->cd();
        gPad->SetLogy();
        gPad->SetGrid();

		// draw a frame to enforce axis limits (ensures limits are used regardless of graph autoscaling)
		run3Graph->GetXaxis()->SetTitle("p_{T} (GeV/#it{c})");
		run3Graph->GetYaxis()->SetTitle("dN/dp_{T} (GeV/#it{c})^{-1}");
		run3Graph->SetTitle(Form("%s comparison", spec.label.c_str()));
		gPad->DrawFrame(2.0, yMin, 8.0, yMax);
		run3Graph->Draw("P SAME");
		gPad->Modified();
		gPad->Update();

		auto legend = std::make_unique<TLegend>(0.58, 0.72, 0.92, 0.92);
		legend->SetFillStyle(0);
		legend->SetBorderSize(0);
		legend->SetTextFont(42);
		legend->AddEntry(run3Graph.get(), "Run3 apass5 (BDT)", "lep");
		legend->Draw();

		TLatex header;
		header.SetTextFont(42);
		header.SetTextSize(0.045);
		header.SetNDC();
		header.DrawLatex(0.16, 0.92, Form("#sqrt{s_{NN}} = 5.36 TeV (Run3 apass5) - %s", spec.label.c_str()));

		std::string outputFile = outputDir + "/H3l_run3_apass5_vs_run2_" + spec.label + ".pdf";
		canvas->SaveAs(outputFile.c_str());
		delete run3Hist;
		return;
	}

	canvas->SetLeftMargin(0.12);
	canvas->SetRightMargin(0.05);
	canvas->SetTopMargin(0.08);
	canvas->SetBottomMargin(0.12);

	// position top and bottom pads so they don't overlap
	auto topPad = std::make_unique<TPad>(Form("pad_main_%s", spec.label.c_str()), "", 0.0, 0.33, 1.0, 1.0);
	topPad->SetLeftMargin(0.12);
	topPad->SetRightMargin(0.05);
	topPad->SetTopMargin(0.07);
	topPad->SetBottomMargin(0.05);
	topPad->SetLogy();
	topPad->SetGrid();
	topPad->Draw();
	topPad->cd();

	run3Graph->GetXaxis()->SetTitle("p_{T} (GeV/#it{c})");
	run3Graph->GetYaxis()->SetTitle("dN/dp_{T} (GeV/#it{c})^{-1}");
	run3Graph->SetTitle(Form("%s comparison", spec.label.c_str()));

	    // draw a frame that enforces x/y ranges, then draw graphs on top
	    gPad->DrawFrame(2.0, yMin, 8.0, yMax);
	    run3Graph->Draw("P SAME");
	    gPad->Modified();
	    gPad->Update();
	if (run2Graph) {
		run2Graph->Draw("P SAME");
	}
	if (run2TF) {
		run2TF->SetLineColor(kAzure + 2);
		run2TF->SetLineStyle(2);
		run2TF->SetLineWidth(2);
		run2TF->Draw("SAME");
	}

	auto legend = std::make_unique<TLegend>(0.58, 0.66, 0.92, 0.88);
	legend->SetFillStyle(0);
	legend->SetBorderSize(0);
	legend->SetTextFont(42);
	legend->AddEntry(run3Graph.get(), "Run3 apass5 (BDT)", "lep");
	if (run2Graph) legend->AddEntry(run2Graph.get(), "Run2 reference", "lep");
	else if (run2TF) legend->AddEntry(run2TF.get(), "Run2 BW fit", "l");
	legend->Draw();

	TLatex header;
	header.SetTextFont(42);
	header.SetTextSize(0.045);
	header.SetNDC();
	header.DrawLatex(0.16, 0.92, Form("#sqrt{s_{NN}} = 5.36 TeV (Run3 apass5) vs 5.02 TeV (Run2) - %s", spec.label.c_str()));

	canvas->cd();
	auto bottomPad = std::make_unique<TPad>(Form("pad_ratio_%s", spec.label.c_str()), "", 0.0, 0.0, 1.0, 0.33);
	bottomPad->SetLeftMargin(0.12);
	bottomPad->SetRightMargin(0.05);
	bottomPad->SetTopMargin(0.03);
	bottomPad->SetBottomMargin(0.25);
	bottomPad->SetGridy();
	bottomPad->Draw();
	bottomPad->cd();

	if (ratioGraph) {
		ratioGraph->SetMarkerStyle(20);
		ratioGraph->SetMarkerSize(0.9);
		ratioGraph->SetLineColor(kGray + 2);
		ratioGraph->SetLineWidth(2);
		ratioGraph->GetXaxis()->SetTitle("p_{T} (GeV/#it{c})");
		ratioGraph->GetYaxis()->SetTitle("Run3 / Run2");
		ratioGraph->GetYaxis()->SetNdivisions(505);
		ratioGraph->GetYaxis()->SetTitleOffset(1.1);
		ratioGraph->GetXaxis()->SetTitleFont(42);
		ratioGraph->GetYaxis()->SetTitleFont(42);
		ratioGraph->GetXaxis()->SetLabelFont(42);
		ratioGraph->GetYaxis()->SetLabelFont(42);
		ratioGraph->GetXaxis()->SetLabelSize(0.04);
		ratioGraph->GetYaxis()->SetLabelSize(0.04);
		ratioGraph->SetTitle("");
			ratioGraph->Draw("AP");
			ratioGraph->GetXaxis()->SetLimits(2.0, 8.0);
			ratioGraph->GetXaxis()->SetRangeUser(2.0, 8.0);
			gPad->Modified();
			gPad->Update();

		TLine line;
		double xMin = ratioGraph->GetXaxis()->GetXmin();
		double xMax = ratioGraph->GetXaxis()->GetXmax();
		line.SetLineStyle(2);
		line.SetLineColor(kGray + 3);
		line.DrawLine(xMin, 1.0, xMax, 1.0);
	}

	std::string outputFile = outputDir + "/H3l_run3_apass5_vs_run2_" + spec.label + ".pdf";
	canvas->SaveAs(outputFile.c_str());

	delete run3Hist;
}

}  // namespace

void SpectrumVsRun2() {
	gStyle->SetOptStat(0);
	std::string outputDir = "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/results/run3_vs_run2";
	gSystem->mkdir(outputDir.c_str(), true);

	std::vector<CentralitySpec> specs = {
		{"0_10",
		 "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID/BdtSpectrum/cen0-10/pt_analysis_pbpb.root",
		 "/Users/zhengqingwang/alice/data/h3l_spec_run2/h3l_0_10.root",
		 "Graph1D_y1",
		 "h_corrected_counts",
		 "std",
		 "(Anti)hypertriton spectrum in 0-10% V0M centrality class",
         "BlastWave_H3L_0_10"
		},
        {"10_30",
         "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID/BdtSpectrum/cen10-30/pt_analysis_pbpb.root",
         "/Users/zhengqingwang/alice/data/h3l_spec_run2/h3l_10_30.root",
         "Graph1D_y1",
         "h_corrected_counts",
         "std",
         "(Anti)hypertriton spectrum in 10-30% V0M centrality class",
         "BlastWave_H3L_10_30"
        },
        {"30_50",
         "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID/BdtSpectrum/cen30-50/pt_analysis_pbpb.root",
         "/Users/zhengqingwang/alice/data/h3l_spec_run2/h3l_30_50.root",
         "Graph1D_y1",
         "h_corrected_counts",
         "std",
         "(Anti)hypertriton spectrum in 30-50% V0M centrality class",
         "BlastWave_H3L_30_50"
        },
        {"50_80",
         "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/LHC23_PbPb_pass5_CustomV0s_HadronPID/BdtSpectrum/cen50-80/pt_analysis_pbpb.root",
         "/Users/zhengqingwang/alice/data/h3l_spec_run2/h3l_50_80.root",
         "Graph1D_y1",
         "h_corrected_counts",
         "std",
		 "(Anti)hypertriton spectrum in 50-80% V0M centrality class",
		 ""
        }
	};

	for (const auto& spec : specs) {
		DrawComparison(spec, outputDir);
	}
}
