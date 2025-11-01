"""MLMaker: helper class to encapsulate ML training and application logic used in ct_extraction.py

This module factors out ML-specific code from ct_extraction.py (roughly lines 220-635) into a reusable class.

Features:
- prepare_data: shift fNSigmaHe, optional preselections
- train: train XGBoost via hipe4ml ModelHandler
- apply: apply trained model to TreeHandler / DataFrame
- score_from_efficiency_array: robust method to map target efficiencies to score thresholds with fallback
- plot_bdtrange: plotting helpers for QA

Usage example:
    m = MLMaker(training_variables=vars, hyperparams=params)
    m.prepare_data(bin_mc_hdl, bin_data_hdl, training_preselections)
    m.train(train_test_data)
    m.apply(bin_data_hdl)

Note: this file depends on hipe4ml and XGBoost (same environment as original script).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from hipe4ml.model_handler import ModelHandler
import xgboost as xgb
import os


class MLMaker:
    """Encapsulate ML training and application logic.

    Public methods:
    - prepare_data(mc_hdl, data_hdl, shift_nsigma=True, training_preselections='') -> train_test_data
    - train(train_test_data, hyperparams, output_margin=True) -> ModelHandler
    - apply(model_hdl, data_hdl, column_name='model_output') -> None
    - score_from_efficiency_array(test_labels, y_pred, eff_arr) -> np.ndarray
    - plot_output_train_test(...) simple wrapper calling pu.plot_output_train_test if available

    This class aims to preserve behavior from ct_extraction.py while being reusable.
    """

    def __init__(self, training_variables: List[str], random_state: int = 42):
        self.training_variables = training_variables
        self.random_state = random_state
        self.model_hdl: Optional[ModelHandler] = None

    def prepare_data(self, bin_mc_hdl, bin_data_hdl, training_preselections: str = '',
                     balance_bkg_frac: Optional[float] = None, test_size: float = 0.2):
        """Prepare train/test datasets used for hipe4ml training.

        - shift fNSigmaHe in MC to have mean 0
        - apply training_preselections to both MC and data if provided
        - returns train_test_data as used by au.train_test_generator in original code
        """
        # Copy handlers
        mc_hdl_local = bin_mc_hdl
        data_hdl_local = bin_data_hdl

        # Apply preselections if requested
        if training_preselections:
            mc_hdl_local = mc_hdl_local.apply_preselections(training_preselections, inplace=False)
            data_hdl_local = data_hdl_local.apply_preselections(training_preselections, inplace=False)

        # Shift He3 nSigma in MC to 0 (original code shifted df_mcH)
        try:
            df_mc = mc_hdl_local.get_data_frame()
            if 'fNSigmaHe' in df_mc.columns:
                df_mc['fNSigmaHe'] = df_mc['fNSigmaHe'] - df_mc['fNSigmaHe'].mean()
                mc_hdl_local.set_data_frame(df_mc)
        except Exception:
            pass

        # create train/test data using hipe4ml's helper if available
        # The original code used au.train_test_generator([bin_mc_hdl_ML, bin_data_hdl_ML], [1,0], ...)
        from hipe4ml.analysis_utils import train_test_generator as ttg
        train_test_data = ttg([mc_hdl_local, data_hdl_local], [1, 0], test_size=test_size, random_state=self.random_state)
        return train_test_data

    def train(self, train_test_data, hyperparams: dict, output_margin: bool = True) -> ModelHandler:
        """Train an XGBoost model via hipe4ml ModelHandler and return it.

        hyperparams are passed to model_hdl.set_model_params
        """
        model_hdl = ModelHandler(xgb.XGBClassifier(), self.training_variables)
        model_hdl.set_model_params(hyperparams)
        model_hdl.train_test_model(train_test_data, False, output_margin=output_margin)
        self.model_hdl = model_hdl
        return model_hdl

    def apply(self, model_hdl: ModelHandler, target_hdl, column_name: str = 'model_output'):
        """Apply model to a TreeHandler (or DataFrame wrapper) and add predictions column.

        This mirrors bin_data_hdl.apply_model_handler(model_hdl, column_name="model_output")
        """
        target_hdl.apply_model_handler(model_hdl, column_name=column_name)

    def score_from_efficiency_array(self, test_labels: np.ndarray, y_pred: np.ndarray, eff_arr: np.ndarray) -> np.ndarray:
        """Map desired efficiencies to score thresholds robustly.

        First try to use original `au.score_from_efficiency_array` logic if available; if it fails
        (e.g., no polynomial root), fall back to threshold scanning (deterministic).

        Returns array of scores (floats) with same length as eff_arr. If an efficiency is
        unreachable, returns np.nan for that entry.
        """
        # Try to import original implementation
        try:
            import hipe4ml.analysis_utils as au
            try:
                return au.score_from_efficiency_array(test_labels, y_pred, eff_arr)
            except Exception:
                pass
        except Exception:
            pass

        # Fallback: threshold scan
        uniq_scores = np.sort(np.unique(y_pred))
        # if uniq_scores are in descending order in original usage, ensure thresholds descending
        thresholds = np.sort(uniq_scores)

        tot_pos = np.sum(np.array(test_labels) == 1)
        if tot_pos == 0:
            return np.full_like(eff_arr, np.nan, dtype=float)

        tprs = []
        for thr in thresholds:
            preds = (y_pred >= thr).astype(int)
            tp = np.sum((preds == 1) & (np.array(test_labels) == 1))
            tprs.append(tp / tot_pos)
        tprs = np.array(tprs)

        scores = []
        for eff in eff_arr:
            if eff < tprs.min() - 1e-12 or eff > tprs.max() + 1e-12:
                scores.append(np.nan)
            else:
                idx = np.argmin(np.abs(tprs - eff))
                scores.append(thresholds[idx])
        return np.array(scores)

    def plot_product_vs_score(self, df_working_point: pd.DataFrame, output_path: str):
        """Simple plotting helper to reproduce the product vs BDT score plot.
        """
        ds = pd.DataFrame({
            'score': df_working_point['BDT_score'],
            'center': df_working_point['product'],
            'upper': df_working_point['product_up'],
            'lower': df_working_point['product_down']
        }).sort_values('score')
        plt.figure(figsize=(10, 6))
        plt.plot(ds['score'], ds['center'], color='black', label='Center', linewidth=2)
        plt.scatter(ds['score'], ds['upper'], color='red', s=10, label='Upper 1σ', alpha=0.5)
        plt.scatter(ds['score'], ds['lower'], color='blue', s=10, label='Lower 1σ', alpha=0.5)
        plt.fill_between(ds['score'], ds['lower'], ds['upper'], color='gray', alpha=0.2, label='1σ Region')
        plt.xlabel('BDT_Score')
        plt.ylabel('Expected Significance x BDT_Efficiency')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def plot_qa_figs(self, model_hdl: ModelHandler, train_test_data, output_dir: str, suffix: str):
        """Plot QA figures for training and testing datasets.

        This is a simple wrapper calling pu.plot_output_train_test if available.
        """
        try:
            import hipe4ml.plotting_utils as pu
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'ML_output_train_test_{suffix}.pdf')
            pu.plot_output_train_test(model_hdl, train_test_data, output_path)
        except Exception:
            pass