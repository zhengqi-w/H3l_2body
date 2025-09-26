import os
import numpy as np
import uproot
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
import hipe4ml.analysis_utils as au
import hipe4ml.plot_utils as pu
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar


class MLMaker:
    """Class to handle machine learning model application to ROOT trees."""
    """Under development"""
    def __init__(self, model_path, tree_path, variables, spectators=None):
        self.model_path = model_path
        self.tree_path = tree_path
        self.variables = variables
        self.spectators = spectators if spectators is not None else []
        self.model_hdl = None
        self.tree_hdl = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} does not exist.")
        self.model_hdl = ModelHandler(self.model_path)
        print(f"Model loaded from {self.model_path}")

    def load_tree(self):
        if not os.path.exists(self.tree_path):
            raise FileNotFoundError(f"Tree file {self.tree_path} does not exist.")
        self.tree_hdl = TreeHandler(self.tree_path)
        print(f"Tree loaded from {self.tree_path}")

    def apply_model(self, output_branch="ml_score"):
        if self.model_hdl is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        if self.tree_hdl is None:
            raise RuntimeError("Tree is not loaded. Call load_tree() first.")

        df = self.tree_hdl.get_df(columns=self.variables + self.spectators)
        scores = self.model_hdl.predict(df[self.variables])
        df[output_branch] = scores

        # Save the new branch back to the ROOT file
        with uproot.recreate(self.tree_path) as f:
            f[self.tree_hdl.tree_name] = uproot.newtree({col: df[col].values for col in df.columns})
        
        print(f"Applied model and saved scores to branch '{output_branch}' in {self.tree_path}")

    def plot_score_distribution(self, output_branch="ml_score", bins=50):
        if self.tree_hdl is None:
            raise RuntimeError("Tree is not loaded. Call load_tree() first.")

        df = self.tree_hdl.get_df(columns=[output_branch])
        plt.hist(df[output_branch], bins=bins, histtype='stepfilled', alpha=0.7)
        plt.xlabel(output_branch)
        plt.ylabel('Entries')
        plt.title(f'Distribution of {output_branch}')
        plt.grid()
        plt.show()