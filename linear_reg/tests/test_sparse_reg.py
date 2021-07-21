from sklearn import datasets
from ..nonlinear_causal.sparse_reg import WLasso

diabetes = datasets.load_diabetes()
X_diabetes, y_diabetes = diabetes.data, diabetes.target
