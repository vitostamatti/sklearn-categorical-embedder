import scipy.stats as st
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd


class CategoricalEmbedder(TransformerMixin, BaseEstimator):

    def __init__(
            self,
            emb_size=32, 
            max_iter=100, 
            random_state=None, 
            verbose=False, 
            missing_values=np.nan, 
            copy=True
        ):

        self.emb_size = emb_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.missing_values = missing_values
        self.copy = copy


    def fit(self, X, y):

        X = self._validate_data(
            X,
            reset=True,
            accept_sparse="csc",
            dtype=object,
            force_all_finite="allow-nan",
            copy=self.copy,
        )

        self.ordinal_encoder_ = OrdinalEncoder(
            handle_unknown = "use_encoded_value", 
            unknown_value = -1, 
            encoded_missing_value=-1
            )

        self.ordinal_encoder_.fit(X)

        self.cardinalities_ = [len(cat)+1 for cat in self.ordinal_encoder_.categories_]

        self.offset_ = np.cumsum([1] + self.cardinalities_[:-1], axis=0)
        
        if getattr(self,'feature_names_in_',None) is not None:
            self.categories_ = [
                [f'{f}_unk'] + list(self.ordinal_encoder_.categories_[i]) 
                for i,f in enumerate(self.feature_names_in_)
                ]
        else:
            self.categories_ = [
                [f'feature_{i}_unk'] + list(self.ordinal_encoder_.categories_[i]) 
                for i,f in enumerate(self.ordinal_encoder_.categories_)
                ]

        self.onehot_encoder_ = OneHotEncoder(categories = self.categories_)
        self.onehot_encoder_.fit(X)
        x_oh = self.onehot_encoder_.transform(X)

        feature_oh_bias = np.zeros(shape=x_oh.shape)
        bias = np.linspace(start=-0.1,stop=0.1,num=len(self.cardinalities_))

        prev_idx = 0
        for i,n in enumerate(self.cardinalities_):
            feature_oh_bias[:,prev_idx:prev_idx+n] = bias[i]
            prev_idx = prev_idx+n

        x_oh = x_oh + feature_oh_bias

        self.mlp_ = MLPClassifier(
            max_iter=self.max_iter,
            hidden_layer_sizes=(self.emb_size,),
            random_state=self.random_state, 
            verbose=self.verbose, 
            early_stopping=True,
            learning_rate_init=1e-5
        )
        self.mlp_.fit(x_oh, y)

        self.emb_ = self.mlp_.coefs_[0]

        return self


    def transform(self, X):
        X = self._validate_data(
            X,
            reset=True,
            accept_sparse="csc",
            dtype=object,
            force_all_finite="allow-nan",
            copy=self.copy,
        )
        x_ord = self.ordinal_encoder_.transform(X)+self.offset_
        x_emb = np.take(a=self.emb_, indices=x_ord.astype(int), axis=0)
        x_emb = x_emb.reshape(x_emb.shape[0], x_emb.shape[1]*x_emb.shape[2])

        return x_emb
