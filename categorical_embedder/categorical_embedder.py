import scipy.stats as st
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt


class CategoricalEmbedder(TransformerMixin, BaseEstimator):
    """Transformer to encode categorical features into vectors of a given size.
    
    On the fit() method computes the following steps:

        1-It uses OrdinalEncoder to convert categories into indexes, adding one
        extra category for unknown or nan values.

        2-t adds an offset to each category index to identify all possible
        values of all features with on specific value.

        3-It uses a OneHotEncoder and adds a constant to each feature column.

        4-It uses the multi-layer perceptron of sklearn to extract embeddings from
        one hot transformed input.

    On the transform() method computes the following steps:

        1-It uses the fitted OrdinalEncoder to get the indexes of the inputs

        2-Uses np.take to get specific feature values embeddings from the embedding matrix.

        3-Reshapes the embeddins in a vector of shape=(n_observations,n_features*emb_size)

    Args:
        emb_size (int, optional): size of the embedding for each unique category. Defaults to 32.
        max_iter (int, optional): max iteration of the MLP. Defaults to 100.
        random_state (_type_, optional): random state. Defaults to None.
        verbose (bool, optional): True or False. Defaults to False.
        missing_values (_type_, optional): Defaults to np.nan.
        copy (bool, optional): Defaults to True.


    Attributes:
        ordinal_encoder_: _description_
        cardinalities_: _description_
        offset_: _description_
        onehot_encoder_: _description_
        categories_: _description_
        mlp_: _description_
        emb_: _description_
    """ 


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




def umap_plot_cat_emb(cat_emb:CategoricalEmbedder, n_neighbors=10, annotation=True):
    import umap
    
    def get_cmap(n, name='viridis'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)
        
    mapper = umap.UMAP(n_neighbors=n_neighbors).fit(cat_emb.emb_)

    emb_umap = mapper.transform(cat_emb.emb_)

    cmap = get_cmap(len(cat_emb.feature_names_in_))

    plt.figure(figsize=(15,15))
    if getattr(cat_emb, "feature_names_in_",None) is not None:
        features = cat_emb.feature_names_in_
    else:
        features = [f"feature_{i}" for i in range(len(cat_emb.cardinalities_))]

    for idx, feature in enumerate(features):
        if idx == len(cat_emb.offset_)-1:
            feature_emb = emb_umap[cat_emb.offset_[idx]:]
        else:
            feature_emb = emb_umap[cat_emb.offset_[idx]:cat_emb.offset_[idx+1]]    
        x = feature_emb[:,0]
        y = feature_emb[:,1]
        c = cmap(idx)
        ann = cat_emb.categories_[idx]
        plt.scatter(x,y,  color=c, label=feature)

        if annotation:
            for i in range(len(x)):
                plt.annotate(ann[i], (x[i]+0.05, y[i] + 0))

    plt.legend()
    plt.show()        