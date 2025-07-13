import pandas as pd
from sklearn.decomposition import PCA

def perform_pca_on_returns(returns: pd.DataFrame, n_components: int = 3):
    """
    Performs Principal Component Analysis (PCA) on a DataFrame of asset returns.

    This is useful for identifying the underlying factors that drive portfolio risk and returns.

    Args:
        returns (pd.DataFrame): DataFrame where each column is the returns for an asset
                                and the index is a datetime.
        n_components (int): The number of principal components to compute.

    Returns:
        tuple:
            - pd.Series: The explained variance ratio for each principal component.
            - pd.DataFrame: The factor exposures (loadings) for each asset on each component.
            - pd.DataFrame: The time series of factor returns for each component.
    """
    # Ensure there are no missing values
    clean_returns = returns.dropna()

    # Initialize and fit the PCA model
    pca = PCA(n_components=n_components)
    pca.fit(clean_returns)

    # Explained variance
    explained_variance = pd.Series(
        pca.explained_variance_ratio_,
        index=[f'PC{i+1}' for i in range(n_components)],
        name='ExplainedVariance'
    )

    # Factor exposures (component loadings)
    factor_exposures = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=clean_returns.columns
    )

    # Factor returns (transformed data)
    factor_returns = pd.DataFrame(
        pca.transform(clean_returns),
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=clean_returns.index
    )

    return explained_variance, factor_exposures, factor_returns
