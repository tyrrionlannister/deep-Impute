import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer

def preprocess_data(data):
    # Step 1: Remove non-numeric columns
    data_numeric = data.select_dtypes(include=[np.number])

    # Step 2: Log Normalization
    data_log = data_numeric.apply(lambda x: np.log1p(x))

    # Step 3: Variance Filtering - Keep genes with high variance-to-mean ratio
    var_mean_ratio = data_log.var() / data_log.mean()
    data_filtered = data_log.loc[:, var_mean_ratio > 0.5]
    if data_filtered.empty:
        raise ValueError("All genes were filtered out during variance filtering. Try lowering the threshold.")

    # Step 4: Cell and Gene Scaling
    # Cell-level scaling (Counts per million)
    data_cpm = data_filtered.div(data_filtered.sum(axis=1), axis=0) * 1e6
    # Gene-level Z-score scaling
    data_scaled = (data_cpm - data_cpm.mean()) / data_cpm.std()

    # Step 5: Dropout Handling - Remove very sparse genes and cells with updated thresholds
    # Conditional sparsity filtering to ensure data is not overly reduced
    if (data_scaled > 0).mean().min() < 0.05 or (data_scaled > 0).sum(axis=1).min() < 100:
        data_sparse_filtered = data_scaled  # Skip filtering if too much data would be removed
    else:
        data_sparse_filtered = data_scaled.loc[:, (data_scaled > 0).mean() > 0.05]  # Genes in >5% of cells
        data_sparse_filtered = data_sparse_filtered[(data_sparse_filtered > 0).sum(axis=1) > 100]  # Cells with >100 genes
    if data_sparse_filtered.empty:
        raise ValueError("All data was filtered out after handling sparse genes/cells. Adjust the sparsity threshold.")

    # Step 6: Outlier Removal - Filter cells with extreme total expression values
    cell_totals = data_sparse_filtered.sum(axis=1)
    z_scores = (cell_totals - cell_totals.mean()) / cell_totals.std()
    data_outlier_filtered = data_sparse_filtered[(z_scores.abs() <= 3)]
    if data_outlier_filtered.empty:
        raise ValueError("All data was filtered out after outlier removal. Adjust outlier thresholds.")

    # Step 7: Dimensionality Reduction (Optional) - PCA for top components
    if data_outlier_filtered.shape[0] > 1:  # Ensure there is more than one sample
        pca = PCA(n_components=50)
        data_reduced = pd.DataFrame(pca.fit_transform(data_outlier_filtered), index=data_outlier_filtered.index)
    else:
        data_reduced = data_outlier_filtered  # Skip PCA if there's not enough data

    # Step 8: KNN Imputation for initial fill (Optional)
    imputer = KNNImputer(n_neighbors=5)
    data_imputed_initial = pd.DataFrame(imputer.fit_transform(data_reduced), index=data_reduced.index)

    return data_imputed_initial

def deep_impute(data):
    # Preprocess data
    processed_data = preprocess_data(data)

    # Define a neural network for imputation
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(processed_data.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(processed_data.shape[1], activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Split data for training
    train_data = processed_data.sample(frac=0.8, random_state=0)
    test_data = processed_data.drop(train_data.index)

    # Train the model
    model.fit(train_data, train_data, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    # Impute missing values
    imputed_data = model.predict(processed_data)
    imputed_data = pd.DataFrame(imputed_data, columns=processed_data.columns)

    # Calculate metrics
    mse = mean_squared_error(processed_data, imputed_data)
    r2 = r2_score(processed_data, imputed_data)
    metrics = {'MSE': mse, 'R2': r2}

    return imputed_data, metrics
