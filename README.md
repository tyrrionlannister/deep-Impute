# **My Imputation App**

A deep learning-powered application for imputing and denoising single-cell RNA-seq (scRNA-seq) data, designed to enhance the quality of downstream biological analysis. The application uses a neural network-based approach to address dropout issues in scRNA-seq datasets, providing accurate and efficient imputation.

---

## **Features**
- **Preprocessing**: Handles log normalization, variance filtering, and dimensionality reduction (PCA).
- **Deep Learning Model**: Implements an autoencoder-based architecture for data imputation.
- **Evaluation Metrics**: Calculates Mean Squared Error (MSE) and R² scores to assess imputation quality.
- **Web Interface**: Allows users to upload datasets, perform imputation, and download results.

---

## **Technologies Used**
- **Backend**: Python, Flask.
- **Frontend**: HTML, CSS, Bootstrap.
- **Machine Learning**: TensorFlow/Keras, scikit-learn.
- **Data Handling**: pandas, NumPy.

