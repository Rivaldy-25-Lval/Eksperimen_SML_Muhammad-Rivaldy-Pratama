"""
Automated Preprocessing Script - Heart Disease Dataset
Muhammad Rivaldy Pratama
Dicoding Submission: Membangun Sistem Machine Learning (MSML)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HeartDiseasePreprocessor:
    """
    Automated preprocessing pipeline for Heart Disease dataset
    Implements the same steps as the experiment notebook
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        
    def load_data(self, filepath_or_url):
        """
        Load Heart Disease dataset from file or URL
        
        Parameters:
        -----------
        filepath_or_url : str
            Path to CSV file or URL
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataset
        """
        logger.info(f"Loading data from: {filepath_or_url}")
        try:
            if filepath_or_url.startswith('http'):
                # Load from UCI ML Repository
                column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                              'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                              'ca', 'thal', 'target']
                df = pd.read_csv(filepath_or_url, names=column_names, na_values='?')
                # Convert target to binary (0=no disease, 1=disease)
                df['target'] = (df['target'] > 0).astype(int)
            else:
                df = pd.read_csv(filepath_or_url)
            
            logger.info(f"✅ Data loaded successfully! Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            raise
    
    def handle_missing_values(self, df):
        """
        Handle missing values using median imputation
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with missing values imputed
        """
        logger.info("Handling missing values...")
        missing_before = df.isnull().sum().sum()
        logger.info(f"Missing values before: {missing_before}")
        
        if missing_before > 0:
            # Get columns with missing values
            cols_with_missing = df.columns[df.isnull().any()].tolist()
            logger.info(f"Columns with missing values: {cols_with_missing}")
            
            # Apply median imputation
            df[cols_with_missing] = self.imputer.fit_transform(df[cols_with_missing])
            
            missing_after = df.isnull().sum().sum()
            logger.info(f"✅ Missing values after imputation: {missing_after}")
        else:
            logger.info("✅ No missing values found")
        
        return df
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """
        Split data into train and test sets with stratification
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        test_size : float
            Proportion of test set (default: 0.2)
        random_state : int
            Random seed (default: 42)
            
        Returns:
        --------
        tuple
            X_train, X_test, y_train, y_test
        """
        logger.info("Splitting data into train and test sets...")
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Save feature names
        self.feature_names = X.columns.tolist()
        
        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"✅ Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
        logger.info(f"✅ Test set:  {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
        logger.info(f"   Train - Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()}")
        logger.info(f"   Test  - Class 0: {(y_test==0).sum()}, Class 1: {(y_test==1).sum()}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """
        Scale features using StandardScaler
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
            
        Returns:
        --------
        tuple
            Scaled X_train, X_test as DataFrames
        """
        logger.info("Scaling features with StandardScaler...")
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        logger.info("✅ Features scaled (mean≈0, std≈1)")
        
        return X_train_scaled, X_test_scaled
    
    def save_preprocessed_data(self, X_train, X_test, y_train, y_test, output_dir='data/preprocessed'):
        """
        Save preprocessed data to CSV files
        
        Parameters:
        -----------
        X_train, X_test : pd.DataFrame
            Scaled features
        y_train, y_test : pd.Series
            Target values
        output_dir : str
            Output directory path
        """
        logger.info(f"Saving preprocessed data to: {output_dir}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Combine features and target
        train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
        test_data = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
        
        # Save to CSV
        train_path = Path(output_dir) / 'train_data.csv'
        test_path = Path(output_dir) / 'test_data.csv'
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        logger.info(f"✅ Saved train_data.csv: {train_data.shape}")
        logger.info(f"✅ Saved test_data.csv: {test_data.shape}")
        logger.info(f"   Train size: {train_path.stat().st_size / 1024:.2f} KB")
        logger.info(f"   Test size:  {test_path.stat().st_size / 1024:.2f} KB")
        
        return train_path, test_path
    
    def save_scaler(self, output_dir='models'):
        """
        Save fitted scaler for future use
        
        Parameters:
        -----------
        output_dir : str
            Output directory for scaler
        """
        logger.info(f"Saving scaler to: {output_dir}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        scaler_path = Path(output_dir) / 'scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"✅ Scaler saved to: {scaler_path}")
        
    def run_full_pipeline(self, data_source, output_dir='data/preprocessed'):
        """
        Run complete preprocessing pipeline
        
        Parameters:
        -----------
        data_source : str
            Path to CSV file or URL
        output_dir : str
            Output directory for preprocessed data
            
        Returns:
        --------
        tuple
            Paths to saved train and test files
        """
        logger.info("="*70)
        logger.info("HEART DISEASE PREPROCESSING PIPELINE")
        logger.info("="*70)
        
        # Step 1: Load data
        df = self.load_data(data_source)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Split data
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # Step 4: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Step 5: Save preprocessed data
        train_path, test_path = self.save_preprocessed_data(
            X_train_scaled, X_test_scaled, y_train, y_test, output_dir
        )
        
        # Step 6: Save scaler
        self.save_scaler()
        
        logger.info("="*70)
        logger.info("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        
        return train_path, test_path


def main():
    """
    Main function to run preprocessing
    """
    # Initialize preprocessor
    preprocessor = HeartDiseasePreprocessor()
    
    # Data source (UCI ML Repository URL or local file)
    data_source = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Alternative: use local file if already downloaded
    local_file = Path("data/raw/heart_disease.csv")
    if local_file.exists():
        data_source = str(local_file)
        logger.info(f"Using local file: {data_source}")
    
    # Run pipeline
    try:
        train_path, test_path = preprocessor.run_full_pipeline(
            data_source=data_source,
            output_dir='data/preprocessed'
        )
        
        print("\n" + "="*70)
        print("📊 SUMMARY")
        print("="*70)
        print(f"✅ Training data: {train_path}")
        print(f"✅ Test data:     {test_path}")
        print(f"✅ Scaler saved:  models/scaler.joblib")
        print(f"✅ Features:      {len(preprocessor.feature_names)}")
        print(f"   {preprocessor.feature_names}")
        print("="*70)
        print("🎉 Ready for model training!")
        print("="*70)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
