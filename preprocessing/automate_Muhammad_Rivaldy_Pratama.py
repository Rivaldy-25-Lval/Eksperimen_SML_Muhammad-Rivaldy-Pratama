import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WineQualityPreprocessor:
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, filepath_or_url):
        logger.info(f"Loading data from: {filepath_or_url}")
        try:
            if filepath_or_url.startswith('http'):
                df = pd.read_csv(filepath_or_url, sep=';')
            else:
                df = pd.read_csv(filepath_or_url)
            logger.info(f"✅ Data loaded successfully! Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            raise
    
    def remove_duplicates(self, df):
        logger.info("Removing duplicate data...")
        before = len(df)
        df_clean = df.drop_duplicates()
        after = len(df_clean)
        logger.info(f"✅ Removed {before - after} duplicate rows")
        return df_clean
    
    def categorize_quality(self, quality):
        if quality <= 5:
            return 0
        elif quality == 6:
            return 1
        else:
            return 2
    
    def feature_engineering(self, df):
        logger.info("Performing feature engineering...")
        df['quality_category'] = df['quality'].apply(self.categorize_quality)
        
        distribution = df['quality_category'].value_counts().sort_index()
        logger.info(f"✅ Quality distribution:\n{distribution}")
        return df
    
    def remove_outliers_iqr(self, df, columns):
        logger.info("Removing outliers using IQR method...")
        df_no_outliers = df.copy()
        total_removed = 0
        
        for col in columns:
            Q1 = df_no_outliers[col].quantile(0.25)
            Q3 = df_no_outliers[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            before = len(df_no_outliers)
            df_no_outliers = df_no_outliers[
                (df_no_outliers[col] >= lower_bound) & 
                (df_no_outliers[col] <= upper_bound)
            ]
            after = len(df_no_outliers)
            removed = before - after
            if removed > 0:
                total_removed += removed
                logger.info(f"  - {col}: {removed} outliers removed")
        
        logger.info(f"✅ Total outliers removed: {total_removed}")
        return df_no_outliers
    
    def scale_features(self, X_train, X_test):
        logger.info("Scaling features using StandardScaler...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(
            X_train_scaled, 
            columns=X_train.columns, 
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled, 
            columns=X_test.columns, 
            index=X_test.index
        )
        
        logger.info("✅ Features scaled successfully!")
        return X_train_scaled, X_test_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        logger.info(f"Splitting data (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"✅ Training set: {X_train.shape[0]} samples")
        logger.info(f"✅ Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessed_data(self, X_train, X_test, y_train, y_test, output_dir):
        logger.info(f"Saving preprocessed data to: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Gabungkan X dan y
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        # Simpan ke CSV
        train_path = os.path.join(output_dir, 'train_data.csv')
        test_path = os.path.join(output_dir, 'test_data.csv')
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"✅ Train data saved: {train_path}")
        logger.info(f"✅ Test data saved: {test_path}")
        logger.info(f"✅ Scaler saved: {scaler_path}")
    
    def preprocess(self, input_path, output_dir='data/preprocessed'):
        logger.info("="*60)
        logger.info("STARTING AUTOMATED PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        df = self.load_data(input_path)
        df = self.remove_duplicates(df)
        df = self.feature_engineering(df)
        
        numeric_cols = df.columns[:-2].tolist()
        df = self.remove_outliers_iqr(df, numeric_cols)
        
        X = df.drop(['quality', 'quality_category'], axis=1)
        y = df['quality_category']
        self.feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        self.save_preprocessed_data(
            X_train_scaled, X_test_scaled, y_train, y_test, output_dir
        )
        
        logger.info("="*60)
        logger.info("✅ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        return X_train_scaled, X_test_scaled, y_train, y_test


def main():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    output_dir = "data/preprocessed"
    preprocessor = WineQualityPreprocessor()
    
    try:
        X_train, X_test, y_train, y_test = preprocessor.preprocess(
            input_path=url,
            output_dir=output_dir
        )
        
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Number of classes: {len(y_train.unique())}")
        print(f"Output directory: {output_dir}")
        print("="*60)
        
        return 0
    
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
