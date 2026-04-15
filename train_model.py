# -*- coding: utf-8 -*-
"""
Fake Job Predictor Model 
Trainig Script with Error Handling and Model Persistence
Author - Saksham Nagar
"""
import re
import pandas as pd
import pickle
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,confusion_matrix,accuracy_score,
    precision_score,recall_score,f1_score,roc_auc_score
)

warnings.filterwarnings('ignore')

class FakeJobModel:
    """
    Production ready model for fake job predictions
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.metrics = {}

    def load_and_preprocess_data(self):
        #Load and Preprocess data
        try:
            print(f"Loading data from {self.data_path}...")
            df = pd.read_csv(self.data_path)
            if df.empty:
                raise ValueError("Dataset is empty")
            print(f"Loaded {len(df)} records")

            #Fill missing Values
            text_cols = ['title', 'description', 'company_profile', 'requirements', 'benefits' ]
            
            for col in text_cols:
                if col in df.columns:
                    df[col] = df[col].fillna('').astype(str)
                else:
                    raise KeyError(f"Column {col} not found in dataset")
            
            # Remove duplicate rows
            initial_shape = df.shape
            df.drop_duplicates(inplace=True)
            print(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")

            # Drop job_id column if it exists
            if 'job_id' in df.columns:
                df.drop(['job_id'], axis=1, inplace=True)
                print("Dropped 'job_id' column")
            else:
                print("'job_id' column not found, skipping drop")
            
            #Create Combined text
            df['text'] = (
                df['title'] + ' ' +
                df['company_profile'] + ' ' +
                df['description'] + ' ' +
                df['requirements'] + ' ' +
                df['benefits']
            )
            def clean_text(text):
                text = text.lower()
                text = re.sub(r'[^a-zA-Z]',' ',text)
                return text
            df['text'] = df['text'].apply(clean_text)
            
            if df['text'].isnull().sum() > 0:
                raise ValueError("Text column still contains null values")
            print("Text preprocessing completed successfully\n")

            self.df = df
        
        except Exception as e:
            print(f"Error in data preprocessing: {str(e)}")
            raise

    #Intialising TFIDF Vectorization
    def tfidf_vectorize(self,X):
        try:
            print("\n"+"="*50)
            print("TF-IDF Vectorization")
            print("\n"+"="*50)

            tfidf = TfidfVectorizer(stop_words='english',max_features=5000)
            X = tfidf.fit_transform(X)
            self.tfidf = tfidf
            print(f"✓ TF-IDF Vectorization completed")
            print(f"✓ Shape: {X.shape}")
            return X,tfidf
        except Exception as e:
            print(f"Error in TF-IDF Vectorization: {str(e)}")
            raise
    
    #Training the model
    def train_model(self,X,y):
        try:
            print("\n" + "=" * 50)
            print("Training Model")
            print("\n" + "=" * 50)

            #Split data
            X_train,X_test,y_train,y_test = train_test_split(
                X,y,test_size=0.2,random_state=42,stratify=y
            )
            print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

            #Train model
            self.model = LogisticRegression(class_weight='balanced', max_iter=1000)
            self.model.fit(X_train, y_train)
            print("Model trained Successfully")

            return X_train,X_test,y_train,y_test

        except Exception as e:
            print(f"Error in training model: {str(e)}")
            raise 

    def evaluate_model(self,X_test,y_test):
        if self.model is None:
            raise ValueError("Model not trained yet")
        else:
            try:
                print("\n"+"="*50)
                print("Evaluating Model")
                print("\n"+"="*50)

                y_pred = self.model.predict(X_test)
                y_proba = self.model.predict_proba(X_test)[:,1]

                #Metrics
                self.metrics = {
                'accuracy': accuracy_score(y_test,y_pred),
                'precision': precision_score(y_test,y_pred,zero_division=0),
                'recall': recall_score(y_test,y_pred,zero_division=0),
                'f1': f1_score(y_test,y_pred,zero_division=0),
                'roc_auc': roc_auc_score(y_test,y_proba)
                }
                print("Classification Report")
                print(classification_report(y_test,y_pred))
                print("\nConfusion Matrix")
                print(confusion_matrix(y_test,y_pred))
                print("\nEvaluation completed successfully")
        
            except Exception as e:
                print(f"Error in evaluating model: {str(e)}")
                raise
    
    #Save model and tfidf
    def save_model(self,output_dir='Trained_Models'):
        try:
            Path(output_dir).mkdir(parents=True,exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = Path(output_dir) / f'fake_job_model_{timestamp}.pkl'
            tfidf_path = Path(output_dir) / f'tfidf_vectorizer_{timestamp}.pkl'
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)

            print(f"Model saved successfully to {model_path}")

            if hasattr(self,'tfidf'):
                with open(tfidf_path, 'wb') as f:
                    pickle.dump(self.tfidf, f)
                print(f"Tfidf saved to {tfidf_path}")
            else:
                print("Tfidf vectorizer wasn't found, skipping save")
            
            return model_path, tfidf_path
        
        except Exception as e:
            print(f"Error in saving model: {str(e)}")
            raise

    def main(self):
        """Main Training Pipeline"""
        try:
            print("\n" + "="*50)
            print("Fake Job Prediction Model Training")
            print("="*50)

            #Load and preprocess data
            self.load_and_preprocess_data()

            #TF-IDF Vectorization
            X,tfidf = self.tfidf_vectorize(self.df['text'])

            #Train model
            if 'fraudulent' not in self.df.columns:
                raise KeyError("Target column Fraudulent not found")
            else:
                X_train,X_test,y_train,y_test = self.train_model(X,self.df['fraudulent'])

            #Evaluate model
            self.evaluate_model(X_test,y_test)

            #Save model
            model_path,tfidf_path = self.save_model()

            print("\n" + "="*50)
            print("Training completed successfully")
            print("="*50)
            print(f"Model saved to: {model_path}")
            print(f"TF-IDF saved to: {tfidf_path}")
            print(f"Metrics: {self.metrics}")

        except Exception as e:
            print(f"\nError in training pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    model = FakeJobModel('Data/cleaned_data.csv')
    model.main()