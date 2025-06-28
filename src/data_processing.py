import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg = X.groupby('CustomerId').agg(
            total_amount=('Amount', 'sum'),
            avg_amount=('Amount', 'mean'),
            std_amount=('Amount', 'std'),
            transaction_count=('TransactionId', 'count'),
            fraud_count=('FraudResult', 'sum')
        ).fillna(0)
        return agg.reset_index()


class TimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, time_col='TransactionStartTime'):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col], errors='coerce')
        df['transaction_hour'] = df[self.time_col].dt.hour
        df['transaction_day'] = df[self.time_col].dt.day
        df['transaction_month'] = df[self.time_col].dt.month
        df['transaction_weekday'] = df[self.time_col].dt.weekday
        return df


def preprocess_data(raw_df):
   
    df = TimeFeatures().transform(raw_df)


    features = ['CustomerId', 'Amount', 'TransactionId', 'FraudResult',
                'transaction_hour', 'transaction_day', 'transaction_month', 'transaction_weekday',
                'ProductCategory', 'ChannelId', 'CurrencyCode']

    df = df[features]

  
    aggregates = AggregateFeatures().transform(df)


    cat_cols = ['ProductCategory', 'ChannelId', 'CurrencyCode']
    latest_cats = df.sort_values('transaction_month').groupby('CustomerId')[cat_cols].last().reset_index()

    final_df = pd.merge(aggregates, latest_cats, on='CustomerId', how='left')


    num_cols = ['total_amount', 'avg_amount', 'std_amount', 'transaction_count', 'fraud_count']
    cat_cols = ['ProductCategory', 'ChannelId', 'CurrencyCode']

 
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(final_df)
    return X_processed, final_df['CustomerId']  

