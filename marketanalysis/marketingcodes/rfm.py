from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from marketingcodes.config import (PROCESSED_DATA_DIR,
                                        RAW_DATA_DIR,EXTERNAL_DATA_DIR,np,
                                        pd,INTERIM_DATA_DIR,
                                        plt,FIGURES_DIR, SimpleImputer,IsolationForest)

import pandas as pd
import numpy as np

def calculate_rfm(df):
    now = pd.Timestamp.now()

    # Frequency
    frequency = df.groupby(['Customer ID', 'InvoiceDate'])['Invoice'].count().reset_index()
    total_bought = df.groupby('Customer ID')['Quantity'].count()
    purchase_times = df.groupby(['Customer ID', 'PurchaseTimeofday'])['Invoice'].nunique().unstack(fill_value=0)

    # Recency
    recency = (now - df.groupby('Customer ID')['InvoiceDate'].max()).dt.days

    # Monetary
    Montary = df.groupby('Customer ID')['total_price'].sum()

    # Reset indices if necessary
    frequency.reset_index(drop=True, inplace=True)
    recency.reset_index(drop=True, inplace=True)
    Montary.reset_index(drop=True, inplace=True)

    RFM = pd.DataFrame({
        'RecencyValue': recency,
        'FrequencyValue': frequency['Invoice'],
        'MonetaryValue': Montary
    })

    Montary = Montary if Montary is not None else 0
    frequency = frequency if frequency is not None else 0

    avg_spending_per_invoice = Montary / frequency['Invoice']
    avg_spending_per_item = df.groupby('Customer ID')['total_price'].sum() / df.groupby('Customer ID')['Quantity'].sum()
    avg_spending_per_item.replace([np.inf, -np.inf], np.nan, inplace=True)
    avg_spending_per_item.dropna(inplace=True)

    customer_df = pd.DataFrame({
        'TotalItemsBought': total_bought,
        'AvgSpendingPerInvoice': avg_spending_per_invoice,
        'AvgSpendingPerItem': avg_spending_per_item
    }).join(RFM)

    customer_df.fillna(0, inplace=True)
    customer_df.reset_index(inplace=True)
    customer_df['Customer ID'] = df['Customer ID']
    customer_df['Customer ID'] = customer_df['Customer ID'].replace('nan', np.nan)
    customer_df = customer_df.dropna(subset=['Customer ID'])
    customer_df = customer_df.dropna()

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    most_recent_invoice_date = df['InvoiceDate'].max()
    customer_recency = df.groupby('Customer ID')['InvoiceDate'].max().reset_index()
    customer_recency['RecencyValue_insidesample'] = (most_recent_invoice_date - customer_recency['InvoiceDate']).dt.days
    customer_recency = customer_recency.drop('InvoiceDate', axis=1)
    customer_df = customer_df.merge(customer_recency, on='Customer ID', how='left')

    bin_edges = [0, 100, 200, 300, 400, 525]
    bin_labels = [5, 4, 3, 2, 1]
    customer_df["Recency_Score"] = pd.cut(customer_df["RecencyValue_insidesample"], bins=bin_edges, labels=bin_labels, include_lowest=True)

    bin_edges = [0, 56, 112, 168, 224, 281]
    bin_labels = [5, 4, 3, 2, 1]
    customer_df["Frequency_Score"] = pd.cut(customer_df["FrequencyValue"], bins=bin_edges, labels=bin_labels, include_lowest=True)

    bin_edges = [-25111.09, 50000, 100000, 200000, 300000, 360260.28]
    bin_labels = [5, 4, 3, 2, 1]
    customer_df["Monetary_Score"] = pd.cut(customer_df["MonetaryValue"], bins=bin_edges, labels=bin_labels, include_lowest=True)

    customer_df['RF_score'] = customer_df["Recency_Score"].astype(int) + customer_df["Frequency_Score"].astype(int)

    seg_map = {
        (1, 3): 'At_risk',
        (4, 4): 'Hibernating',
        (4, 6): 'At_risk',
        (4, 7): 'Cant_loose',
        (6, 4): 'About_to_sleep',
        (6, 6): 'Need_attention',
        (7, 10): 'Loyal_customers',
        (7, 4): 'Promising',
        (10, 4): 'New_customers',
        (7, 6): 'Potential_loyalists',
        (10, 7): 'Champions',
        (5, 5): 'At_risk',
        (8, 8): 'Loyal_customers',
        (9, 9): 'Loyal_customers'
    }

    customer_df["RF Segment"] = customer_df["RF_score"].replace(seg_map, regex=True)

    summary = customer_df[["RF Segment", "RecencyValue", "FrequencyValue", "MonetaryValue"]].groupby("RF Segment").agg(["mean", "sum", "count", "min", "max"])

    return customer_df, summary

# Example usage
# df = pd.read_csv('path_to_your_data.csv')
# customer_df, summary = calculate_rfm(df)
# print(customer_df.head())
# print(summary)


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "cleaned_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "outlier.csv",
    # -----------------------------------------
        ):


    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
