from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from marketingcodes.config import (PROCESSED_DATA_DIR,
                                        RAW_DATA_DIR,EXTERNAL_DATA_DIR,np,
                                        pd,INTERIM_DATA_DIR,
                                        plt,FIGURES_DIR, SimpleImputer,IsolationForest)


def detect_outliers(df):
    # Select features
    X = df[['total_price', 'InvoiceHour']]

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Initialize and fit Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iso_forest.fit(X_imputed)

    # Predict outliers
    outliers = iso_forest.predict(X_imputed)

    # Update the 'Outlier' column
    df['Outlier'] = ['Yes' if x == -1 else 'No' for x in outliers]

    return df


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "cleaned_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "outlier.csv",
    # -----------------------------------------
        ):
    # Assuming df is your DataFrame
    outlierdf = pd.read_csv(input_path)
    outliers = detect_outliers(outlierdf)
    outliers.to_csv(output_path)

    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
