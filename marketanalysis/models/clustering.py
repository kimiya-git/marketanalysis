from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
from marketingcodes.config import (PROCESSED_DATA_DIR,
                                        RAW_DATA_DIR,EXTERNAL_DATA_DIR,np,
                                        pd,INTERIM_DATA_DIR,
                                        plt,FIGURES_DIR,pd,np,
                                   KElbowVisualizer,KMeans,
                                   StandardScaler,sns,plt)



def plot_clusters(customer_df):
    # Select the features you want to visualize
    features = ['TotalItemsBought', 'AvgSpendingPerInvoice', 'AvgSpendingPerItem', 'FrequencyValue', 'MonetaryValue', 'RecencyValue_insidesample']

    # plt.figure(figsize=(3, 3))
    # Create pair plots to visualize the relationships between features
    sns.pairplot(customer_df[features] ,height=2)
    plt.show()




def perform_clustering(customer_df):
    features = customer_df[['TotalItemsBought',
                            'AvgSpendingPerInvoice',
                            'AvgSpendingPerItem',
                            'FrequencyValue',
                            'MonetaryValue',
                            'RecencyValue_insidesample']]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = KMeans(random_state=42)

    visualizer = KElbowVisualizer(model, k=(2, 10), timings=False, locate_elbow=True, metric='distortion')
    visualizer.fit(features_scaled)
    # visualizer.show()

    return features_scaled




def kmeans_clustring(customer_df, features_scaled,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    kmeans.fit(features_scaled)

    customer_df['cluster'] = kmeans.labels_

    return customer_df



app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "customer_rfm.csv",
    output_path: Path = PROCESSED_DATA_DIR / "kdf.csv",
    # -----------------------------------------
        ):

    df = pd.read_csv(input_path)

    # plot_clusters(df)
    features = perform_clustering(df)

    kdf = kmeans_clustring(df,features,5)

    # kdf.to_csv(output_path)



    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
