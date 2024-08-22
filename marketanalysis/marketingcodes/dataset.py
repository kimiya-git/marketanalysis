from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm


from marketingcodes.config import (PROCESSED_DATA_DIR,
                                        RAW_DATA_DIR,EXTERNAL_DATA_DIR,np,
                                        pd,INTERIM_DATA_DIR,plt,FIGURES_DIR)

app = typer.Typer()

print("hi")
def read_sample(sample_size=20000, path='/content/online_retail_II.csv', seed=1155):
    """
    :param sample_size:
    :param path:
    :param seed:
    :return: dataframe and a sample , if you need to work
    with the sample, you need to replace df with sample in the main section
    """
    df = pd.read_csv(path)
    sample = df.sample(sample_size, random_state=seed)
    return df, sample

def missing_values(df):
  """
  :param df: the read dataframe
  :return: the column name and its number and indicating how much
  of it is null
  """
  missing_count = df.isnull().sum()
  missing_Percentages = pd.DataFrame (
      {
          'Column' : missing_count.index,
          'Count' : missing_count.values,
          'Percentage' : missing_count.values / len(df) * 100
      })
  return missing_Percentages


def explore_sample(sample):
  """
  :param sample: dataframe either sample or df
  :return: returns information about the given dataframe
  """
  head = pd.DataFrame(sample.head())
  tail = pd.DataFrame(sample.tail())
  nunique = pd.DataFrame(sample.nunique() , columns = ['number of uniques'])
  describe =  pd.DataFrame(sample.describe())
  dtypes = pd.DataFrame(sample.dtypes , columns=['types'])

  result = {
      'Table 3' : head,
      'Table 4' : tail,
      'Table 5' : nunique,
      'Table 6 ' : missing_values(sample) ,
      'Table 7' : describe,
      'Table 8 ' : dtypes

  }

  return result

def print_sample_exploration(results):
  """
  :param results: gets a dictionary from previous function
  :return: prints result of information about dataframe
  """

  for operation , dataframe in results.items():
    print(f'{operation}')

    if operation == 'Table 6':
      print(' total percentage' , dataframe['Percentage'].sum())
    print(dataframe)


def handle_date(sample):
  """
  :param sample: gets a dataframe with datetime column
  :return: splits datatime into year, months, time,day,etc
  """
  sample['InvoiceDate'] = pd.to_datetime(sample['InvoiceDate'])
  sample['InvoiceYear'] = sample['InvoiceDate'].dt.year
  sample['InvoiceMonth'] = sample['InvoiceDate'].dt.month
  sample['InvoiceDay'] =  sample['InvoiceDate'].dt.day
  sample['Invoicetime'] = sample['InvoiceDate'].dt.time
  sample['InvoiceHour'] = sample['InvoiceDate'].dt.hour
  sample['Invoiceday'] = sample['InvoiceDate'].dt.day_name()
  return sample

def get_total(row) :
  """
  :param row: gets each row of the dataframe
  :return: final price is calculated
  """

  return row['Quantity'] * row['Price']


def purches_time(hour):

    if 6<= hour < 12:
      return 'morning'
    elif 12<=hour<18:
      return 'afternoon'
    else:
      return 'evening'

def clean_customer_id(csv_path: Path):
    """
    :param csv_path: you have to give csv file saved in external folder
    :return: cleaned final dataset
    """
    df = pd.read_csv(csv_path)
    print("Shape of data before removing NaN's CustomerID:", df.shape)
    df.dropna(subset=["Customer ID"], axis=0, inplace=True)
    print("Shape of data after removing NaN's CustomerID:", df.shape)

    print("Number of duplicates before cleaning:", df.duplicated().sum())
    df = df.drop_duplicates(keep="first")
    print("Number of duplicates after cleaning:", df.duplicated().sum())

    return df


def calculate_peak_hour(sample):
    d = {}
    for index, row in sample.iterrows():
        if row['Invoicetime'] in d:
            d[row['Invoicetime']] += row['total_price']
        else:
            d[row['Invoicetime']] = row['total_price']

    peak_hour = max(d, key=d.get)
    return peak_hour


def save_invoice_plot(df, output_path):
    df1 = pd.DataFrame(df.groupby('Invoiceday')['Invoice'].count()).reset_index()
    fig = plt.figure(figsize=(8, 4))
    plt.bar(df1['Invoiceday'], df1['Invoice'], color='maroon', width=0.3)
    plt.xlabel("Days of a week")
    plt.ylabel("No. Invoices")
    plt.title("Number of total Invoices per day")
    plt.savefig(output_path, format='png')
    plt.show()

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = EXTERNAL_DATA_DIR / "online_retail_II.csv",
    output_path_inter : Path = INTERIM_DATA_DIR/ "time_price_edited.csv",
    output_path: Path = PROCESSED_DATA_DIR / "cleaned_dataset.csv",
    figure_path : Path = FIGURES_DIR / "invoice_plot.png"
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")


    df, sample = read_sample(path=input_path)
    sem = pd.DataFrame(sample.sem(numeric_only=True), columns=['SEM'])
    print(sem)

    results = explore_sample(df)
    print_sample_exploration(results)

    df = handle_date(df)
    df['total_price'] = df.apply(get_total, axis=1)
    df['PurchaseTimeofday'] = df['InvoiceHour'].apply(purches_time)
    df.to_csv(output_path_inter, index=False)

    # Clean the Customer ID column
    cleaned_df = clean_customer_id(output_path_inter)
    cleaned_df.to_csv(output_path, index=False)
    logger.success(f"Cleaned dataset saved to {output_path}")

    # print peak of hour in dataset
    peak_hour = calculate_peak_hour(cleaned_df)
    print(f"Peak hour: {peak_hour}")
    print("hi")
    # save an image to report
    save_invoice_plot(cleaned_df, figure_path)
    logger.success(f"Invoice plot saved to {figure_path}")

    # You can now use df and sample as needed
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------

if __name__ == "__main__":
    app()
