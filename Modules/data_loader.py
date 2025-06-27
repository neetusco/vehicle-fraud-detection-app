import pandas as pd
import os
import sys

# Add project root to sys.path to access logger module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)

def load_dataset(filename="vehicle_insurance_fraud_dataset.csv"):
    """
    Loads the vehicle insurance fraud dataset from the data folder.
    """
    filepath = os.path.join("data", filename)
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Dataset loaded successfully with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found at {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def explore_dataset(df):
    """
    Logs basic EDA statistics of the DataFrame.
    """
    if df is not None:
        logger.info(f"Shape of dataset: {df.shape}")
        logger.info(f"Data types:\n{df.dtypes}")
        logger.info(f"Missing values:\n{df.isnull().sum()}")
        logger.info(f"Sample rows:\n{df.head()}")
    else:
        logger.warning("No dataset to explore.")

# Example usage
if __name__ == "__main__":
    df = load_dataset()
    explore_dataset(df)