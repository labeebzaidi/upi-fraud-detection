import pandas as pd
import numpy as np

def generate_full_dataset():
    np.random.seed(42)
    n_samples = 5000

    categories = ['Food & Dining', 'Groceries', 'Electronics', 'P2P Transfer', 'Utilities', 'Travel/Transport', 'Healthcare', 'Gaming/Entertainment']
    devices = ['Mobile Web', 'Native App Android', 'Native App iOS', 'Desktop Web', 'POS Terminal']

    data = {
        'transaction_hour': np.random.randint(0, 24, n_samples),
        'transaction_day': np.random.randint(1, 31, n_samples),
        'transaction_month': np.random.randint(1, 13, n_samples),
        'transaction_year': np.random.choice([2023, 2024], n_samples),
        'transaction_exact_time': np.random.randint(0, 86400, n_samples),
        'transaction_amount': np.random.exponential(scale=2000, size=n_samples),
        'upi_number': [f'upi{i}@bank' for i in range(n_samples)],
        'account_holder_name': [f'User{i}' for i in range(n_samples)],
        'seller_name': [f'Seller{np.random.randint(1, 100)}' for i in range(n_samples)],
        'transaction_category': np.random.choice(categories, n_samples),
        'device_type': np.random.choice(devices, p=[0.2, 0.4, 0.3, 0.05, 0.05], size=n_samples)
    }

    df = pd.DataFrame(data)

    # Some basic synthetic fraud rules
    is_fraud = np.where(
        ((df['transaction_amount'] > 10000) & (df['transaction_hour'] < 5)) | 
        ((df['transaction_category'] == 'Gaming/Entertainment') & (df['transaction_amount'] > 5000)) |
        ((df['transaction_category'] == 'P2P Transfer') & (df['transaction_amount'] > 20000)) |
        ((df['device_type'] == 'Desktop Web') & (df['transaction_amount'] > 15000) & (df['transaction_hour'] > 22)), 1, 0
    )

    # Add 5% noise to make things realistic
    noise_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    is_fraud[noise_indices] = 1 - is_fraud[noise_indices]

    df['transaction_status'] = np.where(is_fraud == 1, 'Fraud', 'Valid')

    # Introduce 5% network/user failure rate before ML processing
    fail_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    # The fail_reasons aren't strictly necessary for the dataset, but 'Failed' status is.
    df.loc[fail_indices, 'transaction_status'] = 'Failed'

    df.to_csv('upi_training_dataset_5k.csv', index=False)
    print("Diverse Dataset generated successfully: upi_training_dataset_5k.csv")

if __name__ == '__main__':
    generate_full_dataset()
