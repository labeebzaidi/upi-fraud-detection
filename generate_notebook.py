import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_code_cell(source):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    })

def add_markdown_cell(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    })

add_markdown_cell(["# UPI Fraud Detection Model Building Phase\n", "This notebook trains and compares various models for UPI fraud detection usnig a diverse feature schema."])

add_code_cell([
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
])

add_markdown_cell(["## 1. Load or Generate Realistic Dataset"])

add_code_cell([
    "np.random.seed(42)\n",
    "n_samples = 5000\n",
    "\n",
    "categories = ['Food & Dining', 'Groceries', 'Electronics', 'P2P Transfer', 'Utilities', 'Travel/Transport', 'Healthcare', 'Gaming/Entertainment']\n",
    "devices = ['Mobile Web', 'Native App Android', 'Native App iOS', 'Desktop Web', 'POS Terminal']\n",
    "\n",
    "data = {\n",
    "    'transaction_hour': np.random.randint(0, 24, n_samples),\n",
    "    'transaction_day': np.random.randint(1, 31, n_samples),\n",
    "    'transaction_month': np.random.randint(1, 13, n_samples),\n",
    "    'transaction_year': np.random.choice([2023, 2024], n_samples),\n",
    "    'transaction_exact_time': np.random.randint(0, 86400, n_samples),\n",
    "    'transaction_amount': np.random.exponential(scale=2000, size=n_samples),\n",
    "    'upi_number': [f'upi{i}@bank' for i in range(n_samples)],\n",
    "    'account_holder_name': [f'User{i}' for i in range(n_samples)],\n",
    "    'seller_name': [f'Seller{np.random.randint(1, 100)}' for i in range(n_samples)],\n",
    "    'transaction_category': np.random.choice(categories, n_samples),\n",
    "    'device_type': np.random.choice(devices, p=[0.2, 0.4, 0.3, 0.05, 0.05], size=n_samples)\n",
    "}\n",
    "\n",
    "df = pd.DataFrame(data)\n",
    "\n",
    "is_fraud = np.where(\n",
    "    ((df['transaction_amount'] > 10000) & (df['transaction_hour'] < 5)) | \n",
    "    ((df['transaction_category'] == 'Gaming/Entertainment') & (df['transaction_amount'] > 5000)) | \n",
    "    ((df['transaction_category'] == 'P2P Transfer') & (df['transaction_amount'] > 20000)) | \n",
    "    ((df['device_type'] == 'Desktop Web') & (df['transaction_amount'] > 15000) & (df['transaction_hour'] > 22)), 1, 0\n",
    ")\n",
    "\n",
    "noise_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)\n",
    "is_fraud[noise_indices] = 1 - is_fraud[noise_indices]\n",
    "df['transaction_status'] = np.where(is_fraud == 1, 'Fraud', 'Valid')\n",
    "fail_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)\n",
    "df.loc[fail_indices, 'transaction_status'] = 'Failed'\n",
    "\n",
    "print(df.head())\n",
    "print(f\"Status Distribution: \\n{df['transaction_status'].value_counts(normalize=True)}\")\n"
])

add_markdown_cell(["## 2. Preprocessing"])

add_code_cell([
    "features_to_drop = ['upi_number', 'account_holder_name', 'seller_name']\n",
    "# The ML model will not evaluate Failed transactions\n",
    "df_filtered = df[df['transaction_status'] != 'Failed'].copy()\n",
    "df_filtered['is_fraud'] = np.where(df_filtered['transaction_status'] == 'Fraud', 1, 0)\n",
    "X = df_filtered.drop(columns=['is_fraud', 'transaction_status'] + features_to_drop)\n",
    "y = df_filtered['is_fraud']\n",
    "\n",
    "# Handle categorical variables dynamically\n",
    "X = pd.get_dummies(X, columns=['transaction_category', 'device_type'])\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n",
    "print(f'Training shape: {X_train.shape}, Testing shape: {X_test.shape}')"
])

add_markdown_cell(["## 3. Compare Initial Models"])

add_code_cell([
    "models = {\n",
    "    'Logistic Regression': LogisticRegression(C=0.01),\n",
    "    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=15),\n",
    "    'Support Vector Classifier': SVC(C=0.5, kernel='rbf'),\n",
    "    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),\n",
    "    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)\n",
    "}\n",
    "\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train, y_train)\n",
    "    y_pred = model.predict(X_test)\n",
    "    acc = accuracy_score(y_test, y_pred)\n",
    "    print(f\"{name} Accuracy: {acc*100:.2f}%\")"
])

add_markdown_cell(["## 4. Convolutional Neural Network (CNN) Model"])

add_code_cell([
    "X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)\n",
    "X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)\n",
    "\n",
    "cnn_model = Sequential([\n",
    "    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),\n",
    "    MaxPooling1D(pool_size=2),\n",
    "    Flatten(),\n",
    "    Dense(128, activation='relu'),\n",
    "    Dropout(0.2),\n",
    "    Dense(64, activation='relu'),\n",
    "    Dense(1, activation='sigmoid')\n",
    "])\n",
    "\n",
    "cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n",
    "history = cnn_model.fit(X_train_cnn, y_train, epochs=20, batch_size=32, validation_data=(X_test_cnn, y_test), verbose=1)\n"
])

add_code_cell([
    "loss, accuracy = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)\n",
    "print(f\"CNN Model Target Accuracy: {accuracy*100:.2f}%\")\n"
])

add_code_cell([
    "cnn_model.save('cnn_model.keras')\n",
    "import joblib\n",
    "joblib.dump(scaler, 'scaler.pkl')\n",
    "print(\"Models saved.\")\n"
])

with open(r'c:\Users\labee\Desktop\upi fraud\model_building.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Generated model_building.ipynb successfully.")
