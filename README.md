# Ex.No: 1B                     CONVERSION OF NON STATIONARY TO STATIONARY DATA
# Date: 31/10/2026

### AIM:
To perform regular differncing,seasonal adjustment and log transformatio on international airline passenger data

# Dataset Name: Retail Sales Dataset
# Software Required: 
- Jupyter Notebook
### ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.
### PROGRAM:
# Developed By: Vijiyalakshmi A
# Register No: 212225240185

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
data = pd.read_csv("retail_sales_dataset.csv")

# Display first five rows
data.head()

# Convert Date into datetime
data['Date'] = pd.to_datetime(data['Date'])

# Check data types
data.dtypes

# Set Date as index
data.set_index('Date', inplace=True)

# Resample daily sales
data = data['Total Amount'].resample('D').sum().to_frame()

# Fill missing values
data['Total Amount'] = data['Total Amount'].interpolate()

# Regular Differencing
data['sales_diff'] = data['Total Amount'] - data['Total Amount'].shift(1)

# Seasonal Adjustment
result = seasonal_decompose(
    data['Total Amount'],
    model='additive',
    period=7
)

data['sales_sea_diff'] = result.resid

# Log Transformation
data['sales_log'] = np.log1p(data['Total Amount'])

# Log Transformation and Regular Differencing
data['sales_log_diff'] = data['sales_log'] - data['sales_log'].shift(1)

# Log Transformation and Seasonal Adjustment
result = seasonal_decompose(
    data['sales_log_diff'].dropna(),
    model='additive',
    period=7
)

data['sales_log_seasonal_diff'] = result.resid

# Plot all transformations
plt.figure(figsize=(16, 16))

plt.subplot(6, 1, 1)
plt.plot(data['Total Amount'], label='Original')
plt.legend(loc='best')
plt.title('Original Data')
plt.xlabel('Date')
plt.ylabel('Total Sales')

plt.subplot(6, 1, 2)
plt.plot(data['sales_diff'], label='Regular Difference')
plt.legend(loc='best')
plt.title('Regular Differencing')
plt.xlabel('Date')
plt.ylabel('Differenced Sales')

plt.subplot(6, 1, 3)
plt.plot(data['sales_sea_diff'], label='Seasonal Adjustment')
plt.legend(loc='best')
plt.title('Seasonal Adjustment')
plt.xlabel('Date')
plt.ylabel('Seasonally Adjusted Sales')

plt.subplot(6, 1, 4)
plt.plot(data['sales_log'], label='Log Transformation')
plt.legend(loc='best')
plt.title('Log Transformation')
plt.xlabel('Date')
plt.ylabel('Log(Total Sales)')

plt.subplot(6, 1, 5)
plt.plot(data['sales_log_diff'], label='Log Transformation and Regular Differencing')
plt.legend(loc='best')
plt.title('Log Transformation and Regular Differencing')
plt.xlabel('Date')
plt.ylabel('Differenced Log Sales')

plt.subplot(6, 1, 6)
plt.plot(data['sales_log_seasonal_diff'], label='Log Transformation and Seasonal Differencing')
plt.legend(loc='best')
plt.title('Log Transformation and Regular Differencing and Seasonal Adjustment')
plt.xlabel('Date')
plt.ylabel('Seasonally Adjusted Log Sales')

plt.tight_layout()
plt.show()

# Plot all data
data.plot(kind='line', figsize=(12, 6))
plt.show()

```

### OUTPUT:

ORIGINAL DATA:
<img width="1503" height="242" alt="image" src="https://github.com/user-attachments/assets/09f83aca-7b5a-4e22-ab6e-5b7c92bec746" />

REGULAR DIFFERENCING:
<img width="1460" height="230" alt="image" src="https://github.com/user-attachments/assets/7ba7d7da-51b3-4496-9953-1e82664ab5ef" />

SEASONAL ADJUSTMENT:
<img width="1512" height="237" alt="image" src="https://github.com/user-attachments/assets/af5a6218-8afb-4593-98fd-d8beeb41e065" />

LOG TRANSFORMATION:
<img width="1467" height="248" alt="image" src="https://github.com/user-attachments/assets/046c494a-b48a-45f8-95c7-b1d22497e076" />

LOG TRANSFORMATION AND REGULAR DIFFERENCING:
<img width="1488" height="238" alt="image" src="https://github.com/user-attachments/assets/0b6a79d8-c55c-4d47-bf52-afac094e5c7f" />

LOG TRANSFORMATION AND REGULAR DIFFERENCING AND SEASONAL ADJUSTMENT:
<img width="1441" height="255" alt="image" src="https://github.com/user-attachments/assets/5e7e786b-0a35-4f42-b110-21882b3048d2" />

OVERALL VIEW:
<img width="985" height="478" alt="image" src="https://github.com/user-attachments/assets/01cfbc70-b04a-4978-8bf7-698baab5527c" />

### RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on international airline passenger
data.
