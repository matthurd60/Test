#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:53:30 2025

@author: matthewhurd
"""

import pandas as pd
import torch
import yfinance as yf
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline
import numpy as np

if torch.backends.mps.is_available():
    device = "mps"  # Use MPS for GPU acceleration on Mac Mini
else:
    device = "cpu"  # Fallback to CPU

# Step 1: Load Historical Data (S&P 500 as an example)
ticker = "^GSPC"  # S&P 500 Index
data = yf.download(ticker, start="2010-01-01", end="2025-01-01", progress=False)

# Step 2: Calculate Daily Returns
data['Returns'] = data['Close'].pct_change()

# Drop the first NaN value due to pct_change()
data = data.dropna()

# Step 3: Prepare the Data for Chronos
# We will use the "Returns" column as our target time series for forecasting
df = data[['Returns']]  # Target: equity returns

# Step 4: Initialize the Chronos Pipeline
# Using a pretrained Chronos model
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",  # You can use "amazon/chronos-bolt-small" for another model variant
    device_map=device,  # Use "mps" or "cuda" if available for faster processing
    torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,  # Adjust dtype based on your system
)

# Step 5: Prepare the data for forecasting (create context tensor)
context = torch.tensor(df['Returns'].values, dtype=torch.float32).unsqueeze(0)  # Unsqueeze for batch size of 1

# Step 6: Forecast the Next 30 Days
forecast = pipeline.predict(context, prediction_length=30)

# Step 7: Display the Forecasted Returns
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)


# Create a plot to visualize the results
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Returns'], label="Historical Returns")
plt.plot(pd.date_range(df.index[-1], periods=31, freq='D')[1:], median, label="Forecasted Returns", linestyle='--', color='orange')
plt.title(f"Equity Market Returns Forecast for {ticker}")
plt.xlabel("Date")
plt.ylabel("Returns")
plt.legend()
plt.show()