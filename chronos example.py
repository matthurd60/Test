#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 19:37:34 2025

@author: matthewhurd
"""

import pandas as pd
import torch
from chronos import BaseChronosPipeline

# Check if MPS (Metal Performance Shaders) is available
if torch.backends.mps.is_available():
    device = "mps"  # Use MPS for GPU acceleration on Mac Mini
else:
    device = "cpu"  # Fallback to CPU

# Load your time series data into a pandas DataFrame
df = pd.read_csv("https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")

# Initialize the Chronos pipeline with a pretrained model
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",  # You can also use "amazon/chronos-bolt-small" for the Chronos-Bolt model
    device_map=device,  # Use MPS or CPU based on availability
    torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,  # MPS supports only float32 and bfloat16
)
# Prepare the context tensor from your data
context = torch.tensor(df["#Passengers"])

# Generate forecasts
forecast = pipeline.predict(
    context=context,
    prediction_length=12
)

# Visualize the forecast
import matplotlib.pyplot as plt
import numpy as np

forecast_index = range(len(df), len(df) + 12)
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

plt.figure(figsize=(8, 4))
plt.plot(df["#Passengers"], color="royalblue", label="historical data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()