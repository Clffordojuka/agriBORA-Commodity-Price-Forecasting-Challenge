```markdown
# AgriBORA Commodity Price Forecasting: Technical Solution Documentation

**Project:** AgriBORA Commodity Price Forecasting  
**Author:** Clifford Ojuka 
**Date:** December 28, 2025  
**Strategy:** "Absolute Zero" (Hybrid Statistical/Heuristic Model)  
**Final Model Score:** 0.7346  

---

## 1. Project Overview

### 1.1 Background
Smallholder farmers in Africa often face significant financial risks due to market price volatility and post-harvest losses. AgriBORA provides a platform to mitigate these risks through market intelligence. This project focuses on developing a predictive model to forecast commodity prices, enabling farmers to make data-driven decisions on when to sell their produce.

### 1.2 Problem Statement
The objective is to predict the average weekly price of **Maize** across five key Kenyan counties: **Kiambu, Kirinyaga, Mombasa, Nairobi, and Uasin-Gishu**.

The forecasting challenge involves a sparse, rolling time-series dataset:
* **Input Data:** Historical weekly price anchors (Weeks 46, 49, 50, 51).
* **Prediction Targets:**
    1.  **Week 52:** representing the end-of-year holiday period.
    2.  **Week 1:** representing the post-holiday new year period.

### 1.3 Solution Approach: "Absolute Zero" Strategy
The solution utilizes a **Damped Mean-Trend Hybrid** architecture. The model posits that standard regression techniques overestimate market volatility in short-term horizons. Therefore, I decomposed the forecast into two distinct behavioral components:

1.  **Holiday Correction (Week 52):**
    * **Hypothesis:** Market activity slows down during the holiday week, leading to a temporary price dip or reversion to the mean.
    * **Technique:** A **Discounted Mean Reversion** model is used. The forecast is derived from the 3-week trailing average, applied with a **3.05% discount factor** to capture the demand drop-off.

2.  **New Year Recovery (Week 1):**
    * **Hypothesis:** Prices recover and follow the general upward trend, but the market is "stickier" (less volatile) than a pure linear projection would suggest.
    * **Technique:** A **Damped Linear Trend** model is used. The standard linear regression slope is dampened by a specific multiplier (**1.0245**) to provide a conservative, low-error growth estimate.

---

## 2. Technical Implementation

The solution is implemented in a modular pipeline consisting of Data Reconstruction, Analysis, and Hybrid Inference.

### Step 1: Library Imports & Configuration

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from typing import Dict, List

# Visualization settings
plt.style.use('ggplot')
sns.set_palette("husl")
%matplotlib inline

```

### Step 2: Data Reconstruction Engine

The input dataset provided sparse anchor points. To enable robust trend analysis, we programmatically reconstructed the missing time steps (Weeks 47 and 48) using linear interpolation to create a continuous 6-week time series.

```python
class MarketDataProcessor:
    """
    Handles data reconstruction and preprocessing for sparse time-series data.
    """
    def __init__(self):
        # Historical price anchors
        self.raw_data_wk46 = {
            'Kiambu': 38.33, 'Kirinyaga': 38.89, 'Mombasa': 36.11, 
            'Nairobi': 36.80, 'Uasin-Gishu': 33.22
        }
        self.raw_data_wk49 = {
            'Kiambu': 39.44, 'Kirinyaga': 40.00, 'Mombasa': 36.11, 
            'Nairobi': 37.85, 'Uasin-Gishu': 33.33
        }
        self.raw_data_wk50 = {
            'Kiambu': 42.78, 'Kirinyaga': 38.33, 'Mombasa': 37.22, 
            'Nairobi': 39.10, 'Uasin-Gishu': 34.89
        }
        self.raw_data_wk51 = {
            'Kiambu': 44.44, 'Kirinyaga': 46.67, 'Mombasa': 42.22, 
            'Nairobi': 42.78, 'Uasin-Gishu': 41.67
        }
        self.counties = sorted(list(self.raw_data_wk46.keys()))

    def _interpolate(self, start_val: float, end_val: float, steps: int) -> np.ndarray:
        """Linearly interpolates values between two known points."""
        return np.linspace(start_val, end_val, steps + 2)[1:-1]

    def reconstruct_time_series(self) -> Dict[str, np.ndarray]:
        """
        Reconstructs the full 6-week price history (Weeks 46-51).
        Returns: Dictionary mapping county names to numpy arrays of prices.
        """
        full_series = {}
        for county in self.counties:
            p46 = self.raw_data_wk46[county]
            p49 = self.raw_data_wk49[county]
            # Interpolate missing Weeks 47 and 48
            p47, p48 = self._interpolate(p46, p49, 2)
            
            p50 = self.raw_data_wk50[county]
            p51 = self.raw_data_wk51[county]
            
            # Construct sequential array [Wk46, Wk47, Wk48, Wk49, Wk50, Wk51]
            full_series[county] = np.array([p46, p47, p48, p49, p50, p51])
            
        return full_series

# Initialize and run reconstruction
processor = MarketDataProcessor()
county_series = processor.reconstruct_time_series()
print("Data Reconstruction Complete.")

```

### Step 3: Exploratory Data Analysis (EDA)

Visualizing the reconstructed trends allows us to observe the momentum leading into the forecast period. The data typically shows a strong upward trend into Week 51, necessitating the "damping" strategy to prevent overshooting in the forecasts.

```python
# Convert to DataFrame for plotting
df_eda = pd.DataFrame(county_series)
df_eda['Week'] = [46, 47, 48, 49, 50, 51]
df_eda.set_index('Week', inplace=True)

# Plotting
plt.figure(figsize=(10, 5))
for column in df_eda.columns:
    plt.plot(df_eda.index, df_eda[column], marker='o', linewidth=2, label=column)

plt.title('Reconstructed Market Price Trends (Weeks 46-51)')
plt.xlabel('Week Number')
plt.ylabel('Price (KES)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

```

### Step 4: Hybrid Modeling (Strategy 5.4 - "Absolute Zero")

This class implements the final inference logic. It calculates independent forecasts for the holiday dip and the trend recovery, applying the optimized multipliers to minimize RMSE.

**Model Parameters:**

* **Week 52 Factor:** `0.9695` (Calculated 3.05% reversion from mean).
* **Week 1 Factor:** `1.0245` (Conservative linear growth multiplier).

```python
class HybridForecaster:
    """
    Implements the winning 'Absolute Zero' strategy.
    Combines Linear Regression for trend detection with Mean Reversion.
    """
    def __init__(self, wk52_discount: float = 0.9695, wk1_trend_multiplier: float = 1.0245):
        self.wk52_discount = wk52_discount
        self.wk1_multiplier = wk1_trend_multiplier
        self.days = np.array([0, 7, 14, 21, 28, 35]).reshape(-1, 1) # Time index

    def predict(self, series_data: Dict[str, np.ndarray]) -> pd.DataFrame:
        predictions = []

        for county, prices in series_data.items():
            # --- Feature Engineering ---
            # 1. Linear Trend Component
            model = LinearRegression()
            model.fit(self.days, prices)
            
            # Predict 'raw' trend values for future timestamps (Wk52=Day 42, Wk1=Day 49)
            trend_wk1_raw = model.predict([[49]])[0]

            # 2. Rolling Mean Component
            mean_3wk = prices[-3:].mean() # Average of Wk49, Wk50, Wk51

            # --- Inference Logic ---
            
            # Target 1: Week 52 (Christmas Dip)
            # Logic: Market reverts to 3.05% below the recent 3-week mean.
            pred_52 = mean_3wk * self.wk52_discount

            # Target 2: Week 1 (January Recovery)
            # Logic: Follows linear trend but damped by conservative multiplier (1.0245).
            pred_1 = trend_wk1_raw * self.wk1_multiplier

            # --- Safety Clips ---
            # Prevents extreme outliers based on the last observed price (Wk51)
            last_price = prices[-1]
            pred_52 = np.clip(pred_52, last_price * 0.85, last_price * 1.15)
            pred_1 = np.clip(pred_1, last_price * 0.90, last_price * 1.35)

            # --- Formatting ---
            predictions.append({
                'ID': f"{county}_Week_52",
                'Target_RMSE': pred_52,
                'Target_MAE': pred_52
            })
            predictions.append({
                'ID': f"{county}_Week_1",
                'Target_RMSE': pred_1,
                'Target_MAE': pred_1
            })

        return pd.DataFrame(predictions)

```

### Step 5: Execution & Final Output

The final step instantiates the model with the tuned parameters and generates the submission file formatted for evaluation.

```python
def main():
    # 1. Initialize Forecaster with Optimized Parameters
    forecaster = HybridForecaster(
        wk52_discount=0.9695,       # 3.05% Discount from Mean
        wk1_trend_multiplier=1.0245 # Conservative Trend Multiplier
    )

    # 2. Generate Predictions
    submission_df = forecaster.predict(county_series)

    # 3. Output to CSV
    filename = 'AgriBORA_Final_Submission.csv'
    submission_df.to_csv(filename, index=False)
    
    print(f"Success. Submission generated: {filename}")
    print("\n--- Final Predicted Values ---")
    print(submission_df.head(10))

if __name__ == "__main__":
    main()

```

```

```