# Forecasting Seasonal Time Series Data Using Bidirectional LSTM Networks

## 👤 Author

| Name            | Role              | LinkedIn                                      |
|-----------------|-------------------|-----------------------------------------------|
| Jason Emmanuel  | Data Scientist | [linkedin.com/in/jasoneml](https://www.linkedin.com/in/jasoneml/) |

## 🌟 Project Overview

This project is focused on developing and evaluating a **Bidirectional Long Short-Term Memory (BiLSTM)** deep learning model for forecasting **seasonal time series data**. Seasonal time series, commonly encountered in fields like meteorology, retail sales, energy consumption, and economics, exhibit repeated patterns over fixed intervals—such as daily, weekly, monthly, or yearly cycles.

Forecasting such data accurately is crucial for planning and decision making. Traditional methods like ARIMA or Holt-Winters exponential smoothing can work well but often struggle to capture complex nonlinear relationships and long-term dependencies. Deep learning, especially Recurrent Neural Networks (RNNs) and LSTM architectures, offer powerful tools to model these sequences because they can learn temporal dependencies effectively.

This project uses TensorFlow to:

- Generate synthetic seasonal time series data combining trend, seasonality, and noise components  
- Prepare the data with sliding windows appropriate for sequence models  
- Build a Bidirectional LSTM model that learns from both past and future context within sequences  
- Train the model with dynamic learning rate adjustment for efficient optimization  
- Evaluate forecasting accuracy with meaningful metrics  
- Visualize both training dynamics and final forecasting performance  

---

## 🎯 Why Forecast Seasonal Time Series with BiLSTM?

### Challenges in Seasonal Time Series Forecasting

Seasonal time series data often contains:

- **Trend:** A long-term increase or decrease in values  
- **Seasonality:** Regular periodic fluctuations (daily, weekly, yearly)  
- **Noise:** Random fluctuations due to unpredictable factors  

Accurate forecasting requires modeling these components simultaneously while capturing intricate temporal dependencies.

### Traditional vs Deep Learning Approaches

- **Traditional statistical models** (e.g., ARIMA, Holt-Winters) assume linear relationships and require manual feature engineering to capture seasonality and trend, which limits flexibility.  
- **Deep learning models**, particularly **LSTM networks**, inherently model long-range dependencies and nonlinearities by maintaining memory states across sequences.  
- **Bidirectional LSTMs** take this further by processing sequences in both forward and backward directions, enabling the model to have richer context when predicting values at each timestep.  

Hence, BiLSTM is well-suited for seasonal data where the value at a time point depends not only on past but also future data patterns within a window.

---

## 🧩 Synthetic Data Generation

To develop and test the model, synthetic time series data was generated to simulate realistic seasonal patterns with noise and trend.

### Components

- **Trend:** A linear upward or downward slope added over time to mimic gradual changes.  
- **Seasonality:** A periodic repeating pattern using sine/cosine functions or piecewise patterns with a fixed period (e.g., 365 days for yearly seasonality).  
- **Noise:** Random Gaussian noise to simulate measurement errors or random external influences.  

### Implementation Details

The following helper functions were implemented:

- `trend(time, slope)`: Returns a linear trend component given a slope.  
- `seasonality(time, period, amplitude)`: Generates a repeating seasonal pattern based on the time vector, period length, and amplitude.  
- `noise(time, noise_level, seed)`: Adds Gaussian noise with specified noise level and random seed for reproducibility.  

Example code snippet for generating a 4-year daily time series:

```python
time = np.arange(4 * 365 + 1, dtype="float32")  # 4 years of daily data
series = trend(time, slope=0.005) + 10
series += seasonality(time, period=365, amplitude=50)
series += noise(time, noise_level=3, seed=51)
```

---

## 📊 Visualizations

![image](https://github.com/user-attachments/assets/f061f31b-144a-4c74-9de0-d7a7bb3377ca)

The graph displays a time series with a repeating pattern characterized by sharp spikes followed by gradual declines, occurring at regular intervals. This cyclical behavior suggests a periodic event or process that resets periodically, with values sharply increasing and then decaying over time. The series also contains some noise or small fluctuations, which is common in real-world data, but the overall trend of recurring peaks is clearly visible. Such a pattern is ideal for time series forecasting models, as it offers predictable structure interspersed with variability.

![image](https://github.com/user-attachments/assets/744bc4bf-ce1f-4492-b9d1-fa851d4224b0)

The graph shows the training loss over 50 epochs, illustrating how the model's performance improves during training. At the beginning, the loss is very high, indicating poor model predictions. However, it rapidly decreases within the first few epochs, showing significant learning progress. After around 10 epochs, the curve starts to flatten, indicating that the model has reached a point of diminishing returns where further training yields only marginal improvements. The small fluctuations in the later epochs suggest some noise but overall stability, implying that the model has likely converged and is no longer significantly overfitting or underfitting.

![image](https://github.com/user-attachments/assets/0bdf76b3-6a3a-4bd3-8372-0b430598432d)

The graph shows a comparison between the actual time series values (in blue) and the model’s predictions (in orange) for the validation set, starting from time step 1100 onward. The actual series exhibits a noisy downward trend with occasional sharp peaks, while the predicted values closely follow the overall pattern with smoother fluctuations. The close alignment between the two lines indicates that the model has successfully learned the temporal dynamics of the series and can generalize well on unseen data, despite some minor deviations where the actual values spike abruptly.

---

## 🛠️ Tools and Libraries Used

| Tool / Library          | Description                                         | Purpose in Project                                    |
|------------------------|-----------------------------------------------------|-----------------------------------------------------|
| **Python**             | High-level programming language                     | Core language for data generation, modeling, and evaluation |
| **NumPy**              | Numerical computing library                          | Handling arrays, numerical operations, and synthetic data creation |
| **Matplotlib**         | Plotting library                                    | Visualizing time series, training loss, and forecasts |
| **TensorFlow**         | Open-source deep learning framework                 | Building, training, and evaluating the BiLSTM model |
| **tf.data API**        | TensorFlow data pipeline utilities                   | Creating efficient, batched, and shuffled datasets for training |
| **Pickle**             | Python object serialization library                  | Saving model evaluation metrics for later use       |
| **Jupyter Notebook**   | Interactive coding environment                        | Developing, testing, and documenting code interactively |
| **Google Colab (optional)** | Cloud-based Jupyter notebook environment          | Running the project with free GPU access             |
