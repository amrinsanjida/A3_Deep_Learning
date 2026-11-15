# Stock Price Forecasting with a Stacked LSTM

A four-layer LSTM network in TensorFlow/Keras that predicts the daily opening
price of Google (GOOG) stock from the previous 60 trading days.

## Approach

1. **Data:** daily Google stock prices. 1,258 trading days (2012–2016) for
   training, and the 20 trading days of January 2017 for testing.
2. **Features:** the `Open` price, scaled to [0, 1] with `MinMaxScaler`, fitted on training data only.
3. **Sequences:** sliding windows of 60 days, each predicting the next day (1,198 training samples).
4. **Model:** four stacked LSTM layers (60 → 60 → 80 → 120 units), each followed
   by 20% dropout, then a dense output layer. 185,641 parameters in total.
5. **Training:** Adam optimiser, mean squared error loss, 250 epochs, batch size 32.

## Results on the January 2017 test window

| Metric | Value |
|---|--:|
| RMSE | 34.21 USD |
| MAE | 32.77 USD |
| MSE | 1,170.64 |

With prices around 780–830 USD, the mean absolute error is about 4% of the
price. The predicted curve follows the overall direction of the real price but
lags behind sudden moves, which is typical for a model trained on a single
price feature.

## What I learned

- Framing a time series as supervised learning with sliding windows
- Avoiding data leakage by fitting the scaler on training data only and building test windows from the 60 days before the test period
- The limits of price-only forecasting, and why a naive "tomorrow = today" baseline matters

## Possible next steps

- Compare against naive persistence and ARIMA baselines
- Add volume and technical indicators as extra features
- Use walk-forward validation and early stopping instead of a fixed 250 epochs

## Run it

The notebook was written for Google Colab and asks you to upload
`Google_Stock_Price_Train.csv` and `Google_Stock_Price_Test.csv`. To run it
locally, put both files next to the notebook and skip the upload cells.

```bash
pip install -r requirements.txt
jupyter notebook A3_Deep_Learning.ipynb
```

*Coursework project, Master of Artificial Intelligence and Machine Learning, University of Adelaide (2024).*
