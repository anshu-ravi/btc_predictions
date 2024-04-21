# Bitcoin Price Forecasting Using Machine Learning & Sentiment 


## About This Project
This repository contains the code and documentation for my BBA thesis at IE University. The study conducts an examination of various machine learning algorithms to predict Bitcoin prices. It utilizes advanced features like news sentiment analysis and the Crypto Greed and Fear Index (CGFI) to enhance prediction accuracy.

## Key Findings

- Ridge Regression emerged as the most effective model, outperforming SVM, XGBoost, LightGBM, and LSTM across the main performance metrics: RMSE (Root Mean Square Error), MAE (Mean Absolute Error), and MAPE (Mean Absolute Percentage Error).
- Incorporating news sentiment and CGFI as features significantly improved the model's precision in forecasting future Bitcoin values.

## Features
- News Sentiment Analysis: Extracted using Newspaper3k and summarized by Pegasus Transformer, with sentiment derived via DistilRoBERTa fine-tuned on financial news.
- Crypto Greed and Fear Index (CGFI): Integrated as a predictive feature to reflect the current sentiment in the cryptocurrency market. Obtained from Alternative.me

## Repository Structure
- modelling.ipynb: Notebook containing the modeling process, feature engineering, and algorithm comparisons.
- back_testing.ipynb: Notebook for backtesting the models to evaluate their performance over different time frames. A simple trading strategy is also implemented here.


## How to Use 
Clone the Repository
```
git clone https://github.com/yourusername/bitcoin-price-forecast.git
cd bitcoin-price-forecast
```

## Author 
- Anshumaan Ravi 