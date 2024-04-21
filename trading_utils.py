import pandas as pd


def get_metrics(returns):
    assert(len(returns) > 0)
    returns = pd.Series(returns)

    win_rate = returns[returns > 0].shape[0] / returns.shape[0]
    loss_rate = returns[returns < 0].shape[0] / returns.shape[0]
    max_win = returns.max()
    max_loss = returns.min()
    avg_gain = returns[returns > 0].mean()
    avg_loss = returns[returns < 0].mean()
    expectancy = abs((win_rate * avg_gain) / (loss_rate * avg_loss))
    


    metrics_names = ['Winning Rate', 'Loosing Rate', 'Max Profit', 'Max Loss', 'Avg Profit', 'Avg Loss', 'Expectancy']
    metrics = [win_rate, loss_rate, max_win, max_loss, avg_gain, avg_loss, expectancy]
    metrics = pd.Series({k:v for k,v in zip(metrics_names, metrics)})
    return metrics

def backtest_strategy(df):

    position = False
    returns = []
    for _, row in df.iterrows():
        if not position and row['signal'] == 1:
            position = True
            entry_price = row['Close']
        elif position and row['signal'] == -1:
            position = False
            exit_price = row['Close']
            returns.append((exit_price - entry_price) / entry_price)
    if position:
        returns.append((row['Close'] - entry_price) / entry_price)
    return returns


def TimeSeriesSplitWalkForwardDT(df, tf='month', n_train=4, n_test=1, rolling=False):
    df_freq_start = {'year': 'YS', 'month': 'MS', 'week': 'W-MON', 'day': 'D', 'hour': 'h', 'minute': 'min'}
    df_freq_end = {'year': 'YE', 'month': 'ME', 'week': 'W-SUN', 'day': 'D', 'hour': 'h', 'minute': 'min'}

    dt_range_start = pd.date_range(start=df.index[0], end=df.index[-1], freq=df_freq_start[tf], normalize=True)
    dt_range_end = pd.date_range(start=df.index[0], end=df.index[-1], freq=df_freq_end[tf], normalize=True)

    dt_range_start = dt_range_start[(dt_range_start >= df.index[0]) & (dt_range_start <= df.index[-1])]
    dt_range_end = dt_range_end[(dt_range_end >= df.index[0]) & (dt_range_end <= df.index[-1])]

    dt_range_start = dt_range_start[:-1]
    if tf in ['year', 'month', 'week']:
        dt_range_end += pd.Timedelta(days = 1, minutes = -1)
        if df.index[0] != dt_range_start[0]:
            dt_range_end = dt_range_end[1:]
    elif tf in ['day', 'hour', 'minute']:
        dt_range_end += pd.Timedelta(minutes = -1)
        dt_range_end = dt_range_end[1:]
    
    dt_range = list(zip(dt_range_start, dt_range_end))
    
    for idx in range(n_train-1, len(dt_range)-n_test, n_test):
        train_start = df.index[0] if rolling else dt_range[idx - (n_train - 1)][0]
        train_end = dt_range[idx][1]
        test_start = dt_range[idx + 1][0]
        test_end = dt_range[idx + n_test][1]

        train_idx = df.index[(df.index >= train_start) & (df.index <= train_end)]
        test_idx = df.index[(df.index >= test_start) & (df.index <= test_end)]
        
        yield(train_idx, test_idx)