import datetime
from binance import Client 
import requests
import csv
import pandas as pd
import os
from pygooglenews import GoogleNews

class CryptoData:
    def __init__(self, asset, interval, start_dt, last_dt):
        self.asset = asset
        self.interval = interval
        self.start_dt = start_dt
        self.last_dt = last_dt
        self.data_interval = False
        self.client = Client()
        self.dir = 'data/prices'
        self.path = os.path.join(self.dir, self.interval)
        
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.path, exist_ok=True)

    
    def download_data(self, symbol, start_date, end_date):
        if isinstance(start_date, datetime.datetime) or isinstance(end_date, datetime.date):
            start_date = str(start_date)
            end_date = str(end_date)
        
        data = self.client.get_historical_klines(symbol=symbol, interval=self.interval, start_str=start_date, end_str=end_date)
        data = pd.DataFrame(data)
        data = data.iloc[:-1, :6]
        data.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        data = data.set_index('Time')
        data.index = pd.to_datetime(data.index, unit='ms').strftime('%d-%m-%Y')
        data = data.astype(float)
        return data
    
    def get_crypto_data(self):
        while (self.start_dt < self.last_dt):
            print(self.start_dt)
            symbol = self.asset + 'USDT'
            end_dt = self.start_dt + datetime.timedelta(days=1)
            data = self.download_data(symbol=symbol, start_date=self.start_dt, end_date=end_dt)
            try:
                self.data_interval = pd.concat([self.data_interval, data], axis=0)
            except:
                self.data_interval = data
            self.data_interval.to_csv(f'{self.path}/{symbol}.csv')
            self.start_dt = end_dt
        print(f'Price Data for {self.asset} downloaded successfully for period {self.start_dt} to {self.last_dt}!')


class CryptoCFGIData:
    def __init__(self, limit=0, format="csv", date_format="world"):
        self.url = f"https://api.alternative.me/fng/?limit={limit}&format={format}&date_format={date_format}"
        self.dir = 'data/cfgi'
        self.filename = 'cfgi.csv'
        os.makedirs(self.dir, exist_ok=True)

    def save_data(self, data):
        filename = os.path.join(self.dir, self.filename)
        data.to_csv(filename, index=False)
    
    def download_data(self):
        response = requests.get(self.url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data from {self.url}")
        data_text = response.text
        reader = csv.reader(data_text.splitlines())
        data = list(reader)
        data = pd.DataFrame(data[4:-5], columns=['Date', 'Value', 'Value_Classification'])
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
        data.sort_values('Date', inplace=True)
        self.save_data(data)
        print(f"CFGI Data downloaded successfully for period {data['Date'].min()} to {data['Date'].max()}!")


class BTCNews:
    def __init__(self, keywords=None, start_date=None, last_date=None):
        self.keywords = keywords if keywords else ["bitcoin", "btc", "cryptocurrency"]
        self.query = ' OR '.join(self.keywords)
        self.start_date = start_date if start_date else "2018-02-01"
        self.last_date = last_date if last_date else "2024-02-12"
        self.gn = GoogleNews(lang = 'en', country = 'US')
        self.dir = 'data/news'
        self.news_data = []

    def extract_news_info(self, news):
        data = []
        for i in range(len(news)):
            title = news[i]['title']
            url = news[i]['link']
            date = news[i]['published_parsed']
            source = news[i]['source']['title']
            data.append({'Title': title, 'URL': url, 'Date': date, 'Source': source})
        
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: datetime.datetime(*x[:6])))
        df = df.sort_values(by='Date', ascending=True)
        df = df.reset_index(drop=True)
        df = df.drop_duplicates(subset='Title')
        
        print(f'Period: {df.Date.min()} to {df.Date.max()} has {len(df)} news articles')
        return df

    def get_news(self):
        while self.start_date <= self.last_date:
            start_date_obj = datetime.datetime.strptime(self.start_date, '%Y-%m-%d') + datetime.timedelta(days=1)
            end_date_obj = start_date_obj + datetime.timedelta(days=5)

            self.start_date = start_date_obj.strftime('%Y-%m-%d')
            end_date = end_date_obj.strftime('%Y-%m-%d')

            news_articles = self.gn.search(self.query, helper=True,
                                           from_=self.start_date, 
                                           to_=end_date, 
                                           proxies=None,
                                           scraping_bee=None)

            if len(news_articles['entries']) > 1:
                data = self.extract_news_info(news_articles['entries'])
                try:
                    self.news_data = pd.concat([self.news_data, data], axis=0)
                except:
                    self.news_data = data
                os.makedirs(self.dir, exist_ok=True)
                self.news_data.to_csv(f'{self.dir}/btc_news.csv', index=False)
            else:
                print(f"No news found for {self.start_date}")

            self.start_date = end_date