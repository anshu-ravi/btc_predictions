from newspaper import Article
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
import numpy as np

def get_article_text(url):
    article = Article(url)
    try:
        article.download()
        article.parse()
        text = article.text
    except:
        # print(f"Error: Could not download article")
        text = None
    return text

class Summarizer:
    def __init__(self):
        self.model_name = "google/pegasus-xsum"
        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def summarize_article(self, text_to_summarize):

        if pd.isnull(text_to_summarize) or text_to_summarize == '':
            return np.nan

        # Encode the text into tensor
        inputs = self.tokenizer.encode("summarize: " + text_to_summarize, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        # Generate the summary
        summary_ids = self.model.generate(inputs, max_length=250, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

class NewsSentiment:
    def __init__(self):
        self.model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.pipeline = pipeline('sentiment-analysis', model=self.model, tokenizer=self.tokenizer, device=self.device)

    def get_sentiment(self, text):
        if isinstance(text, pd.Series):
            text = text.tolist()
        result = self.pipeline(text)
        label = [r['label'] for r in result]
        score = [r['score'] for r in result]
        return label, score
        