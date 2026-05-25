import pandas as pd


POSITIVE_WORDS = [
    "strong", "growth", "expands", "positive",
    "beat", "higher", "demand", "margin"
]

NEGATIVE_WORDS = [
    "weak", "pressure", "cautious", "decline",
    "lower", "risk", "inflation", "faces"
]


def load_sample_news():
    news_data = pd.DataFrame({
        "Ticker": ["SHW", "HD", "SHW", "HD"],
        "Title": [
            "Sherwin-Williams reports strong demand in housing renovation",
            "Home Depot faces pressure from weaker consumer spending",
            "Sherwin-Williams expands margins despite inflation",
            "Home Depot announces cautious outlook"
        ]
    })

    return news_data


def simple_sentiment(text):
    text = text.lower()
    score = 0

    for word in POSITIVE_WORDS:
        if word in text:
            score += 1

    for word in NEGATIVE_WORDS:
        if word in text:
            score -= 1

    return score


def compute_sentiment(news_data=None):
    if news_data is None:
        news_data = load_sample_news()

    news_data = news_data.copy()

    news_data["Sentiment_Score"] = news_data["Title"].apply(
        simple_sentiment
    )

    sentiment_by_stock = (
        news_data
        .groupby("Ticker")["Sentiment_Score"]
        .mean()
        .reset_index()
    )

    return news_data, sentiment_by_stock