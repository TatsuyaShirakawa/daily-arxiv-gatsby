import os
import time
import dotenv
dotenv.load_dotenv()
import twitter

TWITTER_CONSUMER_KEY = os.environ['TWITTER_CONSUMER_KEY'],
TWITTER_CONSUMER_SECRET = os.environ['TWITTER_CONSUMER_SECRET'],
TWITTER_ACCESS_TOKEN = os.environ['TWITTER_ACCESS_TOKEN'],
TWITTER_ACCESS_SECRET = os.environ['TWITTER_ACCESS_SECRET']


API = twitter.Api(consumer_key=os.environ['TWITTER_CONSUMER_KEY'],
                  consumer_secret=os.environ['TWITTER_CONSUMER_SECRET'],
                  access_token_key=os.environ['TWITTER_ACCESS_TOKEN'],
                  access_token_secret=os.environ['TWITTER_ACCESS_SECRET'])


def search_tweets(arxiv_id, limit=100):
    count = 0
    while True:
        try:
            query = f'q=arxiv%20url%3A{arxiv_id}&result_type=recent&count={limit}'
            results = API.GetSearch(raw_query=query)
        except Exception:
            print(f'sleep {2 ** count} sec.')
            time.sleep(2 ** count)
            count += 1
            continue
        break
    return results
