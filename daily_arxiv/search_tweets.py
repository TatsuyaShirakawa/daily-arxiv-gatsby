import twint


def search_tweets(arxiv_id, limit=100):
    tweets = []
    config = twint.Config()
    config.Search = f'url:arxiv url:{arxiv_id}'
    config.limit = limit
    config.Store_object = True
    config.Hide_output = True
    config.Store_object_tweets_list = tweets
    twint.run.Search(config)
    result = [vars(tweet) for tweet in tweets]
    result = [t for t in result if 'arxiv' in t['tweet'] and str(arxiv_id) in t['tweet']]
    return result
