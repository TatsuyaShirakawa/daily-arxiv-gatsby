import re
import os
import datetime
from pathlib import Path
from tqdm import tqdm
import twitter

try:
    import dotenv
    dotenv.load_dotenv()
except Exception:
    print('skpped loading environment variables from .env')


def tweet_score(tweet):
    retweet_count = int(tweet['retweets_count'])
    favorite_count = int(tweet['likes_count'])
    return retweet_count + favorite_count


def total_retweet_count(paper):
    return sum([int(tweet['retweets_count']) for tweet in paper['tweets']])


def total_favorite_count(paper):
    return sum([int(tweet['likes_count']) for tweet in paper['tweets']])


def paper_score(paper):
    all_tweets = paper['tweets']
    return sum([tweet_score(tweet) for tweet in all_tweets])


class HotPaperBlogWriter:

    def __init__(self,
                 favorite_tags=['cs.CV',
                                'cs.CL',
                                'cs.LG',
                                'cs.DS',
                                'cs.IR',
                                'cs.NE',
                                'stat.ML',
                                'cs.AR'],
                 unfavorite_tags=['cs.CR',
                                  'cs.IT',
                                  'cs.LO',
                                  'cs.NI',
                                  'cs.PL',
                                  'cs.RO',
                                  'cs.SE'
                                  ],
                 paper_score_threshold=50,
                 tweet_score_threshold=25
                 ):
        assert(len(set(favorite_tags) & set(unfavorite_tags)) == 0)
        self.favorite_tags = favorite_tags[:]
        self.unfavorite_tags = unfavorite_tags[:]
        self.paper_score_threshold = paper_score_threshold
        self.tweet_score_threshold = tweet_score_threshold

        api = twitter.Api(consumer_key=os.environ['TWITTER_CONSUMER_KEY'],
                          consumer_secret=os.environ['TWITTER_CONSUMER_SECRET'],
                          access_token_key=os.environ['TWITTER_ACCESS_TOKEN'],
                          access_token_secret=os.environ['TWITTER_ACCESS_SECRET'])
        self.t = api

    def get_tweet_string(self, tweet):
        return self.t.GetStatusOembed(url=tweet['link'])['html']

    def save_markdown(self, data, result_file, min_tweet_topk=1, max_tweet_topk=10):

        metadata = data['meta']
        papers = data['papers']

        r_subject = re.compile(r'.+ \((?P<subject>.+)\)')

        def write_paper(no, paper, fout):
            title = paper['title']
            authors = ', '.join(paper['authors'])
            links = paper['links']
            link_pdf = None if not "Download PDF" in links else links['Download PDF']
            subjects = [r_subject.search(_).group('subject')
                        for _ in paper['subjects']]
            abstract = paper["summary"].replace('\n', ' ')
            total_retweet = total_retweet_count(paper)
            total_favorite = total_favorite_count(paper)

            print(file=fout)
            print(f'# {no}. {title}', file=fout)
            print(file=fout)            
            print(authors, file=fout)
            print(file=fout)
            now = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
            print(
                f'- retweets: {total_retweet}, favorites: {total_favorite} ({now})',
                file=fout
            )
            print(file=fout)                        
            print(f'- links: [abs]({links["Abstract"]})', file=fout, end='')
            if link_pdf:            
                print(f' | [pdf]({links["Download PDF"]})', file=fout, end='')
            print(file=fout)            
            print(
                '- ' + ' | '.join([f'[{subject}](https://arxiv.org/list/{subject}/recent)'
                                   for subject in subjects]),
                file=fout)
            print(file=fout)
            print(f'{abstract}', file=fout)
            print(file=fout)
            for i, tweet in enumerate(list(
                    reversed(sorted(paper['tweets'],
                                    key=tweet_score)))[:max_tweet_topk]):
                retweet_count = int(tweet['retweets_count'])
                favorite_count = int(tweet['likes_count'])

                if i + 1 >= min_tweet_topk and retweet_count + favorite_count < self.tweet_score_threshold:
                    continue

                tweet_str = self.get_tweet_string(tweet)
                print(tweet_str, file=fout)

            print(file=fout)
            print(file=fout)


        tags = []
        for paper in papers:
            tags.append([r_subject.search(_).group('subject')
                         for _ in paper['subjects']])
        
        favorites = [p for p, t in zip(papers, tags)
                     if paper_score(p) >= self.paper_score_threshold
                     # and any([_ in self.favorite_tags for _ in t])
                     and 'Download PDF' in p['links']]

        favorites = list(reversed(sorted(favorites, key=paper_score)))

        with open(result_file, 'w') as fout:
            start_date = metadata['since']
            start_date = datetime.datetime.strftime(
                datetime.datetime.strptime(start_date, '%Y/%m/%d'),
                '%Y-%m-%d'
            )
            end_date = metadata['until']
            end_date = datetime.datetime.strftime(
                datetime.datetime.strptime(end_date, '%Y/%m/%d') - datetime.timedelta(days=1),
                '%Y-%m-%d'
                )
            if start_date == end_date:
                date_str = start_date
            else:
                date_str = start_date + ' - ' + end_date
            now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%ZZ')
            print(f'''---
title: Hot Papers {date_str}
date: {now}
template: "post"
draft: false
slug: "hot-papers-{date_str}"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers {date_str}"
socialImage: "/media/flying-marine.jpg"

---''', file=fout)

            for i, paper in enumerate(tqdm(favorites)):
                write_paper(i + 1, paper, fout)
                

def write_blog(papers, filename, paper_score_threshold=50, tweet_score_threshold=25):
    writer = HotPaperBlogWriter(paper_score_threshold=paper_score_threshold,
                                tweet_score_threshold=tweet_score_threshold)
    writer.save_markdown(papers, filename)
