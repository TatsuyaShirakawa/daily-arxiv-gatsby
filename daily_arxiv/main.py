import os
import sys
import time
from pathlib import Path
import datetime
from tqdm import tqdm
import subprocess
import dotenv
from daily_arxiv.crawl_arxiv import crawl_arxiv
from daily_arxiv.search_tweets import search_tweets
from daily_arxiv.create_markdown import write_blog


dotenv.load_dotenv()


TARGETS = ['cs', 'stat.ML']
SINCE = 1
UNTIL = 1
SLEEP = 0.5
BLOG_DIR = 'content/posts'
PAPER_SCORE_THRESHOLD = 50
TWEET_SCORE_THRESHOLD = 25
GITHUB_USER = 'TatsuyaShirakawa'
# sanity check for environmental variables
GITHUB_TOKEN = os.environ['GITHUB_TOKEN']
TWITTER_CONSUMER_KEY = os.environ['TWITTER_CONSUMER_KEY'],
TWITTER_CONSUMER_SECRET = os.environ['TWITTER_CONSUMER_SECRET'],
TWITTER_ACCESS_TOKEN = os.environ['TWITTER_ACCESS_TOKEN'],
TWITTER_ACCESS_SECRET = os.environ['TWITTER_ACCESS_SECRET']


def get_blog_filename(papers):
    metadata = papers['meta']
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
        date_str = start_date + '-' + end_date
    return  Path(BLOG_DIR) / f'{date_str}---Hot-Papers.md'


def commit_and_push(blog_filename):
    print('git add')
    ret = subprocess.run(['git', 'add', blog_filename],
                         stdin=sys.stdin, stdout=sys.stdout)
    print('git commit')
    ret = subprocess.run(['git', 'commit', '-m', '"new post"'],
                         stdin=sys.stdin, stdout=sys.stdout)

    print('git push')
    url = f'https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/TatsuyaShirakawa/daily-arxiv-gatsby.git'
    ret = subprocess.run(['git', 'push', url],
                         stdin=sys.stdin, stdout=sys.stdout)
    
    

def main():
    
    papers = crawl_arxiv(targets=TARGETS,
                         since=SINCE,
                         until=UNTIL)

    if len(papers['papers']) == 0:
        print('no papers found')
        return 

    for paper in tqdm(papers['papers']):
        paper['tweets'] = search_tweets(paper['id'])
        time.sleep(SLEEP)

    blog_filename = get_blog_filename(papers)
    write_blog(papers,
               blog_filename,
               paper_score_threshold=PAPER_SCORE_THRESHOLD,
               tweet_score_threshold=TWEET_SCORE_THRESHOLD)

    commit_and_push(blog_filename)


if __name__ == '__main__':

    main()
