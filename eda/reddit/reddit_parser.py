import praw
import pandas as pd
import signal
import sys


def save_comments():
    out = 'reddit_ru.csv'
    df = pd.DataFrame(comments, columns=['comment'])
    df.to_csv(out, index=False)
    print(f'saved to {out}')


def signal_handler(sig, frame):
    save_comments()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


reddit = praw.Reddit(client_id ='',
                     client_secret ='',
                     user_agent ='Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36')

# subreddit = reddit.subreddit('politota')
subreddit = reddit.subreddit('ru_Anime')
comments = []

try:
    for submission in subreddit.hot(limit=None):
        submission.comments.replace_more(limit=None)
        for comment in submission.comments.list():
            comments.append(comment.body)

        print(len(comments))

    print('OK')
    save_comments()
except Exception as e:
    print('error ocurred')
    save_comments()
