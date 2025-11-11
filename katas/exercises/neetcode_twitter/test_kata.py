"""Tests for Design Twitter kata."""

try:
    from user_kata import Twitter
except ImportError:
    from .reference import Twitter


def test_twitter_basic():

    twitter = Twitter()
    twitter.post_tweet(1, 5)
    assert twitter.get_news_feed(1) == [5]
    twitter.follow(1, 2)
    twitter.post_tweet(2, 6)
    assert twitter.get_news_feed(1) == [6, 5]
    twitter.unfollow(1, 2)
    assert twitter.get_news_feed(1) == [5]

def test_twitter_multiple_users():

    twitter = Twitter()
    twitter.post_tweet(1, 1)
    twitter.post_tweet(2, 2)
    twitter.post_tweet(3, 3)
    twitter.follow(1, 2)
    twitter.follow(1, 3)
    feed = twitter.get_news_feed(1)
    assert set(feed) == {1, 2, 3}

def test_twitter_ten_tweets():

    twitter = Twitter()
    for i in range(15):
        twitter.post_tweet(1, i)
    feed = twitter.get_news_feed(1)
    assert len(feed) == 10
    assert feed[0] == 14  # Most recent
