"""Design Twitter - LeetCode 355 - Reference Solution"""
import heapq
from collections import defaultdict

class Twitter:
    def __init__(self):
        self.tweets = defaultdict(list)  # userId -> [(timestamp, tweetId)]
        self.following = defaultdict(set)  # userId -> set of followed userIds
        self.timestamp = 0

    def post_tweet(self, user_id: int, tweet_id: int) -> None:
        self.tweets[user_id].append((self.timestamp, tweet_id))
        self.timestamp += 1

    def get_news_feed(self, user_id: int) -> list[int]:
        # Merge tweets from user and followed users using max heap
        max_heap = []

        # Add user's own tweets
        if user_id in self.tweets:
            for timestamp, tweet_id in self.tweets[user_id]:
                heapq.heappush(max_heap, (-timestamp, tweet_id))

        # Add tweets from followed users
        for followee_id in self.following[user_id]:
            if followee_id in self.tweets:
                for timestamp, tweet_id in self.tweets[followee_id]:
                    heapq.heappush(max_heap, (-timestamp, tweet_id))

        # Get 10 most recent
        result = []
        for _ in range(min(10, len(max_heap))):
            result.append(heapq.heappop(max_heap)[1])

        return result

    def follow(self, follower_id: int, followee_id: int) -> None:
        if follower_id != followee_id:
            self.following[follower_id].add(followee_id)

    def unfollow(self, follower_id: int, followee_id: int) -> None:
        self.following[follower_id].discard(followee_id)
