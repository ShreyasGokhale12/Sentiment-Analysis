from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import time
import sentiment_mod as s

ckey="bwUA9QzIsU6PEszgF81zVFABm"
csecret="H2lQ3G6ot3fYpqmwSRrInBwjCW3aWzbu34F5CVzQ5wmrMXdI23"
atoken="982109105174147073-cUWfMQ70HABuqW21DlMjm0pEBnZ035y"
asecret="IMBMP0w7ImyLmLXppf0WKDwYXRdUqIoPktGMNvfRzfDiH"

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        tweet = ascii(all_data["text"])

        sentiment_value , confidence = s.sentiment(tweet)
        print(tweet , sentiment_value , confidence)
        if confidence*100 >=80 :
            output = open('twitter-out.txt','a')
            output.write(sentiment_value)
            output.close()
            
        return True

    def on_error(self, status):
        print (status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["car"])

