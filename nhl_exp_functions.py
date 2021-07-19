#----------------------------------------------
# Load dependencies
#----------------------------------------------
import pandas as pd
import base64
import tweepy as tw
import re
import numpy as np
import string
import unicodedata
import nltk
import gensim
import pyLDAvis.gensim_models
import pickle 
import pyLDAvis
import os
import matplotlib.pyplot as plt
import gensim.corpora as corpora
import streamlit as st
from pprint import pprint
from nltk.util import bigrams
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from streamlit_metrics import metric, metric_row
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('stopwords')


#----------------------------------------------
# DEFINE VARIABLES
#----------------------------------------------

# English stopwords
stopwords_en = nltk.corpus.stopwords.words('english')

# French stopwords
stopwords_fr = nltk.corpus.stopwords.words('french')
    
# Read in teams & accounts CSVs
teams = pd.read_csv('assets/nhl_app_teams.csv')
accounts = pd.read_csv('assets/nhl_app_accounts.csv')
accounts['username'] = accounts['account_id'].str.replace("@", "")



#----------------------------------------------
# DEFINE FUNCTIONS
#----------------------------------------------
@st.cache(suppress_st_warning=True,allow_output_mutation=True)
# Function 1
#-----------------
def get_table_download_link(df):
    # Reference: https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
    """
    Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="tweets.csv">Download Raw Data CSV file</a>'
    return href

# Function 2: 
#----------------
# Hit twitter api & add basic features & output 2 dataframes
# @st.cache(suppress_st_warning=True,allow_output_mutation=True)
def twitter_get_nhl(num_of_tweets):  
    
    with st.spinner('Getting NHL Expansion Draft data from Twitter...'):

        # Set up Twitter API access
        # Define access keys and tokens
        consumer_key = st.secrets['consumer_key']
        consumer_secret = st.secrets['consumer_secret']
        access_token = st.secrets['access_token']
        access_token_secret = st.secrets['access_token_secret']

        # Tweepy auth handler
        auth = tw.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tw.API(auth, wait_on_rate_limit = True)

        # Define search terms
        # https://developer.twitter.com/en/docs/twitter-api/v1/rules-and-filtering/search-operators
        user_word = f'expansion draft OR expansiondraft'

        # Filter out retweets
        user_word = user_word + ' -filter:retweets'
        # The following is based on user language selection

        # English tweets only
        language = 'en'

        # Run search using defined params
        tweets = tw.Cursor(api.search,
                                q = user_word,
                                tweet_mode = "extended",
                                lang = language).items(num_of_tweets)

        # Store as dataframe
        tweet_metadata = [[tweet.created_at, tweet.id, tweet.full_text, tweet.user.screen_name, tweet.retweet_count, tweet.favorite_count, tweet.user.followers_count, tweet.user.verified] for tweet in tweets]    
        df_tweets = pd.DataFrame(data=tweet_metadata, columns=['created_at', 'id', 'full_text', 'user', 'rt_count', 'fav_count', 'follower_ct', 'verified'])

        # Add a new data variable
        df_tweets['created_dt'] = df_tweets['created_at'].dt.date

        # Add a new time variable
        df_tweets['created_time'] = df_tweets['created_at'].dt.time

        # Create a new text variable to do manipulations on 
        df_tweets['clean_text'] = df_tweets.full_text

        # Create a tidy dataframe to later display to users 
        df_new = df_tweets[["created_dt", "created_time", "full_text", "user", "rt_count", "fav_count", "follower_ct", "verified"]]
        df_new = df_new.rename(columns = {"created_dt": "Date", 
                                        "created_time": "Time", 
                                        "full_text": "Tweet", 
                                        "user": "Username", 
                                        "rt_count": "Retweets",  
                                        "fav_count": "Favourites",
                                        "follower_ct": "Followers",
                                        "verified": "Verified"})
    return df_tweets, df_new


# Function 3a
#----------------
# Function to create dataframe of most recent 400 tweets from a specific user
def get_user_tweets(screen_name):

    with st.spinner('Getting NHL Insider data from Twitter...'):

        #Twitter only allows access to a users most recent 3240 tweets with this method
        #Adapted by: https://gist.github.com/yanofsky/5436496?fbclid=IwAR12gb56FOxTNI6R3SfiwpnbPpTvKLoeGR3kP0peQ1nGilcwsF8bR0LSVqE
        
        # Set up Twitter API access
        # Define access keys and tokens
        consumer_key = st.secrets['consumer_key']
        consumer_secret = st.secrets['consumer_secret']
        access_token = st.secrets['access_token']
        access_token_secret = st.secrets['access_token_secret']

        auth = tw.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tw.API(auth, wait_on_rate_limit = True)

        #initialize a list to hold all the tweepy Tweets
        alltweets = []  
        
        #make initial request for most recent tweets (200 is the maximum allowed count)
        new_tweets = api.user_timeline(screen_name = screen_name,count=200)
        
        #save most recent tweets
        alltweets.extend(new_tweets)
        
        #save the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        
        #keep grabbing tweets until there are no tweets left to grab
        #loop through once (0 to 0) to get 100 tweets
        for i in range(0):
            
            #all subsiquent requests use the max_id param to prevent duplicates
            new_tweets = api.user_timeline(screen_name = screen_name,count=100,max_id=oldest)
            
            #save most recent tweets
            alltweets.extend(new_tweets)
            
            #update the id of the oldest tweet less one
            oldest = alltweets[-1].id - 1
            i +=1
        
        #transform the tweepy tweets into a 2D array that will populate the csv 
        outtweets = [[tweet.user.screen_name, tweet.id_str, tweet.created_at, tweet.text, tweet.retweet_count, tweet.favorite_count, tweet.user.followers_count, tweet.user.verified, tweet.retweeted, tweet.in_reply_to_status_id] for tweet in alltweets]

        #transform 2D array into pandas dataframe
        df_itweets = pd.DataFrame(data=outtweets, columns=['user', 'id', 'created_at', 'full_text', 'rt_count', 'fav_count', 'follower_ct', 'verified', 'is_rt', 'reply_id'])
        
        # Create date column
        df_itweets['created_dt'] = df_itweets['created_at'].dt.normalize()

        # Date filter max
        df_itweets['filter_dt'] = pd.to_datetime("now") - (pd.to_timedelta(48, unit='h'))

        # Filter based on results of filter_dt
        df_itweets = df_itweets[df_itweets.created_at > df_itweets.filter_dt]

        # Create a new text variable to do manipulations on 
        df_itweets['clean_text'] = df_itweets.full_text

    return df_itweets


# Function 3b
#----------------
# Return recent tweets for list of Twitter accounts specified in /assets/nhl_app_accounts.csv
def insider_recent_tweets():

    # Get list of Twitter accounts
    df_accounts = pd.read_csv('assets/nhl_app_accounts.csv')

    # Remove @ from username
    df_accounts['clean_account_id'] = df_accounts['account_id'].str.replace("@", "")

    # Create list of accounts
    list_accounts = df_accounts['clean_account_id'].tolist()

    # Create empty dataframe to append tweets in the for loop below
    df_user_tweets = pd.DataFrame()

    # Iterates through list of accounts, gets most recents tweets for each account, and appends to dataframe
    for i in range(len(list_accounts)):
        # Uses function 3 to get tweets for each account
        new_user_tweets = get_user_tweets(list_accounts[i])
        # Append tweets to dataframe
        df_user_tweets = df_user_tweets.append(new_user_tweets)

    # Reset dataframe index
    df_user_tweets = df_user_tweets.reset_index(drop=True)

    return pd.DataFrame(df_user_tweets)


# Function 4a: 
#----------------
# INSIDERS: takes in user selections and filters out the df accordingly
# this is an input to the topic model
# filter based on df_sentiment?

# df 1 = df_sentiment (has dups, has team classifications)
def filter_fan_rows(team_choice, df):
    
    # Select columns from text_sentiment
    df = df[['id', 'user', 'created_at',  'is_rt', 'reply_id', 'company', 'account_type', 'nhl_team_abbr', 'nhl_team', 'multiple_teams', 'expansion_type', 'full_text', 'clean_text', 'sentiment', 'positive_score', 'negative_score', 'neutral_score', 'compound_score']]

    # Create ind variable: if reply_id is null, False else True
    df['is_reply'] = np.where(pd.isnull(df.reply_id), 'False', 'True')

    # If team_choice "All ", dont filter else filter by selected teams + Kraken
    if 'All' not in team_choice:
        team_choice.append("SEA") # always include Kraken
        boolean_series = df.nhl_team_abbr.isin(team_choice) # list to filter by
        df = df[boolean_series] # filter df_sentiment by list
        #msg1 = "Filtered list of NHL teams"

    return df

# Function 4b: 
#----------------
# INSIDERS: takes in user selections and filters out the df accordingly
# this is an input to the topic model
# filter based on df_sentiment?

# df 1 = df_sentiment (has dups, has team classifications)
def filter_insider_rows(team_choice, rt_choice, reply_choice, account_choice, df):
    
    # Select columns from text_sentiment
    df = df[['id', 'user', 'created_at',  'is_rt', 'reply_id', 'company', 'account_type', 'nhl_team_abbr', 'nhl_team', 'multiple_teams', 'expansion_type', 'full_text', 'clean_text', 'sentiment', 'positive_score', 'negative_score', 'neutral_score', 'compound_score']]

    # Create ind variable: if reply_id is null, False else True
    df['is_reply'] = np.where(pd.isnull(df.reply_id), 'False', 'True')

    # If team_choice "All ", dont filter else filter by selected teams + Kraken
    if 'All' not in team_choice:
        team_choice.append("SEA") # always include Kraken
        boolean_series = df.nhl_team_abbr.isin(team_choice) # list to filter by
        df = df[boolean_series] # filter df_sentiment by list
        #msg1 = "Filtered list of NHL teams"

    # If company_choice choice "All", dont filter else filter by selected teams + Kraken
    if account_choice == 'Both':
        df = df
        msg2 = "All Hockey Insiders (Analytics & Reporters)"
    # If company_choice choice "All", dont filter else filter by selected teams + Kraken
    elif account_choice == 'Hockey Analytics':
        df = df.loc[df['account_type'] == 'analytics']# filter df_sentiment by list
        msg2 = "Only Hockey Analytics accounts"
    # If company_choice choice "All", dont filter else filter by selected teams + Kraken
    elif account_choice == 'Hockey Reporters':
        df = df.loc[df['account_type'] == 'insider']# filter df_sentiment by list
        msg2 = "Only Hockey Reporter accounts"
    
    # If rt choice is false, dont filter else filter by selected teams + Kraken
    if rt_choice == 'Yes':
        df = df
        msg3 = "Retweets included"
    # No selection
    elif rt_choice == 'No':
        df = df[df['full_text'].str.contains("RT") == False] # filter out rows that contain with RT
        msg3 = "Retweets not included"

    # If reply choice is false, only select rows where reply is false
    if reply_choice == 'Yes':
        df = df
        msg4 = "Replies included "
    # No selection
    elif reply_choice == 'No':
        df = df.loc[df['is_reply'] == 'False']# filter df_sentiment by list
        msg4 = "Retweets mpt included"

    return df, msg2,  msg3, msg4


# Function 5a: 
#----------------
# takes in pandas dataframe after first twitter scrape
# returns a pandas dataframe that has classified each tweet as relating to an nhl team

def classify_nhl_team(df):

    # Create a new and smaller dataframe to work with called df
    df = df[["id", "user", "created_at", 'rt_count', 'fav_count', 'follower_ct', 'verified', "full_text", "clean_text"]]
    
    # NHL Team Classification: If a team's keywords come up, classify as a team specific indicator, with value = team name
    df['ANA'] = pd.np.where(df['clean_text'].str.contains('ANA'), 'Anaheim Ducks', '0')
    df['ARZ'] = pd.np.where(df['clean_text'].str.contains('ARZ'), 'Arizona Coyotes', '0')
    df['BOS'] = pd.np.where(df['clean_text'].str.contains('BOS'), 'Boston Bruins', '0')
    df['BUF'] = pd.np.where(df['clean_text'].str.contains('BUF'), 'Buffalo Sabres', '0')
    df['CGY'] = pd.np.where(df['clean_text'].str.contains('CGY'), 'Calgary Flames', '0')
    df['CAR'] = pd.np.where(df['clean_text'].str.contains('CAR'), 'Carolina Hurricanes', '0')
    df['CHI'] = pd.np.where(df['clean_text'].str.contains('CHI'), 'Chicago Blackhawks', '0')
    df['COL'] = pd.np.where(df['clean_text'].str.contains('COL'), 'Colorado Avalanche', '0')
    df['CBJ'] = pd.np.where(df['clean_text'].str.contains('CBJ'), 'Columbus Blue Jackets', '0')
    df['DAL'] = pd.np.where(df['clean_text'].str.contains('DAL'), 'Dallas Stars', '0')
    df['DET'] = pd.np.where(df['clean_text'].str.contains('DET'), 'Detroit Red Wings', '0')
    df['EDM'] = pd.np.where(df['clean_text'].str.contains('EDM'), 'Edmonton Oilers', '0')
    df['FLA'] = pd.np.where(df['clean_text'].str.contains('FLA'), 'Florida Panthers', '0')
    df['LAK'] = pd.np.where(df['clean_text'].str.contains('LAK'), 'Los Angeles Kings', '0')
    df['MIN'] = pd.np.where(df['clean_text'].str.contains('MIN'), 'Minnesota Wild', '0')
    df['MTL'] = pd.np.where(df['clean_text'].str.contains('MTL'), 'Montreal Canadiens', '0')
    df['NSH'] = pd.np.where(df['clean_text'].str.contains('NSH'), 'Nashville Predators', '0')
    df['NJD'] = pd.np.where(df['clean_text'].str.contains('NJD|NJ'), 'New Jersey Devils', '0')
    df['NYI'] = pd.np.where(df['clean_text'].str.contains('NYI'), 'New York Islanders', '0')
    df['NYR'] = pd.np.where(df['clean_text'].str.contains('NYR'), 'New York Rangers', '0')
    df['OTT'] = pd.np.where(df['clean_text'].str.contains('OTT'), 'Ottawa Senators', '0')
    df['PHI'] = pd.np.where(df['clean_text'].str.contains('PHI'), 'Philadelphia Flyers', '0')
    df['PIT'] = pd.np.where(df['clean_text'].str.contains('PIT'), 'Pittsburgh Penguins', '0')
    df['SJS'] = pd.np.where(df['clean_text'].str.contains('SJ|SJS'), 'San Jose Sharks', '0')
    df['SEA'] = pd.np.where(df['clean_text'].str.contains('SEA'), 'Seattle Kraken', '0')
    df['STL'] = pd.np.where(df['clean_text'].str.contains('STL'), 'St Louis Blues', '0')
    df['TBL'] = pd.np.where(df['clean_text'].str.contains('TB|TBL'), 'Tampa Bay Lightning', '0')
    df['TOR'] = pd.np.where(df['clean_text'].str.contains('TOR'), 'Toronto Maple Leafs', '0')
    df['VAN'] = pd.np.where(df['clean_text'].str.contains('VAN'), 'Vancouver Canucks', '0') 
    df['VGK'] = pd.np.where(df['clean_text'].str.contains('VEG|VGK'), 'Vegas Golden Knights', '0')
    df['WSH'] = pd.np.where(df['clean_text'].str.contains('WSH|WASH'), 'Washington Capitals', '0')
    df['WPG'] = pd.np.where(df['clean_text'].str.contains('WPG'), 'Winnipeg Jets', '0')

    # Convert tweet to lower
    df.clean_text = df.clean_text.str.lower()  
    
    # NHL Team Classification: If a team's keywords come up, classify as a team specific indicator, with value = team name
    df['ANA'] = pd.np.where(df['clean_text'].str.contains('anaheim|ducks|flytogether'), 'Anaheim Ducks', df.ANA)
    df['ARZ'] = pd.np.where(df['clean_text'].str.contains('arizona|coyotes|yotes'), 'Arizona Coyotes', df.ARZ)
    df['BOS'] = pd.np.where(df['clean_text'].str.contains('boston|bruins|nhlbruins'), 'Boston Bruins', df.BOS)
    df['BUF'] = pd.np.where(df['clean_text'].str.contains('buffalo|sabres|letsgobuffalo'), 'Buffalo Sabres', df.BUF)
    df['CGY'] = pd.np.where(df['clean_text'].str.contains('calgary|flames|cofred'), 'Calgary Flames', df.CGY)
    df['CAR'] = pd.np.where(df['clean_text'].str.contains('carolina|hurricanes|canes|letsgocanes'), 'Carolina Hurricanes', df.CAR)
    df['CHI'] = pd.np.where(df['clean_text'].str.contains('chicago|blackhawks|blackhawks'), 'Chicago Blackhawks', df.CHI)
    df['COL'] = pd.np.where(df['clean_text'].str.contains('colorado|avalanche|GoAvsGo|avs'), 'Colorado Avalanche', df.COL)
    df['CBJ'] = pd.np.where(df['clean_text'].str.contains('columbus|bluejackets|jackets|cbj'), 'Columbus Blue Jackets', df.CBJ)
    df['DAL'] = pd.np.where(df['clean_text'].str.contains('dallas|stars|gostars'), 'Dallas Stars', df.DAL)
    df['DET'] = pd.np.where(df['clean_text'].str.contains('detroit|redwings|lgrw'), 'Detroit Red Wings', df.DET)
    df['EDM'] = pd.np.where(df['clean_text'].str.contains('edmonton|oilers|oil'), 'Edmonton Oilers', df.EDM)
    df['FLA'] = pd.np.where(df['clean_text'].str.contains('florida|panthers|flapanthers'), 'Florida Panthers', df.FLA)
    df['LAK'] = pd.np.where(df['clean_text'].str.contains('los angeles|kings|gokingsgo'), 'Los Angeles Kings', df.LAK)
    df['MIN'] = pd.np.where(df['clean_text'].str.contains('minnesota|wild|mnwild|minny|guerin'), 'Minnesota Wild', df.MIN)
    df['MTL'] = pd.np.where(df['clean_text'].str.contains('montreal|canadiens|habs|gohabsgo'), 'Montreal Canadiens', df.MTL)
    df['NSH'] = pd.np.where(df['clean_text'].str.contains('nashville|predators|preds'), 'Nashville Predators', df.NSH)
    df['NJD'] = pd.np.where(df['clean_text'].str.contains('new jersey|devils|njdevils'), 'New Jersey Devils', df.NJD)
    df['NYI'] = pd.np.where(df['clean_text'].str.contains('new york islanders|islanders|isles'), 'New York Islanders', df.NYI)
    df['NYR'] = pd.np.where(df['clean_text'].str.contains('new york rangers|rangers|nyr'), 'New York Rangers', df.NYR)
    df['OTT'] = pd.np.where(df['clean_text'].str.contains('ottawa|senators|sens|gosensgo'), 'Ottawa Senators', df.OTT)
    df['PHI'] = pd.np.where(df['clean_text'].str.contains('philadelphia|flyers|anytimeanywhere'), 'Philadelphia Flyers', df.PHI)
    df['PIT'] = pd.np.where(df['clean_text'].str.contains('pittsburgh|penguins|pens|letsgopens|pitts'), 'Pittsburgh Penguins', df.PIT)
    df['SJS'] = pd.np.where(df['clean_text'].str.contains('san jose|sharks|sjsharks'), 'San Jose Sharks', df.SJS)
    df['SEA'] = pd.np.where(df['clean_text'].str.contains('seattle|kraken|seakraken'), 'Seattle Kraken', df.SEA)
    df['STL'] = pd.np.where(df['clean_text'].str.contains('stlouis|st. louis|st louis|blues|stlblues'), 'St Louis Blues', df.STL)
    df['TBL'] = pd.np.where(df['clean_text'].str.contains('tampa bay|lightning|tampa|gobolts'), 'Tampa Bay Lightning', df.TBL)
    df['TOR'] = pd.np.where(df['clean_text'].str.contains('toronto|maple leafs|leafsforever|leafs|dubas'), 'Toronto Maple Leafs', df.TOR)
    df['VAN'] = pd.np.where(df['clean_text'].str.contains('vancouver|canucks|nucks|benning'), 'Vancouver Canucks', df.VAN) 
    df['VGK'] = pd.np.where(df['clean_text'].str.contains('vegas|golden knights|knights|#vegasborn'), 'Vegas Golden Knights', df.VGK)
    df['WSH'] = pd.np.where(df['clean_text'].str.contains('washington|capitals|caps|allcaps'), 'Washington Capitals', df.WSH)
    df['WPG'] = pd.np.where(df['clean_text'].str.contains('winnipeg|jets|gojetsgo'), 'Winnipeg Jets', df.WPG)

    # Define columns to concatenate
    cols = ['ANA', 'ARZ', 'BOS', 'BUF', 'CGY', 'CAR', 'CHI', 'COL', 'CBJ', 'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NSH', 'NJD', 'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'SJS', 'SEA', 'STL', 'TBL', 'TOR', 'VAN', 'VGK', 'WSH', 'WPG']
    # Concatenate columns
    df['teams_concat'] = df[cols].apply(lambda x: ','.join(x), axis=1)
    # Replace 0s with nothing
    df['teams_concat'] = df.teams_concat.str.replace('0,|0|,0','').str.strip()
    # ind variable - if multiple commas exist (proxy for num of teams), then 1 else 0
    df['multiple_teams'] = np.where(df.teams_concat.str.contains(","), 1, 0)
    # ind variable - if the length of teams_concat is equal to 0 (proxy for no teams matched), then 1 else 0
    df['no_matches'] = np.where(df.teams_concat.str.len() == 0, 1, 0)

    # Stash a dataframe with those tweets that were paired with a keyword
    df_match = df.loc[df['no_matches'] != 1]
    # Stash a dataframe with those tweets that were never paired to a keyword 
    df_nomatch = df.loc[df['no_matches'] == 1]
    # Select columns
    df_nomatch = df_nomatch[['id', 'user', 'created_at', 'full_text', 'clean_text', 'multiple_teams', 'no_matches', 'teams_concat', 'rt_count', 'fav_count', 'follower_ct', 'verified']]

    # Melt the dataframe such that each row is equal to a tweet that was matched to a team's keyword (introducing dups to tweets)
    melted_df = df.melt(
                    id_vars = ['id', 'user', 'created_at', 'full_text', 'clean_text', 'multiple_teams', 'no_matches', 'teams_concat', 'rt_count', 'fav_count', 'follower_ct', 'verified'],
                    value_vars = ['ANA', 'ARZ', 'BOS', 'BUF', 'CGY', 'CAR', 'CHI', 'COL', 'CBJ', 'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NSH', 'NJD', 'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'SJS', 'SEA', 'STL', 'TBL', 'TOR', 'VAN', 'VGK', 'WSH', 'WPG'],
                    var_name = 'nhl_team_abbr',
                    value_name = "nhl_team"
                    )
    # Filter out 0s 
    melted_df = melted_df.loc[melted_df['nhl_team'] != '0']

    # Add feature parity to df_nomatch
    df_nomatch['nhl_team_abbr'] = 'Unknown' 
    df_nomatch['nhl_team'] = 'Unknown' 

    # Append df_nomatch to melted_df to get df_clean
    df_clean = melted_df.append(df_nomatch)

    # Show a few rows of data
    df_clean.head(5)
    #print('total rows:', len(df_clean),'melted_rows:', (len(melted_df)), 'nomatch_rows:', len(df_nomatch))

    # Extend df_clean by joining in data about the team
    df_merged = pd.merge(df_clean, teams, on = 'nhl_team', how = 'left', indicator = True).astype("object")
    
    return df_merged, df, df_match, df_nomatch


# Function 5b: 
#----------------
# takes in pandas dataframe after first twitter scrape
# returns a pandas dataframe that has classified each tweet as relating to an nhl team

def classify_nhl_team_insider(df1):

    # Bring in pre-loaded data on accounts
    df_accounts = accounts

    # Extend df_clean by joining in data about the account
    df = df1.merge(df_accounts, left_on='user', right_on='username', how='left', indicator=True).astype('object')
    
    # Create a new and smaller dataframe to work with called df
    df = df[["id", "user", "created_at", 'rt_count', 'fav_count', 'follower_ct', 'verified', 'is_rt', 'reply_id', "full_text", "clean_text", "company", "account_type"]]
    
    # NHL Team Classification: If a team's keywords come up, classify as a team specific indicator, with value = team name
    df['ANA'] = pd.np.where(df['clean_text'].str.contains('ANA'), 'Anaheim Ducks', '0')
    df['ARZ'] = pd.np.where(df['clean_text'].str.contains('ARZ'), 'Arizona Coyotes', '0')
    df['BOS'] = pd.np.where(df['clean_text'].str.contains('BOS'), 'Boston Bruins', '0')
    df['BUF'] = pd.np.where(df['clean_text'].str.contains('BUF'), 'Buffalo Sabres', '0')
    df['CGY'] = pd.np.where(df['clean_text'].str.contains('CGY'), 'Calgary Flames', '0')
    df['CAR'] = pd.np.where(df['clean_text'].str.contains('CAR'), 'Carolina Hurricanes', '0')
    df['CHI'] = pd.np.where(df['clean_text'].str.contains('CHI'), 'Chicago Blackhawks', '0')
    df['COL'] = pd.np.where(df['clean_text'].str.contains('COL'), 'Colorado Avalanche', '0')
    df['CBJ'] = pd.np.where(df['clean_text'].str.contains('CBJ'), 'Columbus Blue Jackets', '0')
    df['DAL'] = pd.np.where(df['clean_text'].str.contains('DAL'), 'Dallas Stars', '0')
    df['DET'] = pd.np.where(df['clean_text'].str.contains('DET'), 'Detroit Red Wings', '0')
    df['EDM'] = pd.np.where(df['clean_text'].str.contains('EDM'), 'Edmonton Oilers', '0')
    df['FLA'] = pd.np.where(df['clean_text'].str.contains('FLA'), 'Florida Panthers', '0')
    df['LAK'] = pd.np.where(df['clean_text'].str.contains('LAK'), 'Los Angeles Kings', '0')
    df['MIN'] = pd.np.where(df['clean_text'].str.contains('MIN'), 'Minnesota Wild', '0')
    df['MTL'] = pd.np.where(df['clean_text'].str.contains('MTL'), 'Montreal Canadiens', '0')
    df['NSH'] = pd.np.where(df['clean_text'].str.contains('NSH'), 'Nashville Predators', '0')
    df['NJD'] = pd.np.where(df['clean_text'].str.contains('NJD|NJ'), 'New Jersey Devils', '0')
    df['NYI'] = pd.np.where(df['clean_text'].str.contains('NYI'), 'New York Islanders', '0')
    df['NYR'] = pd.np.where(df['clean_text'].str.contains('NYR'), 'New York Rangers', '0')
    df['OTT'] = pd.np.where(df['clean_text'].str.contains('OTT'), 'Ottawa Senators', '0')
    df['PHI'] = pd.np.where(df['clean_text'].str.contains('PHI'), 'Philadelphia Flyers', '0')
    df['PIT'] = pd.np.where(df['clean_text'].str.contains('PIT'), 'Pittsburgh Penguins', '0')
    df['SJS'] = pd.np.where(df['clean_text'].str.contains('SJ|SJS'), 'San Jose Sharks', '0')
    df['SEA'] = pd.np.where(df['clean_text'].str.contains('SEA'), 'Seattle Kraken', '0')
    df['STL'] = pd.np.where(df['clean_text'].str.contains('STL'), 'St Louis Blues', '0')
    df['TBL'] = pd.np.where(df['clean_text'].str.contains('TB|TBL'), 'Tampa Bay Lightning', '0')
    df['TOR'] = pd.np.where(df['clean_text'].str.contains('TOR'), 'Toronto Maple Leafs', '0')
    df['VAN'] = pd.np.where(df['clean_text'].str.contains('VAN'), 'Vancouver Canucks', '0') 
    df['VGK'] = pd.np.where(df['clean_text'].str.contains('VEG|VGK'), 'Vegas Golden Knights', '0')
    df['WSH'] = pd.np.where(df['clean_text'].str.contains('WSH|WASH'), 'Washington Capitals', '0')
    df['WPG'] = pd.np.where(df['clean_text'].str.contains('WPG'), 'Winnipeg Jets', '0')

    # Convert tweet to lower
    df.clean_text = df.clean_text.str.lower()  
    
    # NHL Team Classification: If a team's keywords come up, classify as a team specific indicator, with value = team name
    df['ANA'] = pd.np.where(df['clean_text'].str.contains('anaheim|ducks|flytogether'), 'Anaheim Ducks', df.ANA)
    df['ARZ'] = pd.np.where(df['clean_text'].str.contains('arizona|coyotes|yotes'), 'Arizona Coyotes', df.ARZ)
    df['BOS'] = pd.np.where(df['clean_text'].str.contains('boston|bruins|nhlbruins'), 'Boston Bruins', df.BOS)
    df['BUF'] = pd.np.where(df['clean_text'].str.contains('buffalo|sabres|letsgobuffalo'), 'Buffalo Sabres', df.BUF)
    df['CGY'] = pd.np.where(df['clean_text'].str.contains('calgary|flames|cofred'), 'Calgary Flames', df.CGY)
    df['CAR'] = pd.np.where(df['clean_text'].str.contains('carolina|hurricanes|canes|letsgocanes'), 'Carolina Hurricanes', df.CAR)
    df['CHI'] = pd.np.where(df['clean_text'].str.contains('chicago|blackhawks|blackhawks'), 'Chicago Blackhawks', df.CHI)
    df['COL'] = pd.np.where(df['clean_text'].str.contains('colorado|avalanche|GoAvsGo|avs'), 'Colorado Avalanche', df.COL)
    df['CBJ'] = pd.np.where(df['clean_text'].str.contains('columbus|bluejackets|jackets|cbj'), 'Columbus Blue Jackets', df.CBJ)
    df['DAL'] = pd.np.where(df['clean_text'].str.contains('dallas|stars|gostars'), 'Dallas Stars', df.DAL)
    df['DET'] = pd.np.where(df['clean_text'].str.contains('detroit|redwings|lgrw'), 'Detroit Red Wings', df.DET)
    df['EDM'] = pd.np.where(df['clean_text'].str.contains('edmonton|oilers|oil'), 'Edmonton Oilers', df.EDM)
    df['FLA'] = pd.np.where(df['clean_text'].str.contains('florida|panthers|flapanthers'), 'Florida Panthers', df.FLA)
    df['LAK'] = pd.np.where(df['clean_text'].str.contains('los angeles|kings|gokingsgo'), 'Los Angeles Kings', df.LAK)
    df['MIN'] = pd.np.where(df['clean_text'].str.contains('minnesota|wild|mnwild|minny|guerin'), 'Minnesota Wild', df.MIN)
    df['MTL'] = pd.np.where(df['clean_text'].str.contains('montreal|canadiens|habs|gohabsgo'), 'Montreal Canadiens', df.MTL)
    df['NSH'] = pd.np.where(df['clean_text'].str.contains('nashville|predators|preds'), 'Nashville Predators', df.NSH)
    df['NJD'] = pd.np.where(df['clean_text'].str.contains('new jersey|devils|njdevils'), 'New Jersey Devils', df.NJD)
    df['NYI'] = pd.np.where(df['clean_text'].str.contains('new york islanders|islanders|isles'), 'New York Islanders', df.NYI)
    df['NYR'] = pd.np.where(df['clean_text'].str.contains('new york rangers|rangers|nyr'), 'New York Rangers', df.NYR)
    df['OTT'] = pd.np.where(df['clean_text'].str.contains('ottawa|senators|sens|gosensgo'), 'Ottawa Senators', df.OTT)
    df['PHI'] = pd.np.where(df['clean_text'].str.contains('philadelphia|flyers|anytimeanywhere'), 'Philadelphia Flyers', df.PHI)
    df['PIT'] = pd.np.where(df['clean_text'].str.contains('pittsburgh|penguins|pens|letsgopens|pitts'), 'Pittsburgh Penguins', df.PIT)
    df['SJS'] = pd.np.where(df['clean_text'].str.contains('san jose|sharks|sjsharks'), 'San Jose Sharks', df.SJS)
    df['SEA'] = pd.np.where(df['clean_text'].str.contains('seattle|kraken|seakraken'), 'Seattle Kraken', df.SEA)
    df['STL'] = pd.np.where(df['clean_text'].str.contains('stlouis|st. louis|st louis|blues|stlblues'), 'St Louis Blues', df.STL)
    df['TBL'] = pd.np.where(df['clean_text'].str.contains('tampa bay|lightning|tampa|gobolts'), 'Tampa Bay Lightning', df.TBL)
    df['TOR'] = pd.np.where(df['clean_text'].str.contains('toronto|maple leafs|leafsforever|leafs|dubas'), 'Toronto Maple Leafs', df.TOR)
    df['VAN'] = pd.np.where(df['clean_text'].str.contains('vancouver|canucks|nucks|benning'), 'Vancouver Canucks', df.VAN) 
    df['VGK'] = pd.np.where(df['clean_text'].str.contains('vegas|golden knights|knights|#vegasborn'), 'Vegas Golden Knights', df.VGK)
    df['WSH'] = pd.np.where(df['clean_text'].str.contains('washington|capitals|caps|allcaps'), 'Washington Capitals', df.WSH)
    df['WPG'] = pd.np.where(df['clean_text'].str.contains('winnipeg|jets|gojetsgo'), 'Winnipeg Jets', df.WPG)

    # Define columns to concatenate
    cols = ['ANA', 'ARZ', 'BOS', 'BUF', 'CGY', 'CAR', 'CHI', 'COL', 'CBJ', 'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NSH', 'NJD', 'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'SJS', 'SEA', 'STL', 'TBL', 'TOR', 'VAN', 'VGK', 'WSH', 'WPG']
    # Concatenate columns
    df['teams_concat'] = df[cols].apply(lambda x: ','.join(x), axis=1)
    # Replace 0s with nothing
    df['teams_concat'] = df.teams_concat.str.replace('0,|0|,0','').str.strip()
    # ind variable - if multiple commas exist (proxy for num of teams), then 1 else 0
    df['multiple_teams'] = np.where(df.teams_concat.str.contains(","), 1, 0)
    # ind variable - if the length of teams_concat is equal to 0 (proxy for no teams matched), then 1 else 0
    df['no_matches'] = np.where(df.teams_concat.str.len() == 0, 1, 0)

    # Stash a dataframe with those tweets that were paired with a keyword
    df_match = df.loc[df['no_matches'] != 1]
    # Stash a dataframe with those tweets that were never paired to a keyword 
    df_nomatch = df.loc[df['no_matches'] == 1]
    # Select columns
    df_nomatch = df_nomatch[['id', 'user', 'created_at', 'full_text', 'clean_text', 'multiple_teams', 'no_matches', 'teams_concat', 'rt_count', 'fav_count', 'follower_ct', 'verified', 'is_rt', 'reply_id', "company", "account_type"]]

    # Melt the dataframe such that each row is equal to a tweet that was matched to a team's keyword (introducing dups to tweets)
    melted_df = df.melt(
                    id_vars = ['id', 'user', 'created_at', 'full_text', 'clean_text', 'multiple_teams', 'no_matches', 'teams_concat', 'rt_count', 'fav_count', 'follower_ct', 'verified',  'is_rt', 'reply_id', "company", "account_type"],
                    value_vars = ['ANA', 'ARZ', 'BOS', 'BUF', 'CGY', 'CAR', 'CHI', 'COL', 'CBJ', 'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NSH', 'NJD', 'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'SJS', 'SEA', 'STL', 'TBL', 'TOR', 'VAN', 'VGK', 'WSH', 'WPG'],
                    var_name = 'nhl_team_abbr',
                    value_name = "nhl_team"
                    )
    # Filter out 0s 
    melted_df = melted_df.loc[melted_df['nhl_team'] != '0']

    # Add feature parity to df_nomatch
    df_nomatch['nhl_team_abbr'] = 'Unknown' 
    df_nomatch['nhl_team'] = 'Unknown' 

    # Append df_nomatch to melted_df to get df_clean
    df_clean = melted_df.append(df_nomatch)

    # Show a few rows of data
    df_clean.head(5)
    #print('total rows:', len(df_clean),'melted_rows:', (len(melted_df)), 'nomatch_rows:', len(df_nomatch))

    # Extend df_clean by joining in data about the team
    df_merged = pd.merge(df_clean, teams, on = 'nhl_team', how = 'left', indicator = True).astype("object")
    
    return df_merged, df, df_match, df_nomatch


# Function 6
#-----------------
def create_topics_df(df1, df2):
    # where df1 = dataframe that has been classified with NHL teams but has duplicate rows
    # where df2 = dataframe that has been groomed for text analysis, has no dups, and no NHL teams classification
    # logic = filter IDs in df2 by df1 to ensure user input is filtering the dataset for topic modeling & wordcloud

    id_keep_list = df1.id.tolist()
    filter_series = df2.id.isin(id_keep_list) # list to filter by
    df_topics = df2[filter_series] # filter df_sentiment by list 

    #df_topics2 = pd.merge(df1, df_topics[['id', 'sentiment','compound_score', 'positive_score', 'neutral_score','negative_score']], on = 'id', how = 'inner', indicator = True).astype("object")

    return df_topics


# Function... not used (but, kept for nowas it can be used later)!
#-----------------
def feature_extract(df):
    #TODO: add emoticons and emojis to this! and other punctuation

    # Create pre-clean character count feature
    df['character_ct'] = df.full_text.apply(lambda x: len(x))
    # Create stopword count features (english and french)
    df['stopword_en_ct'] = df.full_text.apply(lambda x: len([x for x in x.split() if x in stopwords_en]))
    df['stopword_fr_ct'] = df.full_text.apply(lambda x: len([x for x in x.split() if x in stopwords_fr]))
    # Create hashtag count feature
    df['hashtag_ct'] = df.full_text.apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
    # Create link count feature
    df['link_ct'] = df.full_text.apply(lambda x: len([x for x in x.split() if x.startswith('https')]))
    # Create @ sign count feature
    df['atsign_ct'] = df.full_text.apply(lambda x: len([x for x in x.split() if x.startswith('@')]))
    # Create numeric count feature
    df['numeric_ct'] = df.full_text.apply(lambda x: len([x for x in x.split() if x.isdigit()]))
    # Create an uppercase count feature
    df['uppercase_ct'] = df.full_text.apply(lambda x: len([x for x in x.split() if x.isupper()]))
    return df

# Function 7a
#-------------
def round1_text_clean(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text) # remove emoji
    text = ' ' + text # added space because there was some weirdness for first word (strip later)
    text = text.lower() # convert all text to lowercase
    text = re.sub(r'(\s)@\w+', '', text) # remove whole word if starts with @
    text = re.sub(r'(\s)\w*\d\w*\w+', '', text) # remove whole word if starts with number
    text = re.sub(r'https\:\/\/t\.co\/*\w*', '', text) # remove https links
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # removes punctuation
    text = re.sub('\[.*?\]', '', text) # removes text in square brackets
    text = text.replace("’s", '') # replace apostrophes with empty string
    text = text.replace("'s", '') # replace apostrophes with empty string
    text = text.replace("’t", '') # replace apostrophes with empty string
    text = text.replace("'t", '') # replace apostrophes with empty string
    #text = re.sub('\w*\d\w*', '', text) # remove whole word if starts with number
    #text = re.sub(r'(\s)#\w+', '', text) # remove whole word if starts with #
    text = text.strip() # strip text
    return text

# Function 7b
#-------------
text_clean_round1 = lambda x: round1_text_clean(x)

# Function 8
#-------------
def text_clean_round2(text):
    """
    A simple function to clean up the data. All the words that
    are not designated as a stop word is then lemmatized after
    encoding and basic regex parsing are performed.
    """
    nltk.download('wordnet')
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore'))
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]

# Function 9
#-------------
def text_clean_round3(text):
    #TODO: add emoticons and emojis to this!
    # Load in stopwords
    stopwords_en = nltk.corpus.stopwords.words('english')
    stopwords_fr = nltk.corpus.stopwords.words('french')
    stopwords = stopwords_en + stopwords_fr
    # Create pre-clean character count feature
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))
    return text

# Function 10a
#-----------------
def tweets_ngrams(n, top_n, df):
    """
    Generates series of top ngrams
    n: number of words in the ngram
    top_n: number of ngrams with highest frequencies
    """
    text = df.clean_text
    words = text_clean_round2(''.join(str(text.tolist())))
    result = (pd.Series(data = nltk.ngrams(words, n), name = 'frequency').value_counts())[:top_n]
    return result.to_frame()

# Function 10b
#-----------------
def all_ngrams(top_n, df):
    text = df.clean_text
    words = text_clean_round2(''.join(str(text.tolist())))
    unigram = ((pd.Series(data = nltk.ngrams(words, 1), name = 'freq').value_counts())[:top_n]).to_frame()
    unigram['ngram'] = 'unigram'
    bigram = ((pd.Series(data = nltk.ngrams(words, 2), name = 'freq').value_counts())[:top_n]).to_frame()
    bigram['ngram'] = 'bigram'
    trigram = ((pd.Series(data = nltk.ngrams(words, 3), name = 'freq').value_counts())[:top_n]).to_frame()
    trigram['ngram'] = 'trigram'
    result = unigram.append([bigram, trigram])
    result['ngram_nm'] = result.index
    return result

# Function 11
#----------------

# Credit: https://jackmckew.dev/sentiment-analysis-text-cleaning-in-python-with-vader.html
sid_analyzer = SentimentIntensityAnalyzer()

# Get sentiment
def get_sentiment(text:str, analyser, desired_type:str='pos'):
    # Get sentiment from text
    sentiment_score = analyser.polarity_scores(text)
    return sentiment_score[desired_type]

# Get Sentiment scores
def get_sentiment_scores(df, data_column):
    df[f'positive_score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'pos'))
    df[f'negative_score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'neg'))
    df[f'neutral_score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'neu'))
    df[f'compound_score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'compound'))
    return df

# Function 12
#----------------
# Credit: https://www.dataquest.io/blog/tutorial-add-column-pandas-dataframe-based-on-if-else-condition/

# classify based on VADER readme rules
def sentiment_classifier(df, data_column):

    # create a list of our conditions
    conditions = [
        (df[data_column] >= 0.05),
        (df[data_column] > -0.05) & (df[data_column] < 0.05),
        (df[data_column] <= -0.05),
        ]

    # create a list of the values we want to assign for each condition
    values = ['Positive', 'Neutral', 'Negative']
    
    # apply
    df['sentiment'] = np.select(condlist = conditions, choicelist = values)
    return df

# Function 13
#----------------

# Credit: https://ourcodingclub.github.io/tutorials/topic-modelling-python/

def lda_topics(data, number_of_topics, no_top_words, min_df, max_df):
    with st.spinner('Setting up LDA model..'):
        # the vectorizer object will be used to transform text to vector form
        vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, token_pattern='\w+|\$[\d\.]+|\S+')

        # apply transformation
        tf = vectorizer.fit_transform(data).toarray()

        # tf_feature_names tells us what word each column in the matrix represents
        tf_feature_names = vectorizer.get_feature_names()

        model = LDA(n_components=number_of_topics, random_state=0)

        model.fit(tf)

        topic_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(tf_feature_names[i])
                            for i in topic.argsort()[:-no_top_words - 1:-1]]
            topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                            for i in topic.argsort()[:-no_top_words - 1:-1]]

        topic_df = pd.DataFrame(topic_dict)

    return pd.DataFrame(topic_df)


# Function 14
#----------------
# Credit: https://ourcodingclub.github.io/tutorials/topic-modelling-python/

def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


# Function 15
#---------------
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

#TODO: This function does not work in the streamlit app. Returns "BrokenPipeError: [Errno 32] Broken pipe"
def LDA_viz(data):
    data_words = list(sent_to_words(data))

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # number of topics
    num_topics = 10
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics)
    # Print the Keyword in the 10 topics
    # pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    # Visualize the topics
    pyLDAvis.enable_notebook()
    LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html')
    
    return LDAvis_prepared

# Function 16
#----------------
# Credit: https://jackmckew.dev/sentiment-analysis-text-cleaning-in-python-with-vader.html

def print_top_n_tweets(df, sent_type, num_rows):
    text = 'full_text'
    top = df.nlargest(num_rows, sent_type)
    top_tweets = top[[sent_type,text]].reset_index()
    top_tweets = top_tweets.drop(columns = 'index')
    top_tweets.index = top_tweets.index + 1 
    return top_tweets

# Function 17
#----------------
# Function to convert  
def word_cloud_all(df, wordcloud_words): 
    # convert text_cleaned to word
    text = df.clean_text
    word_list = text_clean_round2(''.join(str(text.tolist())))
    # initialize an empty string
    str1 = " " 
    # return string  
    str2 = str1.join(word_list)
    # generate word cloud
    wordcloud = WordCloud(max_font_size=80, max_words=wordcloud_words, background_color="white", height=100).generate(str2)
    return wordcloud

# Function 18
#----------------
# Function to convert  
def word_cloud_sentiment(df, sent_type, num_rows, wordcloud_words): 
    # slice df to top n rows for sentiment type selected
    sliced_df = df.nlargest(num_rows, sent_type)
    # convert text_cleaned to word
    text = sliced_df.clean_text
    word_list = text_clean_round2(''.join(str(text.tolist())))
    # initialize an empty string
    str1 = " " 
    # return string  
    str2 = str1.join(word_list)
    # generate word cloud
    wordcloud = WordCloud(max_font_size=100, max_words=wordcloud_words, background_color="white").generate(str2)
    return wordcloud

# Function 19
#----------------
# Function to plot default wordcloud
def default_wordcloud(text_sentiment):

    # Hard coded default wordcloud (for show)
    score_type = 'All'
    score_type_nm = 'compound_score'
    wordcloud_words = 15
    top_n_tweets = 5
   
    # Run wordlcloud for top n tweets
    if score_type == 'All':         
        wordcloud = word_cloud_all(text_sentiment, wordcloud_words)
    else:
        wordcloud = word_cloud_sentiment(text_sentiment, score_type_nm, top_n_tweets, wordcloud_words)

    # Display the generated wordcloud image:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # Write word cloud, need to call it still
    st.write('Word Cloud Generator')
    st.pyplot()
    
    return 

# Function 20
#----------------
# Function to plot wordcloud that changes
def plot_wordcloud(submitted2, score_type, text_sentiment, wordcloud_words, top_n_tweets):
    
    # Scenarios
    # Scenario 1: No input (default)
    if submitted2 is not True:
        score_type = 'All'
        score_type_nm = 'compound_score'
        
    # Scenario 2: All
    if score_type == 'All':
        score_type_nm = 'compound_score'

    # Scenario 3: Positive
    if score_type == 'Positive':
        score_type_nm = 'positive_score'

    # Scenario 4: Neutral
    if score_type == 'Neutral':
        score_type_nm = 'neutral_score'

    # Scenario 5: Negative
    if score_type == 'Negative':
        score_type_nm = 'negative_score'

    # Run wordlcloud for top n tweets
    if score_type == 'All':         
        wordcloud = word_cloud_all(text_sentiment, wordcloud_words)
    else:
        wordcloud = word_cloud_sentiment(text_sentiment, score_type_nm, top_n_tweets, wordcloud_words)

    # Display the generated wordcloud image:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # Write word cloud, need to call it still
    st.write('Word Cloud Generator')
    st.pyplot()
    
    return 

# Function 21a
#----------------
# Function to display topics and related keywords
def print_lda_keywords(data, number_of_topics):
    # sets initial topic number to 1 to be used in output
    topic_num = 1
    # takes user input number of topics and loops over topic
    for n in range(0, number_of_topics*2, 2):
        # joins keywords for each topic, separated by comma
        list_topic_words = ", ".join(data.iloc[:, n])
        # prints output
        st.warning('**Theme #**' + str(topic_num) + '**:** ' + list_topic_words)
        # increments topic number by 1 so that each theme printed out will have a new number
        topic_num += 1

# Function 21b
#----------------
# Function to display topics, related keywords, and weights
def print_lda_keywords_weight(data, number_of_topics):
    # sets initial topic number to 1 to be used in output
    topic_num = 1
    # takes user input number of topics and loops over topic and weight
    for n in range(0, number_of_topics*2, 2):
        w = n + 1
        list_topic_words_weight = ", ".join(data.iloc[:, n] + ' (' + data.iloc[:, w] + ')')
        # prints output
        st.warning('**Theme #**' + str(topic_num) + '**:** ' + list_topic_words_weight)
        # increments topic number by 1 so that each theme printed out will have a new number
        topic_num += 1

  
# Function 22
#----------------
# Group and summarize df_sentiment to be used as building blocks for various KPIs
def group_nhl_data(df_sentiment):  
   # Sentiment group dataframe
    expansion_group = df_sentiment.groupby(['expansion_type', 'sentiment']).agg({'id': 'nunique', 'compound_score': ['mean', 'median', 'min', 'max']}).reset_index(level=[0,1])    
    expansion_group2 = df_sentiment.groupby('expansion_type').agg({'id': 'nunique', 'compound_score': ['mean', 'median', 'min', 'max']}).reset_index()
    team_group = df_sentiment.groupby(['nhl_team_abbr', 'nhl_team', 'sentiment']).agg({'id': 'nunique'}).reset_index()
    team_group2 = df_sentiment.groupby(['nhl_team_abbr', 'nhl_team']).agg({'id': 'nunique', 'compound_score': ['mean', 'median', 'min', 'max']}).reset_index(level=[0,1])
    team_group2 = team_group2.loc[team_group2['nhl_team_abbr'] != 'Unknown']

    # columns
    team_group2.columns = ['nhl_team_abbr', 'nhl_team', 'tweets', 'avg_compound_score', 'median_compound_score', 'min_compound_score', 'max_compound_score']    # rename
    expansion_group.columns = ['expansion_type', 'sentiment', 'tweets', 'avg_compound_score', 'median_compound_score', 'min_compound_score', 'max_compound_score']
    expansion_group2.columns = ['expansion_type',  'tweets', 'avg_compound_score', 'median_compound_score', 'min_compound_score', 'max_compound_score']
    # rename
    expansion_group.rename(columns={"id": "tweets"}, inplace = True)
    team_group.rename(columns={"id": "tweets"}, inplace = True)

    # Join team_group to teams data to expand the dataframe
    # Extend df_clean by joining in data about the team
    team_group = pd.merge(team_group,
                        teams,
                        on = 'nhl_team',
                        how = 'left',
                        indicator = True)

    # Summary metrics -- Kraken
    kraken = expansion_group.loc[expansion_group['expansion_type'] == 'Kraken']
    kraken_total = kraken.tweets.sum()
    kraken_negative  = kraken.loc[kraken['sentiment'] == 'Negative'].tweets.max()
    kraken_neutral = kraken.loc[kraken['sentiment'] == 'Neutral'].tweets.max()
    kraken_positive = kraken.loc[kraken['sentiment'] == 'Positive'].tweets.max()

    #Summary metrics -- Rest of the league
    rol = expansion_group.loc[expansion_group['expansion_type'] == 'Rest of League']
    rol_total = rol.tweets.sum()
    rol_negative  = rol.loc[rol['sentiment'] == 'Negative'].tweets.max()
    rol_neutral = rol.loc[rol['sentiment'] == 'Neutral'].tweets.max()
    rol_positive = rol.loc[rol['sentiment'] == 'Positive'].tweets.max()

    #Summary metrics -- team = unknown
    unknown = team_group.loc[team_group['nhl_team'] == 'Unknown']
    unknown_total = unknown.tweets.sum()
    unknown_negative  = unknown.loc[unknown['sentiment'] == 'Negative'].tweets.max()
    unknown_neutral = unknown.loc[unknown['sentiment'] == 'Neutral'].tweets.max()
    unknown_positive = unknown.loc[unknown['sentiment'] == 'Positive'].tweets.max()
    
    return expansion_group2, team_group2, kraken, kraken_total, kraken_negative, kraken_neutral, kraken_positive, rol, rol_total, rol_negative, rol_neutral, rol_positive, unknown, unknown_total, unknown_negative, unknown_negative, unknown_neutral, unknown_positive
  
  
  
# Function 23
#----------------
def load_fan_message(user_num_tweets):
    st.success('🎈Done! We got you the last ' + 
                user_num_tweets + 
                ' tweets about the NHL Expansion Draft')

# Function 23
#----------------
def load_insider_message(total_tweets):
    tweets_num = str(total_tweets)
    st.success('🎈Done! After filtering, we got you  ' +
                tweets_num +
                ' tweets from NHL Insiders in the last 48h')
