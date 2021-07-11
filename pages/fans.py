#--------------------------------------------------
# PART 1: LOAD DEPENDENCIES & TODO
#--------------------------------------------------
# - 1.1: Load libraries
# - 1.2: Load custom library
# - 1.3: TODO items
#--------------------------------------------------

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

# 1.1: Load libraries
#------------------------------------#
import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from streamlit_metrics import metric_row
import altair as alt
import time
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

# 1.2: Load custom library
#------------------------------------#
import nhl_exp_functions as nf # custom functions file


# Everything must be defined within the app function
def app():

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
# PART 2: APP UI SETUP
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
# - 2.1: Main panel setup 
# - 2.2: Sidebar setup
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-

    ## 2.1.1: Main Panel Title
    ##----------------------------------##
    st.title('Fans View')

    col_small, col_large = st.beta_columns((1,2)) # col1 is 2x greater than col2

    ## 2.2.1: Sidebar Title
    ##----------------------------------##
    st.sidebar.header('Choose Search Inputs') #sidebar title
    

    ## 2.2.2: Sidebar Input Fields
    ##----------------------------------##
    with st.form(key ='form_1'):
        with st.sidebar:
            user_word_entry = st.text_input("1. Enter one keyword", "stanleycup", help='Ensure that keyword does not contain spaces')    
            select_hashtag_keyword = st.radio('2. Search hashtags, or all keywords?', ('Hashtag', 'Keyword'), help='Searching only hashtags will return fewer results')
            select_language = st.radio('3. Tweet language', ('All', 'English', 'French'), help = 'Select the language you want the Analyzer to search Twitter for')
            num_of_tweets = st.number_input('4. Maximum number of tweets', min_value=100, max_value=10000, value = 150, step = 50, help = 'Returns the most recent tweets within the last 7 days')
            st.sidebar.text("") # spacing
            submitted1 = st.form_submit_button(label = 'Run Tweet Analyzer 🚀', help = 'Re-run analyzer with the current inputs')


#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
# PART 3: APP DATA SETUP
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
# - 3.1: Twitter data ETL
# - 3.2: Define key variables
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-

    # 3.1: Twitter Data ETL

    # Layout
    #------------------------------------#

    # Run function 2: Get twitter data 
    df_tweets, df_new = nf.twitter_get(select_hashtag_keyword, select_language, user_word_entry, num_of_tweets)

    # Run function 3: Get classified nhl teams data
    df_nhl = nf.classify_nhl_team(df_tweets)

    # Run function #4: Feature extraction
    df_nhl = nf.feature_extract(df_nhl)

    # Run function #5: Round 1 text cleaning (convert to lower, remove numbers, @, punctuation, numbers. etc.)
    df_nhl['clean_text'] = df_nhl.clean_text.apply(nf.text_clean_round1)

    ## Run function #7: Round 3 text cleaning (remove stop words)
    df_nhl.clean_text  = nf.text_clean_round3(df_nhl.clean_text)

    #Read in teams & accounts CSVs
    teams = pd.read_csv('assets/nhl_app_teams.csv')

    # if team is kraken, then kraken else rest of league
    teams['expansion_type'] = np.where(teams.nhl_team.str.contains("Kraken"), "Kraken", "Rest of League")


    #~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=


    # 3.2: Define Key Variables
    #------------------------------------#
    user_num_tweets =str(num_of_tweets)
    total_tweets = len(df_nhl['full_text']) #TODO dups in here! need to do distinct count of IDs
    #highest_retweets = max(df_nhl['rt_count'])
    #highest_likes = max(df_nhl['fav_count'])

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
    # PART 4: APP DATA & VISUALIZATIONS
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
# - 4.1: UX messaging
# - 4.2: Sentiment analysis
# - 4.3: Descriptive analysis
# - 4.4: Topic model analysis
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-

    #~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

    # 4.1: UX Messaging
    #------------------------------------#

    # Loading message for users
    with st.spinner('Getting data from Twitter...'):
        time.sleep(5)
        # Keyword or hashtag
        if select_hashtag_keyword == 'Hashtag':
            st.success('🎈Done! You searched for the last ' + 
                user_num_tweets + 
                ' tweets that used #' + 
                user_word_entry)

        else:
            st.success('🎈Done! You searched for the last ' + 
                user_num_tweets + 
                ' tweets that used they keyword ' + 
                user_word_entry)

    #~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

    # 4.2: Sentiment Analysis
    #------------------------------------#

    # Subtitle
    st.header('❤️ Fan Sentiment')

    # Get sentiment scores on raw tweets
    text_sentiment = nf.get_sentiment_scores(df_nhl, 'full_text')

    # Add sentiment classification
    text_sentiment = nf.sentiment_classifier(df_nhl, 'compound_score')

    # Select columns to output
    df_sentiment = text_sentiment[['id', 'created_at', 'nhl_team_abbr', 'nhl_team', 'expansion_type', 'full_text', 'sentiment', 'positive_score', 'negative_score', 'neutral_score', 'compound_score']]

    # Sentiment group dataframe
    sentiment_group = df_sentiment.groupby(['sentiment']).agg({'id': 'nunique'}).reset_index()
    expansion_group = df_sentiment.groupby(['expansion_type', 'sentiment']).agg({'id': 'nunique'}).reset_index()
    team_group = df_sentiment.groupby(['nhl_team_abbr', 'nhl_team', 'sentiment']).agg({'id': 'nunique'}).reset_index()
    # rename
    sentiment_group.rename(columns={"id": "tweets"}, inplace = True)
    expansion_group.rename(columns={"id": "tweets"}, inplace = True)
    team_group.rename(columns={"id": "tweets"}, inplace = True)


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


    ## 4.2.1: Summary Card Metrics
    ##----------------------------------##

    # KPI Cards for Seattle Kraken sentiment summary
    st.subheader('Kraken Fan Scoring')
    metric_row(
        {
            "% 😡 Negative Tweets": "{:.0%}".format(kraken_negative/kraken_total),
            "% 😑 Neutral Tweets": "{:.0%}".format(kraken_neutral/kraken_total),
            "% 😃 Positive Tweets": "{:.0%}".format(kraken_positive/kraken_total),   
        }
    )

    # KPI Cards for Rest of League sentiment summary
    st.subheader('Rest of League Fan Scoring')
    metric_row(
        {
            "% 😡 Negative Tweets": "{:.0%}".format(rol_negative/rol_total),
            "% 😑 Neutral Tweets": "{:.0%}".format(rol_neutral/rol_total),
            "% 😃 Positive Tweets": "{:.0%}".format(rol_positive/rol_total),   
        }
    )



    ## 4.2.2: Sentiment by team bar chart
    ##----------------------------------##

    # Altair chart: sentiment bart chart by day
    sentiment_bar = alt.Chart(df_sentiment).mark_bar().encode(
                        x = alt.X('count(id):Q', stack="normalize", axis = alt.Axis(title = 'Percent of Total Tweets', format='%')),
                        y = alt.Y('monthdate(created_at):O', axis = alt.Axis(title = 'Date')),
                        tooltip = [alt.Tooltip('sentiment', title = 'Sentiment Group'), 'count(id):Q', alt.Tooltip('average(compound_score)', title = 'Avg Compound Score'), alt.Tooltip('median(compound_score)', title = 'Median Compound Score')],
                        color=alt.Color('sentiment',
                            scale=alt.Scale(
                            domain=['Positive', 'Neutral', 'Negative'],
                            range=['forestgreen', 'lightgray', 'indianred']))
                    ).properties(
                        height = 400
                    ).interactive()

    # Write the chart
    col_large.subheader('Classifying Tweet Sentiment by Day')
    col_large.altair_chart(sentiment_bar, use_container_width=True)

    ## 4.2.2: Sentiment by team bar chart
    ##----------------------------------##

    # Altair chart: sentiment bart chart by day
    sentiment_bar = alt.Chart(df_sentiment).mark_bar().encode(
                        x = alt.X('count(id):Q', stack="normalize", axis = alt.Axis(title = 'Percent of Total Tweets', format='%')),
                        y = alt.Y('monthdate(created_at):O', axis = alt.Axis(title = 'Date')),
                        tooltip = [alt.Tooltip('sentiment', title = 'Sentiment Group'), 'count(id):Q', alt.Tooltip('average(compound_score)', title = 'Avg Compound Score'), alt.Tooltip('median(compound_score)', title = 'Median Compound Score')],
                        color=alt.Color('sentiment',
                            scale=alt.Scale(
                            domain=['Positive', 'Neutral', 'Negative'],
                            range=['forestgreen', 'lightgray', 'indianred']))
                    ).properties(
                        height = 400
                    ).interactive()

    # Write the chart
    col_large.subheader('Classifying Tweet Sentiment by Day')
    col_large.altair_chart(sentiment_bar, use_container_width=True)






    ## 4.2.4: Sentiment Expander Bar
    ##----------------------------------##
    sentiment_expander = st.beta_expander('Expand to see more sentiment analysis', expanded=False)


    ## 4.2.5: Compound Score Histogram
    ##----------------------------------##

    # Histogram for VADER compound score
    sentiment_histo= alt.Chart(df_sentiment).mark_bar().encode(
                        x = alt.X('compound_score:O', axis = alt.Axis(title = 'VADER Compound Score (Binned)'), bin=alt.Bin(extent=[-1, 1], step=0.25)),
                        y = alt.Y('count(id):Q', axis = alt.Axis(title = 'Number of Tweets')),
                        tooltip = [alt.Tooltip('sentiment', title = 'Sentiment Group'), 'count(id):Q', alt.Tooltip('average(compound_score)', title = 'Average Compound Score'), alt.Tooltip('median(compound_score)', title = 'Median Compound Score')] ,
                        color=alt.Color('sentiment',
                            scale=alt.Scale(
                            domain=['Positive', 'Neutral', 'Negative'],
                            range=['forestgreen', 'lightgray', 'indianred']))
                    ).properties(
                        height = 400
                    ).interactive()

    # Write the chart
    sentiment_expander.subheader('Checking Sentiment Skewness')
    sentiment_expander.write('VADER Compound Scores Histogram')
    sentiment_expander.altair_chart(sentiment_histo, use_container_width=True)    


    ## 4.2.6: Download raw sentiment data
    ##----------------------------------##

    # Show raw data if selected
    if sentiment_expander.checkbox('Show VADER results for each Tweet'):
        sentiment_expander.subheader('Raw data')
        sentiment_expander.write(df_sentiment)

    # Click to download raw data as CSV
    sentiment_expander.markdown(nf.get_table_download_link(df_sentiment), unsafe_allow_html=True)