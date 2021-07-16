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
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=
# 1.3: #TODO
#------------------------------------#

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
# PART 2: APP UI SETUP
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
# - 2.1: Main panel setup 
# - 2.2: Sidebar setup
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-

# Everything must be defined within the app function
def app():

    ## 2.1.1: Main Panel Title
    ##----------------------------------##
    st.title('NHL Insiders View')



#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
# PART 3: APP DATA SETUP
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
# - 3.1: Twitter data ETL
# - 3.2: Define key variables
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-

    # 3.1: Twitter Data ETL

    # Layout
    #------------------------------------#
    # Run function 21: Get recent tweets for each insider account
    df_insider_tweets = nf.insider_recent_tweets()

    st.write(df_insider_tweets)

    # Run function #4: Feature extraction
    df_tweets = nf.feature_extract(df_insider_tweets)

    # Run function #5: Round 1 text cleaning (convert to lower, remove numbers, @, punctuation, numbers. etc.)
    df_tweets['clean_text'] = df_tweets.full_text.apply(nf.text_clean_round1)

    ## Run function #7: Round 3 text cleaning (remove stop words)
    df_tweets.clean_text  = nf.text_clean_round3(df_tweets.clean_text)

    # Run function 3: Get classified nhl teams data
    df_nhl, df_original, df_match, df_nomatch = nf.classify_nhl_team(df_insider_tweets)

    #Read in teams & accounts CSVs
    teams = pd.read_csv('assets/nhl_app_teams.csv')

    # if team is kraken, then kraken else rest of league
    teams['expansion_type'] = np.where(teams.nhl_team.str.contains("Kraken"), "Kraken", "Rest of League")

    # Get sentiment scores on raw tweets
    text_sentiment = nf.get_sentiment_scores(df_nhl, 'full_text')

    # Add sentiment classification
    text_sentiment = nf.sentiment_classifier(df_nhl, 'compound_score')

    # Select columns to output
    df_sentiment = text_sentiment[['user', 'id', 'created_at', 'nhl_team_abbr', 'nhl_team', 'expansion_type', 'full_text', 'sentiment', 'positive_score', 'negative_score', 'neutral_score', 'compound_score']]

    # Sentiment group dataframe
    sentiment_group = df_sentiment.groupby(['sentiment']).agg({'id': 'nunique'}).reset_index()
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
    sentiment_group.rename(columns={"id": "tweets"}, inplace = True)
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


    #~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

    # 3.2: Define Key Variables
    #------------------------------------#
    total_tweets = len(df_original['full_text'])
    match_tweets = len(df_match['full_text'])
    nomatch_tweets = len(df_nomatch['full_text'])

    #highest_retweets = max(df_nhl['rt_count'])
    #highest_likes = max(df_nhl['fav_count'])

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
    # PART 4: APP DATA & VISUALIZATIONS
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
# - 4.2: Sentiment analysis
# - 4.3: Descriptive analysis
# - 4.4: Topic model analysis
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-


    #~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

    # 4.2: Sentiment Analysis
    #------------------------------------#

    # Subtitle
    st.header('❤️ Fan Sentiment')

    ## 4.2.1: Sentiment by team bar chart
    ##----------------------------------##
    col1, col2 = st.beta_columns((1,1)) # 2 same sized columns

    # Altair chart: sentiment bart chart by expansion team
    sentiment_exp_bar = alt.Chart(expansion_group2).mark_bar().encode(
                        x = alt.X('avg_compound_score:Q', axis = alt.Axis(title = 'Average Compound Score')),
                        y = alt.Y('expansion_type:O', sort = '-x', axis = alt.Axis(title = 'Expansion Type')),
                        tooltip = [alt.Tooltip('expansion_type:O', title = 'Expansions Type'),alt.Tooltip('sum(tweets):Q', title = 'Number of Tweets'), alt.Tooltip('avg_compound_score', title = 'Average Score'), alt.Tooltip('median_compound_score', title = 'Median Score'),alt.Tooltip('min_compound_score', title = 'Worst Score'), alt.Tooltip('max_compound_score', title = 'Best Score')],
                        color=alt.condition(alt.datum.avg_compound_score > 0,
                            alt.value("#99D9D9"),  # The positive color
                            alt.value("E9072B"))  # The negative color
                    ).transform_filter( # filters
                            (alt.datum.expansion_type != 'Unknown')               
                    ).properties(
                        height = 400
                    ).interactive()
    # Write the chart
    col1.subheader('Kraken vs Rest of League')
    col1.altair_chart(sentiment_exp_bar, use_container_width=True)

   # Altair chart: sentiment bart chart by day (color condition #TODO)
    sentiment_bar = alt.Chart(team_group2).mark_bar().encode(
                        x = alt.X('avg_compound_score:Q', axis = alt.Axis(title = 'Average Compound Score')),
                        y = alt.Y('nhl_team_abbr:O', sort = '-x', axis = alt.Axis(title = 'NHL Team')),
                        tooltip = [alt.Tooltip('nhl_team:O', title = 'NHL Team'),alt.Tooltip('sum(tweets):Q', title = 'Number of Tweets'), alt.Tooltip('avg_compound_score', title = 'Average Score'), alt.Tooltip('median_compound_score', title = 'Median Score'),alt.Tooltip('min_compound_score', title = 'Worst Score'), alt.Tooltip('max_compound_score', title = 'Best Score')],
                        color = alt.condition(alt.datum.avg_compound_score > 0,
                                    alt.value("#99D9D9"),  # The positive color
                                    alt.value("#E9072B"))  # The negative color                            
                    ).properties(
                        height = 400
                    ).interactive()

    # Write the chart
    col2.subheader('Tweet Sentiment by Team')
    col2.altair_chart(sentiment_bar, use_container_width=True)
    

    ## 4.2.3: Sentiment Expander Bar
    ##----------------------------------##
    sentiment_expander = st.beta_expander('Expand to see sentiment distribution', expanded=False)


    ## 4.2.4: Compound Score Histogram
    ##----------------------------------##

    # Histogram for VADER compound score

    base = alt.Chart(df_sentiment)

    sentiment_histo= base.mark_bar().encode(
                        x = alt.X('compound_score:Q', axis = alt.Axis(title = 'VADER Compound Score'), bin=alt.Bin(extent=(-1, 1), maxbins=50)),
                        y = alt.Y('count(id):Q', axis = alt.Axis(title = 'Number of Tweets')),
                        tooltip = [alt.Tooltip('sentiment', title = 'Sentiment Group'), 'count(id):Q', alt.Tooltip('average(compound_score)', title = 'Average Compound Score'), alt.Tooltip('median(compound_score)', title = 'Median Compound Score')] ,
                        color=alt.Color('sentiment',
                            scale=alt.Scale(
                            domain=['Positive', 'Neutral', 'Negative'],
                            range=['#99D9D9', '#D3D3D3', '#E9072B']))
                    ).properties(
                        height = 400
                    ).interactive()


    # Write the chart
    sentiment_expander.subheader('Checking Sentiment Skewness')
    sentiment_expander.write('VADER Compound Scores Histogram')
    sentiment_expander.altair_chart(sentiment_histo, use_container_width=True)    



    ## 4.2.5: Summary Card Metrics
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



    ## 4.2.5: Compound Score Violin
    ##----------------------------------##
    
    # TODO WOULD BE COOL TO HAVE THIS VIZ :)
    #https://github.com/altair-viz/altair/issues/2173



    ## 4.2.6: Download raw sentiment data
    ##----------------------------------##

    # Show raw data if selected
    if sentiment_expander.checkbox('Show VADER results for each Tweet'):
        sentiment_expander.subheader('Raw data')
        sentiment_expander.write(df_sentiment)

    # Click to download raw data as CSV
    sentiment_expander.markdown(nf.get_table_download_link(df_sentiment), unsafe_allow_html=True)


 #~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

    # 4.3: Wordclouds & Tweets
    #------------------------------------#

    # Subtitle
    st.header('☁️🔝 Wordcloud & Top Tweets')


    ## 4.3.1: Sentiment Expander Bar
    ##----------------------------------##

    # Setup expander
    wordcloud_expander = st.beta_expander('Expand to customize wordcloud & top tweets', expanded=False)

    # Sentiment Wordcloud subheader & note
    wordcloud_expander.subheader('Advanced Settings')


    ## 4.3.2: Wordcloud expander submit form
    ##----------------------------------##

    # Sentiment expander form submit for the wordcloud & top tweets
    with wordcloud_expander.form('form_2'):    
        score_type = st.selectbox('Select sentiment', ['All', 'Positive', 'Neutral', 'Negative'])
        wordcloud_words = st.number_input('Choose the max number of words for the word cloud', 15)
        top_n_tweets =  st.number_input('Choose the top number of tweets *', 3)
        submitted2 = st.form_submit_button('Regenerate Wordcloud', help = 'Re-run the Wordcloud with the current inputs')


    ## 4.3.3: Plot wordcloud
    ##----------------------------------##
    nf.plot_wordcloud(submitted2, score_type, text_sentiment, wordcloud_words, top_n_tweets)


    ## 4.3.4: Plot top tweets
    ##----------------------------------##

    # Scenarios

    # Scenario 1: All
    if score_type == 'All':
        score_type_nm = 'compound_score'
        score_nickname = 'All'

    # Scenario 2: Positive
    if score_type == 'Positive':
        score_type_nm = 'positive_score'
        score_nickname = 'Positive'

    # Scenario 3: Neutral
    if score_type == 'Neutral':
        score_type_nm = 'neutral_score'
        score_nickname = 'Neutral'

    # Scenario 4: Negative
    if score_type == 'Negative':
        score_type_nm = 'negative_score'
        score_nickname = 'Negative'

    # Run the top n tweets
    top_tweets_res = nf.print_top_n_tweets(df_sentiment, score_type_nm, top_n_tweets)

    # Conditional title
    str_num_tweets = str(top_n_tweets)
    show_top = str('Showing top ' + 
                    str_num_tweets + 
                    ' ' +
                    score_nickname + 
                    ' tweets ranked by '+ 
                    score_type_nm)

    # Write conditional
    st.write(show_top)

    # Show top n tweets
    for i in range(top_n_tweets):
        i = i + 1
        st.info('**Tweet #**' + str(i) + '**:** ' + top_tweets_res['full_text'][i] + '  \n **Score:** ' + str(top_tweets_res[score_type_nm][i]))

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

    # 4.4: Descriptive Analysis
    #------------------------------------#

    # Subtitle
    st.header('📊 Descriptive Analysis')


    # KPI for printing test
    #st.write("All Tweets:", total_tweets)
    #st.write("Matched Tweets:", match_tweets)
    #st.write("Not Matched Tweets:", nomatch_tweets)
    #st.write("Kraken:", kraken_total)
    #st.write("Rest of League:",  rol_total)
    #st.write("No team: ", unknown_total)


    ## 4.4.1: Summary Metric Cards
    ##----------------------------------##

    # KPI Cards for descriptive summary
    st.subheader('Tweet Summary')
    metric_row(
        {
            "Number of tweets": total_tweets,
            "% Tweets about the Kraken": "{:.0%}".format(kraken_total/total_tweets),
            "% Tweets about another NHL Team": "{:.0%}".format(rol_total/total_tweets),
        }
    )

    most_retweets_index = df_tweets['rt_count'].idxmax()
    most_likes_index = df_tweets['fav_count'].idxmax()

    if most_retweets_index == most_likes_index:
        st.info('**Same tweet had the most retweets and likes: **' + df_tweets['full_text'][most_retweets_index])
    else:
        st.info('**Tweet with most retweets: **' + df_tweets['full_text'][most_retweets_index])
        st.info('**Tweet with most likes: **' + df_tweets['full_text'][most_likes_index])


    ## 4.4.2: Descriptive Expander Bar
    ##----------------------------------##
    descriptive_expander = st.beta_expander('Expand to see more descriptive analysis', 
                                            expanded=False)


    ## 4.4.3: Tweets by day bar chart
    ##----------------------------------##

    # Subtitle
    descriptive_expander.subheader('Number of Tweets by Day')

    # Altair chart: number of total tweets by day
    #TODO: declutter x-axis. Unreadable when there are multiple dates
    line = alt.Chart(df_tweets).mark_line(interpolate = 'basis').encode(
                        x = alt.X('hoursminutes(created_at):O', axis = alt.Axis(title = 'Hour of Day')),
                        y = alt.Y('count(id):Q', axis = alt.Axis(title = 'Number of Total Tweets', tickMinStep=1)),
                        color = "count(id):Q"
                    # tooltip = [alt.Tooltip('monthdatehours(created_at):O', title = 'Tweet Date'), alt.Tooltip('count(id):Q', title = 'Number of Tweets')]
                    ).properties(
                        height = 350
                    ).interactive()  

    # Plot with altair
    descriptive_expander.altair_chart(line, use_container_width=True)


    ## 4.4.4: Ngram Word Counts
    ##----------------------------------##

    # Subtitle
    descriptive_expander.subheader('Word Frequency and Ngrams')

    # User selections
    ngram_option = descriptive_expander.selectbox(
                    'Select the number of ngrams',
                    ('Single', 'Bigram', 'Trigram'))

    # Scenarios

    # Scenario 1: Single ngram
    if ngram_option == 'Single':
        ngram_num = 1
        ngram_nm = 'Single Word Frequencies'

    # Scenario 2: Bigrams
    if ngram_option == 'Bigram':
        ngram_num = 2
        ngram_nm = 'Bigram Word Frequencies'

    # Scenario 3: Trigrams
    if ngram_option == 'Trigram':
        ngram_num = 3
        ngram_nm = 'Trigram Word Frequencies'

    # Display ngram based on selection
    ngram_visual = nf.tweets_ngrams(ngram_num, 10, df_tweets)
    ngram_visual['ngram'] = ngram_visual.index

    # Conditional subtitle
    descriptive_expander.write(ngram_nm)

    # Altair chart: ngram word frequencies
    ngram_bar = alt.Chart(ngram_visual).mark_bar().encode(
                        x = alt.X('frequency', axis = alt.Axis(title = 'Word Frequency')),
                        y = alt.Y('ngram', axis = alt.Axis(title = 'Ngram'), sort = '-x'),
                        tooltip = [alt.Tooltip('frequency', title = 'Ngram Frequency')],#,  alt.Tooltip('Ngram', title = 'Ngram Word(s)')] ,
                    ).properties(
                        height = 350
                    )

    descriptive_expander.altair_chart(ngram_bar, use_container_width=True)


    ## 4.4.5: Download raw descriptive data
    ##----------------------------------##

    # Show raw data if selected
    if descriptive_expander.checkbox('Show raw data'):
        descriptive_expander.subheader('Raw data')

    # Click to download raw data as CSV
    descriptive_expander.markdown(nf.get_table_download_link(df_tweets), unsafe_allow_html=True)

    #~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=

    # 4.5: Topic Modeling
    #------------------------------------#

    # Subtitle
    st.header('🧐 Top Themes')

    ## 4.5.1: Topic Expander Bar
    ##----------------------------------##
    topic_expander = st.beta_expander('Expand to see more topic modeling options', 
                                            expanded=False)

    ## 4.5.2: Topic Model table
    ##----------------------------------##

    # Define data variable
    data = df_tweets['clean_text']

    topic_view_option = topic_expander.radio('Choose display options', ('Default view', 'Analyst view (advanced options)'))



    if topic_view_option == 'Default view':
        # Topic model expander form submit for topic model table & visual
        with topic_expander.form('form_3'):
            number_of_topics = st.number_input('Choose the number of topics. Start with a larger number and decrease if you see topics that are similar.',min_value=1, value=5)
            no_top_words = st.number_input('Choose the number of words in each topic you want to see.',min_value=1, value=5)
            submitted2 = st.form_submit_button('Regenerate topics', help = 'Re-run topic model analysis with the current inputs')
        df_lda = nf.lda_topics(data, number_of_topics, no_top_words, 0.1, 0.9)
        nf.print_lda_keywords(df_lda, number_of_topics)
    else:
        with topic_expander.form('form_3'):
            number_of_topics = st.number_input('Choose the maximum number of topics. Start with a larger number and decrease if you see topics that are similar.',min_value=1, value=5)
            no_top_words = st.number_input('Choose the maximum number of words in each topic you want to see.',min_value=1, value=5)
            min_df = st.number_input('Ignore words that appear less than the specified proportion (decimal number between 0 and 1).',min_value=0.0, max_value=1.0, value=0.1)
            max_df = st.number_input('Ignore words that appear more than the specified proportion (decimal number between 0 and 1).',min_value=0.0, max_value=1.0, value=0.9)
            submitted2 = st.form_submit_button('Regenerate topics', help = 'Re-run topic model analysis with the current inputs')
        df_lda = nf.lda_topics(data, number_of_topics, no_top_words, min_df, max_df)
        st.write('Weights shown in brackets represent how important the word is to each topic')
        nf.print_lda_keywords_weight(df_lda, number_of_topics)
