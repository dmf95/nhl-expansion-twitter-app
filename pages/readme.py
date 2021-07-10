#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
# PART 1: LOAD DEPENDENCIES & TODO
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
# - 1.1: Load libraries
# - 1.2: #TODO items
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-
# 1.1: Load libraries
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-
import streamlit as st

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-
# 1.2: #TODO items
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-
#TODO Replace the content to serve as a README file

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
# PART 2: Load
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
# - 1.1: Load libraries
# - 1.2: #TODO items
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-

def app():
    st.title('Read Me')

    st.write('This is the `home page` of this multi-page app.')

    st.write('In this app, we will be building a simple classification model using the Iris dataset.')

    ## Part : About the App
    ##----------------------------------##

    ## About the app title
    st.text("") # spacing
    st.header('About the App')

    # General expander section
    about_expander = st.beta_expander("General")
    about_expander.markdown("""
    * **Creators:** [Shannon Lo](https://shannonhlo.github.io/) & [Domenic Fayad](https://www.fullstaxx.com/)
    * **References:**
    * https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
    * https://jackmckew.dev/sentiment-analysis-text-cleaning-in-python-with-vader.html
    * https://www.dataquest.io/blog/tutorial-add-column-pandas-dataframe-based-on-if-else-condition/
    * https://ourcodingclub.github.io/tutorials/topic-modelling-python/
    """)

    # Methodology expander section
    method_expander= st.beta_expander("Methodology")
    method_expander.markdown("""
    * Applying the [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) library to our text data
    * [VADER](https://github.com/cjhutto/vaderSentiment#vader-sentiment-analysis) (**V**alence **A**ware **D**ictionary and s**E**ntiment **R**easoner) = lexicon and rule-based sentiment analysis tool, specifically attuned to sentiments expressed in social media
    * [Compound score](https://github.com/cjhutto/vaderSentiment#about-the-scoring) = computed by summing the valence scores of each word in the lexicon, adjusted according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive)
    * Positive sentiment: compound score >= 0.05
    * Neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
    * Negative sentiment: compound score <= -0.05
    """)



    