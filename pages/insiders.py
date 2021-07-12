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

    #col1, col2 = st.beta_columns((1,1)) # 2 same sized columns

    ## 2.2.1: Sidebar Title
    ##----------------------------------##
    st.sidebar.header('Choose Search Inputs') #sidebar title
    

    ## 2.2.2: Sidebar Input Fields
    ##----------------------------------##
    with st.form(key ='form_1'):
        with st.sidebar:
            insider_account_name = st.text_input("1. Enter one Twitter account", help='Enter a Twitter account handle without the @')
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
    df_user_tweets = nf.get_user_tweets(insider_account_name)

    st.write(df_user_tweets)