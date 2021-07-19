#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
# PART 1: LOAD DEPENDENCIES & TODO
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
# - 1.1: Load libraries
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-


#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-
# 1.1: Load libraries
#--------------------------------------------------
import streamlit as st
from PIL import Image
from pages import readme ,fans, insiders
from app_multipage import multi_app # import your app modules here
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-


#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
# PART 2: APP UI SETUP
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-
# - 2.1: Main panel setup 
# - 2.2: Sidebar setup
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~--=~-=~-=~-=~-=~-=~-

#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-
# 2.1: Main Panel Setup
#--------------------------------------------------

## 2.1.1: Main Configs
##----------------------------------##
st.set_page_config(layout="wide") # page expands to full width
col1, col2 = st.beta_columns((2,1)) # col1 is 2x greater than col2

## 2.1.2: Main Title/Logo
##----------------------------------##
image = Image.open('assets/nhl_app_logo.png') #logo
st.image(image, width = 340) #logo width
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-


#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-
# 2.2: Sidebar Setup
#--------------------------------------------------

## 2.2.1: Sidebar Configs
##----------------------------------##
side = st.sidebar

## 2.2.2: Sidebar Title/Logo
##----------------------------------##
side.header('Page Navigation') #sidebar title

## 2.2.3: Sidebar Page Navigation
##----------------------------------##
app = multi_app() # Load app_multipage.py class functions

# Add all your application here
app.add_app("Read Me", readme.app)
app.add_app("NHL Fans", fans.app)
app.add_app("NHL Insiders", insiders.app)

app.run() # Run the main app

## 2.2.4: Sidebar Social
##----------------------------------##
side.text("") # spacing
side.header('Developer Contact')
side.write("[![Star](https://img.shields.io/github/stars/dmf95/nhl-expansion-twitter-app.svg?logo=github&style=social)](https://github.com/dmf95/nhl-expansion-twitter-app)")
side.write("[![Follow](https://img.shields.io/twitter/follow/DomenicFayad?style=social)](https://twitter.com/DomenicFayad)")
side.write("[![Follow](https://img.shields.io/twitter/follow/shannonhlo26?style=social)](https://twitter.com/shannonhlo26)")
#~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-=~-