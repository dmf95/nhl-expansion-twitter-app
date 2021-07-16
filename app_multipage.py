# Reference: https://www.youtube.com/watch?v=nSw96qUbK9o&t=194s&ab_channel=DataProfessor
# Reference: https://towardsdatascience.com/creating-multipage-applications-using-streamlit-efficiently-b58a58134030

"""Frameworks for running multiple Streamlit applications as a single app.
"""
import streamlit as st

class multi_app:
    """Framework for combining multiple streamlit applications.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()
    It is also possible keep each application in a separate file.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()
    """
    def __init__(self):
        self.pages = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.pages.append({
            "title": title,
            "function": func
        })

    def run(self):
        # app = st.sidebar.radio(
        app = st.sidebar.selectbox(
            'Choose a page',
            self.pages,
            format_func=lambda app: app['title'],
            help = 'Change the page to see new goodies 😏')

        app['function']()