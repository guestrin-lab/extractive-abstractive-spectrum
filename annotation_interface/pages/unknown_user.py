import streamlit as st

## Page configs
st.set_page_config(initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)
st.markdown('''# Sorry, we don\'t recognize that Worker ID :slightly_frowning_face:\n### Please try the link again.''')