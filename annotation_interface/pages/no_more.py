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
st.markdown('''# All tasks completed! :hibiscus:\n### No more tasks at this time.\n### Please return this HIT.''')