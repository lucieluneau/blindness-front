import streamlit as st

345435


if st.checkbox('do you want a button'):
    if st.button('click me'):
        # print is visible in server output, not in the page
        print('button clicked!')
        st.write('I was clicked 🎉')
        st.write('Further clicks are not visible but are executed')
    else:
        st.write('I was not clicked 😞')
