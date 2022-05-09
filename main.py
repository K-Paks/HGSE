import streamlit as st

from src.model.nlp_model import NLPModel

# header
st.title("Healthy Gamer Search Engine")
st.subheader("Improve your mental health with the improved search engine!")
query = st.text_input("What's on your mind?")


nlpmodel = NLPModel()

if query:
    st.write('Results for: ', query)

    # get embeddings
    score_df = nlpmodel.get_scores(query)
    suggestions = nlpmodel.get_suggestions(score_df)

    for sugg in suggestions:
        # show results
        col1, mid, col2 = st.columns([1, 3, 20])
        with col1:
            st.image(
                f'https://i.ytimg.com/vi/{sugg.video_id}/default.jpg', width=100)
        with col2:
            st.markdown(
                f'[{sugg.title}](https://www.youtube.com/watch?v={sugg.video_id})', unsafe_allow_html=True)

            if sugg.name != 'None':
                sugg.name = sugg.name.replace('- ', '')
                st.markdown(
                    f'Relatable part: [{sugg.name}](https://youtu.be/{sugg.video_id}?t={sugg.time_raw}). '
                    f'Starts at {sugg.time_start}')
