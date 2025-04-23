# import streamlit as st
# import  numpy as np
# import pickle
# xgb_model=pickle.load(open('movie_rating_model.pkl','rb'))
# sentence_model=pickle.load(open('sentence_model.pkl','rb'))
# avg_actor_rating=pickle.load(open("avg_actor_rating.pkl",'rb'))
# genre_columns=pickle.load(open('genre_columns_model.pkl','rb'))
# avg_director_rating=pickle.load(open('avg_director_rating.pkl','rb'))

# st.title("Movie rating prediction")
# st.markdown("enter movie details below  to  predict  the rating!")

# selected_genres=st.multiselect("select genre(s)",genre_columns)

# director=st.text_input("director's name","")
# actor1 = st.text_input("actor1 name", "")
# actor2 = st.text_input("actor2 name", "")
# actor3 = st.text_input("actor3 name", "")


# if st.button("predict rating"):
#     try:
#         genre_vector=[1 if genre in selected_genres else 0 for genre in genre_columns]
#         actor_list = [actor1, actor2, actor3]
#         actor_rating = np.mean([avg_actor_rating.get(actor, 5.0) for actor in actor_list])
#         director_rating=avg_director_rating.get(director,5.0)
        
#         final_features = np.array(genre_vector+[actor_rating,director_rating]).reshape(1,-1)
#         predicted_rating = xgb_model.predict(final_features)[0]
#         # st.success(f"The predicted IMDb rating is: **{round(predicted_rating:.2f)}**")
#         st.success(f"The predicted IMDb rating is: **{predicted_rating:.2f}**")

#     except Exception as e:
#         st.error(f"something went wrong: {str(e)}")    
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load models and data
xgb_model = pickle.load(open('movie_rating_model.pkl', 'rb'))
sentence_model = pickle.load(open('sentence_model.pkl', 'rb'))
avg_actor_rating = pickle.load(open("avg_actor_rating.pkl", 'rb'))
genre_columns = pickle.load(open('genre_columns_model.pkl', 'rb'))
avg_director_rating = pickle.load(open('avg_director_rating.pkl', 'rb'))

# Load dataset for dropdown values
df = pd.read_csv("data/IMDb Movies India.csv",encoding='latin1')  # Replace with your dataset

# Get unique directors and actors
all_directors = df['Director'].dropna().unique()
# Extract unique actors and remove NaNs or non-string values
all_actors = pd.unique(df[['Actor 1', 'Actor 2', 'Actor 3']].values.ravel('K'))
all_actors = [actor for actor in all_actors if isinstance(actor, str)]


st.title("🎬 Movie Rating Prediction")
st.markdown("Enter movie details below to predict the rating!")

# Genre multi-select
selected_genres = st.multiselect("Select genre(s)", genre_columns)

# Dropdowns for director and actors
director = st.selectbox("Select Director", sorted(all_directors))
actor1 = st.selectbox("Select Actor 1", sorted(all_actors))
actor2 = st.selectbox("Select Actor 2", sorted(all_actors))
actor3 = st.selectbox("Select Actor 3", sorted(all_actors))

# Predict button
if st.button("Predict Rating"):
    try:
        genre_vector = [1 if genre in selected_genres else 0 for genre in genre_columns]
        actor_list = [actor1, actor2, actor3]
        actor_rating = np.mean([avg_actor_rating.get(actor, 5.0) for actor in actor_list])
        director_rating = avg_director_rating.get(director, 5.0)

        final_features = np.array(genre_vector + [actor_rating, director_rating]).reshape(1, -1)
        predicted_rating = xgb_model.predict(final_features)[0]

        st.success(f"The predicted IMDb rating is: **{predicted_rating:.2f}**")

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
