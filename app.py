import streamlit as st
import pandas as pd
import joblib

# Load the data and models
df1 = pd.read_csv('Bihar Assembly Election - AE.csv')
loaded_catboost_model = joblib.load('catboost_model.pkl')
loaded_logistic_model = joblib.load('logistic_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

def predict_votes_and_win_prob(ac_no, party, alliance):
    # Predict normalized votes using CatBoost
    input_data = pd.DataFrame({
        'AC No': [ac_no],
        'Party': [party],
        'Alliance': [alliance]
    })
    
    # Predict normalized votes
    predicted_normalized_votes = loaded_catboost_model.predict(input_data)
    
    # Factor for valid votes
    factor = df1[(df1['AC No'] == ac_no) & (df1['Year'] == 2020)]['Valid Votes'].values[0] * 1.1
    predicted_vote_count = round(predicted_normalized_votes[0] * factor)
    
    # Use the predicted normalized votes in Logistic Regression to predict win probability
    normalized_votes_scaled = loaded_scaler.transform([[predicted_normalized_votes[0]]])
    win_prob = loaded_logistic_model.predict_proba(normalized_votes_scaled)
    
    # Return both predicted normalized votes (adjusted) and win probabilities
    return {
        'Predicted Normalized Votes (adjusted)': predicted_vote_count,
        'Prob of Loss (Win=0)': win_prob[0][0],
        'Prob of Win (Win=1)': win_prob[0][1]
    }

# Streamlit app
st.title('Bihar Election Prediction')

# Input fields
ac_no = st.number_input('AC No.', min_value=1, max_value=243, step=1)

parties = df1['Party'].unique().tolist()
party = st.selectbox('Party', parties)

alliances = ['NDA', 'MGB', 'Other', 'GDSF', 'Unknown']
alliance = st.selectbox('Alliance', alliances)

if st.button('Predict'):
    if alliance == 'Unknown':
        alliance = None
    results = predict_votes_and_win_prob(ac_no, party, alliance)
    
    st.header('Prediction Results:')
    st.subheader(f"Predicted Vote Count: {results['Predicted Normalized Votes (adjusted)']:,}")
    st.subheader(f"Probability of Loss: {results['Prob of Loss (Win=0)']:.2%}")
    st.subheader(f"Probability of Win: {results['Prob of Win (Win=1)']:.2%}")
    
    # Visualize win probability
    import plotly.graph_objects as go
    
    fig = go.Figure(go.Bar(
        x=['Loss', 'Win'],
        y=[results['Prob of Loss (Win=0)'], results['Prob of Win (Win=1)']],
        text=[f"{results['Prob of Loss (Win=0)']:.2%}", f"{results['Prob of Win (Win=1)']:.2%}"],
        textposition='auto',
        marker_color=['red', 'green']
    ))
    fig.update_layout(title='Win Probability', yaxis_title='Probability')
    st.plotly_chart(fig)

st.info('Note: This prediction is based on the trained CatBoost and Logistic Regression models.')