import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from DLR import read_config,get_prediction_models
import os

##State
if 'models_exist' not in st.session_state:
    st.session_state.models_exist = False

if 'models_training' not in st.session_state:
    st.session_state.models_training = False

# Create the directory
os.makedirs('models', exist_ok=True)

def models_exist():
    return len(os.listdir('models'))>=1
def delete_models():
    try:
        # List all items in the directory
        items = os.listdir('models')
        for item in items:
            item_path = os.path.join('models', item)
            if os.path.isfile(item_path):
                os.remove(item_path)
    except:
        print('Error')
    return 0
# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Topology","Historical Data", "Server Configuration","Results"])

with tab1:
    # Set the title of the app
    st.title("Topology File Upload")
    # Create a file uploader widget
    uploaded_file = st.file_uploader("Choose a excel file", type=["xlsx", "xls"], key=2)

with tab2:
    # Set the title of the app
    st.title("Historical Data Upload")
    # Create a file uploader widget
    uploaded_file2 = st.file_uploader("Choose a csv file", type=["csv"], key=1)
    if (uploaded_file2 is not None):
        st.write(pd.read_csv(uploaded_file2).head())
    st.subheader("Machine Learning Models Status:")
    if st.session_state.models_exist:
        st.write('Models trained')
        if st.button('Delete Models', key=3):
            delete_models()
            st.session_state.models_exist = False
    else:
        if models_exist():
            st.session_state.models_exist = True
            st.write('Models trained')
        if not(st.session_state.models_exist):
            st.write('Models not trained')
            if (uploaded_file is None):
                st.write('Upload Topology Data')
            if (uploaded_file2 is None):
                st.write('Upload Historical Data')
            if (uploaded_file is not None) & (uploaded_file2 is not None):
                if st.button('Train Models', key=4):
                    uploaded_file2.seek(0)
                    st.session_state.models_training = True
                    if st.session_state.models_training :
                        st.write('Model Training ...')
                    models = get_prediction_models(uploaded_file,uploaded_file2)
                    st.session_state.models_training = False
                    st.session_state.models_exist = True

with tab3:
    st.title("MQQT Connection")

    st.subheader("Server Address:")
    IP = st.text_input("Host IP", '')
    P = st.text_input("Host Port", '')
    topic_reading = st.text_input("Measurements Reading Topic", '')
    topic_write = st.text_input("Results Writing Topic", '')
    st.subheader("Credentials:")

    name = st.text_input("Username", '')
    password = st.text_input("Password", '')


with tab4:
    st.subheader("Line Loading Forecasts")
    try:
        with open('loading_plot.html', 'r', encoding='utf-8') as file:
            html_content = file.read()
        components.html(html_content, width=800, height=400,  scrolling=True)
    except FileNotFoundError:
        st.error("Results not found.")
    ##########
    st.subheader("Conductor Temperature Forecasts")
    try:
        with open('temperature_plot.html', 'r', encoding='utf-8') as file:
            html_content = file.read()
        components.html(html_content, width=800, height=400, scrolling=True)
    except FileNotFoundError:
        st.error("Results not found.")
