import pandas as pd 
import warnings
import streamlit as st
import joblib
from sklearn.preprocessing import FunctionTransformer
import sklearn


def preprocessing (x):
    x['NO_Trainings_LstYear'] = x['NO_Trainings_LstYear'].apply(lambda c : '1' if c == 1 else '>1')
    EDU_Mapper = joblib.load('EDU_Mapper.h5')
    x['Education Level'] = x['Education Level'].map(EDU_Mapper)
    return x 


Model = joblib.load('XGB.h5')
Model.steps.insert(0 , ('preprocessing' , FunctionTransformer(preprocessing)))

def Predict(Department, Region_Employment, Education_Level,
       Gender, Recruitment_Channel, NO_Trainings_LstYear, Age,
       previous_year_rating, Service_Length, Awards,
       Avg_Training_Score):
    test = pd.DataFrame(columns=['Department', 'Region_Employment', 'Education Level',
       'Gender', 'Recruitment Channel', 'NO_Trainings_LstYear', 'Age',
       'previous_year_rating', 'Service Length', 'Awards',
       'Avg_Training_Score'])
    
    test.at[0,['Department']] = Department
    test.at[0,['Region_Employment']] = Region_Employment
    test.at[0,['Education Level']] = Education_Level
    test.at[0,['Gender']] = Gender
    test.at[0,['Recruitment Channel']] = Recruitment_Channel
    test.at[0,['NO_Trainings_LstYear']] = NO_Trainings_LstYear
    test.at[0,['Age']] = Age
    test.at[0,['previous_year_rating']] = previous_year_rating
    test.at[0,['Service Length']] = Service_Length
    test.at[0,['Awards']] = Awards
    test.at[0,['Avg_Training_Score']] = Avg_Training_Score
        
    return Model.predict(test)[0]


def main():
    st.header('Employees Promotion Classifier App'.capitalize())
    
    Department = st.selectbox('what is your Department'.title() ,['Sales & Marketing', 'Operations', 'Technology',
                                                                  'Analytics','R&D', 'Procurement', 'Finance', 'HR',
                                                                  'Legal'] )
    
    
    NO_Trainings_LstYear = st.slider('Number of trainings last year'.title() ,
                                     min_value = 1 , max_value=10 , value=2 , step=1)
    
    
    Region_Employment = st.selectbox('what is your Region Employment'.title() ,
                                    ['region_7', 'region_22', 'region_19', 'region_23', 'region_26',
       'region_2', 'region_20', 'region_34', 'region_1', 'region_4',
       'region_29', 'region_31', 'region_15', 'region_14', 'region_11',
       'region_5', 'region_28', 'region_17', 'region_13', 'region_16',
       'region_25', 'region_10', 'region_27', 'region_30', 'region_12',
       'region_21', 'region_8', 'region_32', 'region_6', 'region_33',
       'region_24', 'region_3', 'region_9', 'region_18'])
    
    
    Education_Level = st.selectbox('what is your Education Level'.title() ,
                                   ["Master's & above", "Bachelor's",'Below Secondary'])
    
    
    Gender = st.selectbox('what is your Gender'.title() , ['f', 'm'])
    
    
    Recruitment_Channel = st.selectbox('what is your Recruitment Channel'.title() 
                                       , ['sourcing', 'other','referred'])
    
    
    Avg_Training_Score = st.slider('what is your Avgerage Training Score'.title()
                                   ,min_value = 39.0 , max_value=99.0 , value=60.0 , step=0.5)
    
    
    Age = st.slider('what is your age'.title() ,min_value = 16.0 , max_value=60.0 , value=20.0 , step=1.0)
    
    
    previous_year_rating = st.slider('what is your previous year rating'.title() ,min_value = 1 , max_value=5 , 
                                     value=3 , step=1 )
    
    Service_Length = st.slider('what is your Service Length'.title()
                               ,min_value = 1 , max_value=37 , value=20 , step=1)
    
    
    Awards = st.selectbox('are you have any Awards'.title() ,['NO', 'YES'])
    
    if st.button('predict'.title()):
        ans = Predict(Department, Region_Employment, Education_Level,
       Gender, Recruitment_Channel, NO_Trainings_LstYear, Age,
       previous_year_rating, Service_Length, Awards,
       Avg_Training_Score)
        st.write(ans)
main()
