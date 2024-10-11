import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model():
    with open("pickled/modelAttrition.pkl", 'rb') as pFile:
        model = pickle.load(pFile)
    return model

def prep_data(input_data:dict)->pd.DataFrame:
    df = pd.DataFrame(input_data, index=[1])

    with open('pickled/labelEncoder.pkl', 'rb') as enc_file:
        encoder = pickle.load(enc_file)
        nonint_attributes = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

        for attribute in nonint_attributes:
            df[attribute] = encoder.fit_transform(df[attribute])

    return df

model = load_model()

st.title("Employee Attrition Prediction")

with st.form("attrition_form"):
    st.header("Enter Employee Information")

    # Input fields for all the variables
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    business_travel = st.selectbox('Business Travel', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
    daily_rate = st.number_input('Daily Rate', min_value=0, max_value=1000, value=500)
    department = st.selectbox('Department', ['Sales', 'Research & Development', 'Human Resources'])
    distance_from_home = st.number_input('Distance From Home (miles)', min_value=0, max_value=100, value=10)
    education = st.select_slider('Education', [1, 2, 3, 4, 5])
    education_field = st.selectbox('Education Field', ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'])
    environment_satisfaction = st.select_slider('Environment Satisfaction', [1, 2, 3, 4])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    hourly_rate = st.number_input('Hourly Rate', min_value=0, max_value=100, value=50)
    job_involvement = st.select_slider('Job Involvement', [1, 2, 3, 4])
    job_level = st.select_slider('Job Level', [1, 2, 3, 4, 5])
    job_role = st.selectbox('Job Role', ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
    job_satisfaction = st.select_slider('Job Satisfaction', [1, 2, 3, 4])
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    monthly_income = st.number_input('Monthly Income', min_value=0, max_value=20000, value=5000)
    monthly_rate = st.number_input('Monthly Rate', min_value=0, max_value=20000, value=10000)
    num_companies_worked = st.number_input('Num Companies Worked', min_value=0, max_value=10, value=1)
    overtime = st.toggle('OverTime', value=False)
    percent_salary_hike = st.number_input('Percent Salary Hike', min_value=0, max_value=100, value=10)
    performance_rating = st.select_slider('Performance Rating', [1, 2, 3, 4])
    relationship_satisfaction = st.select_slider('Relationship Satisfaction', [1, 2, 3, 4])
    stock_option_level = st.select_slider('Stock Option Level', [0, 1, 2, 3])
    total_working_years = st.number_input('Total Working Years', min_value=0, max_value=40, value=5)
    training_times_last_year = st.number_input('Training Times Last Year', min_value=0, max_value=10, value=3)
    work_life_balance = st.select_slider('Work Life Balance', [1, 2, 3, 4])
    years_at_company = st.number_input('Years at Company', min_value=0, max_value=50, value=5)
    years_in_current_role = st.number_input('Years in Current Role', min_value=0, max_value=20, value=3)
    years_since_last_promotion = st.number_input('Years Since Last Promotion', min_value=0, max_value=20, value=2)
    years_with_curr_manager = st.number_input('Years with Current Manager', min_value=0, max_value=20, value=3)

    # Submit button
    submitted = st.form_submit_button("Predict Attrition")

    if submitted:
        input_data = {
            'Age': age,
            'BusinessTravel': business_travel,
            'DailyRate': daily_rate,
            'Department': department,
            'DistanceFromHome': distance_from_home,
            'Education': education,
            'EducationField': education_field,
            'EnvironmentSatisfaction': environment_satisfaction,
            'Gender': gender,
            'HourlyRate': hourly_rate,
            'JobInvolvement': job_involvement,
            'JobLevel': job_level,
            'JobRole': job_role,
            'JobSatisfaction': job_satisfaction,
            'MaritalStatus': marital_status,
            'MonthlyIncome': monthly_income,
            'MonthlyRate': monthly_rate,
            'NumCompaniesWorked': num_companies_worked,
            'OverTime': overtime,
            'PercentSalaryHike': percent_salary_hike,
            'PerformanceRating': performance_rating,
            'RelationshipSatisfaction': relationship_satisfaction,
            'StockOptionLevel': stock_option_level,
            'TotalWorkingYears': total_working_years,
            'TrainingTimesLastYear': training_times_last_year,
            'WorkLifeBalance': work_life_balance,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_current_role,
            'YearsSinceLastPromotion': years_since_last_promotion,
            'YearsWithCurrManager': years_with_curr_manager
        }
        
        input_df = prep_data(input_data)

        prediction = model.predict(input_df)
        if prediction[0]:
            st.success("The predicted attrition is: Likely")
        else:
            st.success("The predicted attrition is: Unlikely")
