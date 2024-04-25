import streamlit as st
import pandas as pd
import pickle

# Load the Random Forest model from the pickle file
try:
    model = pickle.load(open('grid_search_rf_model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading the model: {str(e)}")



def predict_churn_status(input_data):
    # Preprocess the input data
    
    #st.write(input_data)
    # Make predictions using the loaded model
    try:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        return "Approved" if prediction[0] == 1 else "Rejected"
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
    # probability = model.predict_proba(input_df)[:, 1]

    
    # return prediction[0], probability[0]

# Create the Streamlit app

def main():
    st.title("Telecom Churn Prediction")
    st.write("Enter the customer details below to predict churn.")

    # Create input fields for user input
    tenure = st.slider("Tenure (months)", 0, 100, 1)
    gender = st.selectbox("Gender", [0, 1])
    st.write("0: Female, 1: Male")
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    st.write("0: No, 1: Yes")
    Partner = st.selectbox("Partner", [0, 1])
    st.write("0: No, 1: Yes")
    Dependents = st.selectbox("Dependents", [0, 1])
    st.write("0: No, 1: Yes")
    MultipleLines = st.selectbox("Multiple Lines", [0, 1])
    st.write("0: No, 1: Yes")
    PhoneService = st.selectbox("Phone Service", [0, 1])
    st.write("0: No, 1: Yes")
    OnlineSecurity = st.selectbox("Online Security", [0, 1])
    st.write("0: No, 1: Yes")
    OnlineBackup = st.selectbox("Online Backup", [0, 1])
    st.write("0: No, 1: Yes")
    DeviceProtection = st.selectbox("Device Protection", [0, 1])
    st.write("0: No, 1: Yes")
    TechSupport = st.selectbox("Tech Support", [0, 1])
    st.write("0: No, 1: Yes")
    StreamingTV = st.selectbox("Streaming TV", [0, 1])
    st.write("0: No, 1: Yes")
    StreamingMovies = st.selectbox("Streaming Movies", [0, 1])
    st.write("0: No, 1: Yes")
    PaperlessBilling = st.selectbox("Paperless Billing", [0, 1])
    st.write("0: No, 1: Yes")
    InternetService_DSL = st.selectbox("InternetService_DSL", [0, 1])
    st.write("0: No, 1: Yes")
    InternetService_Fiber = st.selectbox("InternetService_Fiber Optics", [0, 1])
    st.write("0: No, 1: Yes")
    InternetService_No = st.selectbox("No InternetService", [0, 1])
    st.write("0: No, 1: Yes")
    Contract_Month = st.selectbox("Contract_Month to Month", [0, 1])
    st.write("0: No, 1: Yes")
    Contract_One = st.selectbox("Contract_One Year", [0, 1])
    st.write("0: No, 1: Yes")
    Contract_Two = st.selectbox("Contract_Two Year", [0, 1])
    st.write("0: No, 1: Yes")
    PaymentMethod_Bank = st.selectbox("PaymentMethod_Bank Tranfer", [0, 1])
    st.write("0: No, 1: Yes")
    PaymentMethod_Credit = st.selectbox("PaymentMethod_Credit Card", [0, 1])
    st.write("0: No, 1: Yes")
    PaymentMethod_Electronic = st.selectbox("PaymentMethod_Electronic Check", [0, 1])
    st.write("0: No, 1: Yes")
    PaymentMethod_Mailed = st.selectbox("PaymentMethod_Mailed Check", [0, 1])
    st.write("0: No, 1: Yes")
    MonthlyCharges = st.number_input("Monthly Charges")
    TotalCharges = st.number_input("Total Charges")


    
    # Create a dictionary to store the user input
    input_data = {
        'tenure': tenure,
        'gender':gender,
        'SeniorCitizen':SeniorCitizen,
        'Partner':Partner,
        'Dependents':Dependents,
        'MultipleLines':MultipleLines,
        'PhoneService':PhoneService,
        'OnlineSecurity':OnlineSecurity,
        'OnlineBackup':OnlineBackup,
        'DeviceProtection':DeviceProtection,
        'TechSupport':TechSupport,
        'StreamingTV':StreamingTV,
        'StreamingMovies':StreamingMovies,
        'PaperlessBilling':PaperlessBilling,
        'InternetService_DSL':InternetService_DSL,
        'InternetService_Fiber optic':InternetService_Fiber,
        'InternetService_No':InternetService_No,
        'Contract_Month-to-month':Contract_Month,
        'Contract_One year':Contract_One,
        'Contract_Two year':Contract_Two,
        'PaymentMethod_Bank transfer (automatic)':PaymentMethod_Bank,
        'PaymentMethod_Credit card (automatic)':PaymentMethod_Credit,
        'PaymentMethod_Electronic check':PaymentMethod_Electronic,
        'PaymentMethod_Mailed check':PaymentMethod_Mailed,
        'MonthlyCharges':MonthlyCharges,
        'TotalCharges':TotalCharges
    }

    # Button to trigger predictions
    if input_data is not None and st.button('Churn Prediction Status'):
        # Make predictions and get labels
        prediction_label = predict_churn_status(input_data)

        # Display prediction label
        st.subheader('Prediction:')

        if prediction_label == "Approved":
            st.success("Churned Customer")
            st.write("Model Accuracy: 86%")
        else:
            st.error("Not a Churned Customer")
            st.write("Model Accuracy: 86%")


# Run the Streamlit app
if __name__ == '__main__':
    main()