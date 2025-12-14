import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download


MODEL_REPO_ID = "BujjiProjectPrep/Tourism-Customer-Prediction-1212"
MODEL_FILENAME = "best_tourist_customer_xgb_model.joblib"


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model",
    )
    model = joblib.load(model_path)
    return model

def main():
    st.title("Tourist Customer Wellness Package Purchase Prediction")
    st.write(
        "This app predicts whether a customer is likely to purchase the "
        "Wellness Tourism Package for the company 'Visit With Us'."
    )

    model = load_model()

    st.header("Enter Customer Details")

    # Collect inputs for all model features
    Age = st.number_input("Age", min_value=18, max_value=100, value=35)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    DurationOfPitch = st.number_input(
        "Duration of Pitch (minutes)", min_value=0.0, max_value=120.0, value=15.0, step=1.0
    )

    Occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Free Lancer", "Small Business", "Large Business", "Govt", "Other"],
    )

    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input(
        "Number of Persons Visiting", min_value=1, max_value=20, value=2, step=1
    )
    NumberOfFollowups = st.number_input(
        "Number of Followups", min_value=0.0, max_value=20.0, value=3.0, step=1.0
    )
    ProductPitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],
    )
    PreferredPropertyStar = st.selectbox(
        "Preferred Property Star", [1.0, 2.0, 3.0, 4.0, 5.0]
    )
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    NumberOfTrips = st.number_input(
        "Number of Trips per Year", min_value=0.0, max_value=50.0, value=1.0, step=1.0
    )
    Passport = st.selectbox("Passport (0 = No, 1 = Yes)", [0, 1])
    PitchSatisfactionScore = st.selectbox(
        "Pitch Satisfaction Score (1 = lowest, 5 = highest)", [1, 2, 3, 4, 5]
    )
    OwnCar = st.selectbox("Own Car (0 = No, 1 = Yes)", [0, 1])
    NumberOfChildrenVisiting = st.number_input(
        "Number of Children Visiting", min_value=0.0, max_value=10.0, value=0.0, step=1.0
    )
    Designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
    )
    MonthlyIncome = st.number_input(
        "Monthly Income", min_value=0.0, max_value=1000000.0, value=50000.0, step=1000.0
    )

    # Build input dictionary
    input_dict = {
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome,
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    st.subheader("Input Preview")
    st.dataframe(input_df)

    if st.button("Predict Purchase Likelihood"):
        proba = model.predict_proba(input_df)[0, 1]
        pred = model.predict(input_df)[0]

        st.write(f"**Predicted Probability of Purchase:** {proba:.2f}")

        if pred == 1:
            st.success(
                "The model predicts that this customer is **LIKELY** to purchase the Wellness Package."
            )
        else:
            st.warning(
                "The model predicts that this customer is **UNLIKELY** to purchase the Wellness Package."
            )

if __name__ == "__main__":
    main()
