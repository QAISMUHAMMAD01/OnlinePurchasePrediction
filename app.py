import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Online Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("online_shoppers_50000.csv")

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🛒 Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Dataset", "Prediction"]
)

# ===========================================================
# HOME
# ===========================================================

if page == "Home":

    st.title("🛒 Online Purchase Prediction")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Total Features", df.shape[1]-1)
    col3.metric("Target", "Revenue")

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

# ===========================================================
# DATASET
# ===========================================================

elif page == "Dataset":

    st.title("📊 Dataset Analysis")

    st.write(df)

    st.write("### Shape")

    st.write(df.shape)

    st.write("### Missing Values")

    st.write(df.isnull().sum())

    st.write("### Statistics")

    st.write(df.describe())

# ===========================================================
# PREDICTION
# ===========================================================

else:

    st.title("🤖 Purchase Prediction")

    col1, col2 = st.columns(2)

    with col1:

        Administrative = st.number_input("Administrative", 0, 30, 0)

        Administrative_Duration = st.number_input(
            "Administrative Duration",
            0.0,
            5000.0,
            0.0
        )

        Informational = st.number_input(
            "Informational",
            0,
            30,
            0
        )

        Informational_Duration = st.number_input(
            "Informational Duration",
            0.0,
            5000.0,
            0.0
        )

        ProductRelated = st.number_input(
            "Product Related",
            0,
            1000,
            1
        )

        ProductRelated_Duration = st.number_input(
            "Product Related Duration",
            0.0,
            100000.0,
            10.0
        )

    with col2:

        BounceRates = st.number_input(
            "Bounce Rates",
            0.0,
            1.0,
            0.02
        )

        ExitRates = st.number_input(
            "Exit Rates",
            0.0,
            1.0,
            0.05
        )

        PageValues = st.number_input(
            "Page Values",
            0.0,
            500.0,
            0.0
        )

        SpecialDay = st.slider(
            "Special Day",
            0.0,
            1.0,
            0.0
        )

        Month = st.selectbox(
            "Month",
            encoders["Month"].classes_
        )

        OperatingSystems = st.selectbox(
            "Operating System",
            [1,2,3,4,5,6,7,8]
        )

        Browser = st.selectbox(
            "Browser",
            [1,2,3,4,5,6,7,8,9,10,11,12,13]
        )

        Region = st.selectbox(
            "Region",
            [1,2,3,4,5,6,7,8,9]
        )

        TrafficType = st.selectbox(
            "Traffic Type",
            list(range(1,21))
        )

        VisitorType = st.selectbox(
            "Visitor Type",
            encoders["VisitorType"].classes_
        )

        Weekend = st.selectbox(
            "Weekend",
            [False, True]
        )

        # Encode categorical values
    Month = encoders["Month"].transform([Month])[0]
    VisitorType = encoders["VisitorType"].transform([VisitorType])[0]
    Weekend = encoders["Weekend"].transform([Weekend])[0]

    if st.button("🔮 Predict Purchase", use_container_width=True):

        # IMPORTANT: Column order must match train_model.py
        data = pd.DataFrame([[
            Administrative,
            Administrative_Duration,
            Informational,
            Informational_Duration,
            ProductRelated,
            ProductRelated_Duration,
            BounceRates,
            ExitRates,
            PageValues,
            SpecialDay,
            OperatingSystems,
            Browser,
            Region,
            TrafficType,
            Month,
            VisitorType,
            Weekend
        ]], columns=[
            "Administrative",
            "Administrative_Duration",
            "Informational",
            "Informational_Duration",
            "ProductRelated",
            "ProductRelated_Duration",
            "BounceRates",
            "ExitRates",
            "PageValues",
            "SpecialDay",
            "OperatingSystems",
            "Browser",
            "Region",
            "TrafficType",
            "Month",
            "VisitorType",
            "Weekend"
        ])

        # Prediction
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0]

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Purchase Probability",
                f"{probability[1]*100:.2f}%"
            )

        with col2:
            st.metric(
                "No Purchase Probability",
                f"{probability[0]*100:.2f}%"
            )

        st.write("Prediction Value:", prediction)

        if prediction == 1:
            st.success("✅ Customer is likely to PURCHASE.")
            st.balloons()
        else:
            st.success("✅ Customer is likely to PURCHASE.")