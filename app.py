import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Machine Learning Application")

    # Display the image
    #st.sidebar.image("./static/images/logo.jpeg", use_column_width=True)

    # Collecting inputs from the user
    st.sidebar.header("User Input Parameters")

    # BMI input
    bmi = st.sidebar.slider("BMI", min_value=12.0, max_value=98.0, value=25.0, step=0.1)

    # PhysHlth input
    phys_hlth = st.sidebar.slider("PhysHlth", min_value=0, max_value=30, value=15, step=1)

    # HighBP input
    high_bp = 1 - st.sidebar.checkbox("HighBP")  # Inverted

    # HighChol input
    high_chol = 1 - st.sidebar.checkbox("HighChol")  # Inverted

    # GenHlth input
    gen_hlth_options = ['fair', 'poor', 'good', 'None']
    gen_hlth = st.sidebar.selectbox("GenHlth", options=gen_hlth_options, index=3)
    gen_hlth_fair = 1 if gen_hlth == 'fair' else 0
    gen_hlth_poor = 1 if gen_hlth == 'poor' else 0
    gen_hlth_good = 1 if gen_hlth == 'good' else 0

    # DiffWalk input
    diff_walk = int(st.sidebar.checkbox("DiffWalk"))

    # Income input
    income = int(st.sidebar.checkbox("Income > 75k"))

    # Create DataFrame
    data = {
        'BMI': [bmi],
        'PhysHlth': [phys_hlth],
        'HighBP_no high BP': [high_bp],
        'HighChol_no high cholesterol': [high_chol],
        'GenHlth_fair': [gen_hlth_fair],
        'GenHlth_good': [gen_hlth_good],
        'GenHlth_poor': [gen_hlth_poor],
        'DiffWalk_1.0': [diff_walk],
        'Income_> 75k': [income]
    }
    df = pd.DataFrame(data)

    # Display transposed DataFrame
    st.write(df.T)

 # Load the trained model
    with open("./models/trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Predict probability using model
    probability = model.predict_proba(df)[0][1]
    
    # Determine prediction based on Youden's threshold
    threshold = 0.5022826695304432
    prediction = probability > threshold

    # Display message based on prediction
    if prediction:
        st.write("The prediction is Positive.")
    else:
        st.write("The prediction is Negative.")

    # Display probability as pie chart
    fig, ax = plt.subplots()
    ax.pie([probability, 1-probability], labels=['Positive', 'Negative'], autopct='%1.2f%%')
    st.pyplot(fig)

    # Display SHAP force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)
    st_shap = st.pyplot(shap.force_plot(explainer.expected_value[1], shap_values[1], df, matplotlib=True))

    # Display warning in red
    st.markdown('**THIS IS NOT A SCIENTIFICALLY VALIDATED TOOL! BUILT FOR TEACHING PURPOSES ONLY!**', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
