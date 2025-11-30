import pandas as pd
import streamlit as st
import plotly.express as px


# ----------------------------------------------------------
# 1. DATASET
# ----------------------------------------------------------

results = {
    'Model': [
        'DenseNet121 (Color)', 'ResNet50 (Color)', 'VGG16 (Color)',
        'MobileNetV2 (Color)', 'InceptionV3 (Color)',
        'DenseNet121 (Grayscale)', 'ResNet50 (Grayscale)',
        'VGG16 (Grayscale)', 'MobileNetV2 (Grayscale)',
        'InceptionV3 (Grayscale)'
    ],
    'Accuracy': [
        0.9107, 0.9196, 0.8839, 0.8973, 0.8750,
        0.5223, 0.5223, 0.5223, 0.5223, 0.5223
    ],
    'Loss': [
        0.2151, 0.1845, 0.2809, 0.2759, 0.3006,
        0.6916, 0.6922, 0.6921, 0.6922, 0.6922
    ],
    'Training Time (minutes)': [
        7.09, 8.98, 6.58, 6.81, 9.90,
        17.39, 13.18, 16.77, 4.45, 13.57
    ]
}

results_df = pd.DataFrame(results)


# ----------------------------------------------------------
# 2. PLOTTING FUNCTION
# ----------------------------------------------------------

def plot_accuracy_comparison_plotly(df):

    # Extract base model + image type
    extracted = df['Model'].str.extract(r'(.+)\s\((Color|Grayscale)\)')
    extracted.columns = ['Base Model', 'Image Type']

    # Combine with accuracy
    plot_df = pd.concat([extracted, df[['Accuracy']]], axis=1)

    # Create Plotly chart
    fig = px.bar(
        plot_df,
        x='Base Model',
        y='Accuracy',
        color='Image Type',
        barmode='group',
        height=600,
        color_discrete_sequence=px.colors.sequential.Spectral
    )

    fig.update_layout(
        title="Accuracy Comparison: Color vs Grayscale CNN Models",
        xaxis_title="Base Model",
        yaxis_title="Accuracy",
        title_x=0.5,
        legend_title="Image Type"
    )

    fig.update_xaxes(tickangle=45)

    return fig


# ----------------------------------------------------------
# 3. STREAMLIT PAGE RENDERING
# ----------------------------------------------------------

st.set_page_config(layout="wide")

st.title("Model Performance Results 📊")
st.subheader("Comparison of CNNs on Color vs. Grayscale Images")

# --- Display Table ---
st.dataframe(results_df, use_container_width=True)

st.markdown("---")
st.caption("This data frame can be used as input for plotting.")

# --- Display Chart ---
st.header("📊 Accuracy Comparison (Plotly Bar Chart)")
chart = plot_accuracy_comparison_plotly(results_df)
st.plotly_chart(chart, use_container_width=True)
