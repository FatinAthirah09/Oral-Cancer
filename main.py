import pandas as pd
import streamlit as st
import plotly.express as px

# =========================================================
# 1. DATASET
# =========================================================
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

# =========================================================
# 2. STREAMLIT PAGE SETTINGS
# =========================================================
st.set_page_config(layout="wide")
st.title("Model Performance Dashboard 📊")
st.subheader("Comparison of CNNs on Color vs. Grayscale Images")

# --- Display Table ---
st.dataframe(results_df, use_container_width=True)

# --- Hyperparameters Section ---
st.markdown("---")
st.subheader("⚙️ Hyperparameters Used")

hyperparams = {
    "Batch Size": 16,
    "Epochs": 50,
    "Learning Rate": 0.001,
    "Patience": 8,
    "Validation Split": 0.2
}

# Convert to DataFrame for pretty display
hyperparams_df = pd.DataFrame(list(hyperparams.items()), columns=["Parameter", "Value"])
st.table(hyperparams_df)

st.markdown("---")
st.caption("The table shows Accuracy, Loss, and Training Time for each model.")


# =========================================================
# 3. PLOTLY BAR CHART FUNCTIONS
# =========================================================

def plot_accuracy(df):
    extracted = df['Model'].str.extract(r'(.+)\s\((Color|Grayscale)\)')
    extracted.columns = ['Base Model', 'Image Type']
    plot_df = pd.concat([extracted, df[['Accuracy']]], axis=1)

    fig = px.bar(
        plot_df,
        x='Base Model',
        y='Accuracy',
        color='Image Type',
        barmode='group',
        height=500,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        title="Accuracy Comparison",
        xaxis_title="Base Model",
        yaxis_title="Accuracy",
        title_x=0.5,
        legend_title="Image Type"
    )
    fig.update_xaxes(tickangle=45)
    return fig

def plot_loss(df):
    extracted = df['Model'].str.extract(r'(.+)\s\((Color|Grayscale)\)')
    extracted.columns = ['Base Model', 'Image Type']
    plot_df = pd.concat([extracted, df[['Loss']]], axis=1)

    fig = px.bar(
        plot_df,
        x='Base Model',
        y='Loss',
        color='Image Type',
        barmode='group',
        height=500,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        title="Loss Comparison (Lower is Better)",
        xaxis_title="Base Model",
        yaxis_title="Loss",
        title_x=0.5,
        legend_title="Image Type"
    )
    fig.update_xaxes(tickangle=45)
    fig.add_hline(
        y=0.693,
        line_dash="dot",
        annotation_text="Random Guess BCE Loss (0.693)",
        annotation_position="top left"
    )
    return fig

def plot_training_time(df):
    extracted = df['Model'].str.extract(r'(.+)\s\((Color|Grayscale)\)')
    extracted.columns = ['Base Model', 'Image Type']
    plot_df = pd.concat([extracted, df[['Training Time (minutes)']]], axis=1)

    fig = px.bar(
        plot_df,
        x='Base Model',
        y='Training Time (minutes)',
        color='Image Type',
        barmode='group',
        height=500,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        title="Training Time Comparison",
        xaxis_title="Base Model",
        yaxis_title="Training Time (minutes)",
        title_x=0.5,
        legend_title="Image Type"
    )
    fig.update_xaxes(tickangle=45)
    return fig


# =========================================================
# 4. RENDER CHARTS IN STREAMLIT
# =========================================================

st.header("📊 Accuracy Comparison")
st.plotly_chart(plot_accuracy(results_df), use_container_width=True)

st.header("📉 Loss Comparison")
st.plotly_chart(plot_loss(results_df), use_container_width=True)

st.header("⏱ Training Time Comparison")
st.plotly_chart(plot_training_time(results_df), use_container_width=True)
