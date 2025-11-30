import pandas as pd
import streamlit as st # <-- Necessary import for Streamlit

# --- 1. Data Preparation (Optimized) ---

# Define the results using a more compact list-of-lists or dictionary-of-lists structure.
# This is cleaner and less repetitive than appending line-by-line.
results = {
    'Model': [
        'DenseNet121 (Color)', 'ResNet50 (Color)', 'VGG16 (Color)', 'MobileNetV2 (Color)', 'InceptionV3 (Color)', 
        'DenseNet121 (Grayscale)', 'ResNet50 (Grayscale)', 'VGG16 (Grayscale)', 'MobileNetV2 (Grayscale)', 'InceptionV3 (Grayscale)'
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

# Create the DataFrame
results_df = pd.DataFrame(results)

# --- 2. Streamlit Display ---

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("Model Performance Results Table 📊")
    st.subheader("Comparison of CNNs on Color vs. Grayscale Images")
    
    # Display the table using st.dataframe()
    # This renders an interactive, searchable table in your Streamlit app.
    st.dataframe(results_df, use_container_width=True)
    
    st.markdown("---")
    st.caption("This data frame can be used as input for plotting.")
