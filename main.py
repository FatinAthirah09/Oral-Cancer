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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# --- 1. Data Preparation (Already Optimized) ---

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

results_df = pd.DataFrame(results)


# --- 2. Plotting Function (Streamlit Version) ---

def plot_accuracy_comparison(results_df):
    # Extract base model + type
    extracted = results_df['Model'].str.extract(r'(.+)\s\((Color|Grayscale)\)')
    extracted.columns = ['Base Model', 'Image Type']

    # Combine with accuracy values
    plot_df = pd.concat([extracted, results_df[['Accuracy']]], axis=1)

    # Pivot so each model has two accuracy values
    comparison_df = plot_df.pivot(index='Base Model', columns='Image Type', values='Accuracy').reset_index()

    # Rename columns
    comparison_df.columns = ['Base Model', 'Color Accuracy', 'Grayscale Accuracy']

    # Melt for seaborn
    melted_df = comparison_df.melt(
        id_vars='Base Model',
        value_vars=['Color Accuracy', 'Grayscale Accuracy'],
        var_name='Image Type',
        value_name='Accuracy'
    )

    # --- Streamlit Plot ---
    plt.figure(figsize=(12, 7))
    sns.barplot(
        x='Base Model',
        y='Accuracy',
        hue='Image Type',
        data=melted_df,
        palette='Spectral'
    )

    # Styling
    plt.title('Accuracy Comparison: Color vs. Grayscale Models', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.ylim(0, 1.05)
    sns.despine()

    # Horizontal reference line
    plt.axhline(0.5, linestyle='--', color='red', alpha=0.6)

    st.pyplot(plt)


# --- 3. Streamlit App Section ---

def render_accuracy_chart():
    st.header("📊 Accuracy Comparison Bar Chart")
    st.caption("Color vs. Grayscale CNN Performance Comparison")
    plot_accuracy_comparison(results_df)

