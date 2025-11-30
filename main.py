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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st 

# --- 1. Data Preparation (Optimized from your Colab table code) ---

# Define the results using the clean dictionary-of-lists structure.
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

# --------------------------------------------------------------------------------

# --- 2. Plotting Function (Plotly Bar Chart - Final, Final Attempt) ---

def plot_accuracy_comparison_plotly(results_df: pd.DataFrame, accuracy_col: str = 'Accuracy', model_col: str = 'Model'):
    
    # 1. Data Preparation: Extract Base Model and Image Type
    extracted_data = results_df[model_col].str.extract(r'(.+)\s\((Color|Grayscale)\)')
    if extracted_data.empty or extracted_data.shape[1] < 2:
        st.error("Error: Model names could not be parsed for plotting.")
        return go.Figure()
        
    extracted_data.columns = ['Base Model', 'Image Type']
    plot_df = pd.concat([
        extracted_data,
        results_df[[accuracy_col]]
    ], axis=1).dropna(subset=['Base Model']) 
    
    # CRITICAL CLEANUP AND TYPE CASTING
    plot_df['Base Model'] = plot_df['Base Model'].astype(str).str.strip()
    plot_df['Image Type'] = plot_df['Image Type'].astype(str).str.strip()
    plot_df[accuracy_col] = pd.to_numeric(plot_df[accuracy_col], errors='coerce')

    plot_df.dropna(subset=['Base Model', 'Image Type', accuracy_col], inplace=True)

    if plot_df.empty:
        st.error("No valid data remaining after cleaning for plotting.")
        return go.Figure()
    
    # Get the ORDER of base models for the explicit category fix
    base_model_order = plot_df['Base Model'].unique().tolist()
    
    # 2. Visualization: Plotly Express
    # Use explicit color list for maximum compatibility (blue/orange pair)
    color_list = ['#1f77b4', '#ff7f0e'] # Deep Blue and Deep Orange
    
    fig = px.bar(
        plot_df,
        x='Base Model',
        y=accuracy_col,
        color='Image Type', 
        barmode='group',
        title='Accuracy Comparison: Color vs. Grayscale Model Performance',
        labels={
            'Base Model': 'Base Model / Algorithm',
            accuracy_col: f'{accuracy_col}',
            'Image Type': 'Input Data Type'
        },
        template='plotly_white',
        color_discrete_sequence=color_list # FIXED: Pass a list instead of a string name
    )

    # 3. Enhance Plot Details
    min_acc = plot_df[accuracy_col].min()
    max_acc = plot_df[accuracy_col].max()
    y_range = [max(0, min_acc - 0.05), min(1.0, max_acc + 0.05)]
    
    fig.update_layout(
        yaxis_range=y_range,
        xaxis_tickangle=-45,
        legend_title='Input Data Type',
        # FIXED: Explicitly define the category order for the X-axis
        xaxis={'categoryorder':'array', 'categoryarray': base_model_order} 
    )
    
    # Use fig.add_shape for horizontal line
    fig.add_shape(
        type='line', xref='paper', yref='y',
        x0=0, y0=0.5, x1=1, y1=0.5,
        line=dict(color="Red", width=1.5, dash="dash")
    )
    fig.add_annotation(
        text="Random Guess Baseline (0.5)",
        xref="paper", yref="y",
        x=1, y=0.5,
        showarrow=False, xanchor='right', yanchor='bottom',
        font=dict(color="Red", size=10)
    )

    return fig
