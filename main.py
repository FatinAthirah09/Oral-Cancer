import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# import plotly.colors as pcolors # <--- No longer needed, can be commented/removed

# ... (Data Preparation Section is unchanged) ...
results = {
    'Model': ['DenseNet121 (Color)', 'ResNet50 (Color)', 'VGG16 (Color)', 'MobileNetV2 (Color)', 'InceptionV3 (Color)', 
              'DenseNet121 (Grayscale)', 'ResNet50 (Grayscale)', 'VGG16 (Grayscale)', 'MobileNetV2 (Grayscale)', 'InceptionV3 (Grayscale)'],
    'Accuracy': [0.9107, 0.9196, 0.8839, 0.8973, 0.8750, 0.5223, 0.5223, 0.5223, 0.5223, 0.5223],
    'Loss': [0.2151, 0.1845, 0.2809, 0.2759, 0.3006, 0.6916, 0.6922, 0.6921, 0.6922, 0.6922],
    'Training Time (minutes)': [7.09, 8.98, 6.58, 6.81, 9.90, 17.39, 13.18, 16.77, 4.45, 13.57]
}
results_df = pd.DataFrame(results)


# --- 2. Plotting Function (Plotly Format - Using String Name) ---

def plot_accuracy_comparison_plotly(results_df: pd.DataFrame, accuracy_col: str = 'Accuracy', model_col: str = 'Model'):
    """
    Compares the accuracy of models trained on color and grayscale images
    using an interactive Plotly grouped bar chart.
    """

    # 1. Data Preparation (Unchanged)
    extracted_data = results_df[model_col].str.extract(r'(.+)\s\((Color|Grayscale)\)')
    if extracted_data.empty or extracted_data.shape[1] < 2:
        print("Error: Model names could not be parsed. Check the expected format 'BaseModel (Type)'.")
        return
        
    extracted_data.columns = ['Base Model', 'Image Type']
    plot_df = pd.concat([
        extracted_data,
        results_df[[accuracy_col]]
    ], axis=1).dropna(subset=['Base Model'])

    # 2. Visualization: Use Plotly Express
    
    # *** FINAL FIX APPLIED HERE: Using the string name 'Spectral' ***
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
        color_discrete_sequence='Spectral' # <--- FIXED LINE (Using string name)
    )

    # 3. Enhance Plot Details
    min_acc = plot_df[accuracy_col].min()
    max_acc = plot_df[accuracy_col].max()
    y_range = [max(0, min_acc - 0.05), min(1.0, max_acc + 0.05)]
    
    fig.update_layout(
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12,
        legend_title_font_size=12,
        yaxis_range=y_range,
        xaxis_tickangle=-45
    )
    
    # Add a horizontal line at 0.5 for reference
    fig.add_trace(
        go.Scatter(
            x=plot_df['Base Model'].unique(),
            y=[0.5] * len(plot_df['Base Model'].unique()), 
            mode='lines',
            name='Random Guess Baseline',
            line=dict(color='red', dash='dash', width=1.5),
            hoverinfo='name'
        )
    )

    # Note: In Streamlit, you must pass the figure to st.plotly_chart()
    # fig.show() 
    return fig # Changed to return the figure for Streamlit

# --- 3. Execution ---

# Call the function (if in Streamlit, you would use st.plotly_chart)
# plot_accuracy_comparison_plotly(results_df).show()
# OR in Streamlit: st.plotly_chart(plot_accuracy_comparison_plotly(results_df))
