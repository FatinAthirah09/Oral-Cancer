import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pcolors # <--- ADDED DEFINITIVE IMPORT

# --- 1. Data Preparation (Unchanged) ---

# Create a dictionary to store the results
results = {
    'Model': [],
    'Accuracy': [],
    'Loss': [],
    'Training Time (minutes)': []
}

# Populate the dictionary with results from each model run
# Color Models
results['Model'].append('DenseNet121 (Color)')
results['Accuracy'].append(0.9107)
results['Loss'].append(0.2151)
results['Training Time (minutes)'].append(7.09)

results['Model'].append('ResNet50 (Color)')
results['Accuracy'].append(0.9196)
results['Loss'].append(0.1845)
results['Training Time (minutes)'].append(8.98)

results['Model'].append('VGG16 (Color)')
results['Accuracy'].append(0.8839)
results['Loss'].append(0.2809)
results['Training Time (minutes)'].append(6.58)

results['Model'].append('MobileNetV2 (Color)')
results['Accuracy'].append(0.8973)
results['Loss'].append(0.2759)
results['Training Time (minutes)'].append(6.81)

results['Model'].append('InceptionV3 (Color)')
results['Accuracy'].append(0.8750)
results['Loss'].append(0.3006)
results['Training Time (minutes)'].append(9.90)

# Grayscale Models
results['Model'].append('DenseNet121 (Grayscale)')
results['Accuracy'].append(0.5223)
results['Loss'].append(0.6916)
results['Training Time (minutes)'].append(17.39)

results['Model'].append('ResNet50 (Grayscale)')
results['Accuracy'].append(0.5223)
results['Loss'].append(0.6922)
results['Training Time (minutes)'].append(13.18)

results['Model'].append('VGG16 (Grayscale)')
results['Accuracy'].append(0.5223)
results['Loss'].append(0.6921)
results['Training Time (minutes)'].append(16.77)

results['Model'].append('MobileNetV2 (Grayscale)')
results['Accuracy'].append(0.5223)
results['Loss'].append(0.6922)
results['Training Time (minutes)'].append(4.45)

results['Model'].append('InceptionV3 (Grayscale)')
results['Accuracy'].append(0.5223)
results['Loss'].append(0.6922)
results['Training Time (minutes)'].append(13.57)

# Create the DataFrame
results_df = pd.DataFrame(results)

# --- 2. Plotting Function (Plotly Format - Final Fix) ---

def plot_accuracy_comparison_plotly(results_df: pd.DataFrame, accuracy_col: str = 'Accuracy', model_col: str = 'Model'):
    """
    Compares the accuracy of models trained on color and grayscale images
    using an interactive Plotly grouped bar chart.
    """

    # 1. Data Preparation
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
    
    # *** FINAL FIX APPLIED HERE: Using the pcolors alias ***
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
        # --- Using the explicitly imported plotly.colors module for reliability ---
        color_discrete_sequence=pcolors.qualitative.Spectral 
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
            x=plot_df['Base Model'].unique(), # Get unique base models for full span
            y=[0.5] * len(plot_df['Base Model'].unique()), 
            mode='lines',
            name='Random Guess Baseline',
            line=dict(color='red', dash='dash', width=1.5),
            hoverinfo='name'
        )
    )

    fig.show()

# --- 3. Execution ---

# Call the new Plotly function
plot_accuracy_comparison_plotly(results_df)
