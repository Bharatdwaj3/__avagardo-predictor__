import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import streamlit as st
import os

def load_model():
    """Load the trained model"""
    model_path = os.path.join('model', 'avocado_yield_model.pkl')
    if not os.path.exists(model_path):
        st.error("Model not found! Please run train.py first to train the model.")
        st.stop()
    return joblib.load(model_path)

def load_data():
    """Load the dataset"""
    try:
        return pd.read_csv('data/data.csv')
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure data.csv is in the data directory.")
        st.stop()

def predict_yield(model, country, region, area, organic_pct, farm_size, production_system):
    """Make yield prediction based on input features"""
    # Create a dataframe with the input features
    input_data = pd.DataFrame({
        'Country': [country],
        'Region': [region],
        'Cultivated Area (Ha)': [area],
        'Organic Production (%)': [organic_pct],
        'Farm Size Category': [farm_size],
        'Production System': [production_system]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    return prediction

def calculate_production(area, yield_value):
    """Calculate total production based on area and yield"""
    return area * yield_value

def main():
    # Page config
    st.set_page_config(page_title="Avocado Agriculture Prediction App", page_icon="🥑", layout="wide")
    
    # Header
    st.title("🥑 Avocado Agriculture Prediction System")
    st.write("""
    This application predicts avocado yield and production based on regional characteristics
    and farming practices using machine learning.
    """)
    
    # Load model and data
    model = load_model()
    df = load_data()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Yield Prediction", "Data Explorer", "Model Info"])
    
    # Tab 1: Prediction Interface
    with tab1:
        st.header("Predict Avocado Yield")
        st.write("Enter the details of your avocado farming operation to predict yield.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Get unique values for categorical features
            countries = sorted(df['Country'].unique())
            
            # Country selection
            country = st.selectbox("Country", countries)
            
            # Filter regions based on selected country
            regions = sorted(df[df['Country'] == country]['Region'].unique())
            region = st.selectbox("Region", regions)
            
            # Farm size category
            farm_sizes = sorted(df['Farm Size Category'].unique())
            farm_size = st.selectbox("Farm Size Category", farm_sizes)
        
        with col2:
            # Numerical inputs
            area = st.number_input("Cultivated Area (Hectares)", min_value=1, max_value=200000, value=1000)
            organic_pct = st.slider("Organic Production (%)", min_value=0, max_value=100, value=15)
            
            # Production system
            production_systems = sorted(df['Production System'].unique())
            production_system = st.selectbox("Production System", production_systems)
        
        # Prediction button
        if st.button("Predict Yield"):
            with st.spinner("Calculating prediction..."):
                # Make prediction
                predicted_yield = predict_yield(model, country, region, area, organic_pct, farm_size, production_system)
                estimated_production = calculate_production(area, predicted_yield)
                
                # Display results
                st.success(f"Predicted Yield: **{predicted_yield:.2f}** Metric Tons/Hectare")
                
                # Create result metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Production", f"{estimated_production:.2f} MT")
                
                # Compare with regional average
                region_avg = df[(df['Country'] == country) & (df['Region'] == region)]['Yield (MT/Ha)'].mean()
                difference = 100 * (predicted_yield - region_avg) / region_avg if region_avg else 0
                col2.metric("Regional Average Yield", f"{region_avg:.2f} MT/Ha", f"{difference:.1f}%" if difference else "N/A")
                
                # Estimated export potential
                export_ratio = df[(df['Country'] == country)]['Export Volume (MT)'].sum() / df[(df['Country'] == country)]['Total Production (MT)'].sum()
                potential_export = estimated_production * export_ratio
                col3.metric("Potential Export", f"{potential_export:.2f} MT")
                
                # Recommendation based on prediction
                st.subheader("Recommendations")
                if predicted_yield > region_avg * 1.1:
                    st.write("✅ Your predicted yield is above the regional average. Your current agricultural practices seem effective.")
                    st.write("Consider increasing organic production percentage to potentially access premium markets.")
                elif predicted_yield < region_avg * 0.9:
                    st.write("⚠️ Your predicted yield is below the regional average. Consider these improvements:")
                    st.write("- Review irrigation practices for the selected region")
                    st.write("- Consider soil analysis and possible amendments")
                    st.write("- Evaluate pest management strategies common in your selected region")
                else:
                    st.write("Your predicted yield is close to the regional average. Continue with established best practices.")
    
    # Tab 2: Data Explorer
    with tab2:
        st.header("Avocado Production Data Explorer")
        
        # Country filter
        selected_country = st.selectbox("Select a country to explore", ["All Countries"] + sorted(df['Country'].unique()), key="explore_country")
        
        if selected_country == "All Countries":
            filtered_data = df
        else:
            filtered_data = df[df['Country'] == selected_country]
        
        # Display filtered data
        st.write(f"### Data for {selected_country}")
        st.dataframe(filtered_data)
        
        # Visualizations
        st.subheader("Data Visualizations")
        
        viz_type = st.radio(
            "Select visualization type",
            ["Country Production Comparison", "Yield vs. Area", "Production System Analysis"]
        )
        
        fig = plt.figure(figsize=(10, 6))
        
        if viz_type == "Country Production Comparison":
            # Create country production bar chart
            country_production = df.groupby('Country')['Total Production (MT)'].sum().sort_values(ascending=False).head(10)
            plt.bar(country_production.index, country_production.values/1000)
            plt.xlabel('Country')
            plt.ylabel('Total Production (Thousand MT)')
            plt.title('Top 10 Avocado Producing Countries in Latin America')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
        elif viz_type == "Yield vs. Area":
            # Create scatter plot of yield vs area
            plt.scatter(df['Cultivated Area (Ha)'], df['Yield (MT/Ha)'], alpha=0.6)
            plt.xlabel('Cultivated Area (Ha)')
            plt.ylabel('Yield (MT/Ha)')
            plt.title('Relationship Between Cultivation Area and Yield')
            plt.grid(True, linestyle='--', alpha=0.7)
            
        elif viz_type == "Production System Analysis":
            # Boxplot of yield by production system
            production_systems = df['Production System'].unique()
            yield_by_system = [df[df['Production System'] == system]['Yield (MT/Ha)'] for system in production_systems]
            plt.boxplot(yield_by_system, labels=production_systems)
            plt.xlabel('Production System')
            plt.ylabel('Yield (MT/Ha)')
            plt.title('Yield Distribution by Production System')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
    
    # Tab 3: Model Information
    with tab3:
        st.header("Model Information")
        
        st.write("""
        ### Avocado Yield Prediction Model
        
        This application uses a Random Forest Regression model trained on the Latin American avocado
        production dataset to predict avocado yields based on several input features.
        
        #### Features used in the model:
        - Country and region (geographical location)
        - Cultivated area (hectares)
        - Organic production percentage
        - Farm size category
        - Production system
        
        #### Model Performance:
        - Algorithm: Random Forest Regressor
        - Training method: 80% training data, 20% test data
        - Preprocessing: Standard scaling for numerical features, one-hot encoding for categorical features
        
        #### How to use this app:
        1. Go to the "Yield Prediction" tab
        2. Enter or select your farming parameters
        3. Click the "Predict Yield" button
        4. View your predicted yield and recommendations
        
        #### For additional model details:
        Run `python train.py` to retrain the model and see detailed performance metrics.
        """)
        
        # Display model image if available
        performance_img_path = os.path.join('model', 'yield_prediction_performance.png')
        if os.path.exists(performance_img_path):
            st.image(performance_img_path, caption="Model Performance: Actual vs Predicted Yield")
        else:
            st.info("Performance visualization not available. Run train.py to generate it.")

if __name__ == "__main__":
    main()