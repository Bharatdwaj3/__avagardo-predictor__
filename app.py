import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import streamlit as st
import os

def load_model():
    """Load the trainded model"""
    model_path = os.path.join('model','avacado_yield_model.pkl')
    if not os.path.exists(model_path):
            st.error("Model not found! Please run train.py first to train the model")
            st.stop()
            return joblib.load(model_path)

def load_data():
      """Load the dataset"""
      try:
            return pd.read_csv('data/data.csv')
      except FileNotFoundError:
            st.error("Dataset not found! Please ensure data.csv is in the data directory")
            st.stop()
def predict_yield(model, country, region, area, organic_pct, farm_size, production_system):
      """"Make yield production based on input features"""
      input_data = pd.DataFrame({
            'Country': [country],
            'Region': [region],
            'Cultivated Area (Ha)': [area],
            'Organic Production (%)': [organic_pct],
            'Farm Size Category': [farm_size],
            'Production System': [production_system],
      })

      prediction=model.predict(input_data)[0]   
      return prediction      

def calculate_production(area, yield_value):
      """Calculate total production based on are and yield"""
      return area * yield_value
def main():
      st.set_page_config(page_title="Avocado Agriculture predection App", 
                         page_icon="🥑",layout="wide")
      st.title("🥑 Avocado Agriculture Prediction System")
      st.write("""This Application preddicts avocado yield and production
               based on regional characteristcs
               and farming practices using machine
               learning""")
      model=load_model()
      df=load_data()

      tab1, tab2, tab3 = st.tabs(["Yield Predection","Data Explorer","Model Info"])

      with tab1:
            st.header("Predict Avocado Yield")
            st.write("Enter the details of your avocado farming operation to predict yeild")
            col1, col2 =  st.columns(2)

            with col1:
                  countries=sorted(df['Country'].unique())
                  country=st.selectbox("Country",countries)
                  regions=sorted(df[df['Country']==country]['Region'].unique())
                  region=st.selectbox("Region",regions)
                  farm_sizes=sorted(df['Farm Size Category'].unique())
                  farm_size=st.selectbox("Farm Size Category",farm_sizes)

                  with col2:
                        area=st.number_input("Cultivated Area (Hectares)",min_value=1,max_value=20000,value=1000)
                        organic_pct=st.slider("Organic Production (%)",min_value=0,max_value=100,value=15)

                        production_systems=sorted(df['Production System'].unique())
                        production_system=st.selectbox('Production System', production_systems)

                  if st.button("Production Yield"):
                        with st.spinner("Calculating predection..."):
                              predicted_yield=predict_yield(model, country, region, area, organic_pct, farm_size, production_system)
                              estimated_production=calculate_production(area, predict_yield)

                              st.success(f"Predicted Yield: **{predicted_yield:.2f}** Metric Tons/Hectare")
                              col1,col2,col3=st.columns(3)
                              col1.metric("Total Production",f"{estimated_production:2f} MT")

                              region_avg = df(df['Region']==region)
                              difference=100*(predicted_yield-region_avg)/region_avg if region_avg else 0
                              col2.metric1("Region Average Yield",f"{region_avg:.2f}MT/Ha",f"{difference:.1f}% if difference else N/A")

                              export_ratio=df[(df['Country']==country)]['Export Volume (MT)'].sum()/df(df['Country']==country)['Total Production (MT)'].sum()
                              potential_export=estimated_production*export_ratio
                              col3.metric("Potential Export",f"{potential_export:.2f}MT")

                              st.subheader("Recommendation")
                              if predict_yield>region_avg*1.1:
                                    st.write("✅ Your Predicted yield is above the regional average Your current agricultural practices seem effective")
                                    st.write("Consider increasing organic percentage to potentially access markets")
                              elif predict_yield<region_avg*0.9:
                                    st.write("Your predicted yeild is below the regional average Consider these imporvements:")
                                    st.write("_Review Irrigation practices for the selected region")
                                    st.write("-Consider soil analysis and possible amendments")
                                    st.write("-Evaluate pest management strategies common in your selected region")
                              else:
                                    st.write("Your predicted yeild is close to the orignal average .Continue with the established best practices")

      with tab2:
            st.header("Avocado Production Data Explorer")
            selected_country=st.selectbox("Select a country to explore",["All Countries"]+sorted(df['Country'].unique()),key="explore country")
            if selected_country == "All Countries":
                  filtered_data = df
            else:
                  filtered_data = df[df['Country'] == selected_country]

                  st.write(f"### Data for {selected_country}")

                  st.dataframe(filtered_data)

                  st.subheader("Data Visualization")

                  viz_type=st.radio(
                        "Select visualizatio type",
                        ["Country Production Comparison","Yield vs Area","Production System Analysis"]
                  )

                  fig=plt.figure(figsize=(10,6))

            if viz_type=="Counrty Production Comparison":
                  country_production = df.groupby('Country')['Total Production (MT)'].sum().sort_values(ascending=False).head(10)
                  plt.bar(country_production.index, country_production.values/1000)
                  plt.xlabel('Country')
                  plt.ylabel('Total Production (Thousand MT)')
                  plt.title('Top 10 Avocado Producing Countries in Latin America')
                  plt.xticks(rotation=45, ha='right')
                  plt.tight_layout()

            elif viz_type=="Yield vs Area":
                  plt.scatter(df['Cultivated Area (Ha)'],df['Yield (MT/Ha)'],alpha=0.6)
                  plt.xlabel('Cultivated Area')
                  plt.ylabel('Yield (Thousand MT)')
                  plt.title('Relationship between Cultivation Area and Yield')
                  plt.grid(True,linestyle="--",alpha=0.7)
            
            elif viz_type == "Production System Analysis":

                  production_systems = df['Production System'].unique()
                  yield_by_system = [df[df['Production System']==system]['Yield (MT/Ha)'] for system in production_systems]
                  plt.boxplot(yield_by_system, labels=production_systems)
                  plt.xlabel('Production System ')
                  plt.ylabel('Yield (MT/Ha')
                  plt.title('Yield Distribution by Production System')
                  plt.grid(True,axis='y',linestyle="--",alpha=0.7)

            st.pyplot(fig)

      with tab3:
            st.header("Model Information")
            st.write("""
            ### Avocado Yield Predection Model:
                  This Application uses a Random Forest model trained on the Latin American avocado production dataset to predict avocado yields based on several features.
            
            ### Features uded in the model:
                  - Country and Region(geographical features)
                  - Cultivated Area(hectares)
                  - Organic production percentage
                  - Farm size category
                  - Production systems

            ### Model Performance:
                  - Algorithm: Random Forest Regressor
                  - Training method: 80% training data, 20% test data
                  - Preprocessing: Standard scaling for numerical features, one-hot encoding for categorical features
 
            ### How to use this app: 
                  1. Go to the "Yield Predection" tab
                  2. Enter or select your farming parameters
                  3. Click the "Predict Yield" button
                  4. View your predicted yield and recommendations
                     
            ### For Additional model details:
                  Run `python train.py` to retain the model and see detailed performance metrics.
""")
            performance_img_path = os.path.join('model','yield_predection_performance.png')
            if os.path.exists(performance_img_path):
                  st.image(performance_img_path, caption="Model Performance: Actual vs Predicted Yield")
            else:
                  st.info("Performace visualization not avaliable. Run train.py to generate it")
if __name__ == "__main__":
      main()            