import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. Generate Sample Data (Sesuai CSV)
def generate_sample_data():
    variants = ['Espresso Classic', 'Latte Deluxe', 'Mocha Supreme', 
               'Matcha Green', 'Vanilla Dream', 'Caramel Crunch',
               'Decaf Blend', 'Hazelnut Twist', 'Cold Brew', 'Coconut Blend']
    dates = pd.date_range(start='2023-01-01', periods=30)
    
    data = {
        'variant': np.repeat(variants, len(dates)),
        'date': list(dates) * len(variants),
        'quantity': [int(50 + 10*(i%30) + 5*(i%7)) for i in range(len(variants)*len(dates))],
        'promo_event': ['Yes' if x%4 == 0 else 'No' for x in range(len(variants)*len(dates))],
        'bundling_package_variant': ['Standard']*200 + ['Premium']*50 + ['Limited']*50
    }
    return pd.DataFrame(data)

# 2. Preprocessing
def preprocess_data(df):
    # Encode categorical features
    le = LabelEncoder()
    df['promo'] = le.fit_transform(df['promo_event'])
    df['bundling'] = le.fit_transform(df['bundling_package_variant'])
    df['ds'] = pd.to_datetime(df['date'])
    return df

# 3. Forecasting Function
def forecast_variant(df, variant_name):
    variant_data = df[df['variant'] == variant_name].copy()
    
    prophet_df = variant_data[['ds', 'quantity', 'promo', 'bundling']].rename(
        columns={'quantity': 'y'}
    )
    
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.add_regressor('promo')
    model.add_regressor('bundling')
    
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=7)  # Forecast 7 hari
    future['promo'] = 0  # Asumsi tidak ada promo
    future['bundling'] = 0  # Asumsi bundling standard
    
    forecast = model.predict(future)
    return model, forecast

# 4. Main Execution
if __name__ == "__main__":
    # Load data
    raw_data = pd.read_csv("extended_sales_data.csv")/ # Ganti dengan generate_sample_data() untuk data dummy
    # Preprocess
    processed_data = preprocess_data(raw_data)
    
    # Forecast untuk 3 variant contoh
    target_variants = ['Espresso Classic', 'Latte Deluxe', 'Mocha Supreme']
    
    for variant in target_variants:
        print(f"\nForecasting untuk {variant}...")
        model, forecast = forecast_variant(processed_data, variant)
        
        # Hasil forecast besok
        tomorrow = forecast.iloc[-7]  # Hari pertama di forecast period
        print(f"Tanggal: {tomorrow['ds'].date()}")
        print(f"Prediksi: {tomorrow['yhat']:.0f} unit")
        print(f"Interval: [{tomorrow['yhat_lower']:.0f}, {tomorrow['yhat_upper']:.0f}]")
        
        # Plot
        fig = model.plot(forecast)
        plt.title(f'Forecast Penjualan {variant}')
        plt.savefig(f"static_{variant}.png")