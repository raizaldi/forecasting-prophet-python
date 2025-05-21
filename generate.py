import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuration
variants = [
    'Espresso Classic', 'Latte Deluxe', 'Mocha Supreme', 
    'Matcha Green', 'Vanilla Dream', 'Caramel Crunch',
    'Decaf Blend', 'Hazelnut Twist', 'Cold Brew', 'Coconut Blend'
]
bundling_types = [
    'Standard', 'Morning Combo', 'Weekend Special', 
    'Healthy Bundle', 'Afternoon Set', 'Summer Promo'
]

# Generate consistent data
def generate_data(num_days=60, variants_per_day=3):
    data = []
    start_date = datetime(2023, 1, 1)
    
    for day in range(num_days):
        date = start_date + timedelta(days=day)
        is_weekend = date.weekday() >= 5
        is_summer = (date.month == 1) and (15 <= date.day <= 30)
        is_holiday = date.day in [1, 14]
        
        # Select random variants for each day
        daily_variants = np.random.choice(variants, variants_per_day, replace=False)
        
        for variant in daily_variants:
            qty = np.random.randint(70, 150)
            
            # Apply modifiers
            if is_weekend: qty += np.random.randint(10, 30)
            if is_holiday: qty = int(qty * 1.5)
            if variant == 'Cold Brew' and is_summer: qty = int(qty * 1.3)
            if variant == 'Matcha Green': qty += day * 2
            
            promo = np.random.choice(['Yes', 'No'], p=[0.3, 0.7])
            if promo == 'Yes': qty = int(qty * 1.3)
            
            bundling = np.random.choice(
                bundling_types,
                p=[0.5, 0.1, 0.15, 0.1, 0.1, 0.05]
            )
            if bundling == 'Weekend Special': qty += 20
            
            data.append([
                date.strftime('%Y-%m-%d'),
                variant,
                int(qty + np.random.normal(0, 5)),
                promo,
                bundling
            ])
    
    return pd.DataFrame(data, columns=[
        'date', 'variant', 'quantity', 
        'promo_event', 'bundling_package_variant'
    ])

# Generate and save data
df = generate_data(num_days=60, variants_per_day=3)
df.to_csv('extended_sales_data.csv', index=False)
print(f"Generated {len(df)} records")