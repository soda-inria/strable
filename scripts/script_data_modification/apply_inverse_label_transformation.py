import pandas as pd
import os

import numpy as np
from configs.path_configs import path_configs

dataset_summary = pd.read_parquet(f"{path_configs['dataset_summary_wide']}")

#select regression tasks
regression_tasks = dataset_summary[dataset_summary['task'] == 'regression']

#retrieve the tasks where a label transformation was applied (76 regression tasks)
df_with_transformation = ['aijob_ai-ml-ds-salaries', # np.log(data[target_name])
 'california-houses', #np.log(data[target_name]) 
 'college-creditcard-marketing', # np.log1p(data[target_name])
 'college-deposit-product-marketing', #np.log1p(data[target_name])
 'covid-clinical-trials', #np.log1p(data[target_name])
 'global-dams-database', #np.log1p(data[target_name])
 'industry-payments-entity', #np.sign(data[target_name]) * np.log1p(np.abs(data[target_name]))
 'industry-payments-project', #np.sign(data[target_name]) * np.log1p(np.abs(data[target_name]))
 'foreign-gift-and-contract', #np.sign(data[target_name]) * np.log1p(np.abs(data[target_name]))
 'antenna-structure-registration', #np.log(data[target_name])
 'colleges-and-universities', #np.log(data[target_name])
 'electric-retail-service-territories', #np.log(data[target_name])
 'historic-perimeters-wildfires', #np.log1p(data[target_name])
 'electric-generating-plants', #np.log(data[target_name])
 'hospitals', #np.log(data[target_name])
 'oil-natural-gas-platform', #np.log(data[target_name])
 'local-law-enforcements', #np.log(data[target_name])
 'pol-terminal', #np.cbrt(data[target_name])
 'prison-boundaries', #np.cbrt(data[target_name])
 'power-plants', #np.log(data[target_name])
 'transmission-towers', #np.cbrt(data[target_name])
 'schools', #np.log1p(data[target_name])
 'discretionary-grant', #np.log(data[target_name])
 'grant', #np.log(data[target_name])
 'museums', #np.arcsinh(data[target_name])
 'awarded-grants', #np.log(data[target_name])
 'insurance-company-complaints', #np.log1p(data[target_name])
 'first-time-nadac-rates', #np.log(data[target_name])
 'managed-care-enrollment', #np.log1p(data[target_name])
 'financial-management', #np.arcsinh(data[target_name])
 'mlr-summary-reports', #np.log(data[target_name])
 'national-average-drug-acquisition-cost', #np.log(data[target_name])
 'aca-federal-upper-limits-wide', #np.log(data[target_name])
 'conflict-events_wide', #np.arcsinh(data[target_name])
 'fts-funding', #np.sign(data[target_name]) * np.log1p(np.abs(data[target_name]))
 'fts-requirement-and-funding', #np.cbrt(data[target_name])
 'food-prices_wide', #np.log(data[target_name])
 'mercari', #np.log1p(data[target_name])
 'journal-ranking_wide', #np.log1p(data[target_name])
 'summary-of-deposit_wide', #np.log(data[target_name])
 'sf-building-permits', #np.log1p(data[target_name])
 'wine-dataset', #np.log(data[target_name])
 'china-overseas-finance-inventory', #np.log(data[target_name])
 'local-government-renewable-action', #np.log(data[target_name])
 'global-power-plant', #np.log(data[target_name])
 'us-school-bus-fleet', #np.log1p(data[target_name])
 'total-contributions-ibrd-ida-ifc', #np.sign(data[target_name]) * np.log(np.abs(data[target_name]))
 'commitments-in-trust-funds', #np.arcsinh(data[target_name])
 'contributions-to-financial-intermediary-funds', #np.cbrt(data[target_name])
 'corporate-procurement-contract-awards', #np.log(data[target_name])
 'financial-intermediary-funds-cash-transfers', #np.cbrt(data[target_name])
 'disbursements-in-trust-funds', #np.sign(data[target_name]) * np.log1p(np.abs(data[target_name]))
 'financial-intermediary-funds-commitments', #np.log(data[target_name])
 'financial-intermediary-funds-funding-decisions', #np.arcsinh(data[target_name])
 'contract-awards-investment-project-financing', #np.arcsinh(data[target_name])
 'ibrd-statement-loans-guarantees', #np.cbrt(data[target_name])
 'ifc-advisory-services-projects', #np.cbrt(data[target_name])
 'ifc-investment-service-projects', #np.log1p(data[target_name])
 'miga-issued-projects', #np.cbrt(data[target_name])
 'recipient-executed-grants-commitments-disbursements', #np.sign(data[target_name]) * np.log(np.abs(data[target_name]))
 'ida-statement-credits-grants-guarantees' #np.cbrt(data[target_name])
 ]

len(df_with_transformation)/76 #61/76=80% of regression tasks had a label transformation applied. 56% of all tasks had a label transformation applied.

# 1. Define the inverse transformation mapping
inverse_transforms = {
    # np.log -> np.exp
    'aijob_ai-ml-ds-salaries': np.exp,
    'california-houses': np.exp,
    'antenna-structure-registration': np.exp,
    'colleges-and-universities': np.exp,
    'electric-retail-service-territories': np.exp,
    'electric-generating-plants': np.exp,
    'hospitals': np.exp,
    'oil-natural-gas-platform': np.exp,
    'local-law-enforcements': np.exp,
    'power-plants': np.exp,
    'discretionary-grant': np.exp,
    'grant': np.exp,
    'awarded-grants': np.exp,
    'first-time-nadac-rates': np.exp,
    'mlr-summary-reports': np.exp,
    'national-average-drug-acquisition-cost': np.exp,
    'aca-federal-upper-limits-wide': np.exp,
    'food-prices_wide': np.exp,
    'summary-of-deposit_wide': np.exp,
    'wine-dataset': np.exp,
    'china-overseas-finance-inventory': np.exp,
    'local-government-renewable-action': np.exp,
    'global-power-plant': np.exp,
    'corporate-procurement-contract-awards': np.exp,
    'financial-intermediary-funds-commitments': np.exp,

    # np.log1p -> np.expm1
    'college-creditcard-marketing': np.expm1,
    'college-deposit-product-marketing': np.expm1,
    'covid-clinical-trials': np.expm1,
    'global-dams-database': np.expm1,
    'historic-perimeters-wildfires': np.expm1,
    'schools': np.expm1,
    'insurance-company-complaints': np.expm1,
    'managed-care-enrollment': np.expm1,
    'mercari': np.expm1,
    'journal-ranking_wide': np.expm1,
    'sf-building-permits': np.expm1,
    'us-school-bus-fleet': np.expm1,
    'ifc-investment-service-projects': np.expm1,

    # np.sign(y) * np.log1p(np.abs(y)) -> np.sign(y) * np.expm1(np.abs(y))
    'industry-payments-entity': lambda x: np.sign(x) * np.expm1(np.abs(x)),
    'industry-payments-project': lambda x: np.sign(x) * np.expm1(np.abs(x)),
    'foreign-gift-and-contract': lambda x: np.sign(x) * np.expm1(np.abs(x)),
    'fts-funding': lambda x: np.sign(x) * np.expm1(np.abs(x)),
    'disbursements-in-trust-funds': lambda x: np.sign(x) * np.expm1(np.abs(x)),

    # np.cbrt -> x**3
    'pol-terminal': lambda x: x**3,
    'prison-boundaries': lambda x: x**3,
    'transmission-towers': lambda x: x**3,
    'fts-requirement-and-funding': lambda x: x**3,
    'contributions-to-financial-intermediary-funds': lambda x: x**3,
    'financial-intermediary-funds-cash-transfers': lambda x: x**3,
    'ibrd-statement-loans-guarantees': lambda x: x**3,
    'ifc-advisory-services-projects': lambda x: x**3,
    'miga-issued-projects': lambda x: x**3,
    'ida-statement-credits-grants-guarantees': lambda x: x**3,

    # np.arcsinh -> np.sinh
    'museums': np.sinh,
    'financial-management': np.sinh,
    'conflict-events_wide': np.sinh,
    'commitments-in-trust-funds': np.sinh,
    'financial-intermediary-funds-funding-decisions': np.sinh,
    'contract-awards-investment-project-financing': np.sinh,

    # np.sign(y) * np.log(np.abs(y)) -> np.sign(y) * np.exp(np.abs(y))
    'total-contributions-ibrd-ida-ifc': lambda x: np.sign(x) * np.exp(np.abs(x)),
    'recipient-executed-grants-commitments-disbursements': lambda x: np.sign(x) * np.exp(np.abs(x))
}

# 2. Define Output Base Directory
OUTPUT_BASE_DIR = f"{path_configs['path_data_processed']}_inv_trans"

# 3. Process each dataset
for dataset_name in inverse_transforms.keys():
    # Fetch metadata from your dataframe
    row = regression_tasks[regression_tasks['data_name'] == dataset_name]
    
    if row.empty:
        print(f"Warning: {dataset_name} not found in regression_tasks DataFrame. Skipping.")
        continue
        
    row = row.iloc[0]
    target_col = row['target_column']
    data_path = row['data_path']
    
    # Determine the source directory based on whether data_path points to a folder or the parquet file
    if data_path.endswith('.parquet'):
        src_dir = os.path.dirname(data_path)
    else:
        src_dir = data_path

    # Define exact source file paths
    src_parquet = os.path.join(src_dir, 'data.parquet')
    src_config = os.path.join(src_dir, 'config.json')

    # Define exact destination file paths
    dest_dir = os.path.join(OUTPUT_BASE_DIR, dataset_name)
    dest_parquet = os.path.join(dest_dir, 'data.parquet')
    dest_config = os.path.join(dest_dir, 'config.json')

    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    try:
        # Load the data
        df = pd.read_parquet(src_parquet)
        
        # Apply the inverse transformation
        transform_func = inverse_transforms[dataset_name]
        df[target_col] = transform_func(df[target_col])
        
        # Save the transformed data to the new location
        df.to_parquet(dest_parquet, index=False)
        
        # Copy the config.json file as requested
        if os.path.exists(src_config):
            import shutil
            shutil.copy2(src_config, dest_config)
        else:
            print(f"Note: config.json not found for {dataset_name} in {src_dir}")
            
        print(f"Successfully processed and saved: {dataset_name}")
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")

print("\nAll tasks completed.")

