"""Script for containing the experiment information."""

from copy import deepcopy

# Total list of wide data
data_list_wide = ['hospitals',
 'aijob_ai-ml-ds-salaries',
 'beer-ratings',
 'california-houses',
 'commonlit_clear-corpus',
 'ohca_conflict-events-wide',
 'college-creditcard-marketing',
 'college-deposit-product-marketing',
 'financial-product-complaint',
 'prepaid-financial-product',
 'terms-cc-plans',
 'covid-clinical-trials',
 'global-dams-database',
 'industry-payments-entity',
 'clear-corpus',
 'chocolate-bar-ratings',
 'yelp_business',
 'meta-critic_whisky',
 'wine-dataset',
 'ramen-ratings',
 'sf-building-permits',
 'osha-accidents',
 'mercari',
 'insurance-company-complaints',
 'it-salary-survey',
 'kickstarter-projects',
 'lending-club-loan',
 'animalandveterinary-event',
 'cosmetic-event',
 'device-classification',
 'device-covid19serology',
 'fda_device-covid19serology',
 'device-pma',
 'drug-drugsfda',
 'drug-enforcement',
 'drug-ndc',
 'drug-shortages',
 'food-enforcement',
 'food-event',
 'tobacco-problem',
 'medicines',
 'michelin-ratings',
 'orphan-designations',
 'paediatric-investigation-plan',
 'rasff_window',
 'rasnf_notification_list',
 'broadband-availability',
 'discretionary-grant',
 'grant',
 'museums',
 'community-banking_wide',
 'industry-payments-project',
 'tax-incentives',
 'vehicles',
 'journal-ranking_wide',
 'media-ranking_wide',
 'food-prices_wide',
 'conflict-events_wide',
 'summary-of-deposit_wide',
 'china-overseas-finance-inventory',
 'global-power-plant',
 'local-government-renewable-action',
 'us-school-bus-fleet',
 'gainful-employment',
 'cohort-default-rate',
 'foreign-gift-and-contract',
 'fts-incoming-funding',
 'fts-internal-funding',
 'fts-outgoing-funding',
 'fts-funding',
 'fts-requirement-and-funding',
 'awarded-grants',
 'external-clinician-dashboard',
 'health-professional-shortage-areas',
 'hypertension-control-wide',
 'medically-underserved-areas-populations',
 'organ-donation-transplantation',
 'workforce-demographics-wide',
 'first-time-nadac-rates',
 'aca-federal-upper-limits-wide',
 'child-adult-healthcare-quality',
 'managed-care-enrollment',
 'financial-management',
 'mlr-summary-reports',
 'national-average-drug-acquisition-cost',
 'antenna-structure-registration',
 'colleges-and-universities',
 'electric-retail-service-territories',
 'historic-perimeters-wildfires',
 'historical-volcanic-locations',
 'local-law-enforcements',
 'mobile-home-parks',
 'oil-natural-gas-platform',
 'pol-terminal',
 'schools',
 'prison-boundaries',
 'public-refrigerated-warehouses',
 'transmission-lines',
 'transmission-towers',
 'historical-earthquake-locations',
 'electric-generating-plants',
 'power-plants',
 'commitments-in-trust-funds',
 'contract-awards-investment-project-financing',
 'contributions-to-financial-intermediary-funds',
 'corporate-procurement-contract-awards',
 'disbursements-in-trust-funds',
 'financial-intermediary-funds-cash-transfers',
 'miga-issued-projects',
 'ifc-advisory-services-projects',
 'ibrd-statement-loans-guarantees']

# Total list of panel data
data_list_panel = ['ifc-statement-cumulative-gross-committment',
 'college-deposit-product-marketing',
 'ibrda-ida-net-flows-committments',
 'summary-of-deposit_panel',
 'community-banking_panel',
 'managed-care-enrollment',
 'institution-ranking_panel',
 'aca-federal-upper-limits-panel',
 'hypertension-control-panel',
 'journal-ranking_panel',
 'workforce-projection-panel',
 'media-ranking_panel',
 'conflict-events_panel',
 'college-creditcard-marketing',
 'world-bank-program-budget-allfunds',
 'workforce-demographics-panel',
 'food-prices_panel',
 'mlr-summary-reports']

data_list_full_dtype = [
    "RASFF_window",
    "flavors-of-cacao_chocolate-bar-ratings",
    "medicaid-first-time-NADAC-Rates",
    "FSA-Foreign-Gift-and-Contract",
    "conflict-events_2021-2025_wide",
    "awarded-grants-hrsa",
    "institute-museum-library-service_broadband",
    "wri-local-government-renewable-action",
    "institute-museum-library-service_museums",
    "yelp_business",
    "fueleconomy-tax-incentives",
    "medicaid-Medicaid-Financial-Management",
    "ohca-fts-incoming-funding",
    "institute-museum-library-service_grant",
    "scimago-journal-ranking_2019-2024_wide",
    "wri-us-school-bus-fleet",
    "wri-global-power-plant",
    "commonlit_clear-corpus",
    "fueleconomy-vehicles",
    "euro-med-agency_medicines",
    "hypertension-control-hrsa-wide",
    "ohca-fts-requirement-and-funding",
    "FSA-Gainful-Employment",
    "ohca-fts-outgoing-funding",
    "euro-med-agency_paediatric-investigation-plan",
    "workforce-demographics-hrsa-wide",
    "scimago-institution-ranking_2021-2025_wide",
    "fdic-community-banking_2019-2024_wide",
    "external-clinician-dashboard-hrsa",
    "fdic-summary-of-deposit_2020-2024_wide",
    "medicaid-ACA-federal-upper-limits-wide",
    "workforce-projection-hrsa-wide",
    "euro-med-agency_orphan-designations",
    "wri-china-overseas-finance-inventory",
    "aijob_ai-ml-ds-salaries",
    "institute-museum-library-service_discretionary-grant",
    "medically-underserved-areas-populations-hrsa",
    "ohca-fts-internal-funding",
    "Health-Professional-Shortage-Areas-hrsa",
    "FSA-cohort-default-rate",
]

# LLM configurations
llm_configs = dict()

# llm-llama-3.2-1b
temp_configs = dict()
temp_configs["hf_model_name"] = "meta-llama/Llama-3.2-1B"
llm_configs["llm-llama-3.2-1b"] = deepcopy(temp_configs)

# llm-llama-3.2-3b
temp_configs = dict()
temp_configs["hf_model_name"] = "meta-llama/Llama-3.2-3B"
llm_configs["llm-llama-3.2-3b"] = deepcopy(temp_configs)

# llm-llama-3.1-8b
temp_configs = dict()
temp_configs["hf_model_name"] = "meta-llama/Llama-3.1-8B"
llm_configs["llm-llama-3.1-8b"] = deepcopy(temp_configs)

# llm-qwen3-8b
temp_configs = dict()
temp_configs["hf_model_name"] = "Qwen/Qwen3-Embedding-8B"
llm_configs["llm-qwen3-8b"] = deepcopy(temp_configs)

# llm-qwen3-4b
temp_configs = dict()
temp_configs["hf_model_name"] = "Qwen/Qwen3-Embedding-4B"
llm_configs["llm-qwen3-4b"] = deepcopy(temp_configs)

# llm-qwen3-0.6b
temp_configs = dict()
temp_configs["hf_model_name"] = "Qwen/Qwen3-Embedding-0.6B"
llm_configs["llm-qwen3-0.6b"] = deepcopy(temp_configs)

# llm-opt-6.7b
temp_configs = dict()
temp_configs["hf_model_name"] = "facebook/opt-6.7b"
llm_configs["llm-opt-6.7b"] = deepcopy(temp_configs)

# llm-opt-2.7b
temp_configs = dict()
temp_configs["hf_model_name"] = "facebook/opt-2.7b"
llm_configs["llm-opt-2.7b"] = deepcopy(temp_configs)

# llm-opt-1.3b
temp_configs = dict()
temp_configs["hf_model_name"] = "facebook/opt-1.3b"
llm_configs["llm-opt-1.3b"] = deepcopy(temp_configs)

# llm-e5-large-v2
temp_configs = dict()
temp_configs["hf_model_name"] = "intfloat/e5-large-v2"
llm_configs["llm-e5-large-v2"] = deepcopy(temp_configs)

# llm-e5-base-v2
temp_configs = dict()
temp_configs["hf_model_name"] = "intfloat/e5-base-v2"
llm_configs["llm-e5-base-v2"] = deepcopy(temp_configs)

# llm-e5-small-v2
temp_configs = dict()
temp_configs["hf_model_name"] = "intfloat/e5-small-v2"
llm_configs["llm-e5-small-v2"] = deepcopy(temp_configs)

# llm-roberta-large
temp_configs = dict()
temp_configs["hf_model_name"] = "FacebookAI/roberta-large"
llm_configs["llm-roberta-large"] = deepcopy(temp_configs)

# llm-roberta-base
temp_configs = dict()
temp_configs["hf_model_name"] = "FacebookAI/roberta-base"
llm_configs["llm-roberta-base"] = deepcopy(temp_configs)

# llm-all-MiniLM-L6-v2
temp_configs = dict()
temp_configs["hf_model_name"] = "sentence-transformers/all-MiniLM-L6-v2"
llm_configs["llm-all-MiniLM-L6-v2"] = deepcopy(temp_configs)

#llm-fasttext
temp_configs = dict()
temp_configs["hf_model_name"] = "fasttext"
llm_configs["llm-fasttext"] = deepcopy(temp_configs)

#llama-nemotron-embed-1b-v2
temp_configs = dict()
temp_configs["hf_model_name"] = "nvidia/llama-nemotron-embed-1b-v2"
llm_configs["llm-llama-nemotron-embed-1b-v2"] = deepcopy(temp_configs)