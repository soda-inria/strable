"""Script for containing the experiment information."""

from copy import deepcopy

# Embedding configurations
embed_configs = dict()
embed_configs['non_cache_methods'] = [
    'tabpfn',
    'catboost',
    'tarte-ft',
    'tabstar',
    'contexttab',
]

# Estimator configurations
estim_configs = dict()

## Specific configurations
# Ridge
temp_configs = dict()
temp_configs["search_method"] = 'no-search'
temp_configs["fit_with_val"] = False
estim_configs['ridge'] = deepcopy(temp_configs)

# TabPFNv2.5
temp_configs = dict()
temp_configs["search_method"] = 'no-search'
temp_configs["fit_with_val"] = False
estim_configs['tabpfn'] = deepcopy(temp_configs)

# RealTabPFNv2.5
temp_configs = dict()
temp_configs["search_method"] = 'no-search'
temp_configs["fit_with_val"] = False
estim_configs['realtabpfn'] = deepcopy(temp_configs)

# XGBoost
temp_configs = dict()
temp_configs["search_method"] = 'random-search'
temp_configs["fit_with_val"] = True
estim_configs['xgb'] = deepcopy(temp_configs)

# CatBoost
temp_configs = dict()
temp_configs["search_method"] = 'random-search'
temp_configs["fit_with_val"] = True
estim_configs['catboost'] = deepcopy(temp_configs)

# ExtraTrees
temp_configs = dict()
temp_configs["search_method"] = 'random-search'
temp_configs["fit_with_val"] = False
estim_configs['extrees'] = deepcopy(temp_configs)

# RandomForest
temp_configs = dict()
temp_configs["search_method"] = 'random-search'
temp_configs["fit_with_val"] = False
estim_configs['randomforest'] = deepcopy(temp_configs)

# TARTE
temp_configs = dict()
temp_configs["search_method"] = 'no-search'
temp_configs["fit_with_val"] = False
estim_configs['tarte'] = deepcopy(temp_configs)

## LLM configurations
llm_configs = dict()

# llm-llama-3.2-1b
temp_configs = dict()
temp_configs["hf_model_name"] = "meta-llama/Llama-3.2-1B"
temp_configs["num_params"] = 1_240_000_000
llm_configs["llm-llama-3.2-1b"] = deepcopy(temp_configs)

# llm-llama-3.2-3b
temp_configs = dict()
temp_configs["hf_model_name"] = "meta-llama/Llama-3.2-3B"
temp_configs["num_params"] = 3_210_000_000
llm_configs["llm-llama-3.2-3b"] = deepcopy(temp_configs)

# llm-llama-3.1-8b
temp_configs = dict()
temp_configs["hf_model_name"] = "meta-llama/Llama-3.1-8B"
temp_configs["num_params"] = 8_030_000_000
llm_configs["llm-llama-3.1-8b"] = deepcopy(temp_configs)

# llm-qwen3-8b
temp_configs = dict()
temp_configs["hf_model_name"] = "Qwen/Qwen3-Embedding-8B"
temp_configs["num_params"] = 7_570_000_000
llm_configs["llm-qwen3-8b"] = deepcopy(temp_configs)

# llm-qwen3-4b
temp_configs = dict()
temp_configs["hf_model_name"] = "Qwen/Qwen3-Embedding-4B"
temp_configs["num_params"] = 4_020_000_000
llm_configs["llm-qwen3-4b"] = deepcopy(temp_configs)

# llm-qwen3-0.6b
temp_configs = dict()
temp_configs["hf_model_name"] = "Qwen/Qwen3-Embedding-0.6B"
temp_configs["num_params"] = 596_000_000
llm_configs["llm-qwen3-0.6b"] = deepcopy(temp_configs)

# llm-opt-6.7b
temp_configs = dict()
temp_configs["hf_model_name"] = "facebook/opt-6.7b"
temp_configs["num_params"] = 6_700_000_000
llm_configs["llm-opt-6.7b"] = deepcopy(temp_configs)

# llm-opt-2.7b
temp_configs = dict()
temp_configs["hf_model_name"] = "facebook/opt-2.7b"
temp_configs["num_params"] = 2_700_000_000
llm_configs["llm-opt-2.7b"] = deepcopy(temp_configs)

# llm-opt-1.3b
temp_configs = dict()
temp_configs["hf_model_name"] = "facebook/opt-1.3b"
temp_configs["num_params"] = 1_300_000_000
llm_configs["llm-opt-1.3b"] = deepcopy(temp_configs)

# llm-opt-0.3b
temp_configs = dict()
temp_configs["hf_model_name"] = "facebook/opt-350m"
temp_configs["num_params"] = 350_000_000
llm_configs["llm-opt-0.3b"] = deepcopy(temp_configs)

# llm-opt-0.1b
temp_configs = dict()
temp_configs["hf_model_name"] = "facebook/opt-125m"
temp_configs["num_params"] = 125_000_000
llm_configs["llm-opt-0.1b"] = deepcopy(temp_configs)

# llm-e5-large-v2
temp_configs = dict()
temp_configs["hf_model_name"] = "intfloat/e5-large-v2"
temp_configs["num_params"] = 300_000_000
llm_configs["llm-e5-large-v2"] = deepcopy(temp_configs)

# llm-e5-base-v2
temp_configs = dict()
temp_configs["hf_model_name"] = "intfloat/e5-base-v2"
temp_configs["num_params"] = 109_000_000
llm_configs["llm-e5-base-v2"] = deepcopy(temp_configs)

# llm-e5-small-v2
temp_configs = dict()
temp_configs["hf_model_name"] = "intfloat/e5-small-v2"
temp_configs["num_params"] = 33_400_000
llm_configs["llm-e5-small-v2"] = deepcopy(temp_configs)

# llm-roberta-large
temp_configs = dict()
temp_configs["hf_model_name"] = "FacebookAI/roberta-large"
temp_configs["num_params"] = 355_000_000
llm_configs["llm-roberta-large"] = deepcopy(temp_configs)

# llm-roberta-base
temp_configs = dict()
temp_configs["hf_model_name"] = "FacebookAI/roberta-base"
temp_configs["num_params"] = 125_000_000
llm_configs["llm-roberta-base"] = deepcopy(temp_configs)

# llm-all-MiniLM-L6-v2
temp_configs = dict()
temp_configs["hf_model_name"] = "sentence-transformers/all-MiniLM-L6-v2"
temp_configs["num_params"] = 22_700_000
llm_configs["llm-all-MiniLM-L6-v2"] = deepcopy(temp_configs)

# llm-all-MiniLM-L12-v2
temp_configs = dict()
temp_configs["hf_model_name"] = "sentence-transformers/all-MiniLM-L12-v2"
temp_configs["num_params"] = 33_400_000
llm_configs["llm-all-MiniLM-L12-v2"] = deepcopy(temp_configs)

# llm-all-mpnet-base-v2
temp_configs = dict()
temp_configs["hf_model_name"] = "sentence-transformers/all-mpnet-base-v2"
temp_configs["num_params"] = 100_000_000
llm_configs["llm-all-mpnet-base-v2"] = deepcopy(temp_configs)

# llm-sentence-t5-base
temp_configs = dict()
temp_configs["hf_model_name"] = "sentence-transformers/sentence-t5-base"
temp_configs["num_params"] = 110_000_000
llm_configs["llm-sentence-t5-base"] = deepcopy(temp_configs)

# llm-sentence-t5-large
temp_configs = dict()
temp_configs["hf_model_name"] = "sentence-transformers/sentence-t5-large"
temp_configs["num_params"] = 335_000_000
llm_configs["llm-sentence-t5-large"] = deepcopy(temp_configs)

# llm-sentence-t5-xl
temp_configs = dict()
temp_configs["hf_model_name"] = "sentence-transformers/sentence-t5-xl"
temp_configs["num_params"] = 1_240_000_000
llm_configs["llm-sentence-t5-xl"] = deepcopy(temp_configs)

# llm-sentence-t5-xxl
temp_configs = dict()
temp_configs["hf_model_name"] = "sentence-transformers/sentence-t5-xxl"
temp_configs["num_params"] = 4_800_000_000
llm_configs["llm-sentence-t5-xxl"] = deepcopy(temp_configs)

# llm-jasper-token-comp-0.6b
temp_configs = dict()
temp_configs["hf_model_name"] = "infgrad/Jasper-Token-Compression-600M"
temp_configs["num_params"] = 596_000_000
llm_configs["llm-jasper-token-comp-0.6b"] = deepcopy(temp_configs)

# llm-modernbert-base
temp_configs = dict()
temp_configs["hf_model_name"] = "answerdotai/ModernBERT-base"
temp_configs["num_params"] = 149_000_000
llm_configs["llm-modernbert-base"] = deepcopy(temp_configs)

# llm-modernbert-large
temp_configs = dict()
temp_configs["hf_model_name"] = "answerdotai/ModernBERT-large"
temp_configs["num_params"] = 395_000_000
llm_configs["llm-modernbert-large"] = deepcopy(temp_configs)

# llm-bge-large
temp_configs = dict()
temp_configs["hf_model_name"] = "BAAI/bge-large-en-v1.5"
temp_configs["num_params"] = 300_000_000
llm_configs["llm-bge-large"] = deepcopy(temp_configs)

# llm-bge-base
temp_configs = dict()
temp_configs["hf_model_name"] = "BAAI/bge-base-en-v1.5"
temp_configs["num_params"] = 100_000_000
llm_configs["llm-bge-base"] = deepcopy(temp_configs)

# llm-bge-small
temp_configs = dict()
temp_configs["hf_model_name"] = "BAAI/bge-small-en-v1.5"
temp_configs["num_params"] = 33_400_000
llm_configs["llm-bge-small"] = deepcopy(temp_configs)

# llm-f2llm-4b
temp_configs = dict()
temp_configs["hf_model_name"] = "codefuse-ai/F2LLM-4B"
temp_configs["num_params"] = 4_000_000_000
llm_configs["llm-f2llm-4b"] = deepcopy(temp_configs)

# llm-f2llm-1.7b
temp_configs = dict()
temp_configs["hf_model_name"] = "codefuse-ai/F2LLM-1.7B"
temp_configs["num_params"] = 1_700_000_000
llm_configs["llm-f2llm-1.7b"] = deepcopy(temp_configs)

# llm-f2llm-0.6b
temp_configs = dict()
temp_configs["hf_model_name"] = "codefuse-ai/F2LLM-0.6B"
temp_configs["num_params"] = 600_000_000
llm_configs["llm-f2llm-0.6b"] = deepcopy(temp_configs)

# llm-kalm-embed
temp_configs = dict()
temp_configs["hf_model_name"] = "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"
temp_configs["num_params"] = 500_000_000
llm_configs["llm-kalm-embed"] = deepcopy(temp_configs)

# llm-uae-large
temp_configs = dict()
temp_configs["hf_model_name"] = "WhereIsAI/UAE-Large-V1"
temp_configs["num_params"] = 300_000_000
llm_configs["llm-uae-large"] = deepcopy(temp_configs)

# llm-gemma-0.3b
temp_configs = dict()
temp_configs["hf_model_name"] = "google/embeddinggemma-300m"
temp_configs["num_params"] = 300_000_000
llm_configs["llm-gemma-0.3b"] = deepcopy(temp_configs)

# llm-deberta-v3-large
temp_configs = dict()
temp_configs["hf_model_name"] = "microsoft/deberta-v3-large"
temp_configs["num_params"] = 435_000_000
llm_configs["llm-deberta-v3-large"] = deepcopy(temp_configs)

# llm-deberta-v3-base
temp_configs = dict()
temp_configs["hf_model_name"] = "microsoft/deberta-v3-base"
temp_configs["num_params"] = 184_000_000
llm_configs["llm-deberta-v3-base"] = deepcopy(temp_configs)

# llm-deberta-v3-small
temp_configs = dict()
temp_configs["hf_model_name"] = "microsoft/deberta-v3-small"
temp_configs["num_params"] = 142_000_000
llm_configs["llm-deberta-v3-small"] = deepcopy(temp_configs)

# llm-deberta-v3-xsmall
temp_configs = dict()
temp_configs["hf_model_name"] = "microsoft/deberta-v3-xsmall"
temp_configs["num_params"] = 66_000_000
llm_configs["llm-deberta-v3-xsmall"] = deepcopy(temp_configs)

#llm-fasttext
temp_configs = dict()
temp_configs["hf_model_name"] = "fasttext"
temp_configs["num_params"] = 2_100_000_000
llm_configs["llm-fasttext"] = deepcopy(temp_configs)
