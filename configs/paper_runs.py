"""Registry of the evaluation runs the paper figures read.

Each entry is one run of a ``scripts/evaluate_scripts/script_evaluate*.py``
script. Its run name ``<ABLATION>/<save_dir>`` is the folder the script writes
under ``results/`` (the ABLATION segment is fixed by the script and its flags,
the save_dir is the ``--save_dir`` given here). ``scripts/compile_results.py
<run name>`` then writes ``results/compiled_results/result_<flat name>.csv``,
which is the file the figure scripts read via ``figures._main.load_run(key)``.

Fields: ``run`` (run name), ``script``, ``flags`` (extra CLI flags besides the
common ``-dn all-wide -ns 3 -fi all``), ``methods`` (``-m`` tokens; ``None`` =
the full catalogue of ``METHOD_NAMES.md``) and ``figures`` (scripts reading it).

``python configs/paper_runs.py`` prints the evaluate + compile commands.
"""

from itertools import product

LLAMA, QWEN, NEMOTRON = 'llm-llama-3.1-8b', 'llm-qwen3-8b', 'llm-llama-nemotron-embed-1b-v2'
MINILM, E5_BASE, E5_SMALL = 'llm-all-MiniLM-L6-v2', 'llm-e5-base-v2', 'llm-e5-small-v2'
OPT, BGE, E5_LARGE = 'llm-opt-6.7b', 'llm-bge-large', 'llm-e5-large-v2'
FASTTEXT, JASPER = 'llm-fasttext', 'llm-jasper-token-comp-0.6b'
CLASSIC = ('ridge', 'xgb', 'extrees')

EVAL = 'scripts/evaluate_scripts/script_evaluate.py'
STANDSCAL = '-norm True'   # standard scaling before the 30-PCA
NO_PCA = '-nopca True'     # first 30 embedding dimensions instead of PCA


def _grid(encoders, learners):
    """``num-str_<encoder>_<learner>`` for every encoder x learner pair."""
    return tuple(f'num-str_{e}_{l}' for e, l in product(encoders, learners))


def _run(run, methods, figures, script=EVAL, flags=''):
    return dict(run=run, script=script, flags=flags, methods=methods, figures=figures)


_E2E = ('num-str_contexttab_contexttab',)
_POSTPROC = ('figure_2', 'figure_E2')   # post-processing comparison
_EXTRA = ('figure_3', 'figure_4a', 'figure_E5')  # pipelines added to the main benchmark

PAPER_RUNS = {
    # --- Main benchmark (30-PCA) ------------------------------------------
    'main': _run(
        'default/benchmark_main', None, ('all figures',), flags='-ti all'),
    'tabicl': _run(
        'default/tabicl', _grid(('tabvec', LLAMA, QWEN), ('tabicl',)),
        _POSTPROC + ('figure_3', 'figure_4a')),
    'tabicl_minilm_e5_nemotron': _run(
        'default/tabicl_minilm_e5_nemotron', _grid((MINILM, E5_BASE, NEMOTRON), ('tabicl',)),
        _POSTPROC),
    'opt_bge_tabpfn_tabicl': _run(
        'default/opt_bge_tabpfn_tabicl', _grid((OPT, BGE), ('tabpfn', 'tabicl')), _POSTPROC),
    'nemotron_classic': _run(
        'default/nemotron_ridge_xgb_extrees', _grid((NEMOTRON,), CLASSIC),
        _POSTPROC + ('figure_E11',)),
    'tabm': _run(
        'default/tabm', _grid(('tabvec', MINILM, QWEN, LLAMA), ('tabm',)), _EXTRA),
    'realmlp': _run(
        'default/realmlp', _grid(('tabvec', MINILM, QWEN, LLAMA), ('realmlp',)),
        _EXTRA + ('figure_E10',)),
    'mambular': _run(
        'default/mambular', ('num-str_mambular_mambular',), _EXTRA),
    'llama_tabpfn_pca60': _run(
        'default/llama_tabpfn_pca60', _grid((LLAMA,), ('tabpfn',)),
        ('figure_E4a', 'figure_E4b'), flags='-ndim 60'),
    'llama_tabpfn_pca120': _run(
        'default/llama_tabpfn_pca120', _grid((LLAMA,), ('tabpfn',)),
        ('figure_E4a', 'figure_E4b'), flags='-ndim 120'),

    # --- Standard scaling before the 30-PCA -------------------------------
    'standscal_tabpfn': _run(
        'default/standscal_tabpfn',
        _grid((MINILM, E5_BASE, JASPER, LLAMA, 'llm-llama-3.2-1b', 'llm-llama-3.2-3b',
               NEMOTRON), ('tabpfn',)),
        _POSTPROC + _EXTRA, flags=STANDSCAL),
    'standscal_llama_tabicl': _run(
        'default/standscal_llama_tabicl', _grid((LLAMA,), ('tabicl',)),
        _POSTPROC + ('figure_3', 'figure_4a'), flags=STANDSCAL),
    'standscal_qwen_tabpfn_tabicl': _run(
        'default/standscal_qwen_tabpfn_tabicl', _grid((QWEN,), ('tabpfn', 'tabicl')),
        _POSTPROC, flags=STANDSCAL),
    'standscal_tabicl_minilm_e5_nemotron': _run(
        'default/standscal_tabicl_minilm_e5_nemotron',
        _grid((MINILM, E5_BASE, NEMOTRON), ('tabicl',)), _POSTPROC, flags=STANDSCAL),
    'standscal_llama_qwen_nemotron_classic': _run(
        'default/standscal_llama_qwen_nemotron_ridge_xgb_extrees',
        _grid((LLAMA, QWEN, NEMOTRON), CLASSIC), _POSTPROC, flags=STANDSCAL),
    'standscal_minilm_bge_e5large_classic': _run(
        'default/standscal_minilm_bge_e5large_ridge_xgb_extrees',
        _grid((MINILM, BGE, E5_LARGE), CLASSIC), _POSTPROC, flags=STANDSCAL),
    'standscal_e5base_classic': _run(
        'default/standscal_e5base_ridge_xgb_extrees', _grid((E5_BASE,), CLASSIC),
        _POSTPROC, flags=STANDSCAL),
    'standscal_opt_bge_classic': _run(
        'default/standscal_opt_bge_ridge_xgb_extrees', _grid((OPT, BGE), CLASSIC),
        _POSTPROC, flags=STANDSCAL),
    'standscal_opt_bge_tabpfn_tabicl': _run(
        'default/standscal_opt_bge_tabpfn_tabicl', _grid((OPT, BGE), ('tabpfn', 'tabicl')),
        _POSTPROC, flags=STANDSCAL),

    # --- No PCA: first 30 embedding dimensions ----------------------------
    'nopca_qwen_tabpfn': _run(
        'no-pca/qwen_tabpfn', _grid((QWEN,), ('tabpfn',)),
        _POSTPROC + _EXTRA, flags=NO_PCA),
    'nopca_qwen_tabicl': _run(
        'no-pca/qwen_tabicl', _grid((QWEN,), ('tabicl',)),
        _POSTPROC + ('figure_3', 'figure_4a'), flags=NO_PCA),
    'nopca_llama_minilm_e5_nemotron_tabpfn_tabicl': _run(
        'no-pca/llama_minilm_e5_nemotron_tabpfn_tabicl',
        _grid((LLAMA, MINILM, E5_BASE, NEMOTRON), ('tabpfn', 'tabicl')),
        _POSTPROC, flags=NO_PCA),
    'nopca_llama_qwen_nemotron_classic': _run(
        'no-pca/llama_qwen_nemotron_ridge_xgb_extrees',
        _grid((LLAMA, QWEN, NEMOTRON), CLASSIC), _POSTPROC, flags=NO_PCA),
    'nopca_minilm_bge_e5large_classic': _run(
        'no-pca/minilm_bge_e5large_ridge_xgb_extrees',
        _grid((MINILM, BGE, E5_LARGE), CLASSIC), _POSTPROC, flags=NO_PCA),
    'nopca_e5base_classic': _run(
        'no-pca/e5base_ridge_xgb_extrees', _grid((E5_BASE,), CLASSIC),
        _POSTPROC, flags=NO_PCA),
    'nopca_opt_bge_classic': _run(
        'no-pca/opt_bge_ridge_xgb_extrees', _grid((OPT, BGE), CLASSIC),
        _POSTPROC, flags=NO_PCA),
    'nopca_opt_bge_tabpfn_tabicl': _run(
        'no-pca/opt_bge_tabpfn_tabicl', _grid((OPT, BGE), ('tabpfn', 'tabicl')),
        _POSTPROC, flags=NO_PCA),

    # --- Cardinality threshold 30 (low-cardinality strings not embedded) --
    'ct30_ohe_llama_qwen_ridge': _run(
        'ct30-ohe/llama_qwen_ridge', _grid((LLAMA, QWEN), ('ridge',)),
        _EXTRA, script='scripts/evaluate_scripts/script_evaluate_30_thres_OHE.py'),
    'ct30_ohe_llama_qwen_nemotron_classic': _run(
        'ct30-ohe/llama_qwen_nemotron_ridge_xgb_extrees',
        _grid((LLAMA, QWEN, NEMOTRON), CLASSIC), ('figure_E11',),
        script='scripts/evaluate_scripts/script_evaluate_30_thres_OHE.py'),
    'ct30_passthrough_llama_qwen_xgb_tabpfn': _run(
        'ct30-passthrough/llama_qwen_xgb_tabpfn', _grid((LLAMA, QWEN), ('xgb', 'tabpfn')),
        ('figure_E11',),
        script='scripts/evaluate_scripts/script_evaluate_30_thres_lowcard_passthrough.py'),
    'ct30_passthrough_llama_qwen_nemotron_xgb': _run(
        'ct30-passthrough/llama_qwen_nemotron_xgb', _grid((LLAMA, QWEN, NEMOTRON), ('xgb',)),
        ('figure_E11',),
        script='scripts/evaluate_scripts/script_evaluate_30_thres_lowcard_passthrough.py'),
    'ct30_passthrough_nemotron_tabpfn': _run(
        'ct30-passthrough/nemotron_tabpfn', _grid((NEMOTRON,), ('tabpfn',)),
        ('figure_E11',),
        script='scripts/evaluate_scripts/script_evaluate_30_thres_lowcard_passthrough.py'),

    # --- Data-variant ablations -------------------------------------------
    'feature_eng': _run(
        'feature-eng/benchmark_main',
        _grid(('tabvec', MINILM, E5_SMALL, FASTTEXT, JASPER, LLAMA, QWEN),
              CLASSIC + ('tabpfn',)) + _E2E,
        ('figure_E7',), script='scripts/evaluate_scripts/script_evaluate_feature_eng.py'),
    'raw_targets': _run(
        'no-target-transform/benchmark_main',
        _grid(('tabvec', MINILM, E5_SMALL, FASTTEXT, JASPER, LLAMA, QWEN),
              CLASSIC + ('tabpfn',))
        + _E2E + ('num-str_catboost_catboost', 'num-str_tabstar_tabstar'),
        ('figure_E8',), script='scripts/evaluate_scripts/script_evaluate_skewness.py'),
    'imputed': _run(
        'imputed/uniform_impute',
        _grid(('tabvec', MINILM, E5_SMALL, FASTTEXT, LLAMA, QWEN), CLASSIC + ('tabpfn',))
        + _grid((JASPER,), CLASSIC)
        + ('num-str_mambular_mambular', 'num-str_tabstar_tabstar'),
        ('figure_E9',), script='scripts/evaluate_scripts/script_evaluate_impute_missing_val.py',
        flags='-mvp uniform_impute'),
    'full_data': _run(
        'full-data/benchmark_main',
        _grid(('tabvec', MINILM, E5_BASE, FASTTEXT, JASPER, LLAMA, QWEN),
              CLASSIC + ('realmlp',)) + _E2E,
        ('figure_E10',), script='scripts/evaluate_scripts/script_evaluate_FULL.py'),
}


def commands():
    """Evaluate + compile command lines for every registry entry."""
    lines = []
    for key, spec in PAPER_RUNS.items():
        ablation, save_dir = spec['run'].split('/', 1)
        methods = ' '.join(spec['methods']) if spec['methods'] else '<all of METHOD_NAMES.md>'
        lines.append(f"# {key}  ->  {', '.join(spec['figures'])}")
        lines.append(
            f"python {spec['script']} --save_dir {save_dir} -m {methods} "
            f"-dn all-wide -ns 3 -fi all -dv cuda -cf True -oc False {spec['flags']}".rstrip()
        )
        lines.append(f"python scripts/compile_results.py {spec['run']}\n")
    return '\n'.join(lines)


if __name__ == '__main__':
    print(commands())
