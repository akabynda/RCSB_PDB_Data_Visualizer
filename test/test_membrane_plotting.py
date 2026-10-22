"""Cohort counts, explicit omissions, and standard plot exports."""
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.pdb_plot import PDBScientificPlotter, PlotConfig
from src.plotting.membrane import TMDET_STATUSES


def cohort_table():
    return pd.DataFrame({
        'entry_id': ['1aaa', '1bbb', '1ccc', '1ddd', '1eee', '1fff'],
        'year': [2022, 2022, 2022, 2024, 2024, 2025],
        'deposited_model_count': [20] * 6,
        'tmdet_status': [*TMDET_STATUSES, 'membrane_detected'],
        'tmdet_modeled_residue_count': [100, 80, 70, 10, 90, 100],
        'tmdet_tm_residue_count': [25, 0, 20, None, None, 30],
        'tmdet_other_residue_count': [75, 80, None, None, None, 70],
    })


@pytest.fixture
def plotter():
    return PDBScientificPlotter(PlotConfig())


def test_entry_counts_and_plot_cutoff(plotter):
    table = plotter._prepare_solution_nmr_membrane_table(cohort_table())
    annual, statuses = plotter._summarize_solution_nmr_membrane_table(table)
    assert table['entry_id'].tolist() == ['1AAA', '1BBB', '1CCC', '1DDD', '1EEE']
    assert annual.index.tolist() == [2022, 2023, 2024]
    assert annual.tolist() == [3, 0, 2]
    assert statuses.tolist() == [1, 1, 1, 1, 1]


def test_empty_retains_status_categories(plotter):
    annual, statuses = plotter._summarize_solution_nmr_membrane_table(
        plotter._prepare_solution_nmr_membrane_table(cohort_table().iloc[:0]))
    assert annual.empty
    assert statuses.tolist() == [0] * 5


def test_unknown_year_omissions_are_visible(plotter):
    table = cohort_table().astype(object)
    table.loc[0, 'year'] = None
    with pytest.warns(RuntimeWarning, match='1 entries.*1AAA'):
        prepared = plotter._prepare_solution_nmr_membrane_table(table)
    assert len(prepared) == 4


@pytest.mark.parametrize('column,value', [
    ('entry_id', None), ('entry_id', '1BBB'), ('year', -1), ('year', np.inf),
    ('year', 2023.5), ('deposited_model_count', 1), ('deposited_model_count', None),
    ('tmdet_status', 'unknown'), ('tmdet_tm_residue_count', -1),
    ('tmdet_tm_residue_count', None), ('tmdet_other_residue_count', 100),
])
def test_invalid_metadata_never_plots_as_valid(plotter, column, value):
    table = cohort_table().astype(object)
    table.loc[0, column] = value
    with pytest.raises(ValueError):
        plotter._prepare_solution_nmr_membrane_table(table)


def test_unknown_result_not_coerced_to_zero(plotter):
    table = plotter._prepare_solution_nmr_membrane_table(cohort_table())
    assert table.loc[table['tmdet_status'] == 'error', 'tmdet_tm_residue_count'].isna().all()
    assert table.loc[table['tmdet_status'] == 'no_membrane_detected', 'tmdet_tm_residue_count'].tolist() == [0]


@pytest.mark.parametrize('empty', [False, True])
def test_all_standard_variants_and_data_counts(plotter, empty):
    data = cohort_table().iloc[:0] if empty else cohort_table()
    rendered = []
    def inspect(**kwargs):
        ax = kwargs['fig'].axes[0]
        name = kwargs['output_png'].name
        rendered.append(name)
        if name.startswith(('annual', 'status')):
            assert sum(p.get_height() for p in ax.patches) == (0 if empty else 5)
        else:
            assert sum(len(c.get_offsets()) for c in ax.collections) == (0 if empty else 2)
        assert ax.get_ylim()[0] == 0
    with patch.object(plotter, '_read_csv', return_value=data), patch.object(plotter, '_save_figure_files', side_effect=inspect):
        plotter.plot_solution_nmr_membrane_proteins(Path('data.csv'), *[Path(f'{stem}.{ext}')
            for stem in ('annual', 'status', 'residues') for ext in ('png', 'svg')])
    assert len(rendered) == 12
    assert plt.get_fignums() == []
