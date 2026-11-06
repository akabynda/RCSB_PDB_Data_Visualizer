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
from src.plotting.constants import AXIS_MAJOR_TICK_LENGTH, AXIS_MINOR_TICK_LENGTH
from src.plotting.membrane import residue_map_axis_limit


def cohort_table():
    return pd.DataFrame({
        'entry_id': ['1aaa', '1bbb', '1ccc', '1ddd', '1eee', '1fff'],
        'year': [2022, 2022, 2022, 2024, 2024, 2025],
        'deposited_model_count': [20] * 6,
        'opm_status': ['counted', 'counted', 'counted', 'no_opm_annotation', 'coordinates_unavailable', 'counted'],
        # 1CCC lacks 10 deposited residues in its OPM file: 70 modeled, 60 positioned.
        'opm_absent_residue_count': [0, 0, 10, None, None, 0],
        'opm_modeled_residue_count': [100, 80, 60, None, None, 100],
        'opm_membrane_residue_count': [25, 0, 20, None, None, 30],
        'opm_other_residue_count': [75, 80, 40, None, None, 70],
        # 1AAA is a homodimer of 50-residue chains with 12 and 13 membrane residues.
        'max_protein_entity_chain_count': [2, 1, 1, 1, 1, 1],
        'opm_copy_mean_modeled_residue_count': [50, 80, 60, None, None, 100],
        'opm_copy_mean_membrane_residue_count': [12.5, 0, 20, None, None, 30],
        'opm_copy_mean_other_residue_count': [37.5, 80, 40, None, None, 70],
        'opm_copy_min_membrane_residue_count': [12, 0, 20, None, None, 30],
        'opm_copy_max_membrane_residue_count': [13, 0, 20, None, None, 30],
    })


@pytest.fixture
def plotter():
    return PDBScientificPlotter(PlotConfig())


def test_entry_counts_and_plot_cutoff(plotter):
    table = plotter._prepare_solution_nmr_membrane_table(cohort_table())
    annual = plotter._summarize_solution_nmr_membrane_table(table)
    assert table['entry_id'].tolist() == ['1AAA', '1BBB', '1CCC', '1DDD', '1EEE']
    assert annual.index.tolist() == [2022, 2023, 2024]
    assert annual.tolist() == [3, 0, 2]


def test_empty_has_no_annual_counts(plotter):
    annual = plotter._summarize_solution_nmr_membrane_table(
        plotter._prepare_solution_nmr_membrane_table(cohort_table().iloc[:0]))
    assert annual.empty


def test_unknown_year_omissions_are_visible(plotter):
    table = cohort_table().astype(object)
    table.loc[0, 'year'] = None
    with pytest.warns(RuntimeWarning, match='1 entries.*1AAA'):
        prepared = plotter._prepare_solution_nmr_membrane_table(table)
    assert len(prepared) == 4


@pytest.mark.parametrize('column,value', [
    ('entry_id', None), ('entry_id', '1BBB'), ('year', -1), ('year', np.inf),
    ('year', 2023.5), ('deposited_model_count', 1), ('deposited_model_count', None),
    ('opm_status', 'unknown'), ('opm_membrane_residue_count', -1),
    ('opm_membrane_residue_count', None), ('opm_other_residue_count', 100),
    ('opm_modeled_residue_count', 90),  # 25 + 75 != 90
])
def test_invalid_metadata_never_plots_as_valid(plotter, column, value):
    table = cohort_table().astype(object)
    table.loc[0, column] = value
    with pytest.raises(ValueError):
        plotter._prepare_solution_nmr_membrane_table(table)


@pytest.mark.parametrize('row,column', [
    (3, 'opm_modeled_residue_count'),  # no annotation has no counts
    (4, 'opm_membrane_residue_count'),  # an unavailable file has no counts
])
def test_counts_without_counted_status_raise(plotter, row, column):
    table = cohort_table().astype(object)
    table.loc[row, column] = 90
    with pytest.raises(ValueError):
        plotter._prepare_solution_nmr_membrane_table(table)


@pytest.mark.parametrize('status', ['error', 'structure_mismatch'])
def test_entry_without_counts_stays_out_of_the_map(plotter, status):
    table = cohort_table().astype(object)
    table.loc[2, 'opm_status'] = status
    table.loc[2, [column for column in table.columns if column.startswith(('opm_copy', 'opm_modeled', 'opm_membrane', 'opm_other'))]] = None
    prepared = plotter._prepare_solution_nmr_membrane_table(table)
    _, included, _ = plotter._prepare_membrane_copy_counts(prepared)
    assert included.tolist() == [True, True, False, False, False]


@pytest.mark.parametrize('row,column,value', [
    (0, 'max_protein_entity_chain_count', 0), (0, 'max_protein_entity_chain_count', None),
    (0, 'max_protein_entity_chain_count', 1.5),
    (0, 'opm_copy_mean_other_residue_count', 38),  # 12.5 + 38 != 50
    (0, 'opm_copy_min_membrane_residue_count', 13), (0, 'opm_copy_max_membrane_residue_count', 12),
    (0, 'opm_copy_min_membrane_residue_count', 11.5), (0, 'opm_copy_mean_membrane_residue_count', -1),
    (0, 'opm_copy_mean_membrane_residue_count', np.inf),
    (0, 'opm_copy_mean_modeled_residue_count', 101),  # exceeds the entry total
    (1, 'opm_copy_mean_membrane_residue_count', 1),  # zero membrane residues in the entry total
    (1, 'opm_copy_mean_modeled_residue_count', 40),  # single copy must equal total
    (1, 'opm_copy_mean_other_residue_count', None),  # partial set of copy counts
    (2, 'opm_copy_mean_modeled_residue_count', 70),  # residues absent from OPM must not count
])
def test_invalid_copy_counts_never_plot_as_valid(plotter, row, column, value):
    table = cohort_table().astype(object)
    table.loc[row, column] = value
    with pytest.raises(ValueError):
        plotter._prepare_membrane_copy_counts(plotter._prepare_solution_nmr_membrane_table(table))


def test_copy_counts_for_uncounted_result_raise(plotter):
    table = cohort_table().astype(object)
    table.loc[4, ['opm_copy_mean_modeled_residue_count', 'opm_copy_mean_membrane_residue_count',
                  'opm_copy_mean_other_residue_count', 'opm_copy_min_membrane_residue_count',
                  'opm_copy_max_membrane_residue_count']] = [90, 20, 70, 20, 20]
    with pytest.raises(ValueError, match='only for counted'):
        plotter._prepare_membrane_copy_counts(plotter._prepare_solution_nmr_membrane_table(table))


def test_missing_copy_columns_raise(plotter):
    table = plotter._prepare_solution_nmr_membrane_table(cohort_table())
    with pytest.raises(ValueError, match='opm_copy_max_membrane_residue_count'):
        plotter._prepare_membrane_copy_counts(table.drop(columns='opm_copy_max_membrane_residue_count'))


def test_counted_entry_without_copy_counts_is_omitted_visibly(plotter):
    table = cohort_table().astype(object)
    table.loc[0, ['opm_copy_mean_modeled_residue_count', 'opm_copy_mean_membrane_residue_count',
                  'opm_copy_mean_other_residue_count', 'opm_copy_min_membrane_residue_count',
                  'opm_copy_max_membrane_residue_count']] = None
    with pytest.warns(RuntimeWarning, match='1 counted entries.*1AAA'):
        _, included, repeated = plotter._prepare_membrane_copy_counts(
            plotter._prepare_solution_nmr_membrane_table(table))
    assert included.tolist() == [False, True, True, False, False]
    assert repeated.tolist() == [True, False, False, False, False]


def test_unknown_result_not_coerced_to_zero(plotter):
    table = plotter._prepare_solution_nmr_membrane_table(cohort_table())
    assert table.loc[table['opm_status'] != 'counted', 'opm_membrane_residue_count'].isna().all()
    assert table.loc[table['entry_id'] == '1BBB', 'opm_membrane_residue_count'].tolist() == [0]


@pytest.mark.parametrize('empty', [False, True])
def test_all_standard_variants_and_data_counts(plotter, tmp_path, empty):
    data = cohort_table().iloc[:0] if empty else cohort_table()
    rendered = []
    def inspect(**kwargs):
        ax = kwargs['fig'].axes[0]
        name = kwargs['output_png'].name
        rendered.append(name)
        if name.startswith('annual'):
            assert sum(p.get_data().values.sum() for p in ax.patches) == (0 if empty else 5)
        else:
            assert sum(len(c.get_offsets()) for c in ax.collections) == (0 if empty else 3)
            # The scatter panel alone: no marginal histograms.
            assert len(kwargs['fig'].axes) == 1
            if not empty:
                # Copy means, not entry totals: the dimer is plotted at (25/2, 75/2);
                # the third entry counts only the residues in its OPM file (20, 40).
                np.testing.assert_array_equal(ax.collections[0].get_offsets(), [[0, 80], [20, 40]])
                np.testing.assert_array_equal(ax.collections[1].get_offsets(), [[12.5, 37.5]])
                # Both groups are circles of one size; only the colour differs.
                circle, pink = ax.collections
                np.testing.assert_array_equal(circle.get_paths()[0].vertices, pink.get_paths()[0].vertices)
                assert circle.get_sizes().tolist() == pink.get_sizes().tolist()
                assert circle.get_edgecolor().tolist() != pink.get_edgecolor().tolist()
            assert ax.get_legend() is None
            assert ax.get_xlabel() == 'Number of modeled residues in membrane regions'
            assert ax.get_ylabel() == 'Number of modeled residues in extramembrane regions'
            assert ax.get_xscale() == ax.get_yscale() == 'symlog'
            assert all(tick.gridline.get_visible() for tick in ax.xaxis.get_minor_ticks())
            assert all(tick.gridline.get_visible() for tick in ax.yaxis.get_minor_ticks())
            # Visible major and minor ticks on both symlog axes.
            for axis in (ax.xaxis, ax.yaxis):
                assert axis.get_major_ticks()[0].tick1line.get_markersize() == AXIS_MAJOR_TICK_LENGTH
                assert axis.get_minor_ticks()[0].tick1line.get_markersize() == AXIS_MINOR_TICK_LENGTH
            fig = kwargs['fig']
            width, height = fig.get_size_inches()
            main = ax.get_position()
            # A square panel; only float rounding of figure fractions differs.
            assert main.width * width == pytest.approx(main.height * height, rel=1e-12, abs=0)
            # Both axes end at the first major tick above the data: 100 > 80; 10 when empty.
            assert ax.get_xlim() == ax.get_ylim() == ((0, 10) if empty else (0, 100))
            assert ax.get_xticks().tolist() == ax.get_yticks().tolist()
            title = ax.get_title()
            if '_no_title' in name:
                assert title == ''
            else:
                assert title == plotter.config.nmr_membrane_residues_title and title.count('\n') == 1
            assert kwargs['savefig_bbox_inches'] == 'tight'
            assert ax.spines['top'].get_visible() == ('open_axes' not in name)
        assert ax.get_ylim()[0] == 0
    with patch.object(plotter, '_read_csv', return_value=data), patch.object(plotter, '_save_figure_files', side_effect=inspect):
        plotter.plot_solution_nmr_membrane_proteins(Path('data.csv'), *[tmp_path / f'{stem}.{ext}'
            for stem in ('annual', 'residues') for ext in ('png', 'svg')])
    assert len(rendered) == 8
    assert plt.get_fignums() == []
    exported = pd.read_csv(tmp_path / 'residues/residues.csv')
    if empty:
        assert exported.empty
    else:
        assert exported['entry_id'].tolist() == ['1AAA', '1BBB', '1CCC', '1DDD', '1EEE']
        assert exported['included_in_residue_plot'].tolist() == [True, True, True, False, False]
        # The fraction uses residues in the OPM file: 20 / 60, not 20 / 70.
        assert exported['opm_membrane_fraction'].iloc[:3].tolist() == [.25, 0, 20 / 60]
        assert exported['opm_membrane_fraction'].iloc[3:].isna().all()
        assert exported[['opm_absent_residue_count', 'opm_modeled_residue_count']].iloc[2].tolist() == [10, 60]
        # Entry totals and copy means are both retained.
        assert exported['opm_membrane_residue_count'].iloc[0] == 25
        assert exported['opm_copy_mean_membrane_residue_count'].iloc[0] == 12.5
        assert exported[['opm_copy_min_membrane_residue_count',
                         'opm_copy_max_membrane_residue_count']].iloc[0].tolist() == [12, 13]
        assert exported['repeated_protein_chains'].tolist() == [True, False, False, False, False]
        assert exported['opm_copy_mean_membrane_residue_count'].iloc[3:].isna().all()


@pytest.mark.parametrize('maximum,limit', [(0, 10), (9.5, 10), (10, 20), (80, 100), (448, 500), (500, 1000)])
def test_residue_map_axis_limit_is_first_major_tick_above_data(maximum, limit):
    assert residue_map_axis_limit(maximum) == limit
