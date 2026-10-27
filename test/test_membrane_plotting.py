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
from matplotlib.scale import SymmetricalLogTransform

from src.plotting.constants import AXIS_MAJOR_TICK_LENGTH, AXIS_MINOR_TICK_LENGTH
from src.plotting.membrane import (
    RESIDUE_MAP_LINSCALE, RESIDUE_MAP_LINTHRESH, TMDET_STATUSES, residue_map_bin_edges,
)
from src.plotting.membrane_statistics import _fraction_statistics, membrane_time_statistics


def cohort_table():
    return pd.DataFrame({
        'entry_id': ['1aaa', '1bbb', '1ccc', '1ddd', '1eee', '1fff'],
        'year': [2022, 2022, 2022, 2024, 2024, 2025],
        'deposited_model_count': [20] * 6,
        'tmdet_status': [*TMDET_STATUSES, 'membrane_detected'],
        'tmdet_modeled_residue_count': [100, 80, 70, 10, 90, 100],
        # 1CCC excludes an unannotated 10-residue chain: 70 modeled, 60 counted.
        'tmdet_excluded_chain_count': [0, 0, 1, 0, None, 0],
        'tmdet_excluded_residue_count': [0, 0, 10, 0, None, 0],
        'tmdet_counted_residue_count': [100, 80, 60, None, None, 100],
        'tmdet_membrane_residue_count': [25, 0, 20, None, None, 30],
        'tmdet_other_residue_count': [75, 80, 40, None, None, 70],
        # 1AAA is a homodimer of 50-residue chains with 12 and 13 membrane residues.
        'max_protein_entity_chain_count': [2, 1, 1, 1, 1, 1],
        'tmdet_copy_mean_modeled_residue_count': [50, 80, 60, None, None, 100],
        'tmdet_copy_mean_membrane_residue_count': [12.5, 0, 20, None, None, 30],
        'tmdet_copy_mean_other_residue_count': [37.5, 80, 40, None, None, 70],
        'tmdet_copy_min_membrane_residue_count': [12, 0, 20, None, None, 30],
        'tmdet_copy_max_membrane_residue_count': [13, 0, 20, None, None, 30],
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
    ('tmdet_status', 'unknown'), ('tmdet_membrane_residue_count', -1),
    ('tmdet_membrane_residue_count', None), ('tmdet_other_residue_count', 100),
    ('tmdet_counted_residue_count', 90),  # complete results count every modeled residue
])
def test_invalid_metadata_never_plots_as_valid(plotter, column, value):
    table = cohort_table().astype(object)
    table.loc[0, column] = value
    with pytest.raises(ValueError):
        plotter._prepare_solution_nmr_membrane_table(table)


@pytest.mark.parametrize('row,column,value', [
    (2, 'tmdet_counted_residue_count', 61),  # 20 + 40 != 61
    (2, 'tmdet_counted_residue_count', None),  # other count without counted residues
    (2, 'tmdet_counted_residue_count', 80), (2, 'tmdet_other_residue_count', 60),  # above modeled 70
    (4, 'tmdet_counted_residue_count', 90),  # failed calculations have no counts
])
def test_invalid_partial_or_unavailable_counts_raise(plotter, row, column, value):
    table = cohort_table().astype(object)
    table.loc[row, column] = value
    with pytest.raises(ValueError):
        plotter._prepare_solution_nmr_membrane_table(table)


def test_partial_entry_without_annotated_chains_is_unavailable(plotter):
    table = cohort_table().astype(object)
    table.loc[2, ['tmdet_counted_residue_count', 'tmdet_membrane_residue_count', 'tmdet_other_residue_count',
                  'tmdet_copy_mean_modeled_residue_count', 'tmdet_copy_mean_membrane_residue_count',
                  'tmdet_copy_mean_other_residue_count', 'tmdet_copy_min_membrane_residue_count',
                  'tmdet_copy_max_membrane_residue_count']] = None
    prepared = plotter._prepare_solution_nmr_membrane_table(table)
    _, included, _ = plotter._prepare_membrane_copy_counts(prepared)
    assert included.tolist() == [True, True, False, False, False]
    counts, _ = membrane_time_statistics(prepared)
    assert counts['n_unavailable'].tolist() == [1, 0, 2]


@pytest.mark.parametrize('row,column,value', [
    (0, 'max_protein_entity_chain_count', 0), (0, 'max_protein_entity_chain_count', None),
    (0, 'max_protein_entity_chain_count', 1.5),
    (0, 'tmdet_copy_mean_other_residue_count', 38),  # 12.5 + 38 != 50
    (0, 'tmdet_copy_min_membrane_residue_count', 13), (0, 'tmdet_copy_max_membrane_residue_count', 12),
    (0, 'tmdet_copy_min_membrane_residue_count', 11.5), (0, 'tmdet_copy_mean_membrane_residue_count', -1),
    (0, 'tmdet_copy_mean_membrane_residue_count', np.inf),
    (0, 'tmdet_copy_mean_modeled_residue_count', 101),  # exceeds the entry total
    (1, 'tmdet_copy_mean_membrane_residue_count', 1),  # valid negative must stay zero
    (1, 'tmdet_copy_mean_modeled_residue_count', 40),  # single copy must equal total
    (1, 'tmdet_copy_mean_other_residue_count', None),  # partial set of copy counts
    (2, 'tmdet_copy_mean_modeled_residue_count', 70),  # excluded chain must not count
])
def test_invalid_copy_counts_never_plot_as_valid(plotter, row, column, value):
    table = cohort_table().astype(object)
    table.loc[row, column] = value
    with pytest.raises(ValueError):
        plotter._prepare_membrane_copy_counts(plotter._prepare_solution_nmr_membrane_table(table))


def test_copy_counts_for_unavailable_result_raise(plotter):
    table = cohort_table().astype(object)
    table.loc[4, ['tmdet_copy_mean_modeled_residue_count', 'tmdet_copy_mean_membrane_residue_count',
                  'tmdet_copy_mean_other_residue_count', 'tmdet_copy_min_membrane_residue_count',
                  'tmdet_copy_max_membrane_residue_count']] = [90, 20, 70, 20, 20]
    with pytest.raises(ValueError, match='only for usable'):
        plotter._prepare_membrane_copy_counts(plotter._prepare_solution_nmr_membrane_table(table))


def test_missing_copy_columns_raise(plotter):
    table = plotter._prepare_solution_nmr_membrane_table(cohort_table())
    with pytest.raises(ValueError, match='tmdet_copy_max_membrane_residue_count'):
        plotter._prepare_membrane_copy_counts(table.drop(columns='tmdet_copy_max_membrane_residue_count'))


def test_usable_entry_without_copy_counts_is_omitted_visibly(plotter):
    table = cohort_table().astype(object)
    table.loc[0, ['tmdet_copy_mean_modeled_residue_count', 'tmdet_copy_mean_membrane_residue_count',
                  'tmdet_copy_mean_other_residue_count', 'tmdet_copy_min_membrane_residue_count',
                  'tmdet_copy_max_membrane_residue_count']] = None
    with pytest.warns(RuntimeWarning, match='1 usable entries.*1AAA'):
        _, included, repeated = plotter._prepare_membrane_copy_counts(
            plotter._prepare_solution_nmr_membrane_table(table))
    assert included.tolist() == [False, True, True, False, False]
    assert repeated.tolist() == [True, False, False, False, False]


def test_unknown_result_not_coerced_to_zero(plotter):
    table = plotter._prepare_solution_nmr_membrane_table(cohort_table())
    assert table.loc[table['tmdet_status'] == 'error', 'tmdet_membrane_residue_count'].isna().all()
    assert table.loc[table['tmdet_status'] == 'no_membrane_detected', 'tmdet_membrane_residue_count'].tolist() == [0]


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
        elif name.startswith('status'):
            assert sum(p.get_height() for p in ax.patches) == (0 if empty else 5)
        else:
            assert sum(len(c.get_offsets()) for c in ax.collections) == (0 if empty else 3)
            assert len(kwargs['fig'].axes) == 3
            top, right = kwargs['fig'].axes[1:]
            for marginal in (top, right):
                distinct, stacked = marginal.patches[:2]
                assert distinct.get_data().values.sum() == (0 if empty else 2)
                assert stacked.get_data().values.sum() == (0 if empty else 3)
                np.testing.assert_array_equal(stacked.get_data().baseline, distinct.get_data().values)
            if not empty:
                # Copy means, not entry totals: the dimer is plotted at (25/2, 75/2);
                # the partial entry counts only its annotated chains (20, 40).
                np.testing.assert_array_equal(ax.collections[0].get_offsets(), [[0, 80], [20, 40]])
                np.testing.assert_array_equal(ax.collections[1].get_offsets(), [[12.5, 37.5]])
                # Both groups are circles of one size; only the colour differs.
                circle, pink = ax.collections
                np.testing.assert_array_equal(circle.get_paths()[0].vertices, pink.get_paths()[0].vertices)
                assert circle.get_sizes().tolist() == pink.get_sizes().tolist()
                assert circle.get_edgecolor().tolist() != pink.get_edgecolor().tolist()
                np.testing.assert_array_equal(top.patches[1].get_data().edges, residue_map_bin_edges(80))
                # x = 0 is in [0, 2), 12.5 in [10, 15.8) and 20 in [15.8, 25.1).
                np.testing.assert_array_equal(top.patches[1].get_data().values, [1, 0, 0, 0, 0, 1, 1, 0, 0, 0])
                # y = 37.5 is in [25.1, 39.8), 40 in [39.8, 63.1) and 80 in [63.1, 100).
                np.testing.assert_array_equal(right.patches[0].get_data().values, [0] * 8 + [1, 1])
                np.testing.assert_array_equal(right.patches[1].get_data().values, [0] * 7 + [1, 1, 1])
            assert ax.get_legend() is None
            assert ax.get_xlabel() == 'Number of modeled residues in membrane regions (symlog)'
            assert ax.get_ylabel() == 'Number of modeled residues in extramembrane regions (symlog)'
            assert ax.get_xscale() == ax.get_yscale() == 'symlog'
            # Marginal bins have equal widths on the drawn symlog axis.
            widths = np.diff(ax.xaxis.get_transform().transform(top.patches[1].get_data().edges))
            np.testing.assert_allclose(widths, widths[0], rtol=1e-12, atol=0)
            assert all(tick.gridline.get_visible() for tick in ax.xaxis.get_minor_ticks())
            assert all(tick.gridline.get_visible() for tick in ax.yaxis.get_minor_ticks())
            # Every panel has visible major and minor ticks on both axes, symlog ones included.
            for panel in (ax, top, right):
                for axis in (panel.xaxis, panel.yaxis):
                    assert axis.get_major_ticks()[0].tick1line.get_markersize() == AXIS_MAJOR_TICK_LENGTH
                    assert axis.get_minor_ticks()[0].tick1line.get_markersize() == AXIS_MINOR_TICK_LENGTH
            assert ax.get_shared_x_axes().joined(ax, top)
            assert ax.get_shared_y_axes().joined(ax, right)
            fig = kwargs['fig']
            fig.canvas.draw()
            width, height = fig.get_size_inches()
            main, upper, side = (panel.get_position() for panel in (ax, top, right))
            # Equal inches per residue (1:1); only float rounding of figure fractions differs.
            x_scale = main.width * width / np.diff(ax.get_xlim())[0]
            y_scale = main.height * height / np.diff(ax.get_ylim())[0]
            assert x_scale == pytest.approx(y_scale, rel=1e-12, abs=0)
            assert (upper.x0, upper.width) == pytest.approx((main.x0, main.width), rel=1e-12, abs=0)
            assert (side.y0, side.height) == pytest.approx((main.y0, main.height), rel=1e-12, abs=0)
            # Both axes span zero to the first bin edge above the data: 100 > 80; 10 when empty.
            assert ax.get_xlim() == ax.get_ylim() == ((0, 10) if empty else (0, 100))
            assert ax.get_xticks().tolist() == ax.get_yticks().tolist()
            assert kwargs['savefig_bbox_inches'] == 'tight'
            assert all(panel.spines['top'].get_visible() == ('open_axes' not in name)
                       for panel in (ax, top, right))
        assert ax.get_ylim()[0] == 0
    with patch.object(plotter, '_read_csv', return_value=data), patch.object(plotter, '_save_figure_files', side_effect=inspect):
        plotter.plot_solution_nmr_membrane_proteins(Path('data.csv'), *[tmp_path / f'{stem}.{ext}'
            for stem in ('annual', 'status', 'residues') for ext in ('png', 'svg')])
    assert len(rendered) == 12
    assert plt.get_fignums() == []
    exported = pd.read_csv(tmp_path / 'residues/residues.csv')
    if empty:
        assert exported.empty
    else:
        assert exported['entry_id'].tolist() == ['1AAA', '1BBB', '1CCC', '1DDD', '1EEE']
        assert exported['included_in_residue_plot'].tolist() == [True, True, True, False, False]
        # The partial entry's fraction uses counted residues: 20 / 60, not 20 / 70.
        assert exported['tmdet_membrane_fraction'].iloc[:3].tolist() == [.25, 0, 20 / 60]
        assert exported['tmdet_membrane_fraction'].iloc[3:].isna().all()
        assert exported[['tmdet_excluded_chain_count', 'tmdet_excluded_residue_count',
                         'tmdet_counted_residue_count']].iloc[2].tolist() == [1, 10, 60]
        # Entry totals and copy means are both retained.
        assert exported['tmdet_membrane_residue_count'].iloc[0] == 25
        assert exported['tmdet_copy_mean_membrane_residue_count'].iloc[0] == 12.5
        assert exported[['tmdet_copy_min_membrane_residue_count',
                         'tmdet_copy_max_membrane_residue_count']].iloc[0].tolist() == [12, 13]
        assert exported['repeated_protein_chains'].tolist() == [True, False, False, False, False]
        assert exported['tmdet_copy_mean_membrane_residue_count'].iloc[3:].isna().all()


def test_residue_map_bins_have_equal_symlog_widths():
    # 0-10 in steps of 2, then 5 bins per decade up to the first edge above the maximum.
    np.testing.assert_array_equal(residue_map_bin_edges(0), [0, 2, 4, 6, 8, 10])
    edges = residue_map_bin_edges(100)
    np.testing.assert_allclose(edges[5:], 10 ** np.arange(1, 2.4, .2), rtol=1e-12, atol=0)
    assert edges[-2] <= 100 < edges[-1]
    # Equal widths under matplotlib's own symlog transform with the plotted parameters.
    transform = SymmetricalLogTransform(base=10, linthresh=RESIDUE_MAP_LINTHRESH,
                                        linscale=RESIDUE_MAP_LINSCALE)
    widths = np.diff(transform.transform(edges))
    np.testing.assert_allclose(widths, widths[0], rtol=1e-12, atol=0)


def test_fraction_statistics_weight_entries_and_preserve_missing_years(plotter):
    table = plotter._prepare_solution_nmr_membrane_table(cohort_table())
    counts, fractions = membrane_time_statistics(table)
    assert counts['period_start'].tolist() == [2022, 2023, 2024]
    assert counts['n_entries'].tolist() == [3, 0, 2]
    assert counts['n_usable'].tolist() == [3, 0, 0]
    assert counts['n_with_membrane'].tolist() == [2, 0, 0]
    assert counts['n_without_membrane'].tolist() == [1, 0, 0]
    assert counts['n_unavailable'].tolist() == [0, 0, 2]
    assert counts[list(TMDET_STATUSES)].sum(axis=1).tolist() == [3, 0, 2]
    all_entries = fractions.loc[fractions['group'] == 'all_usable'].set_index('period_start')
    positive = fractions.loc[fractions['group'] == 'with_membrane'].set_index('period_start')
    # Mean of 25/100, 0/80 and 20/60, not the pooled residue fraction 45/240.
    # Tolerance covers float rounding of the thirds only.
    assert all_entries.loc[2022, 'mean'] == pytest.approx((.25 + 0 + 1 / 3) / 3, rel=0, abs=1e-15)
    # Linear quartiles of (0, 1/4, 1/3).
    assert all_entries.loc[2022, ['q1', 'med', 'q3']].tolist() == pytest.approx(
        [.125, .25, (.25 + 1 / 3) / 2], rel=0, abs=1e-15)
    assert positive.loc[2022, 'mean'] == pytest.approx((.25 + 1 / 3) / 2, rel=0, abs=1e-15)
    assert all_entries.loc[[2023, 2024], 'mean'].isna().all()
    assert positive.loc[[2023, 2024], 'mean'].isna().all()


def test_zero_fraction_is_observed_but_positive_subgroup_is_empty(plotter):
    table = plotter._prepare_solution_nmr_membrane_table(cohort_table().iloc[[1]])
    counts, fractions = membrane_time_statistics(table)
    assert counts['n_without_membrane'].tolist() == [1]
    assert fractions['n'].tolist() == [1, 0]
    assert fractions.iloc[0]['mean'] == 0
    assert np.isnan(fractions.iloc[1]['mean'])


def test_five_year_calendar_boundaries_and_empty_bins(plotter):
    table = cohort_table().iloc[[0, 1, 3]].copy()
    table['year'] = [1999, 2000, 2010]
    counts, fractions = membrane_time_statistics(
        plotter._prepare_solution_nmr_membrane_table(table), period_years=5)
    assert counts['period_start'].tolist() == [1995, 2000, 2005, 2010]
    assert counts['period_end'].tolist() == [1999, 2004, 2009, 2014]
    assert counts['n_entries'].tolist() == [1, 1, 0, 1]
    assert counts['n_unavailable'].tolist() == [0, 0, 0, 1]
    assert fractions.loc[fractions['period_start'].isin([2005, 2010]), 'mean'].isna().all()


def test_box_summary_includes_extreme_observations_and_zero_iqr():
    # Known quartiles: Q1=Q2=Q3=0; the positive observation remains a flier.
    stats = _fraction_statistics([0, 0, 0, 0, 1])
    assert stats == dict(n=5, mean=.2, q1=0, med=0, q3=0,
                         whislo=0, whishi=0, fliers=[1])
    equal = _fraction_statistics([.5, .5, .5])
    assert equal['whislo'] == equal['whishi'] == .5
    assert equal['fliers'] == []


@pytest.mark.parametrize('empty', [False, True])
def test_trend_exports_include_samples_and_omit_unknown_fractions(plotter, tmp_path, empty):
    data = cohort_table().iloc[:0] if empty else cohort_table()
    rendered = []

    def inspect(**kwargs):
        axes = kwargs['fig'].axes
        ax = axes[0]
        name = kwargs['output_png'].name
        rendered.append(name)
        if name.startswith('composition'):
            assert len(ax.collections) == (0 if empty else 6)
            if not empty:
                # Three shares sum to 100%; the empty 2023 bin remains entirely blank.
                for area, x_bounds, y_bounds in zip(ax.collections,
                    [(2021.5, 2022.5)] * 3 + [(2023.5, 2024.5)] * 3,
                    [(0, 200 / 3), (200 / 3, 100), (100, 100),
                     (0, 0), (0, 0), (0, 100)],
                ):
                    vertices = area.get_paths()[0].vertices
                    np.testing.assert_equal([vertices[:, 0].min(), vertices[:, 0].max()], x_bounds)
                    # Only rounding in conversion to percent is tolerated.
                    np.testing.assert_allclose([vertices[:, 1].min(), vertices[:, 1].max()],
                                               y_bounds, rtol=0, atol=1e-12)
        elif name.startswith('fraction'):
            assert len(axes) == 2
            # Mean membrane fractions (%) of (0, 1/4, 1/3) and of the positive (1/4, 1/3).
            for panel, fraction in zip(axes, ((.25 + 1 / 3) / 3 * 100, (.25 + 1 / 3) / 2 * 100)):
                assert len(panel.collections) == (0 if empty else 2)
                for area, bounds in zip(panel.collections, [(0, fraction), (fraction, 100)]):
                    vertices = area.get_paths()[0].vertices
                    np.testing.assert_equal([vertices[:, 0].min(), vertices[:, 0].max()], [2021.5, 2022.5])
                    np.testing.assert_allclose([vertices[:, 1].min(), vertices[:, 1].max()],
                                               bounds, rtol=0, atol=1e-12)
                assert panel.spines['top'].get_visible() == ('open_axes' not in name)
        else:
            assert len(ax.patches) == (0 if empty else 2)
        assert ax.get_ylim()[0] == 0

    with patch.object(plotter, '_read_csv', return_value=data), patch.object(
        plotter, '_save_figure_files', side_effect=inspect,
    ):
        plotter.plot_solution_nmr_membrane_trends(Path('data.csv'), *[
            tmp_path / f'{stem}.{ext}' for stem in ('composition', 'fraction', 'period')
            for ext in ('png', 'svg')])
    assert len(rendered) == 12
    assert plt.get_fignums() == []
    assert len(list(tmp_path.rglob('*.csv'))) == 3
    annual = pd.read_csv(tmp_path / 'fraction/fraction.csv')
    period = pd.read_csv(tmp_path / 'period/period.csv')
    if empty:
        assert annual.empty and period.empty
    else:
        assert annual['n'].sum() == 5
        assert annual['mean'].isna().sum() == 4
        assert period['period_start'].tolist() == [2020, 2020]
        assert period['n_entries'].tolist() == [5, 5]
        assert period['n_unavailable'].tolist() == [2, 2]
        assert period['fliers'].tolist() == ['[]', '[]']
