"""Entry-level first-model TmDet statistics for the solution-NMR cohort."""

from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, MaxNLocator, StrMethodFormatter
from matplotlib.patches import Patch

from ..dataset.io.common import _atomic_write_csv_rows
from .constants import AXIS_MAJOR_TICK_LENGTH, AXIS_MINOR_TICK_LENGTH, TITLE_FONTSIZE, YEAR_MAJOR_TICK_STEP
from .membrane_statistics import (
    FRACTION_GROUPS, TMDET_STATUSES, membrane_time_statistics, usable_tmdet_results,
)

TMDET_LABELS = (
    'Membrane detected\ncomplete annotation', 'No membrane\ndetected',
    'Partial\nannotation', 'No analyzable\nprotein', 'Calculation\nerror',
)
COPY_COUNT_COLUMNS = (
    'tmdet_copy_mean_modeled_residue_count', 'tmdet_copy_mean_membrane_residue_count',
    'tmdet_copy_mean_other_residue_count', 'tmdet_copy_min_membrane_residue_count',
    'tmdet_copy_max_membrane_residue_count',
)
# Copy means are rationals with small denominators stored as floats; real
# inconsistencies are at least 1/copies residues, far above this rounding bound.
COPY_COUNT_ATOL = 1e-9
# Residue map: side of the square scatter panel, in inches.
RESIDUE_MAP_PANEL_INCHES = 5.0
# Both residue-map axes are symlog: linear below 10 residues, logarithmic above.
# matplotlib draws the linear range linscale / (1 - 1/10) decades wide, so 0.9
# makes 0-10 as wide as one decade.
RESIDUE_MAP_LINTHRESH = 10
RESIDUE_MAP_LINSCALE = .9
RESIDUE_MAP_TICKS = (0, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)


def residue_map_axis_limit(maximum):
    """Upper limit of both residue-map axes: the first major tick of at least 10 above `maximum`."""
    return next(tick for tick in RESIDUE_MAP_TICKS if tick >= RESIDUE_MAP_LINTHRESH and tick > maximum)


class MembranePlotsMixin:
    """Count entries once and keep failed/incomplete calculations visible."""

    @staticmethod
    def _prepare_membrane_copy_counts(table):
        """Validate copy-mean counts; return numeric copy table, plotted rows and oligomer flags."""
        name = 'SOLUTION NMR membrane residue map'
        missing = {'max_protein_entity_chain_count', *COPY_COUNT_COLUMNS} - set(table.columns)
        if missing:
            raise ValueError(f'{name} is missing required columns: {", ".join(sorted(missing))}')
        copies = pd.DataFrame(index=table.index)
        for column in ('max_protein_entity_chain_count', *COPY_COUNT_COLUMNS):
            raw = table[column].astype('string').str.strip()
            copies[column] = pd.to_numeric(raw.mask(raw == ''), errors='raise')
        chains = copies['max_protein_entity_chain_count']
        if chains.isna().any() or (chains < 1).any() or (chains % 1 != 0).any():
            raise ValueError(f'{name}: invalid max_protein_entity_chain_count')
        known = copies[list(COPY_COUNT_COLUMNS)].notna()
        usable = usable_tmdet_results(table)
        if (known.any(axis=1) != known.all(axis=1)).any() or (known.all(axis=1) & ~usable).any():
            raise ValueError(f'{name}: copy counts must be all present, and only for usable TmDet results')
        included = usable & known.all(axis=1)
        unavailable = table.loc[usable & ~included, 'entry_id'].tolist()
        if unavailable:
            warnings.warn(f'{name}: omitting {len(unavailable)} usable entries without copy counts: '
                          + ', '.join(unavailable), RuntimeWarning, stacklevel=3)
        rows = copies.loc[included]
        modeled, membrane, other, low, high = (rows[column] for column in COPY_COUNT_COLUMNS)
        totals = table.loc[included]
        single = rows['max_protein_entity_chain_count'] == 1
        atol = COPY_COUNT_ATOL
        if (not np.isfinite(rows[list(COPY_COUNT_COLUMNS)]).all().all() or (rows < 0).any().any()
                or (modeled <= 0).any() or (low % 1 != 0).any() or (high % 1 != 0).any()
                or ((membrane + other - modeled).abs() > atol).any()
                or (low > membrane + atol).any() or (membrane > high + atol).any()
                or (modeled > totals['tmdet_counted_residue_count'] + atol).any()
                or (membrane > totals['tmdet_membrane_residue_count'] + atol).any()
                or (membrane[totals['tmdet_status'] == 'no_membrane_detected'] != 0).any()
                or (modeled[single] != totals.loc[single, 'tmdet_counted_residue_count']).any()
                or (membrane[single] != totals.loc[single, 'tmdet_membrane_residue_count']).any()):
            raise ValueError(f'{name}: inconsistent copy-mean residue counts')
        return copies, included, chains > 1

    def _render_membrane_residue_map(self, table, output_png, output_svg):
        """Plot copy-mean counts on a square panel with one symlog scale on both axes."""
        copies, included, repeated = self._prepare_membrane_copy_counts(table)
        exported = table.reindex(columns=[
            'entry_id', 'year', 'title', 'membrane_annotation_sources',
            'deposited_model_count', 'protein_entity_count', 'protein_chain_count',
            'tmdet_status', 'tmdet_error', 'tmdet_modeled_residue_count',
            'tmdet_excluded_chain_count', 'tmdet_excluded_residue_count', 'tmdet_counted_residue_count',
            'tmdet_membrane_residue_count', 'tmdet_other_residue_count',
        ]).copy()
        exported['max_protein_entity_chain_count'] = copies['max_protein_entity_chain_count']
        for column in COPY_COUNT_COLUMNS:
            exported[column] = copies[column]
        exported = exported.join(table.reindex(columns=[
            'assembly_global_symmetry', 'assembly_global_stoichiometry', 'assembly_oligomeric_state',
        ]))
        exported['repeated_protein_chains'] = repeated
        exported['included_in_residue_plot'] = included
        exported['tmdet_membrane_fraction'] = (
            table['tmdet_membrane_residue_count'] / table['tmdet_counted_residue_count']
        ).where(included)
        _atomic_write_csv_rows(self._grouped_figure_output_path(output_png).with_suffix('.csv'),
                               exported.columns, exported.astype(object).where(
                                   exported.notna(), None).itertuples(index=False, name=None))
        x = copies.loc[included, 'tmdet_copy_mean_membrane_residue_count'].to_numpy(dtype=float)
        y = copies.loc[included, 'tmdet_copy_mean_other_residue_count'].to_numpy(dtype=float)
        oligomer = repeated[included].to_numpy(dtype=bool)
        # One symlog range on both axes of a square panel keeps the map 1:1.
        limit = residue_map_axis_limit(max(x.max(), y.max()) if len(x) else 0)
        major = [tick for tick in RESIDUE_MAP_TICKS if tick <= limit]
        minor = [tick for tick in (*range(1, 10), *(m * 10 ** d for d in range(1, 4) for m in range(2, 10)))
                 if tick <= limit]
        for with_title, open_axes in ((True, False), (False, False), (True, True), (False, True)):
            # Generous margins for labels and the two-line title; saving crops to the drawn content.
            left, bottom, right, top = 1.0, .8, .2, .9 if with_title else .1
            width, height = left + RESIDUE_MAP_PANEL_INCHES + right, bottom + RESIDUE_MAP_PANEL_INCHES + top
            fig = plt.figure(figsize=(width, height))
            ax = fig.add_axes((left / width, bottom / height,
                               RESIDUE_MAP_PANEL_INCHES / width, RESIDUE_MAP_PANEL_INCHES / height))
            ax.set_xscale('symlog', linthresh=RESIDUE_MAP_LINTHRESH, linscale=RESIDUE_MAP_LINSCALE)
            ax.set_yscale('symlog', linthresh=RESIDUE_MAP_LINTHRESH, linscale=RESIDUE_MAP_LINSCALE)
            ax.scatter(x[~oligomer], y[~oligomer], s=22, marker='o', facecolors=(0, .447, .698, .22),
                       edgecolors='#0072B2', linewidths=.55, clip_on=False, label='Each protein chain distinct')
            ax.scatter(x[oligomer], y[oligomer], s=22, marker='o', facecolors=(.8, .475, .655, .3),
                       edgecolors='#CC79A7', linewidths=.55, clip_on=False, label='Identical chains, mean per copy')
            ax.set(xlim=(0, limit), ylim=(0, limit),
                   xlabel='Number of modeled residues in membrane regions',
                   ylabel='Number of modeled residues in extramembrane regions')
            self._configure_minor_ticks(ax, use_year_x_ticks=False)
            self._configure_boxed_axes(ax)
            if open_axes:
                self._configure_open_axes(ax)
            # Set after the shared helper, which would put linear minor locators on these axes.
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(FixedLocator(major))
                axis.set_minor_locator(FixedLocator(minor))
                axis.set_major_formatter(StrMethodFormatter('{x:g}'))
            # The style draws zero-length x ticks, and the shared helper restores them only
            # for uniform major steps, which the symlog axis lacks.
            ax.tick_params(which='major', length=AXIS_MAJOR_TICK_LENGTH)
            ax.tick_params(which='minor', length=AXIS_MINOR_TICK_LENGTH)
            ax.grid(which='minor', linewidth=.5, alpha=.12)
            if with_title:
                self._set_title(ax, self.config.nmr_membrane_residues_title)
            png, svg = output_png, output_svg
            if not with_title:
                png, svg = self._titleless_output_path(png), self._titleless_output_path(svg)
            if open_axes:
                png, svg = self._open_axes_output_path(png), self._open_axes_output_path(svg)
            self._save_figure_files(fig=fig, output_png=png, output_svg=svg, savefig_bbox_inches='tight')
            plt.close(fig)

    def _draw_membrane_share_areas(self, ax, years, shares, labels, colors):
        """Fill complete calendar-year bins, leaving undefined denominators blank."""
        years = np.asarray(years, dtype=int)
        shares = np.asarray(shares, dtype=float)
        valid = np.isfinite(shares).all(axis=0)
        # Split before expanding steps so a missing year cannot erase its neighbour.
        boundaries = np.flatnonzero(np.diff(np.r_[False, valid, False]))
        edges = self._step_edges(years)
        for start, stop in boundaries.reshape(-1, 2):
            base = np.zeros(stop - start)
            for values, label, color, line in zip(shares, labels, colors, ('-', '--', ':')):
                top = base + values[start:stop]
                ax.fill_between(edges[start:stop + 1], self._step_values(base),
                                self._step_values(top), step='post', color=color,
                                alpha=.85, linewidth=.8, linestyle=line,
                                label=label if start == boundaries[0] else '_nolegend_')
                base = top
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        if len(years):
            ax.set_xlim(edges[0], edges[-1])

    def _render_membrane_fraction_panels(self, fractions, output_png, output_svg, labels, use_year_ticks):
        """Keep the overlapping entry groups in separate 100% residue panels."""
        for with_title, open_axes in ((True, False), (False, False), (True, True), (False, True)):
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=self.config.figure_size(1.4))
            for ax, group, label in zip(axes, FRACTION_GROUPS, labels):
                rows = fractions.loc[fractions['group'] == group]
                mean = rows['mean'].to_numpy(dtype=float) * 100
                self._draw_membrane_share_areas(
                    ax, rows['period_start'], [mean, 100 - mean],
                    ('Membrane residues', 'Other modeled residues'), self.config.area_colors[:2])
                if use_year_ticks:
                    self._configure_year_axis_ticks(ax)
                else:
                    ax.set_xticks(rows['period_start'].to_numpy(dtype=int))
                self._remove_zero_y_tick(ax)
                self._configure_minor_ticks(ax=ax, use_year_x_ticks=use_year_ticks)
                self._configure_boxed_axes(ax)
                if open_axes:
                    self._configure_open_axes(ax)
                ax.set_title(label, fontsize=11, pad=8)
                ax.set_ylabel('Mean residue fraction (%)')
                ax.margins(x=0)
            axes[0].tick_params(axis='x', which='both', labelbottom=False)
            axes[-1].set_xlabel(self.config.x_label)
            handles = [Patch(facecolor=color, edgecolor=color, alpha=.85, linestyle=line, label=label)
                       for color, line, label in zip(self.config.area_colors[:2], ('-', '--'),
                                                    ('Membrane residues', 'Other modeled residues'))]
            self._add_legend(axes[-1], handles=handles, loc='upper center',
                             bbox_to_anchor=(.5, -.3), ncol=2, fontsize=10)
            if with_title:
                fig.suptitle(self.config.nmr_membrane_fraction_title,
                             fontsize=TITLE_FONTSIZE, fontweight='bold')
            fig.tight_layout()
            png, svg = output_png, output_svg
            if not with_title:
                png, svg = self._titleless_output_path(png), self._titleless_output_path(svg)
            if open_axes:
                png, svg = self._open_axes_output_path(png), self._open_axes_output_path(svg)
            self._save_figure_files(fig=fig, output_png=png, output_svg=svg)
            plt.close(fig)

    @classmethod
    def _prepare_solution_nmr_membrane_table(cls, df):
        name = 'SOLUTION NMR membrane protein CSV'
        cls._validate_required_columns(df, {
            'entry_id', 'year', 'deposited_model_count', 'tmdet_status',
            'tmdet_modeled_residue_count', 'tmdet_counted_residue_count',
            'tmdet_membrane_residue_count', 'tmdet_other_residue_count',
        }, name)
        table = df.copy()
        ids = table['entry_id'].astype('string').str.strip().str.upper()
        if ids.isna().any() or (ids == '').any() or ids.duplicated().any():
            raise ValueError(f'{name} requires one row per entry with a nonempty ID')
        table['entry_id'] = ids
        for column in ('year', 'deposited_model_count', 'tmdet_modeled_residue_count',
                       'tmdet_counted_residue_count', 'tmdet_membrane_residue_count',
                       'tmdet_other_residue_count'):
            raw = table[column].astype('string').str.strip()
            numbers = pd.to_numeric(raw.mask(raw == ''), errors='raise')
            known = numbers.dropna()
            if not np.isfinite(known).all() or (known < 0).any() or (known % 1 != 0).any():
                raise ValueError(f'{name}: invalid {column}')
            table[column] = numbers
        if (table['year'].dropna() < 1).any():
            raise ValueError('Invalid deposition year')
        if table['deposited_model_count'].isna().any() or (table['deposited_model_count'] <= 1).any():
            raise ValueError('Membrane cohort requires more than one deposited model')
        if not table['tmdet_status'].isin(TMDET_STATUSES).all():
            raise ValueError('Invalid TmDet status')
        complete = table['tmdet_status'].isin(TMDET_STATUSES[:2])
        partial = table['tmdet_status'] == 'partial_annotation'
        counted = table[['tmdet_counted_residue_count', 'tmdet_other_residue_count']].notna()
        usable = complete | (partial & counted.all(axis=1))
        counts = table.loc[usable, ['tmdet_counted_residue_count', 'tmdet_membrane_residue_count',
                                    'tmdet_other_residue_count', 'tmdet_modeled_residue_count']]
        if (counts.isna().any().any() or (counts.iloc[:, 0] <= 0).any()
                or (counts.iloc[:, 1] + counts.iloc[:, 2] != counts.iloc[:, 0]).any()
                or (counts.iloc[:, 0] > counts.iloc[:, 3]).any()
                or (table.loc[complete, 'tmdet_counted_residue_count']
                    != table.loc[complete, 'tmdet_modeled_residue_count']).any()
                or (counted.any(axis=1) & ~usable).any()):
            raise ValueError('Usable TmDet results require consistent counted/membrane/other residue counts')
        negative = table['tmdet_status'] == 'no_membrane_detected'
        if (table.loc[negative, 'tmdet_membrane_residue_count'] != 0).any():
            raise ValueError('Negative TmDet results must have zero membrane residues')
        unknown = table.loc[table['year'].isna(), 'entry_id'].tolist()
        if unknown:
            warnings.warn(f'{name}: omitting {len(unknown)} entries with unknown deposition year: '
                          + ', '.join(unknown), RuntimeWarning, stacklevel=2)
        table = table.loc[table['year'].notna()].copy()
        table['year'] = table['year'].astype(int)
        return cls._limit_year_column(table).sort_values(['year', 'entry_id'])

    @staticmethod
    def _summarize_solution_nmr_membrane_table(table):
        annual = table.groupby('year').size()
        if not annual.empty:
            annual = annual.reindex(range(int(annual.index.min()), int(annual.index.max()) + 1), fill_value=0)
        statuses = table['tmdet_status'].value_counts().reindex(TMDET_STATUSES, fill_value=0)
        return annual.astype(int), statuses.astype(int)

    def plot_solution_nmr_membrane_proteins(
        self, data_path: Path, annual_output_png: Path, annual_output_svg: Path,
        coverage_output_png: Path, coverage_output_svg: Path,
        residues_output_png: Path, residues_output_svg: Path,
    ) -> None:
        table = self._prepare_solution_nmr_membrane_table(self._read_csv(data_path))
        annual, statuses = self._summarize_solution_nmr_membrane_table(table)
        colors = ('#0072B2', '#E69F00', '#009E73', '#999999', '#D55E00')
        hatches = ('//', '\\\\', 'xx', '..', '++')
        use_year_ticks = not annual.empty and annual.index.max() - annual.index.min() >= YEAR_MAJOR_TICK_STEP

        def draw_annual(ax):
            if len(annual):
                self._plot_step_series(ax, annual.index, annual.to_numpy(),
                                       color=self.config.nmr_color, linewidth=2.0)
            maximum = int(annual.max()) if len(annual) else 0
            ax.set_ylim(0, max(1, maximum * 1.18))
            if len(annual):
                ax.set_xlim(annual.index.min() - .6, annual.index.max() + .6)
            if not use_year_ticks:
                ax.set_xticks(annual.index)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        def draw_status(ax):
            bars = ax.bar(range(len(statuses)), statuses.to_numpy(), color=colors, edgecolor='black', linewidth=.8)
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)
            ax.set_xticks(range(len(statuses)), TMDET_LABELS, fontsize=9)
            ax.set_xlim(-.6, len(statuses) - .4)
            ax.set_ylim(0, max(1, int(statuses.max()) * 1.15))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.bar_label(bars, padding=4, fontsize=11)

        with plt.rc_context():
            self._scientific_style()
            self._render_figure(annual_output_png, annual_output_svg, self.config.nmr_membrane_annual_title,
                                'Number of PDB entries', draw_annual, use_year_x_ticks=use_year_ticks)
            self._render_figure(coverage_output_png, coverage_output_svg, self.config.nmr_membrane_coverage_title,
                                'Number of PDB entries', draw_status, x_label='', use_year_x_ticks=False,
                                auto_minor_ticks=False)
            self._render_membrane_residue_map(table, residues_output_png, residues_output_svg)

    def plot_solution_nmr_membrane_trends(
        self, data_path: Path, composition_output_png: Path, composition_output_svg: Path,
        fraction_output_png: Path, fraction_output_svg: Path,
        period_output_png: Path, period_output_svg: Path,
    ) -> None:
        table = self._prepare_solution_nmr_membrane_table(self._read_csv(data_path))
        annual, annual_fractions = membrane_time_statistics(table)
        _, period_fractions = membrane_time_statistics(table, period_years=5)
        colors = ('#0072B2', '#D55E00')
        labels = ('Entries with usable TmDet results', 'Entries with detected membrane regions')
        use_year_ticks = not annual.empty and annual['period_start'].max() - annual['period_start'].min() >= YEAR_MAJOR_TICK_STEP

        def draw_composition(ax):
            total = annual['n_entries'].replace(0, np.nan).to_numpy(dtype=float)
            shares = [annual[column].to_numpy(dtype=float) / total * 100
                      for column in ('n_with_membrane', 'n_without_membrane', 'n_unavailable')]
            self._draw_membrane_share_areas(
                ax, annual['period_start'], shares,
                ('Membrane regions detected', 'No membrane regions detected', 'Incomplete / unavailable'),
                self.config.homolog_timing_colors)
            if not use_year_ticks:
                ax.set_xticks(annual['period_start'].to_numpy(dtype=int))
            if ax.collections:
                self._add_legend(ax, loc='lower right', fontsize=10)

        def draw_period(ax):
            periods = period_fractions['period_start'].drop_duplicates().tolist()
            for group, color, offset, hatch in zip(FRACTION_GROUPS, colors, (-.18, .18), ('//', '')):
                rows = period_fractions.loc[period_fractions['group'] == group]
                boxes, positions = [], []
                for index, row in enumerate(rows.to_dict('records')):
                    position = index + offset
                    if row['n']:
                        box = {key: row[key] * 100 for key in ('q1', 'med', 'q3', 'whislo', 'whishi')}
                        box['fliers'] = np.asarray(row['fliers']) * 100
                        boxes.append(box)
                        positions.append(position)
                    ax.text(position, 103, str(row['n']), color=color, ha='center', fontsize=8)
                if boxes:
                    ax.bxp(boxes, positions=positions, widths=.28, manage_ticks=False,
                           patch_artist=True, boxprops={'facecolor': color, 'alpha': .45, 'hatch': hatch},
                           medianprops={'color': 'black', 'linewidth': 1.3,
                                        'clip_on': False, 'zorder': 3},
                           capprops={'clip_on': False, 'zorder': 3},
                           flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': color,
                                       'markeredgecolor': color, 'alpha': .6, 'clip_on': False})
            ax.set_xticks(range(len(periods)), [f'{year}\u2013\n{year + 4}' for year in periods], fontsize=9)
            ax.set_xlim(-.6, max(.6, len(periods) - .4))
            ax.set_ylim(0, 125)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            handles = [Patch(facecolor=color, edgecolor='black', alpha=.45, hatch=hatch, label=label)
                       for color, hatch, label in zip(colors, ('//', ''), labels)]
            self._add_legend(ax, handles=handles, loc='upper center', fontsize=9)

        for output, summary in ((composition_output_png, annual),
                                (fraction_output_png, annual_fractions),
                                (period_output_png, period_fractions)):
            exported = summary.copy()
            if 'fliers' in exported:
                exported['fliers'] = exported['fliers'].map(json.dumps)
            _atomic_write_csv_rows(self._grouped_figure_output_path(output).with_suffix('.csv'),
                                   exported.columns, exported.fillna('').itertuples(index=False, name=None))
        with plt.rc_context():
            self._scientific_style()
            self._render_figure(composition_output_png, composition_output_svg,
                                self.config.nmr_membrane_composition_title, 'Share of PDB entries (%)',
                                draw_composition, use_year_x_ticks=use_year_ticks)
            self._render_membrane_fraction_panels(annual_fractions, fraction_output_png,
                                                  fraction_output_svg, labels, use_year_ticks)
            self._render_figure(period_output_png, period_output_svg,
                                self.config.nmr_membrane_period_title, 'Membrane residue fraction per PDB entry (%)',
                                draw_period, x_label='Deposition period', use_year_x_ticks=False,
                                auto_minor_ticks=False)
