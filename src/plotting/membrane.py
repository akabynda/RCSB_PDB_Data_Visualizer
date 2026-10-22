"""Entry-level first-model TmDet statistics for the solution-NMR cohort."""

from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

from ..dataset.io.common import _atomic_write_csv_rows
from .constants import TITLE_FONTSIZE, YEAR_MAJOR_TICK_STEP
from .membrane_statistics import FRACTION_GROUPS, TMDET_STATUSES, membrane_time_statistics

TMDET_LABELS = (
    'Membrane detected\ncomplete annotation', 'No membrane\ndetected',
    'Partial\nannotation', 'No analyzable\nprotein', 'Calculation\nerror',
)


class MembranePlotsMixin:
    """Count entries once and keep failed/incomplete calculations visible."""

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
                    ('TM residues', 'Other modeled residues'), self.config.area_colors[:2])
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
                                                    ('TM residues', 'Other modeled residues'))]
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
            'tmdet_modeled_residue_count', 'tmdet_tm_residue_count', 'tmdet_other_residue_count',
        }, name)
        table = df.copy()
        ids = table['entry_id'].astype('string').str.strip().str.upper()
        if ids.isna().any() or (ids == '').any() or ids.duplicated().any():
            raise ValueError(f'{name} requires one row per entry with a nonempty ID')
        table['entry_id'] = ids
        for column in ('year', 'deposited_model_count', 'tmdet_modeled_residue_count',
                       'tmdet_tm_residue_count', 'tmdet_other_residue_count'):
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
        counts = table.loc[complete, ['tmdet_modeled_residue_count', 'tmdet_tm_residue_count', 'tmdet_other_residue_count']]
        if (counts.isna().any().any() or (counts.iloc[:, 0] <= 0).any()
                or (counts.iloc[:, 1] + counts.iloc[:, 2] != counts.iloc[:, 0]).any()):
            raise ValueError('Complete TmDet results require consistent modeled/TM/other residue counts')
        negative = table['tmdet_status'] == 'no_membrane_detected'
        if (table.loc[negative, 'tmdet_tm_residue_count'] != 0).any():
            raise ValueError('Negative TmDet results must have zero TM residues')
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
        usable = table.loc[table['tmdet_status'].isin(TMDET_STATUSES[:2])]
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

        def draw_residues(ax):
            for status, label, color, marker in zip(TMDET_STATUSES[:2],
                    ('Membrane detected', 'No membrane detected'), colors[:2], ('o', 'x')):
                rows = usable.loc[usable['tmdet_status'] == status]
                ax.scatter(rows['tmdet_tm_residue_count'].to_numpy(dtype=float),
                           rows['tmdet_other_residue_count'].to_numpy(dtype=float),
                           s=24, alpha=.45, c=color, marker=marker,
                           linewidths=.7, label=f'{label} (n = {len(rows)})', clip_on=False)
            max_x = float(usable['tmdet_tm_residue_count'].max()) if len(usable) else 0
            max_y = float(usable['tmdet_other_residue_count'].max()) if len(usable) else 0
            ax.set_xlim(0, max(1, max_x * 1.05))
            ax.set_ylim(0, max(1, max_y * 1.15))
            ax.text(.98, .03, f'{len(table) - len(usable)} incomplete/unavailable results omitted',
                    transform=ax.transAxes, ha='right', fontsize=9)
            self._add_legend(ax, loc='upper right', fontsize=10)

        with plt.rc_context():
            self._scientific_style()
            self._render_figure(annual_output_png, annual_output_svg, self.config.nmr_membrane_annual_title,
                                'Number of PDB entries', draw_annual, use_year_x_ticks=use_year_ticks)
            self._render_figure(coverage_output_png, coverage_output_svg, self.config.nmr_membrane_coverage_title,
                                'Number of PDB entries', draw_status, x_label='', use_year_x_ticks=False,
                                auto_minor_ticks=False)
            self._render_figure(residues_output_png, residues_output_svg, self.config.nmr_membrane_residues_title,
                                'Other modeled residues per PDB entry', draw_residues,
                                x_label='Modeled residues in TmDet H/B regions per PDB entry', use_year_x_ticks=False)

    def plot_solution_nmr_membrane_trends(
        self, data_path: Path, composition_output_png: Path, composition_output_svg: Path,
        fraction_output_png: Path, fraction_output_svg: Path,
        period_output_png: Path, period_output_svg: Path,
    ) -> None:
        table = self._prepare_solution_nmr_membrane_table(self._read_csv(data_path))
        annual, annual_fractions = membrane_time_statistics(table)
        _, period_fractions = membrane_time_statistics(table, period_years=5)
        colors = ('#0072B2', '#D55E00')
        labels = ('Entries with complete TmDet results', 'Entries with detected TM regions')
        use_year_ticks = not annual.empty and annual['period_start'].max() - annual['period_start'].min() >= YEAR_MAJOR_TICK_STEP

        def draw_composition(ax):
            total = annual['n_entries'].replace(0, np.nan).to_numpy(dtype=float)
            shares = [annual[column].to_numpy(dtype=float) / total * 100
                      for column in ('n_with_tm', 'n_without_tm', 'n_unavailable')]
            self._draw_membrane_share_areas(
                ax, annual['period_start'], shares,
                ('TM regions detected', 'No TM regions detected', 'Incomplete / unavailable'),
                self.config.homolog_timing_colors)
            if not use_year_ticks:
                ax.set_xticks(annual['period_start'].to_numpy(dtype=int))
            if ax.collections:
                self._add_legend(ax, loc='upper left', fontsize=10)

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
            self._add_legend(ax, handles=handles, loc='upper center', bbox_to_anchor=(.5, -.23), fontsize=9)

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
                                self.config.nmr_membrane_period_title, 'TM residue fraction per PDB entry (%)',
                                draw_period, x_label='Deposition period', use_year_x_ticks=False,
                                auto_minor_ticks=False)
