"""Entry-level first-model TmDet statistics for the solution-NMR cohort."""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .constants import MAX_PLOT_YEAR, YEAR_MAJOR_TICK_STEP

TMDET_STATUSES = (
    'membrane_detected', 'no_membrane_detected', 'partial_annotation',
    'no_analyzable_protein', 'error',
)
TMDET_LABELS = (
    'Membrane detected\ncomplete annotation', 'No membrane\ndetected',
    'Partial\nannotation', 'No analyzable\nprotein', 'Calculation\nerror',
)


class MembranePlotsMixin:
    """Count entries once and keep failed/incomplete calculations visible."""

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
            ax.bar(annual.index, annual.to_numpy(), width=.8, color=colors[0], edgecolor='black', linewidth=.6)
            maximum = int(annual.max()) if len(annual) else 0
            ax.set_ylim(0, max(1, maximum * 1.18))
            if len(annual):
                ax.set_xlim(annual.index.min() - .6, annual.index.max() + .6)
            if not use_year_ticks:
                ax.set_xticks(annual.index)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.text(.02, .95, f'n = {len(table)} PDB entries', transform=ax.transAxes, va='top', fontsize=11)

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
