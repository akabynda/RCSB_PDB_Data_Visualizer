"""Descriptive entry-weighted summaries of validated first-model TmDet results."""

import numpy as np
import pandas as pd


TMDET_STATUSES = (
    'membrane_detected', 'no_membrane_detected', 'partial_annotation',
    'no_analyzable_protein', 'error',
)
FRACTION_GROUPS = ('all_usable', 'with_membrane')


def usable_tmdet_results(table):
    """Complete results, plus partial ones with counts over their fully annotated chains."""
    return table['tmdet_status'].isin(TMDET_STATUSES[:3]) & table['tmdet_other_residue_count'].notna()


def _fraction_statistics(values):
    """Linear quartiles and Tukey whiskers; an empty group has no estimates."""
    values = np.asarray(values, dtype=float)
    result = dict.fromkeys(('mean', 'q1', 'med', 'q3', 'whislo', 'whishi'), np.nan)
    result.update(n=len(values), fliers=[])
    if not len(values):
        return result
    q1, median, q3 = np.percentile(values, [25, 50, 75], method='linear')
    iqr = q3 - q1
    inside = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
    # With tiny groups an interpolated quartile can lie beyond an observed whisker.
    low, high = min(q1, inside.min()), max(q3, inside.max())
    result.update(mean=values.mean(), q1=q1, med=median, q3=q3,
                  whislo=low, whishi=high,
                  fliers=values[(values < low) | (values > high)].tolist())
    return result


def membrane_time_statistics(table, period_years=1):
    """Summarize a prepared table in calendar bins, retaining empty intervening bins.

    Each PDB entry has equal weight. Valid negative results contribute zero;
    partial results count only fully annotated chains; unavailable results
    contribute only to cohort composition counts.
    """
    if period_years not in (1, 5):
        raise ValueError('Membrane summaries support annual or five-year bins')
    count_columns = ['period_start', 'period_end', 'n_entries', 'n_usable',
                     'n_with_membrane', 'n_without_membrane', 'n_unavailable', *TMDET_STATUSES]
    fraction_columns = ['period_start', 'period_end', 'group', 'n_entries',
                        'n_unavailable', 'n', 'mean', 'q1', 'med', 'q3',
                        'whislo', 'whishi', 'fliers']
    counts, fractions = [], []
    if table.empty:
        return pd.DataFrame(columns=count_columns), pd.DataFrame(columns=fraction_columns)
    starts = table['year'] // period_years * period_years
    for start in range(int(starts.min()), int(starts.max()) + 1, period_years):
        rows = table.loc[starts == start]
        usable = rows.loc[usable_tmdet_results(rows)]
        values = (usable['tmdet_membrane_residue_count']
                  / usable['tmdet_counted_residue_count']).to_numpy(dtype=float)
        count = dict(period_start=start, period_end=start + period_years - 1,
                     n_entries=len(rows), n_usable=len(values),
                     n_with_membrane=int((values > 0).sum()),
                     n_without_membrane=int((values == 0).sum()),
                     n_unavailable=len(rows) - len(values))
        count.update(rows['tmdet_status'].value_counts().reindex(TMDET_STATUSES, fill_value=0).to_dict())
        counts.append(count)
        for group, sample in zip(FRACTION_GROUPS, (values, values[values > 0])):
            fractions.append({key: count[key] for key in (
                'period_start', 'period_end', 'n_entries', 'n_unavailable')}
                | {'group': group} | _fraction_statistics(sample))
    return pd.DataFrame(counts, columns=count_columns), pd.DataFrame(fractions, columns=fraction_columns)
