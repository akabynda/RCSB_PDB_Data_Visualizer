"""Additional unit tests for the plotting data transformations and CLI wiring."""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

from src import pdb_plot
from src.pdb_plot import (
    MAX_PLOT_YEAR,
    NMR_MONOMER_PROGRAM_CLUSTER_LABELS,
    NMR_WEIGHT_LABELS,
    XRAY_HOMOLOG_TIMING_LABELS,
    PDBScientificPlotter,
    PlotConfig,
    PlotKind,
    parse_args,
    parse_plot_kinds,
    parse_positive_float,
)


class PlotArgumentTests(unittest.TestCase):
    """Exercise the public command-line value parsers and dispatch wiring."""

    def test_plot_config_calculates_scaled_figure_size(self) -> None:
        config = PlotConfig(height_inches=4.0, aspect_ratio=1.5)

        self.assertEqual(config.figure_size(), (6.0, 4.0))
        self.assertEqual(config.figure_size(height_scale=0.5), (6.0, 2.0))

    def test_parse_plot_kinds_supports_all_case_insensitively(self) -> None:
        self.assertEqual(parse_plot_kinds("  ALL  "), list(PlotKind))

    def test_parse_plot_kinds_preserves_explicit_order(self) -> None:
        parsed = parse_plot_kinds(
            " solution_nmr_weight_stats, method_counts,method_counts "
        )

        self.assertEqual(
            parsed,
            [
                PlotKind.SOLUTION_NMR_WEIGHT_STATS,
                PlotKind.METHOD_COUNTS,
                PlotKind.METHOD_COUNTS,
            ],
        )

    def test_parse_plot_kinds_rejects_unknown_and_empty_values(self) -> None:
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError, "Unknown plot 'not-a-plot'"
        ):
            parse_plot_kinds("method_counts,not-a-plot")
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "No plots selected"):
            parse_plot_kinds(" , ")

    def test_parse_positive_float_accepts_positive_and_rejects_bad_values(self) -> None:
        self.assertEqual(parse_positive_float("2.75"), 2.75)
        for value in ("0", "-0.1", "not-a-number"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse_positive_float(value)

    def test_parse_positive_float_rejects_non_finite_values(self) -> None:
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse_positive_float(value)

    def test_parse_args_applies_custom_cli_types(self) -> None:
        argv = [
            "pdb_plot.py",
            "--plots",
            "method_counts,solution_nmr_weight_stats",
            "--counts-input",
            "custom-counts.csv",
            "--svg",
            "--aspect-ratio",
            "2.5",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(
            args.plots,
            [PlotKind.METHOD_COUNTS, PlotKind.SOLUTION_NMR_WEIGHT_STATS],
        )
        self.assertEqual(args.counts_input, Path("custom-counts.csv"))
        self.assertTrue(args.svg)
        self.assertEqual(args.aspect_ratio, 2.5)

    def test_main_dispatches_every_default_plot_without_rendering(self) -> None:
        with patch.object(sys, "argv", ["pdb_plot.py"]):
            args = parse_args()

        with (
            patch.object(pdb_plot, "parse_args", return_value=args),
            patch.object(pdb_plot, "PDBScientificPlotter") as plotter_class,
        ):
            pdb_plot.main()

        plotter = plotter_class.return_value
        self.assertEqual(plotter.plot_solution_nmr_monomer_xray_rmsd.call_count, 2)
        expected_once = {
            "plot_method_counts",
            "plot_membrane_protein_counts",
            "plot_solution_nmr_program_counts",
            "plot_solution_nmr_monomer_program_clusters",
            "plot_solution_nmr_weight_stats",
            "plot_solution_nmr_period_boxplot",
            "plot_solution_nmr_period_area",
            "plot_solution_nmr_period_area_share",
            "plot_solution_nmr_period_area_cumulative_share",
            "plot_solution_nmr_monomer_stride_modeled_first_model",
            "plot_solution_nmr_monomer_precision_stride_modeled_first_model_mean",
            "plot_solution_nmr_monomer_precision_stride_modeled_first_model_median",
            "plot_solution_nmr_monomer_quality",
            "plot_solution_nmr_monomer_xray_homologs",
            "plot_solution_nmr_monomer_xray_homologs_historical",
            "plot_solution_nmr_monomer_xray_homolog_timing_share",
            "plot_solution_nmr_monomer_xray_rmsd_precision_correlation",
        }
        for method_name in expected_once:
            with self.subTest(method=method_name):
                getattr(plotter, method_name).assert_called_once()

    def test_main_skips_unselected_plots(self) -> None:
        with patch.object(sys, "argv", ["pdb_plot.py", "--plots", "method_counts"]):
            args = parse_args()

        with (
            patch.object(pdb_plot, "parse_args", return_value=args),
            patch.object(pdb_plot, "PDBScientificPlotter") as plotter_class,
        ):
            pdb_plot.main()

        self.assertEqual(
            plotter_class.return_value.method_calls,
            [
                call.plot_method_counts(
                    data_path=args.counts_input,
                    annual_output_png=args.annual_output_png,
                    annual_output_svg=args.annual_output_svg,
                    cumulative_output_png=args.cumulative_output_png,
                    cumulative_output_svg=args.cumulative_output_svg,
                )
            ],
        )


class PlotInfrastructureTests(unittest.TestCase):
    """Cover cache, style, axes, path, and lightweight render helpers."""

    def setUp(self) -> None:
        self.plotter = PDBScientificPlotter(PlotConfig(dpi=123))

    def tearDown(self) -> None:
        plt.close("all")
        pdb_plot._has_arial_font.cache_clear()

    def test_has_arial_font_matches_case_and_whitespace(self) -> None:
        pdb_plot._has_arial_font.cache_clear()
        fonts = [SimpleNamespace(name="Other"), SimpleNamespace(name=" ARIAL ")]
        with patch.object(pdb_plot.font_manager.fontManager, "ttflist", fonts):
            self.assertTrue(pdb_plot._has_arial_font())

    def test_has_arial_font_returns_false_when_absent(self) -> None:
        pdb_plot._has_arial_font.cache_clear()
        with patch.object(
            pdb_plot.font_manager.fontManager,
            "ttflist",
            [SimpleNamespace(name="Liberation Sans")],
        ):
            self.assertFalse(pdb_plot._has_arial_font())

    def test_scientific_style_warns_when_arial_is_unavailable(self) -> None:
        with (
            plt.rc_context(),
            patch.object(pdb_plot, "_has_arial_font", return_value=False),
            patch.object(pdb_plot.plt.style, "use") as style_use,
        ):
            with self.assertWarnsRegex(RuntimeWarning, "Arial is not available"):
                self.plotter._scientific_style()

            style_use.assert_called_once_with("seaborn-v0_8-whitegrid")
            self.assertEqual(plt.rcParams["axes.titlesize"], 12.0)
            self.assertEqual(plt.rcParams["axes.titleweight"], "bold")

    def test_read_csv_caches_by_resolved_path(self) -> None:
        frame = pd.DataFrame({"value": [1]})
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            canonical = root / "table.csv"
            alias = root / "nested" / ".." / "table.csv"
            with patch.object(pdb_plot.pd, "read_csv", return_value=frame) as read_csv:
                first = self.plotter._read_csv(canonical)
                second = self.plotter._read_csv(alias)

        self.assertIs(first, frame)
        self.assertIs(second, frame)
        read_csv.assert_called_once_with(canonical.resolve())

    def test_visible_major_step_handles_visibility_and_uniformity(self) -> None:
        axis = Mock()
        axis.get_majorticklocs.return_value = np.array([0.0, 5.0, 10.0, 15.0])
        axis.get_view_interval.return_value = (14.0, 1.0)
        self.assertEqual(self.plotter._visible_major_step(axis), 5.0)

        axis.get_majorticklocs.return_value = np.array([0.0, 5.0, 11.0])
        axis.get_view_interval.return_value = (0.0, 11.0)
        self.assertIsNone(self.plotter._visible_major_step(axis))

        axis.get_majorticklocs.return_value = np.array([5.0])
        self.assertIsNone(self.plotter._visible_major_step(axis))

        axis.get_majorticklocs.return_value = np.array([0.0, 5.0])
        axis.get_view_interval.return_value = (1.0, 4.0)
        self.assertIsNone(self.plotter._visible_major_step(axis))

        axis.get_majorticklocs.return_value = np.array([5.0, 5.0])
        axis.get_view_interval.return_value = (0.0, 10.0)
        self.assertIsNone(self.plotter._visible_major_step(axis))

    def test_integer_minor_subdivisions_selects_readable_divisors(self) -> None:
        cases = {
            20.0: 10,
            16.0: 8,
            15.0: 5,
            12.0: 4,
            6.0: 2,
            7.0: None,
            1.0: None,
            2.5: None,
        }
        for step, expected in cases.items():
            with self.subTest(step=step):
                self.assertEqual(
                    self.plotter._integer_minor_subdivisions(step), expected
                )

    def test_categorical_tick_detection_distinguishes_numeric_labels(self) -> None:
        ax = Mock()
        ax.get_xticklabels.return_value = [Mock(get_text=Mock(return_value=""))]
        self.assertFalse(self.plotter._has_categorical_x_ticks(ax))

        ax.get_xticklabels.return_value = [
            Mock(get_text=Mock(return_value="1")),
            Mock(get_text=Mock(return_value="2.5")),
        ]
        self.assertFalse(self.plotter._has_categorical_x_ticks(ax))

        ax.get_xticklabels.return_value = [
            Mock(get_text=Mock(return_value="1996-2006"))
        ]
        self.assertTrue(self.plotter._has_categorical_x_ticks(ax))

    def test_unicode_minus_tick_is_recognized_as_numeric(self) -> None:
        ax = Mock()
        ax.get_xticklabels.return_value = [
            Mock(get_text=Mock(return_value="−5")),
            Mock(get_text=Mock(return_value="0")),
            Mock(get_text=Mock(return_value="5")),
        ]

        self.assertFalse(self.plotter._has_categorical_x_ticks(ax))

    def test_configure_year_axis_formats_only_supported_integral_years(self) -> None:
        fig, ax = plt.subplots()
        self.plotter._configure_year_axis_ticks(ax)
        formatter = ax.xaxis.get_major_formatter()

        self.assertEqual(formatter(2020.0, 0), "2020")
        self.assertEqual(formatter(float(MAX_PLOT_YEAR), 0), str(MAX_PLOT_YEAR))
        self.assertEqual(formatter(MAX_PLOT_YEAR + 1.0, 0), "")
        self.assertEqual(formatter(2020.25, 0), "")
        plt.close(fig)

    def test_configure_minor_ticks_sets_numeric_axes_only(self) -> None:
        fig, ax = plt.subplots()
        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(0.0, 20.0)
        # Fixed visible ticks keep Matplotlib from adding an out-of-range tick
        # whose Unicode minus sign is not accepted by Python's ``float``.
        ax.set_xticks([0.0, 5.0, 10.0])
        ax.set_yticks([0.0, 10.0, 20.0])

        self.plotter._configure_minor_ticks(ax, use_year_x_ticks=False)

        self.assertIsInstance(ax.xaxis.get_minor_locator(), AutoMinorLocator)
        self.assertIsInstance(ax.yaxis.get_minor_locator(), AutoMinorLocator)
        plt.close(fig)

        mocked_ax = MagicMock()
        with patch.object(self.plotter, "_visible_major_step", return_value=None):
            self.plotter._configure_minor_ticks(mocked_ax, use_year_x_ticks=True)
        mocked_ax.xaxis.set_minor_locator.assert_not_called()

        with (
            patch.object(self.plotter, "_visible_major_step", return_value=None),
            patch.object(self.plotter, "_has_categorical_x_ticks", return_value=False),
        ):
            self.plotter._configure_minor_ticks(mocked_ax, use_year_x_ticks=False)
        mocked_ax.xaxis.set_minor_locator.assert_not_called()

    def test_remove_zero_tick_for_positive_and_signed_ranges(self) -> None:
        fig, ax = plt.subplots()
        ax.set_ylim(0.0, 2.0)
        ax.set_yticks([0.0, 1.0, 2.0])
        self.plotter._remove_zero_y_tick(ax)
        np.testing.assert_allclose(ax.get_yticks(), [1.0, 2.0])

        ax.set_ylim(-2.0, 2.0)
        ax.set_yticks([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.plotter._remove_zero_y_tick(ax)
        np.testing.assert_allclose(ax.get_yticks(), [-2.0, -1.0, 1.0, 2.0])
        plt.close(fig)

    def test_boxed_and_open_axes_toggle_spines(self) -> None:
        fig, ax = plt.subplots()
        self.plotter._configure_boxed_axes(ax, y_tick_labels_on_both_sides=True)
        self.assertTrue(all(spine.get_visible() for spine in ax.spines.values()))

        self.plotter._configure_open_axes(ax)
        self.assertTrue(ax.spines["left"].get_visible())
        self.assertTrue(ax.spines["bottom"].get_visible())
        self.assertFalse(ax.spines["right"].get_visible())
        self.assertFalse(ax.spines["top"].get_visible())
        plt.close(fig)

    def test_title_and_legend_use_consistent_style(self) -> None:
        fig, ax = plt.subplots()
        ax.plot([1, 2], [2, 3], label="series")
        self.plotter._set_title(ax, "A title")
        self.plotter._add_legend(ax, loc="upper left")

        self.assertEqual(ax.get_title(), "A title")
        self.assertEqual(ax.title.get_fontweight(), "bold")
        self.assertAlmostEqual(ax.get_legend().get_frame().get_linewidth(), 0.8)
        plt.close(fig)

    def test_output_path_helpers_group_all_variants(self) -> None:
        base = Path("figures/plot.png")
        self.assertEqual(
            self.plotter._titleless_output_path(base),
            Path("figures/plot_no_title.png"),
        )
        self.assertEqual(
            self.plotter._open_axes_output_path(base),
            Path("figures/plot_open_axes.png"),
        )
        cases = {
            "plot.png": "figures/plot/plot.png",
            "plot_no_title.png": "figures/plot/plot_no_title.png",
            "plot_open_axes.png": "figures/plot/plot_open_axes.png",
            "plot_no_title_open_axes.png": ("figures/plot/plot_no_title_open_axes.png"),
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(
                    self.plotter._grouped_figure_output_path(
                        Path("figures") / filename
                    ),
                    Path(expected),
                )
        already_grouped = Path("figures/plot/plot.png")
        self.assertEqual(
            self.plotter._grouped_figure_output_path(already_grouped),
            already_grouped,
        )

    def test_apply_tight_layout_forwards_optional_rect(self) -> None:
        figure = Mock()
        self.plotter._apply_tight_layout(figure, None)
        self.plotter._apply_tight_layout(figure, (0.1, 0.2, 0.9, 0.8))

        self.assertEqual(
            figure.tight_layout.call_args_list,
            [call(), call(rect=(0.1, 0.2, 0.9, 0.8))],
        )

    def test_save_figure_files_groups_png_and_optional_svg(self) -> None:
        figure = Mock()
        self.plotter.generate_svg = True
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            png = root / "sample.png"
            svg = root / "sample.svg"

            self.plotter._save_figure_files(
                figure,
                png,
                svg,
                savefig_bbox_inches="tight",
                savefig_pad_inches=0.2,
            )

            grouped_png = root / "sample" / "sample.png"
            grouped_svg = root / "sample" / "sample.svg"
            self.assertTrue(grouped_png.parent.is_dir())
            self.assertEqual(
                figure.savefig.call_args_list,
                [
                    call(
                        grouped_png,
                        dpi=123,
                        bbox_inches="tight",
                        pad_inches=0.2,
                    ),
                    call(grouped_svg, bbox_inches="tight", pad_inches=0.2),
                ],
            )

    def test_render_figure_builds_four_title_and_axes_variants(self) -> None:
        figure = MagicMock()
        axes = MagicMock()
        axes.get_ylim.return_value = (-1.0, 5.0)
        draw = Mock()

        with (
            patch.object(pdb_plot.plt, "subplots", return_value=(figure, axes)),
            patch.object(pdb_plot.plt, "close") as close,
            patch.object(self.plotter, "_configure_year_axis_ticks") as year_ticks,
            patch.object(self.plotter, "_remove_zero_y_tick"),
            patch.object(self.plotter, "_configure_minor_ticks"),
            patch.object(self.plotter, "_set_title") as set_title,
            patch.object(self.plotter, "_configure_boxed_axes"),
            patch.object(self.plotter, "_configure_open_axes") as open_axes,
            patch.object(self.plotter, "_apply_tight_layout"),
            patch.object(self.plotter, "_save_figure_files") as save,
        ):
            self.plotter._render_figure(
                output_png=Path("out/plot.png"),
                output_svg=Path("out/plot.svg"),
                title="Title",
                y_label="Y",
                x_label="Custom X",
                draw_fn=draw,
            )

        self.assertEqual(draw.call_count, 4)
        self.assertEqual(year_ticks.call_count, 4)
        self.assertEqual(set_title.call_count, 2)
        self.assertEqual(open_axes.call_count, 2)
        self.assertEqual(close.call_count, 4)
        self.assertEqual(axes.set_ylim.call_count, 4)
        self.assertEqual(
            [item.kwargs["output_png"] for item in save.call_args_list],
            [
                Path("out/plot.png"),
                Path("out/plot_no_title.png"),
                Path("out/plot_open_axes.png"),
                Path("out/plot_no_title_open_axes.png"),
            ],
        )
        axes.set_xlabel.assert_called_with("Custom X")

    def test_step_helpers_and_stairs_forwarding(self) -> None:
        np.testing.assert_array_equal(self.plotter._step_edges([]), [])
        np.testing.assert_allclose(self.plotter._step_edges([4.0]), [3.5, 4.5])
        np.testing.assert_allclose(
            self.plotter._step_edges([1.0, 3.0, 8.0]),
            [0.0, 2.0, 5.5, 10.5],
        )
        np.testing.assert_array_equal(self.plotter._step_values([]), [])
        np.testing.assert_allclose(
            self.plotter._step_values([2.0, 5.0]), [2.0, 5.0, 5.0]
        )

        ax = Mock()
        self.plotter._plot_step_series(
            ax=ax,
            x_values=pd.Index([2000, 2001]),
            y_values=pd.Series([2, 4]),
            color="red",
            linewidth=1.5,
            label="values",
            zorder=7,
        )
        kwargs = ax.stairs.call_args.kwargs
        np.testing.assert_allclose(kwargs["values"], [2.0, 4.0])
        np.testing.assert_allclose(kwargs["edges"], [1999.5, 2000.5, 2001.5])
        self.assertIsNone(kwargs["baseline"])
        self.assertFalse(kwargs["fill"])
        self.assertEqual(kwargs["zorder"], 7)

    def test_line_multi_line_and_bar_renderers_draw_via_callback(self) -> None:
        ax = MagicMock()

        def invoke_draw(**kwargs: object) -> None:
            kwargs["draw_fn"](ax)  # type: ignore[operator]

        with (
            patch.object(self.plotter, "_render_figure", side_effect=invoke_draw),
            patch.object(self.plotter, "_add_legend") as add_legend,
        ):
            self.plotter._render_line_series(
                Path("line.png"),
                Path("line.svg"),
                "line",
                "y",
                pd.Index([1, 2]),
                pd.Series([3, 4]),
                "blue",
                label="series",
                y_limits=(0.0, 5.0),
                x_left=1.0,
            )
        ax.plot.assert_called_once()
        ax.set_ylim.assert_called_with(0.0, 5.0)
        ax.set_xlim.assert_called_with(left=1.0)
        add_legend.assert_called_once_with(ax, loc="upper left")

        ax.reset_mock()
        table = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=[2000, 2001])
        with (
            patch.object(self.plotter, "_render_figure", side_effect=invoke_draw),
            patch.object(self.plotter, "_plot_step_series") as plot_step,
            patch.object(self.plotter, "_add_legend"),
        ):
            self.plotter._render_multi_line_series(
                Path("multi.png"),
                Path("multi.svg"),
                "multi",
                "y",
                table,
                colors={"b": "green"},
                labels={"b": "B"},
                y_bottom=0.0,
                use_step=True,
                draw_order=["missing", "b"],
            )
        plot_step.assert_called_once()
        self.assertEqual(plot_step.call_args.kwargs["label"], "B")
        ax.set_ylim.assert_called_once_with(bottom=0.0)

        ax.reset_mock()
        with (
            patch.object(self.plotter, "_render_figure", side_effect=invoke_draw),
            patch.object(self.plotter, "_add_legend"),
        ):
            self.plotter._render_multi_line_series(
                Path("multi-line.png"),
                Path("multi-line.svg"),
                "multi",
                "y",
                table,
                colors={"a": "red", "b": "green"},
                labels={},
            )
        self.assertEqual(ax.plot.call_count, 2)

        ax.reset_mock()
        with (
            patch.object(self.plotter, "_render_figure", side_effect=invoke_draw),
            patch.object(self.plotter, "_plot_step_series") as plot_step,
        ):
            self.plotter._render_bar_series(
                Path("bar.png"),
                Path("bar.svg"),
                "bar",
                "y",
                pd.Index([1]),
                pd.Series([2]),
                "black",
                y_limits=(0.0, 3.0),
                y_bottom=0.0,
                x_left=1.0,
            )
        plot_step.assert_called_once()
        self.assertEqual(
            ax.set_ylim.call_args_list,
            [call(0.0, 3.0), call(bottom=0.0)],
        )
        ax.set_xlim.assert_called_once_with(left=1.0)


class PlotTableTransformationTests(unittest.TestCase):
    """Verify schema checks, normalization, grouping, and numeric summaries."""

    def setUp(self) -> None:
        self.plotter = PDBScientificPlotter(PlotConfig())

    def test_limit_year_column_filters_copy_but_leaves_unrelated_table(self) -> None:
        source = pd.DataFrame({"year": [2024, 2025], "value": [1, 2]})
        limited = self.plotter._limit_year_column(source)
        self.assertEqual(limited["year"].tolist(), [2024])
        limited.loc[:, "value"] = 99
        self.assertEqual(source["value"].tolist(), [1, 2])

        without_year = pd.DataFrame({"value": [1]})
        self.assertIs(self.plotter._limit_year_column(without_year), without_year)

    def test_required_column_error_is_sorted_and_names_dataset(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"Example CSV is missing required columns: alpha, zebra",
        ):
            self.plotter._validate_required_columns(
                pd.DataFrame({"present": [1]}),
                {"zebra", "present", "alpha"},
                "Example CSV",
            )

    def test_prepare_typed_table_copies_and_casts_requested_columns(self) -> None:
        source = pd.DataFrame({"year": ["2024"], "count": ["3"]})
        prepared = self.plotter._prepare_typed_table(
            source,
            required_columns={"year", "count"},
            column_types={"year": int, "count": float},
            dataset_name="Counts",
        )

        self.assertTrue(pd.api.types.is_integer_dtype(prepared["year"]))
        self.assertTrue(pd.api.types.is_float_dtype(prepared["count"]))
        self.assertEqual(source.iloc[0].tolist(), ["2024", "3"])

    def test_prepare_method_counts_pivots_fills_and_limits_years(self) -> None:
        source = pd.DataFrame(
            {
                "year": [2024, 2023, 2023, 2025],
                "method": ["X-ray", "X-ray", "NMR", "NMR"],
                "count": [4, 2, 1, 99],
            }
        )

        table = self.plotter._prepare_method_count_table(source)

        self.assertEqual(table.index.tolist(), [2023, 2024])
        self.assertEqual(table.loc[2023, "NMR"], 1)
        self.assertEqual(table.loc[2024, "NMR"], 0)
        self.assertEqual(table.loc[2024, "X-ray"], 4)
        self.assertTrue(
            all(pd.api.types.is_integer_dtype(dtype) for dtype in table.dtypes)
        )

    def test_prepare_method_counts_casts_numeric_csv_strings(self) -> None:
        source = pd.DataFrame(
            {
                "year": ["2023", "2024", "2025"],
                "method": ["NMR", "X-ray", "NMR"],
                "count": ["2", "4", "99"],
            }
        )

        table = self.plotter._prepare_method_count_table(source)

        self.assertEqual(table.index.tolist(), [2023, 2024])
        self.assertEqual(table.loc[2023, "NMR"], 2)
        self.assertEqual(table.loc[2024, "X-ray"], 4)
        self.assertTrue(pd.api.types.is_integer_dtype(table.index.dtype))
        self.assertTrue(
            all(pd.api.types.is_integer_dtype(dtype) for dtype in table.dtypes)
        )

    def test_prepare_nmr_program_counts_casts_pivots_and_sorts(self) -> None:
        source = pd.DataFrame(
            {
                "year": ["2024", "2023", "2025"],
                "program": ["ARIA", "CYANA", "ARIA"],
                "count": ["4", "2", "99"],
            }
        )

        table = self.plotter._prepare_nmr_program_count_table(source)

        self.assertEqual(table.index.tolist(), [2023, 2024])
        self.assertEqual(table.loc[2023, "CYANA"], 2)
        self.assertEqual(table.loc[2024, "CYANA"], 0)

    def test_cluster_table_normalizes_labels_order_and_fill_behavior(self) -> None:
        source = pd.DataFrame(
            {
                "year": ["2021", "2020", "2020", "2025"],
                "cluster_id": ["CLUSTER1", "CLUSTER9", "CLUSTER1", "CLUSTER1"],
                "cluster_name": ["AMBER", "OTHER", "AMBER", "AMBER"],
                "structure_count": ["4", "1", "2", "100"],
            }
        )
        prepared = self.plotter._prepare_nmr_monomer_program_cluster_table(source)
        table = self.plotter._build_cluster_yearly_table(
            prepared, "structure_count", fill_value=0.0
        )
        unfilled = self.plotter._build_cluster_yearly_table(
            prepared, "structure_count", fill_value=None
        )

        self.assertEqual(prepared["year"].max(), 2021)
        self.assertEqual(table.columns.tolist(), self.plotter._cluster_column_labels())
        self.assertEqual(table.loc[2020, "AMBER"], 2.0)
        self.assertEqual(table.loc[2020, "OTHER"], 1.0)
        self.assertEqual(table.loc[2021, "OTHER"], 0.0)
        self.assertTrue(math.isnan(unfilled.loc[2021, "OTHER"]))
        self.assertEqual(
            self.plotter._display_cluster_label("CUSTOM", "MY_CLUSTER"),
            "MY CLUSTER",
        )
        self.assertEqual(
            self.plotter._display_cluster_label("CLUSTER4", "ignored"),
            NMR_MONOMER_PROGRAM_CLUSTER_LABELS["CLUSTER4"],
        )

    def test_membrane_weight_and_period_preparation(self) -> None:
        membrane = self.plotter._prepare_membrane_count_table(
            pd.DataFrame({"year": ["2024", "2022", "2025"], "count": ["3", "1", "9"]})
        )
        self.assertEqual(membrane["year"].tolist(), [2022, 2024])
        self.assertEqual(membrane["count"].tolist(), [1, 3])

        weights = self.plotter._prepare_nmr_weight_table(
            pd.DataFrame(
                {
                    "entry_id": ["a", "b", "c", "d"],
                    "year": ["1995", "1996", "2006", "2007"],
                    "molecular_weight_kda": ["1", "2", "3", "4"],
                }
            )
        )
        periods = self.plotter._period_series(weights)
        self.assertEqual(periods["Before 1996"].tolist(), [1.0])
        self.assertEqual(periods["1996-2006"].tolist(), [2.0, 3.0])
        self.assertEqual(periods["After 2006"].tolist(), [4.0])

    def test_specialized_preparers_cast_filter_and_drop_short_queries(self) -> None:
        stride = self.plotter._prepare_monomer_stride_modeled_first_model_table(
            pd.DataFrame(
                {
                    "entry_id": ["a", "future"],
                    "year": ["2024", "2025"],
                    "stride_alpha_helix_fraction": ["0.1", "0.2"],
                    "stride_3_10_helix_fraction": ["0.1", "0.2"],
                    "stride_pi_helix_fraction": ["0.1", "0.2"],
                    "stride_beta_strand_fraction": ["0.1", "0.2"],
                    "stride_isolated_beta_bridge_fraction": ["0.1", "0.2"],
                }
            )
        )
        self.assertEqual(stride["entry_id"].tolist(), ["a"])
        self.assertTrue(
            pd.api.types.is_float_dtype(stride["stride_alpha_helix_fraction"])
        )

        precision = self.plotter._prepare_monomer_precision_table(
            pd.DataFrame(
                {
                    "entry_id": ["a", "future"],
                    "year": ["2024", "2025"],
                    "mean_rmsd_angstrom": ["1.5", "2.5"],
                }
            )
        )
        self.assertEqual(precision["entry_id"].tolist(), ["a"])

        quality = self.plotter._prepare_monomer_quality_table(
            pd.DataFrame(
                {
                    "entry_id": ["a", "future"],
                    "year": ["2024", "2025"],
                    "clashscore": ["1", "2"],
                    "ramachandran_outliers_percent": ["3", "4"],
                    "sidechain_outliers_percent": ["5", "6"],
                }
            )
        )
        self.assertEqual(quality["entry_id"].tolist(), ["a"])

        homolog = self.plotter._prepare_monomer_xray_homolog_table(
            pd.DataFrame(
                {
                    "entry_id": ["a", "short", "future"],
                    "year": ["2024", "2024", "2025"],
                    "sequence_identity_percent": ["95", "95", "95"],
                    "nmr_query_sequence_length": ["11", "10", "11"],
                    "has_xray_homolog": ["1", "0", "1"],
                }
            )
        )
        self.assertEqual(homolog["entry_id"].tolist(), ["a"])

        rmsd = self.plotter._prepare_monomer_xray_rmsd_table(
            pd.DataFrame(
                {
                    "entry_id": ["a", "future"],
                    "year": ["2024", "2025"],
                    "rmsd_ca_angstrom": ["1.1", "2.2"],
                }
            )
        )
        extremes = self.plotter._prepare_monomer_xray_rmsd_extremes_table(
            pd.DataFrame(
                {
                    "entry_id": ["a", "future"],
                    "year": ["2024", "2025"],
                    "best_rmsd_ca_angstrom": ["0.8", "1.8"],
                    "worst_rmsd_ca_angstrom": ["2.0", "3.0"],
                }
            )
        )
        self.assertEqual(rmsd["entry_id"].tolist(), ["a"])
        self.assertEqual(extremes["entry_id"].tolist(), ["a"])

    def test_homolog_share_series_computes_annual_and_running_denominators(
        self,
    ) -> None:
        table = pd.DataFrame(
            {
                "entry_id": ["a", "b", "c", "d", "e"],
                "year": [2020, 2020, 2021, 2021, 2021],
                "has_xray_homolog": [1, 0, 1, 1, 0],
            }
        )

        yearly, cumulative = self.plotter._homolog_share_series(table)

        self.assertAlmostEqual(yearly.loc[2020], 50.0)
        self.assertAlmostEqual(yearly.loc[2021], 200.0 / 3.0)
        self.assertAlmostEqual(cumulative.loc[2020], 50.0)
        self.assertAlmostEqual(cumulative.loc[2021], 60.0)

    def test_homolog_timing_counts_and_shares_cover_all_statuses(self) -> None:
        regular = pd.DataFrame(
            {
                "entry_id": ["prior", "later", "none", "none-2021"],
                "year": [2020, 2020, 2020, 2021],
                "has_xray_homolog": [1, 1, 0, 0],
            }
        )
        historical = pd.DataFrame(
            {
                "entry_id": ["prior", "later", "none"],
                "has_xray_homolog": [1, 0, 0],
            }
        )

        counts = self.plotter._build_xray_homolog_timing_count_table(
            regular, historical
        )
        self.assertEqual(counts.columns.tolist(), list(XRAY_HOMOLOG_TIMING_LABELS))
        self.assertEqual(counts.loc[2020].tolist(), [1, 1, 1])
        self.assertEqual(counts.loc[2021].tolist(), [0, 0, 1])

        counts.loc[2022] = [0, 0, 0]
        shares = self.plotter._xray_homolog_timing_share_from_counts(counts)
        np.testing.assert_allclose(shares.loc[2020], [100 / 3, 100 / 3, 100 / 3])
        np.testing.assert_allclose(shares.loc[2021], [0.0, 0.0, 100.0])
        np.testing.assert_allclose(shares.loc[2022], [0.0, 0.0, 0.0])

    def test_write_homolog_timing_counts_uses_stable_csv_schema(self) -> None:
        counts = pd.DataFrame(
            [[1, 2, 3]],
            index=pd.Index([2020], name="year"),
            columns=XRAY_HOMOLOG_TIMING_LABELS,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "counts.csv"
            self.plotter._write_xray_homolog_timing_counts_csv(counts, output_path)
            written = pd.read_csv(output_path)

        self.assertEqual(
            written.columns.tolist(),
            [
                "year",
                "already_released_xray_homolog_count",
                "later_released_xray_homolog_count",
                "no_xray_homolog_count",
            ],
        )
        self.assertEqual(written.iloc[0].tolist(), [2020, 1, 2, 3])

    def test_xray_rmsd_extremes_yearly_table_aligns_disjoint_years(self) -> None:
        regular = pd.DataFrame(
            {
                "year": [2020, 2020, 2022],
                "rmsd_ca_angstrom": [1.0, 3.0, 7.0],
            }
        )
        extremes = pd.DataFrame(
            {
                "year": [2020, 2020, 2021],
                "best_rmsd_ca_angstrom": [0.5, 1.5, 2.0],
                "worst_rmsd_ca_angstrom": [4.0, 6.0, 8.0],
            }
        )

        yearly = self.plotter._xray_rmsd_extremes_yearly_table(
            regular, extremes, "mean"
        )

        self.assertEqual(yearly.index.tolist(), [2020, 2021, 2022])
        self.assertEqual(yearly.loc[2020].tolist(), [2.0, 1.0, 5.0])
        self.assertTrue(math.isnan(yearly.loc[2021, "best_resolution_rmsd"]))
        self.assertTrue(math.isnan(yearly.loc[2022, "best_rmsd"]))

    def test_precision_correlation_merge_deduplicates_and_filters_invalid_rows(
        self,
    ) -> None:
        precision = pd.DataFrame(
            {
                "entry_id": ["a", "a", "infinite", "future", "nan", "unmatched"],
                "year": [2020, 2020, 2021, 2025, 2022, 2022],
                "mean_rmsd_angstrom": [1.0, 9.0, np.inf, 2.0, 3.0, 4.0],
            }
        )
        extremes = pd.DataFrame(
            {
                "entry_id": ["a", "a", "infinite", "future", "nan", "other"],
                "year": [2020, 2020, 2021, 2025, 2022, 2022],
                "best_rmsd_ca_angstrom": [0.5, 0.9, 1.0, 2.0, np.nan, 5.0],
                "worst_rmsd_ca_angstrom": [1.5, 1.9, 2.0, 3.0, 4.0, 6.0],
            }
        )

        merged = self.plotter._prepare_xray_rmsd_precision_correlation_table(
            precision, extremes
        )

        self.assertEqual(
            merged.columns.tolist(),
            ["entry_id", "year", "precision_rmsd", "min_xray_rmsd"],
        )
        self.assertEqual(
            merged.to_dict("records"),
            [
                {
                    "entry_id": "a",
                    "year": 2020,
                    "precision_rmsd": 1.0,
                    "min_xray_rmsd": 0.5,
                }
            ],
        )

    def test_empty_precision_correlation_input_has_clear_error(self) -> None:
        empty = pd.DataFrame()
        with (
            patch.object(self.plotter, "_read_csv", side_effect=[empty, empty]),
            patch.object(
                self.plotter,
                "_prepare_xray_rmsd_precision_correlation_table",
                return_value=empty,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "No overlapping entries"):
                self.plotter.plot_solution_nmr_monomer_xray_rmsd_precision_correlation(
                    Path("precision.csv"),
                    Path("extremes.csv"),
                    Path("scatter.png"),
                    Path("scatter.svg"),
                    Path("year.png"),
                    Path("year.svg"),
                    Path("cumulative.png"),
                    Path("cumulative.svg"),
                )


class PlotCallbackTests(unittest.TestCase):
    """Exercise drawing callbacks without saving raster or vector files."""

    def setUp(self) -> None:
        self.plotter = PDBScientificPlotter(PlotConfig())

    def tearDown(self) -> None:
        plt.close("all")

    @staticmethod
    def _invoke_render_callback(
        axes: MagicMock,
    ) -> object:
        def invoke(*args: object, **kwargs: object) -> None:
            draw_fn = kwargs.get("draw_fn")
            if draw_fn is None:
                draw_fn = args[4]
            draw_fn(axes)  # type: ignore[operator]

        return invoke

    def test_weight_stackplot_supports_step_and_continuous_layers(self) -> None:
        table = pd.DataFrame(
            {
                NMR_WEIGHT_LABELS[0]: [1.0, 2.0],
                NMR_WEIGHT_LABELS[1]: [3.0, 4.0],
                NMR_WEIGHT_LABELS[2]: [5.0, 6.0],
            },
            index=[2020, 2021],
        )
        ax = MagicMock()
        ax.get_xlim.return_value = (2019.0, 2022.0)
        with (
            patch.object(
                self.plotter,
                "_render_figure",
                side_effect=self._invoke_render_callback(ax),
            ),
            patch.object(self.plotter, "_add_legend") as add_legend,
        ):
            self.plotter._render_weight_category_stackplot(
                table,
                Path("step.png"),
                Path("step.svg"),
                "Step",
                "Y",
                y_limits=(0.0, 100.0),
                x_left=2020.0,
                x_right=2021.0,
                use_step_segments=True,
            )

        self.assertEqual(ax.fill_between.call_count, len(NMR_WEIGHT_LABELS))
        ax.set_ylim.assert_called_once_with(0.0, 100.0)
        ax.set_xlim.assert_called_once_with(left=2019.5, right=2021.5)
        add_legend.assert_called_once_with(ax, loc="upper left", title="Weight range")

        continuous_ax = MagicMock()
        with (
            patch.object(
                self.plotter,
                "_render_figure",
                side_effect=self._invoke_render_callback(continuous_ax),
            ),
            patch.object(self.plotter, "_add_legend"),
        ):
            self.plotter._render_weight_category_stackplot(
                table,
                Path("area.png"),
                Path("area.svg"),
                "Area",
                "Y",
            )
        continuous_ax.stackplot.assert_called_once()

    def test_homolog_timing_stackplot_draws_three_bounded_layers(self) -> None:
        table = pd.DataFrame(
            {
                XRAY_HOMOLOG_TIMING_LABELS[0]: [20.0, 25.0],
                XRAY_HOMOLOG_TIMING_LABELS[1]: [30.0, 25.0],
                XRAY_HOMOLOG_TIMING_LABELS[2]: [50.0, 50.0],
            },
            index=[2020, 2021],
        )
        ax = MagicMock()
        with (
            patch.object(
                self.plotter,
                "_render_figure",
                side_effect=self._invoke_render_callback(ax),
            ),
            patch.object(self.plotter, "_add_legend") as add_legend,
        ):
            self.plotter._render_homolog_timing_stackplot(
                table,
                Path("timing.png"),
                Path("timing.svg"),
                "Timing",
            )

        self.assertEqual(ax.fill_between.call_count, 3)
        ax.set_ylim.assert_called_once_with(0.0, 100.0)
        ax.set_xlim.assert_called_once_with(2019.5, 2021.5)
        add_legend.assert_called_once_with(ax, loc="upper left")

    def test_cluster_stackplot_covers_step_continuous_and_external_legend_paths(
        self,
    ) -> None:
        table = pd.DataFrame(
            {"AMBER": [40.0, 50.0], "OTHER": [60.0, 50.0]},
            index=[2020, 2021],
        )
        continuous_ax = MagicMock()
        with (
            patch.object(
                self.plotter,
                "_render_figure",
                side_effect=self._invoke_render_callback(continuous_ax),
            ),
            patch.object(self.plotter, "_add_legend"),
        ):
            self.plotter._render_cluster_stackplot(
                table,
                Path("continuous.png"),
                Path("continuous.svg"),
                "Continuous",
                "Y",
            )
        continuous_ax.stackplot.assert_called_once()

        step_ax = MagicMock()
        step_ax.get_xlim.return_value = (2019.0, 2022.0)
        with (
            patch.object(
                self.plotter,
                "_render_figure",
                side_effect=self._invoke_render_callback(step_ax),
            ),
            patch.object(self.plotter, "_add_legend"),
        ):
            self.plotter._render_cluster_stackplot(
                table,
                Path("step.png"),
                Path("step.svg"),
                "Step",
                "Y",
                x_left=2020.0,
                x_right=2021.0,
                use_step_segments=True,
            )
        self.assertEqual(step_ax.fill_between.call_count, 2)
        step_ax.set_xlim.assert_called_once_with(left=2019.5, right=2021.5)

        outside_ax = MagicMock()

        def invoke_bottom_legend(*args: object, **kwargs: object) -> None:
            kwargs["draw_fn"](outside_ax)  # type: ignore[operator]

        with patch.object(
            self.plotter,
            "_render_cluster_stackplot_with_bottom_legend",
            side_effect=invoke_bottom_legend,
        ) as bottom_legend:
            self.plotter._render_cluster_stackplot(
                table,
                Path("outside.png"),
                Path("outside.svg"),
                "Outside",
                "Y",
                y_limits=(0.0, 100.0),
                legend_outside=True,
            )
        bottom_legend.assert_called_once()
        outside_ax.set_ylim.assert_called_once_with(0.0, 100.0)

    def test_bottom_legend_renderer_builds_all_four_variants(self) -> None:
        figure = MagicMock()
        axes = MagicMock()
        axes.get_ylim.return_value = (-1.0, 100.0)
        axes.get_legend_handles_labels.return_value = (["handle"], ["label"])
        draw = Mock()
        with (
            patch.object(pdb_plot.plt, "figure", return_value=figure),
            patch.object(pdb_plot.plt, "close") as close,
            patch.object(self.plotter, "_configure_year_axis_ticks"),
            patch.object(self.plotter, "_remove_zero_y_tick"),
            patch.object(self.plotter, "_configure_minor_ticks"),
            patch.object(self.plotter, "_set_title") as set_title,
            patch.object(self.plotter, "_configure_boxed_axes"),
            patch.object(self.plotter, "_configure_open_axes") as open_axes,
            patch.object(self.plotter, "_save_figure_files") as save,
        ):
            figure.add_axes.return_value = axes
            self.plotter._render_cluster_stackplot_with_bottom_legend(
                Path("cluster.png"),
                Path("cluster.svg"),
                "Clusters",
                "Share",
                draw,
            )

        self.assertEqual(draw.call_count, 4)
        self.assertEqual(figure.legend.call_count, 4)
        self.assertEqual(save.call_count, 4)
        self.assertEqual(set_title.call_count, 2)
        self.assertEqual(open_axes.call_count, 2)
        self.assertEqual(close.call_count, 4)


class PlotOrchestrationTests(unittest.TestCase):
    """Verify plot-level aggregation while replacing file rendering with mocks."""

    def setUp(self) -> None:
        self.plotter = PDBScientificPlotter(PlotConfig())

    def tearDown(self) -> None:
        plt.close("all")

    @staticmethod
    def _homolog_frame(flags: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "entry_id": [f"entry-{index}" for index in range(len(flags))],
                "year": [2020 + index // 2 for index in range(len(flags))],
                "sequence_identity_percent": [95] * len(flags),
                "nmr_query_sequence_length": [20] * len(flags),
                "has_xray_homolog": flags,
            }
        )

    @staticmethod
    def _invoke_with_new_axes(captured_axes: list[MagicMock]) -> object:
        def invoke(*args: object, **kwargs: object) -> None:
            draw_fn = kwargs.get("draw_fn")
            if draw_fn is None:
                draw_fn = args[4]
            axes = MagicMock()
            axes.boxplot.return_value = {"boxes": [Mock(), Mock(), Mock()]}
            captured_axes.append(axes)
            draw_fn(axes)  # type: ignore[operator]

        return invoke

    def test_method_count_plot_draws_annual_steps_and_cumulative_lines(self) -> None:
        source = pd.DataFrame(
            {
                "year": [2020, 2020, 2020, 2021, 2021, 2021],
                "method": ["X-ray", "NMR", "cryo-EM"] * 2,
                "count": [2, 1, 3, 4, 2, 1],
            }
        )
        axes: list[MagicMock] = []
        with (
            patch.object(self.plotter, "_read_csv", return_value=source),
            patch.object(self.plotter, "_scientific_style"),
            patch.object(
                self.plotter,
                "_render_figure",
                side_effect=self._invoke_with_new_axes(axes),
            ) as render,
            patch.object(self.plotter, "_plot_step_series") as step,
            patch.object(self.plotter, "_add_legend"),
        ):
            self.plotter.plot_method_counts(
                Path("counts.csv"),
                Path("annual.png"),
                Path("annual.svg"),
                Path("cumulative.png"),
                Path("cumulative.svg"),
            )

        self.assertEqual(render.call_count, 2)
        self.assertEqual(step.call_count, 3)
        self.assertEqual(axes[1].plot.call_count, 3)
        cumulative_nmr = axes[1].plot.call_args_list[1].args[1]
        self.assertEqual(cumulative_nmr.tolist(), [1, 3])

    def test_program_plot_selects_at_least_one_top_program(self) -> None:
        source = pd.DataFrame(
            {
                "year": [2020, 2020, 2021, 2021],
                "program": ["ARIA", "CYANA", "ARIA", "CYANA"],
                "count": [4, 3, 5, 2],
            }
        )
        axes: list[MagicMock] = []
        with (
            patch.object(self.plotter, "_read_csv", return_value=source),
            patch.object(self.plotter, "_scientific_style"),
            patch.object(
                self.plotter,
                "_render_figure",
                side_effect=self._invoke_with_new_axes(axes),
            ),
            patch.object(self.plotter, "_plot_step_series") as step,
            patch.object(self.plotter, "_add_legend"),
        ):
            self.plotter.plot_solution_nmr_program_counts(
                Path("programs.csv"),
                Path("programs.png"),
                Path("programs.svg"),
                top_n=0,
            )

        step.assert_called_once()
        self.assertEqual(step.call_args.kwargs["label"], "ARIA")

        with patch.object(self.plotter, "_read_csv", return_value=source.iloc[0:0]):
            with self.assertRaisesRegex(ValueError, "program count CSV is empty"):
                self.plotter.plot_solution_nmr_program_counts(
                    Path("empty.csv"), Path("x.png"), Path("x.svg")
                )

    def test_cluster_plot_builds_normalized_share_tables(self) -> None:
        source = pd.DataFrame(
            {
                "year": [2020, 2020, 2021, 2021],
                "cluster_id": ["CLUSTER1", "CLUSTER9"] * 2,
                "cluster_name": ["AMBER", "OTHER"] * 2,
                "structure_count": [3, 1, 1, 1],
            }
        )
        with (
            patch.object(self.plotter, "_read_csv", return_value=source),
            patch.object(self.plotter, "_scientific_style"),
            patch.object(self.plotter, "_render_cluster_stackplot") as render,
        ):
            self.plotter.plot_solution_nmr_monomer_program_clusters(
                Path("clusters.csv"),
                Path("share.png"),
                Path("share.svg"),
                Path("without.png"),
                Path("without.svg"),
            )

        self.assertEqual(render.call_count, 2)
        full_share = render.call_args_list[0].kwargs["table"]
        without_other = render.call_args_list[1].kwargs["table"]
        np.testing.assert_allclose(full_share.sum(axis=1), [100.0, 100.0])
        self.assertNotIn("OTHER", without_other.columns)
        np.testing.assert_allclose(without_other.sum(axis=1), [100.0, 100.0])

        with patch.object(self.plotter, "_read_csv", return_value=source.iloc[0:0]):
            with self.assertRaisesRegex(ValueError, "cluster summary CSV is empty"):
                self.plotter.plot_solution_nmr_monomer_program_clusters(
                    Path("empty.csv"),
                    Path("a.png"),
                    Path("a.svg"),
                    Path("b.png"),
                    Path("b.svg"),
                )

    def test_weight_stat_area_and_boxplot_aggregations(self) -> None:
        source = pd.DataFrame(
            {
                "entry_id": ["a", "b", "c", "d"],
                "year": [1995, 2000, 2020, 2020],
                "molecular_weight_kda": [5.0, 15.0, 20.0, 30.0],
            }
        )
        with (
            patch.object(self.plotter, "_read_csv", return_value=source),
            patch.object(self.plotter, "_scientific_style"),
            patch.object(self.plotter, "_render_bar_series") as render_bar,
        ):
            self.plotter.plot_solution_nmr_weight_stats(
                Path("weights.csv"),
                Path("mean.png"),
                Path("mean.svg"),
                Path("median.png"),
                Path("median.svg"),
                Path("max.png"),
                Path("max.svg"),
            )
        self.assertEqual(render_bar.call_count, 3)
        self.assertEqual(
            render_bar.call_args_list[0].kwargs["y_values"].loc[2020], 25.0
        )

        axes: list[MagicMock] = []
        with (
            patch.object(self.plotter, "_read_csv", return_value=source),
            patch.object(self.plotter, "_scientific_style"),
            patch.object(
                self.plotter,
                "_render_figure",
                side_effect=self._invoke_with_new_axes(axes),
            ),
        ):
            self.plotter.plot_solution_nmr_period_boxplot(
                Path("weights.csv"), Path("boxplot.png"), Path("boxplot.svg")
            )
        boxes = axes[0].boxplot.return_value["boxes"]
        for box in boxes:
            box.set_facecolor.assert_called_once()
            box.set_alpha.assert_called_once_with(0.5)

        with (
            patch.object(self.plotter, "_read_csv", return_value=source),
            patch.object(self.plotter, "_scientific_style"),
            patch.object(
                self.plotter, "_render_weight_category_stackplot"
            ) as render_stack,
        ):
            self.plotter.plot_solution_nmr_period_area(
                Path("weights.csv"), Path("area.png"), Path("area.svg")
            )
            self.plotter.plot_solution_nmr_period_area_share(
                Path("weights.csv"), Path("share.png"), Path("share.svg")
            )
            self.plotter.plot_solution_nmr_period_area_cumulative_share(
                Path("weights.csv"),
                Path("cumulative.png"),
                Path("cumulative.svg"),
            )
        self.assertEqual(render_stack.call_count, 3)
        annual_share = render_stack.call_args_list[1].kwargs["table"]
        cumulative_share = render_stack.call_args_list[2].kwargs["table"]
        np.testing.assert_allclose(annual_share.sum(axis=1), [100.0] * 3)
        np.testing.assert_allclose(cumulative_share.sum(axis=1), [100.0] * 3)

    def test_membrane_plot_builds_cumulative_and_method_series(self) -> None:
        counts = pd.DataFrame({"year": [2020, 2021], "count": [2, 3]})
        methods = pd.DataFrame(
            {
                "year": [2020, 2020, 2021, 2021],
                "method": ["X-ray", "NMR", "X-ray", "NMR"],
                "count": [2, 1, 3, 4],
            }
        )
        axes: list[MagicMock] = []
        with (
            patch.object(self.plotter, "_read_csv", side_effect=[counts, methods]),
            patch.object(self.plotter, "_scientific_style"),
            patch.object(self.plotter, "_render_bar_series") as render_bar,
            patch.object(self.plotter, "_render_line_series") as render_line,
            patch.object(
                self.plotter,
                "_render_figure",
                side_effect=self._invoke_with_new_axes(axes),
            ),
            patch.object(self.plotter, "_plot_step_series"),
            patch.object(self.plotter, "_add_legend"),
        ):
            self.plotter.plot_membrane_protein_counts(
                Path("counts.csv"),
                Path("methods.csv"),
                Path("annual.png"),
                Path("annual.svg"),
                Path("cumulative.png"),
                Path("cumulative.svg"),
                Path("method-annual.png"),
                Path("method-annual.svg"),
                Path("method-cumulative.png"),
                Path("method-cumulative.svg"),
            )

        render_bar.assert_called_once()
        self.assertEqual(render_line.call_args.kwargs["y_values"].tolist(), [2, 5])
        self.assertEqual(len(axes), 2)
        self.assertEqual(axes[1].plot.call_count, 2)

    def test_stride_precision_and_quality_plots_aggregate_valid_values(self) -> None:
        stride = pd.DataFrame(
            {
                "entry_id": ["valid", "over-100", "negative"],
                "year": [2020, 2020, 2021],
                "stride_alpha_helix_fraction": [0.2, 0.8, -0.2],
                "stride_3_10_helix_fraction": [0.1, 0.4, 0.0],
                "stride_pi_helix_fraction": [0.0, 0.0, 0.0],
                "stride_beta_strand_fraction": [0.2, 0.0, 0.0],
                "stride_isolated_beta_bridge_fraction": [0.0, 0.0, 0.0],
            }
        )
        axes: list[MagicMock] = []
        with (
            patch.object(self.plotter, "_read_csv", return_value=stride),
            patch.object(self.plotter, "_scientific_style"),
            patch.object(
                self.plotter,
                "_render_figure",
                side_effect=self._invoke_with_new_axes(axes),
            ),
            patch.object(self.plotter, "_plot_step_series") as step,
            patch.object(self.plotter, "_add_legend"),
        ):
            self.plotter.plot_solution_nmr_monomer_stride_modeled_first_model(
                Path("stride.csv"), Path("stride.png"), Path("stride.svg")
            )
        self.assertEqual(axes[0].scatter.call_args.args[0].tolist(), [2020])
        np.testing.assert_allclose(step.call_args.kwargs["y_values"], [50.0])

        precision = pd.DataFrame(
            {
                "entry_id": ["a", "b", "c"],
                "year": [2020, 2020, 2021],
                "mean_rmsd_angstrom": [1.0, 3.0, 5.0],
            }
        )
        with (
            patch.object(self.plotter, "_read_csv", return_value=precision),
            patch.object(self.plotter, "_scientific_style"),
            patch.object(self.plotter, "_render_bar_series") as render_bar,
        ):
            self.plotter._plot_solution_nmr_monomer_precision_stat(
                Path("precision.csv"),
                Path("precision.png"),
                Path("precision.svg"),
                "mean",
                "Mean",
                "RMSD",
            )
        self.assertEqual(render_bar.call_args.kwargs["y_values"].tolist(), [2.0, 5.0])

        with patch.object(
            self.plotter, "_plot_solution_nmr_monomer_precision_stat"
        ) as render_precision:
            self.plotter.plot_solution_nmr_monomer_precision_stride_modeled_first_model_mean(
                Path("p.csv"), Path("p.png"), Path("p.svg")
            )
            self.plotter.plot_solution_nmr_monomer_precision_stride_modeled_first_model_median(
                Path("p.csv"), Path("p.png"), Path("p.svg")
            )
        self.assertEqual(
            [item.kwargs["statistic"] for item in render_precision.call_args_list],
            ["mean", "median"],
        )

        quality = pd.DataFrame(
            {
                "entry_id": ["a", "b"],
                "year": [2020, 2020],
                "clashscore": [1.0, 3.0],
                "ramachandran_outliers_percent": [2.0, 4.0],
                "sidechain_outliers_percent": [5.0, 7.0],
            }
        )
        with (
            patch.object(self.plotter, "_read_csv", return_value=quality),
            patch.object(self.plotter, "_scientific_style"),
            patch.object(self.plotter, "_render_bar_series") as render_quality,
        ):
            self.plotter.plot_solution_nmr_monomer_quality(
                Path("quality.csv"),
                Path("clash.png"),
                Path("clash.svg"),
                Path("rama.png"),
                Path("rama.svg"),
                Path("side.png"),
                Path("side.svg"),
            )
        self.assertEqual(render_quality.call_count, 3)
        self.assertEqual(
            render_quality.call_args_list[0].kwargs["y_values"].iloc[0], 2.0
        )

    def test_homolog_plot_variants_forward_annual_and_cumulative_shares(self) -> None:
        table_95 = self._homolog_frame([1, 0, 1])
        table_100 = self._homolog_frame([0, 0, 1])
        with (
            patch.object(
                self.plotter,
                "_read_csv",
                side_effect=[table_95, table_100, table_95, table_100],
            ),
            patch.object(self.plotter, "_scientific_style"),
            patch.object(self.plotter, "_render_bar_series") as bars,
            patch.object(self.plotter, "_render_line_series") as lines,
        ):
            paths = [Path(f"output-{index}") for index in range(10)]
            self.plotter.plot_solution_nmr_monomer_xray_homologs(
                Path("95.csv"), Path("100.csv"), *paths[:8]
            )
            self.plotter.plot_solution_nmr_monomer_xray_homologs_historical(
                Path("95h.csv"), Path("100h.csv"), *paths[:8]
            )

        self.assertEqual(bars.call_count, 4)
        self.assertEqual(lines.call_count, 4)
        first_annual = bars.call_args_list[0].kwargs["y_values"]
        first_cumulative = lines.call_args_list[0].kwargs["y_values"]
        self.assertEqual(first_annual.loc[2020], 50.0)
        self.assertAlmostEqual(first_cumulative.loc[2021], 200.0 / 3.0)

    def test_homolog_timing_plot_writes_and_renders_both_thresholds(self) -> None:
        regular_95 = self._homolog_frame([1, 1, 0])
        regular_100 = self._homolog_frame([1, 0, 0])
        historical_95 = self._homolog_frame([1, 0, 0])
        historical_100 = self._homolog_frame([0, 0, 0])
        with (
            patch.object(
                self.plotter,
                "_read_csv",
                side_effect=[regular_95, historical_95, regular_100, historical_100],
            ),
            patch.object(self.plotter, "_scientific_style"),
            patch.object(
                self.plotter, "_write_xray_homolog_timing_counts_csv"
            ) as write,
            patch.object(self.plotter, "_render_homolog_timing_stackplot") as render,
        ):
            self.plotter.plot_solution_nmr_monomer_xray_homolog_timing_share(
                Path("regular95.csv"),
                Path("regular100.csv"),
                Path("history95.csv"),
                Path("history100.csv"),
                Path("counts95.csv"),
                Path("counts100.csv"),
                Path("share95.png"),
                Path("share95.svg"),
                Path("share100.png"),
                Path("share100.svg"),
            )

        self.assertEqual(write.call_count, 2)
        self.assertEqual(render.call_count, 2)
        shares_95 = render.call_args_list[0].kwargs["table"]
        np.testing.assert_allclose(shares_95.sum(axis=1), [100.0, 100.0])

    def test_rmsd_plot_builds_six_summaries_and_applies_title_suffix(self) -> None:
        regular = pd.DataFrame(
            {
                "entry_id": ["a", "b", "c"],
                "year": [2020, 2020, 2021],
                "rmsd_ca_angstrom": [1.0, 3.0, 4.0],
            }
        )
        extremes = pd.DataFrame(
            {
                "entry_id": ["a", "b", "c"],
                "year": [2020, 2020, 2021],
                "best_rmsd_ca_angstrom": [0.5, 1.5, 2.0],
                "worst_rmsd_ca_angstrom": [4.0, 6.0, 8.0],
            }
        )
        with (
            patch.object(self.plotter, "_read_csv", side_effect=[regular, extremes]),
            patch.object(self.plotter, "_scientific_style"),
            patch.object(self.plotter, "_render_bar_series") as bars,
            patch.object(self.plotter, "_render_multi_line_series") as lines,
        ):
            paths = [Path(f"rmsd-{index}") for index in range(12)]
            self.plotter.plot_solution_nmr_monomer_xray_rmsd(
                Path("regular.csv"),
                Path("extremes.csv"),
                *paths,
                title_suffix="(historical)",
            )

        self.assertEqual(bars.call_count, 4)
        self.assertEqual(lines.call_count, 2)
        self.assertTrue(
            bars.call_args_list[0].kwargs["title"].endswith(" (historical)")
        )
        self.assertEqual(bars.call_args_list[0].kwargs["y_values"].loc[2020], 2.0)
        self.assertEqual(
            lines.call_args_list[0].kwargs["draw_order"],
            ["worst_rmsd", "best_rmsd", "best_resolution_rmsd"],
        )

    def test_precision_correlation_plot_draws_scatter_and_correlation_series(
        self,
    ) -> None:
        table = pd.DataFrame(
            {
                "entry_id": [f"entry-{index}" for index in range(9)],
                "year": [2019, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2022],
                "precision_rmsd": [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 6.0, 6.0, 6.0],
                "min_xray_rmsd": [1.0, 2.0, 3.0, 2.0, 4.0, 3.0, 2.0, 3.0, 4.0],
            }
        )
        drawn_axes: list[plt.Axes] = []

        def render_callback(*args: object, **kwargs: object) -> None:
            draw_fn = kwargs.get("draw_fn")
            if draw_fn is None:
                draw_fn = args[4]
            figure, axes = plt.subplots()
            draw_fn(axes)  # type: ignore[operator]
            drawn_axes.append(axes)
            plt.close(figure)

        with (
            patch.object(self.plotter, "_read_csv", side_effect=[Mock(), Mock()]),
            patch.object(
                self.plotter,
                "_prepare_xray_rmsd_precision_correlation_table",
                return_value=table,
            ),
            patch.object(self.plotter, "_scientific_style"),
            patch.object(
                self.plotter, "_render_figure", side_effect=render_callback
            ) as render,
        ):
            self.plotter.plot_solution_nmr_monomer_xray_rmsd_precision_correlation(
                Path("precision.csv"),
                Path("extremes.csv"),
                Path("scatter.png"),
                Path("scatter.svg"),
                Path("yearly.png"),
                Path("yearly.svg"),
                Path("cumulative.png"),
                Path("cumulative.svg"),
                yearly_min_count=3,
            )

        self.assertEqual(render.call_count, 3)
        self.assertEqual(len(drawn_axes[0].collections), 1)
        self.assertEqual(len(drawn_axes[0].lines), 1)
        self.assertEqual(len(drawn_axes[1].patches), 1)
        self.assertGreaterEqual(len(drawn_axes[2].patches), 1)


if __name__ == "__main__":
    unittest.main()
