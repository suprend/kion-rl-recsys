"""Stdlib layout contract tests; no Matplotlib/NumPy import or real rendering."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).with_name('plot_results_v3.py')
spec = importlib.util.spec_from_file_location('layout_v3_fixture', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class FakeLocator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeFormatter:
    def __init__(self, template):
        self.template = template


class FakeTicker:
    MaxNLocator, StrMethodFormatter = FakeLocator, FakeFormatter


class FakeXAxis:
    def __init__(self):
        self.locator, self.formatter = None, None

    def set_major_locator(self, value):
        self.locator = value

    def set_major_formatter(self, value):
        self.formatter = value


class FakeAxis:
    def __init__(self):
        self.xaxis = FakeXAxis()
        self.tick_configuration = None
        self.legend_configuration = None
        self.xlim = (-.00651, .00273)
        self.numeric_data = object()

    def tick_params(self, **kwargs):
        self.tick_configuration = kwargs

    def legend(self, **kwargs):
        self.legend_configuration = kwargs


class LayoutContractTests(unittest.TestCase):
    def test_03_and_07_use_sparse_short_ticks_on_every_panel_without_limits_or_data_changes(self):
        for name, count in (('03-paired-uncertainty', 3), ('07-item-contributions', 4)):
            axes = [FakeAxis() for _ in range(count)]
            before = [(axis.xlim, axis.numeric_data) for axis in axes]
            mod.readable_layout(axes, name, FakeTicker)
            for axis in axes:
                self.assertEqual(axis.xaxis.locator.kwargs, {'nbins': 3})
                self.assertEqual(axis.xaxis.formatter.template, '{x:.3g}')
                self.assertEqual(axis.tick_configuration, {'axis': 'x', 'labelsize': 8})
                self.assertIsNone(axis.legend_configuration)
            self.assertEqual(before, [(axis.xlim, axis.numeric_data) for axis in axes])

    def test_01_moves_only_right_legend_to_explicit_middle_band(self):
        axes = [FakeAxis(), FakeAxis()]
        mod.readable_layout(axes, '01-checkpoint-tradeoff', FakeTicker)
        self.assertIsNone(axes[0].legend_configuration)
        self.assertEqual(axes[1].legend_configuration,
                         {'loc': 'center left', 'bbox_to_anchor': (.02, .60), 'frameon': False, 'fontsize': 9})
        self.assertTrue(all(axis.xaxis.locator is None and axis.tick_configuration is None for axis in axes))

    def test_unaffected_figures_keep_existing_tick_and_legend_contracts(self):
        for name in ('02-matched-comparison', '04-exposure-distribution', '05-target-ranks', '06-screen-utilities-behavior'):
            axes = [FakeAxis() for _ in range(4)]
            mod.readable_layout(axes, name, FakeTicker)
            self.assertTrue(all(axis.xaxis.locator is None and axis.xaxis.formatter is None and
                                axis.tick_configuration is None and axis.legend_configuration is None for axis in axes))

    def test_frozen_base_and_v2_title_numeric_helpers_are_unchanged(self):
        v2 = SOURCE.with_name('plot_results_v2.py')
        self.assertEqual(hashlib.sha256(v2.read_bytes()).hexdigest(), '08dc40464172cbb3d6cd4f0925fb97b1753f36beae270d56b0ed5a70760906df')
        self.assertEqual(hashlib.sha256(mod.BASE_SOURCE.read_bytes()).hexdigest(), mod.BASE_SHA256)
        old = {node.name: node for node in ast.parse(v2.read_text(encoding='utf-8')).body if isinstance(node, ast.FunctionDef)}
        new_tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        new = {node.name: node for node in new_tree.body if isinstance(node, ast.FunctionDef)}
        for name in ('base_module', 'validate', 'needed_raw_ids', 'format_label', 'canonical_sha', 'load_labels'):
            self.assertEqual(ast.dump(old[name]), ast.dump(new[name]), name)
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr in
                             ('set_xlim', 'set_ylim', 'set_xscale', 'set_yscale', 'autoscale', 'margins') for node in ast.walk(new_tree)))
        self.assertFalse(any(isinstance(node, ast.Assign) and any(isinstance(target, ast.Subscript) and
                             any(isinstance(part, ast.Name) and part.id == 'data' for part in ast.walk(target)) for target in node.targets)
                             for node in ast.walk(new_tree)))
        top_imports = [alias.name for node in new_tree.body if isinstance(node, ast.Import) for alias in node.names]
        self.assertFalse({'numpy', 'matplotlib', 'torch'} & set(top_imports))


if __name__ == '__main__':
    unittest.main(verbosity=2)
