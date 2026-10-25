"""Final one-line legend/path diff tests; pure stdlib, no actual rendering."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


def read(name):
    return (ROOT / name).read_text(encoding='utf-8')


def load(name):
    spec = importlib.util.spec_from_file_location('final_legend_fixture', ROOT / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeAxis:
    def __init__(self):
        self.legend_options = None
        self.numeric_data = object()
        self.xlim = (-.5, 1050.)

    def legend(self, **kwargs):
        self.legend_options = kwargs


class FinalLegendTests(unittest.TestCase):
    def test_plot_diff_is_exactly_one_legend_bbox_and_original_stays_frozen(self):
        old, new = read('plot_results_v3.py'), read('plot_results_v4.py')
        self.assertEqual(hashlib.sha256(old.encode('utf-8')).hexdigest(), '4020dce961bb6abbc7d3b48e40a94266f5b9ce3c2e8b27b5a88f08537b7c97a1')
        self.assertEqual(new, old.replace('bbox_to_anchor=(.02, .60)', 'bbox_to_anchor=(.52, .72)'))
        module = load('plot_results_v4.py')
        self.assertEqual(hashlib.sha256(module.BASE_SOURCE.read_bytes()).hexdigest(), module.BASE_SHA256)

    def test_only_right_legend_moves_and_data_limits_remain_untouched(self):
        module = load('plot_results_v4.py')
        axes = [FakeAxis(), FakeAxis()]
        before = [(axis.numeric_data, axis.xlim) for axis in axes]
        module.readable_layout(axes, '01-checkpoint-tradeoff', None)
        self.assertIsNone(axes[0].legend_options)
        self.assertEqual(axes[1].legend_options,
                         {'loc': 'center left', 'bbox_to_anchor': (.52, .72), 'frameon': False, 'fontsize': 9})
        self.assertEqual(before, [(axis.numeric_data, axis.xlim) for axis in axes])

    def test_native_wrapper_diff_is_only_v4_selection_and_fresh_figures_v3_path(self):
        old, new = read('render_figures_v2.py'), read('render_figures_v3.py')
        self.assertEqual(hashlib.sha256(old.encode('utf-8')).hexdigest(), 'ca315c84fb7bfff7c9efc8d2895ad9cabebf978eeb61d9b409e782a3d942d5f1')
        expected = old.replace('V3_PLOT_SOURCE', 'V4_PLOT_SOURCE').replace('plot_results_v3.py', 'plot_results_v4.py')
        expected = expected.replace('figures-v2', 'figures-v3').replace('Layout-v2 render', 'Layout-v3 render')
        self.assertEqual(new, expected)
        self.assertIn("allowed = {V4_PLOT_SOURCE.resolve()}", new)
        self.assertIn("ROOT / 'reports' / 'figures-v3' / folder", new)

    def test_portable_replay_preserves_exact_all_integrity_guards_apart_from_versioned_paths(self):
        old, new = read('replay_portable_v2.py'), read('replay_portable_v3.py')
        self.assertEqual(hashlib.sha256(old.encode('utf-8')).hexdigest(), '5c60050c05d861053934bf44a99d82be5901615427bfb47d5d6e73f7a719e2e9')
        expected = old.replace('Level A replay v2', 'Level A replay v3').replace('plot_results_v3.py', 'plot_results_v4.py')
        expected = expected.replace("report / 'figures-v2'", "report / 'figures-v3'")
        self.assertEqual(new, expected)
        self.assertIn("require(sys.dont_write_bytecode", new)
        self.assertIn("require(sys.pycache_prefix is None", new)
        self.assertIn("exec(compile(payload, str(path), 'exec'), module.__dict__)", new)
        self.assertIn("report / 'replay-figures' / folder", new)
        for name in ('plot_results_v4.py', 'render_figures_v3.py', 'replay_portable_v3.py'):
            tree = ast.parse(read(name))
            top_imports = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
            self.assertFalse(top_imports & {'numpy', 'matplotlib', 'torch'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
