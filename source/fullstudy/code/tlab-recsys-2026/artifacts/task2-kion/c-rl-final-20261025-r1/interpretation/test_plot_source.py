"""Stdlib aggregate/source contract tests: no rendering or NumPy import."""
import ast
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).with_name('plot_results.py')
spec = importlib.util.spec_from_file_location('scientific3_plots', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class PlotSourceTests(unittest.TestCase):
    def test_plotting_dependencies_are_lazy(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse({'numpy', 'matplotlib', 'matplotlib.pyplot'} & {a.name for a in node.names})
        source = SOURCE.read_text(encoding='utf-8')
        self.assertIn("for extension in ('svg', 'png')", source)
        self.assertNotIn('np.load', source)
        self.assertNotIn('user_id', source)

    def test_cohort_validation_prevents_proxy_mixing(self):
        value = {'contract': 'kion-scientific3-analysis-v1', 'state': 'completed', 'stage': 'full4096',
                 'rows': 4096, 'metric_denominator': 4096, 'candidate_width': 1024,
                 'unique_users': 3000, 'summaries': {arm: {} for arm in mod.ARMS}, 'screen_diagnostics': None}
        self.assertIs(mod.validate(value), value)
        self.assertIn('4096', mod.cohort_caption(value))
        self.assertIn('Фиксированный β', mod.cohort_caption(value))
        value['screen_diagnostics'] = {}
        with self.assertRaisesRegex(ValueError, 'Proxy cohort'):
            mod.validate(value)

    def test_scientific_labels_and_QA_receipt(self):
        source = SOURCE.read_text(encoding='utf-8')
        for token in ('requires_visual_QA', 'unknown/off-pool', 'условны', 'арифметическое разложение', 'raw item ID'):
            self.assertIn(token, source)
        self.assertEqual(len(set(mod.COLORS)), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
