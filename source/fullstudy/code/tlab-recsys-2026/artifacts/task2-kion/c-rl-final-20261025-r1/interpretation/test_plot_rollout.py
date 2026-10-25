"""Stdlib source and synthetic-summary tests; never import/render Matplotlib."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_plot_rollout_tests',HERE/'plot_rollout.py')
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def fixture():
    data = dict(contract=plot.SUMMARY_CONTRACT,state='completed',trajectory_count=192,position_count=16,steps=5,
        seed=20261024,expected_positions=plot.POSITIONS[:],group_summaries=[],matched_C_comparisons=[],
        RM_greedy_unique5_control=dict(positions=16,mean_unique_generated=5.),
        rollout=dict(path='synthetic/rollout.json',sha256='a'*64),
        source=dict(path='synthetic/rollout_summary.py',sha256='b'*64),
        items_catalog=dict(path='synthetic/items.csv',sha256='c'*64),
        source_metadata=dict(contract='kion-final3-liveprefix-grid-v1',five_step_training=False,
            synthetic_prefix_only=True,RM_observed_s0_fixed=True,cache_sha256='d'*64,reward_sha256='e'*64,
            utility_sha256='f'*64))
    for arm in plot.ARMS:
        for mode,mask in plot.FACETS:
            steps=[]
            for j in range(5):
                offset=plot.ARMS.index(arm)*.01
                values=dict(expected_neutral_proxy=offset+j*.005,
                    expected_neutral_residual_proxy=-offset+j*.002,kl_nats=0. if arm=='C' else offset*(j+1),
                    entropy_nats=6.8-j*.08)
                steps.append(dict(step=j+1,**{name:dict(mean=value,denominator=16,total=16) for name,value in values.items()},
                    original_seen_fraction=(j+1)/16,generated_repeat_fraction=0. if mask=='exclude-generated' else j/16))
            data['group_summaries'].append(dict(arm=arm,selection=mode,repeat_mask=mask,positions=16,per_step=steps))
            if arm!='C':
                data['matched_C_comparisons'].append(dict(arm=arm,reference='C',selection=mode,repeat_mask=mask,matched_positions=16))
    return data


class ValidationTests(unittest.TestCase):
    def test_complete_contract_and_no_item_label_dependency(self):
        data=fixture()
        self.assertIs(plot.validate(data),data)
        self.assertEqual(plot.needed_raw_ids(data),set())
        self.assertNotIn('matplotlib',sys.modules)
        self.assertNotIn('numpy',sys.modules)

    def test_incomplete_contract_grid_and_provenance_rejected(self):
        for field,value in [('contract','old'),('state','running'),('trajectory_count',191),('position_count',15),
                            ('steps',4),('seed',0),('expected_positions',list(reversed(plot.POSITIONS))),
                            ('rollout',None),('items_catalog',{})]:
            data=fixture()
            data[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                plot.validate(data)
        for change in (lambda d:d['group_summaries'].pop(),
                       lambda d:d['group_summaries'].__setitem__(0,copy.deepcopy(d['group_summaries'][1])),
                       lambda d:d['matched_C_comparisons'].pop()):
            data=fixture();change(data)
            with self.assertRaises(ValueError):
                plot.validate(data)

    def test_missing_nonfinite_and_partial_denominator_never_substituted(self):
        for change in (lambda s:s.pop('kl_nats'),
                       lambda s:s['entropy_nats'].__setitem__('mean',math.nan),
                       lambda s:s['expected_neutral_proxy'].__setitem__('mean',None),
                       lambda s:s['expected_neutral_residual_proxy'].__setitem__('denominator',15),
                       lambda s:s.__setitem__('original_seen_fraction',2.),
                       lambda s:s.__setitem__('generated_repeat_fraction',-.1)):
            data=fixture();change(data['group_summaries'][0]['per_step'][0])
            with self.assertRaises(ValueError):
                plot.validate(data)

    def test_scope_and_C_reference_sanity_rejected(self):
        for field,value in [('RM_observed_s0_fixed',False),('synthetic_prefix_only',False),('five_step_training',True)]:
            data=fixture();data['source_metadata'][field]=value
            with self.assertRaises(ValueError):
                plot.validate(data)
        for field,value in [('kl_nats',.1),('entropy_nats',7.),('expected_neutral_proxy',.1),
                            ('expected_neutral_residual_proxy',.1)]:
            data=fixture();data['group_summaries'][0]['per_step'][0][field]['mean']=value
            with self.assertRaises(ValueError):
                plot.validate(data)
        data=fixture();data['group_summaries'][1]['per_step'][1]['generated_repeat_fraction']=.1
        with self.assertRaises(ValueError):
            plot.validate(data)

    def test_unplotted_nonfinite_values_rejected(self):
        data=fixture();data['unused_saved_metric']=math.inf
        with self.assertRaises(ValueError):
            plot.validate(data)


class PlotPlanTests(unittest.TestCase):
    def test_two_figures_preserve_full_mode_mask_grid_and_actual_values(self):
        data=fixture();before=copy.deepcopy(data)
        plans=plot.prepare_series(data)
        self.assertEqual(data,before)
        self.assertEqual([plan['file'] for plan in plans],['08-five-step-panel','08b-five-step-behavior'])
        self.assertTrue(all(len(plan['panels'])==12 for plan in plans))
        first=plans[0]['panels'][0]
        expected=[step['expected_neutral_proxy']['mean'] for step in data['group_summaries'][0]['per_step']]
        self.assertEqual(first['series']['C'],expected)
        behavior=plans[1]['panels'][1]
        self.assertEqual(behavior['metric'],'original_seen_fraction')
        self.assertEqual(behavior['series']['C'],[100*(j+1)/16 for j in range(5)])
        self.assertIn('ТОМ ЖЕ текущем prefix',plans[0]['caption'])
        self.assertIn('не KL к распределению на независимо',plans[0]['caption'])
        self.assertIn('заморожены в s0',plans[0]['caption'])
        self.assertIn('без популяционных интервалов',plans[0]['caption'])

    def test_input_group_order_does_not_change_series(self):
        data=fixture();expected=plot.prepare_series(data)
        data['group_summaries'].reverse()
        self.assertEqual(plot.prepare_series(data),expected)

    def test_shared_metric_limits_and_percentage_scale_are_explicit(self):
        self.assertEqual(plot._limits([0,.5],'original_seen_fraction'),(0.,100.))
        low,high=plot._limits([-.1,.2],'expected_neutral_residual_proxy')
        self.assertLess(low,-.1);self.assertGreater(high,.2)
        self.assertEqual(plot._limits([0,0],'kl_nats'),(0.,.02))
        self.assertNotEqual(plot.MARKERS['C'],plot.MARKERS['RL-completion'])
        self.assertNotEqual(plot.LINESTYLES['C'],plot.LINESTYLES['RL-residual'])

    def test_canonical_input_hash_is_order_stable_and_value_sensitive(self):
        data=fixture();reordered=copy.deepcopy(dict(reversed(list(data.items()))))
        self.assertEqual(plot.canonical_sha(data),plot.canonical_sha(reordered))
        data['group_summaries'][0]['per_step'][1]['expected_neutral_proxy']['mean']+=.001
        self.assertNotEqual(plot.canonical_sha(data),plot.canonical_sha(reordered))

    def test_invalid_render_input_fails_before_import_or_output(self):
        data=fixture();data['trajectory_count']=0
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'uncreated'
            with self.assertRaises(ValueError):
                plot.render(data,output)
            self.assertFalse(output.exists())
        self.assertNotIn('matplotlib',sys.modules)


class SourceBoundaryTests(unittest.TestCase):
    def test_numerical_imports_are_only_inside_explicit_render(self):
        tree=ast.parse((HERE/'plot_rollout.py').read_text(encoding='utf-8'))
        top_imports=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
        names={alias.name.split('.')[0] for node in top_imports if isinstance(node,ast.Import) for alias in node.names}
        names|={node.module.split('.')[0] for node in top_imports if isinstance(node,ast.ImportFrom)}
        self.assertLessEqual(names,{'__future__','hashlib','json','math','pathlib','sys','textwrap'})
        render=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='render')
        heavy=[node for node in ast.walk(render) if isinstance(node,ast.Import) and any(a.name.startswith('matplotlib') for a in node.names)]
        self.assertEqual(len(heavy),2)
        calls={node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
        self.assertFalse(calls&{'load','loadtxt','read_csv','read_parquet','mmap','Popen','run','backward'})
        self.assertFalse(any(isinstance(node,ast.If) and isinstance(node.test,ast.Compare) for node in tree.body))


if __name__=='__main__':
    unittest.main()
