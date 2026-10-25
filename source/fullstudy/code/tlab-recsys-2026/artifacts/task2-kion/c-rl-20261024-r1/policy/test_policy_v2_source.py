"""Stdlib regressions for shape/slot repair; numerical proof is root-owned."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import random
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('policy_v2_subject',HERE/'policy_v2.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ProducerShapeTests(unittest.TestCase):
    def test_exact_partial_sizes_slots_and_peers(self):
        origins=[1]*17+[2]*31+[3]*9
        layout=m.producer_layout(origins)
        self.assertEqual(layout[0],(16,0))
        self.assertEqual(layout[15],(16,15))
        self.assertEqual(layout[16],(1,0))
        self.assertEqual(layout[47],(15,14))
        self.assertEqual(layout[56],(9,8))
        for pos,(size,slot) in enumerate(layout):
            peers,peer_slot=m.producer_peers(origins,pos)
            self.assertEqual((len(peers),peer_slot),(size,slot))
            self.assertEqual(peers[slot],pos)

    def test_interleaved_dates_preserve_original_slots(self):
        self.assertEqual(m.producer_layout([2,1,2,1,2]),[(3,0),(2,0),(3,1),(2,1),(3,2)])

    def test_duplicate_occurrences_cannot_overwrite_slots(self):
        layout=m.producer_layout([1]*9)
        buckets=m.shape_buckets([3]*16,layout)
        self.assertEqual(len(buckets),16)
        self.assertTrue(all(bucket['size']==9 and len(bucket['assignments'])==1 for bucket in buckets))
        self.assertEqual(sorted(item[0] for bucket in buckets for item in bucket['assignments']),list(range(16)))

    def test_distinct_context_same_size_slot_collision(self):
        layout=m.producer_layout([1]*16+[2]*16)
        buckets=m.shape_buckets([0,16,1,17],layout)
        self.assertEqual(len(buckets),2)
        self.assertEqual([bucket['used_slots'] for bucket in buckets],[{0,1},{0,1}])

    def test_randomized_multiset_and_logical_order_contract(self):
        origins=[day for day,count in enumerate((33,9,11,6,12,1)) for _ in range(count)]
        layout=m.producer_layout(origins)
        rng=random.Random(147)
        for _ in range(100):
            positions=[rng.randrange(len(layout)) for _ in range(16)]
            buckets=m.shape_buckets(positions,layout)
            actual={}
            for bucket in buckets:
                self.assertEqual(len(bucket['used_slots']),len(bucket['assignments']))
                for logical,position,slot in bucket['assignments']:
                    self.assertNotIn(logical,actual)
                    actual[logical]=position
                    self.assertEqual((bucket['size'],slot),layout[position])
            self.assertEqual([actual[i] for i in range(16)],positions)

    def test_invalid_logical_positions_rejected(self):
        with self.assertRaises(ValueError):
            m.shape_buckets([],[(1,0)])
        with self.assertRaises(ValueError):
            m.shape_buckets([1],[(1,0)])


class FrozenProtocolTests(unittest.TestCase):
    def test_original_policy_immutable(self):
        self.assertEqual(hashlib.sha256((HERE/'policy.py').read_bytes()).hexdigest(),
            '5635b93e0c9c9a245cbf723cdf257d688ce4f0025162a647354918ae52bd11a1')

    def test_objective_sampler_and_full4096_protocol_unchanged(self):
        original=ast.parse((HERE/'policy.py').read_text(encoding='utf-8'))
        repaired=ast.parse((HERE/'policy_v2.py').read_text(encoding='utf-8'))
        def functions(tree):
            return {node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,ast.FunctionDef)}
        old,new=functions(original),functions(repaired)
        for name in ('exact_loss','effective_advantage','distribution','normalized_preregistration',
                     'select_beta','evaluate_selected','load_rl_head','tensor_batch','head_scores'):
            self.assertEqual(old[name],new[name],name)
        source=(HERE/'policy_v2.py').read_text(encoding='utf-8')
        self.assertIn('positions = [rng.choice(eligible) for _ in range(16)]',source)
        self.assertIn("torch.optim.Adam(model.parameters(), lr=prereg['learning_rate'], weight_decay=0)",source)

    def test_no_bucket_loss_normalization(self):
        tree=ast.parse((HERE/'policy_v2.py').read_text(encoding='utf-8'))
        score=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='score_shaped')
        self.assertIn('torch.stack(logical, dim=0)',ast.unparse(score))
        self.assertNotIn('exact_loss',ast.unparse(score))
        self.assertNotIn('.mean(',ast.unparse(score))

    def test_inert_import_and_native_probe_alias(self):
        tree=ast.parse((HERE/'policy_v2.py').read_text(encoding='utf-8'))
        imports=[ast.unparse(node) for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('numpy' in node or 'torch' in node for node in imports))
        probe=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='probe')
        self.assertIn('repair_probe(request, check)',ast.unparse(probe))

    def test_repair_proof_has_no_optimizer_call(self):
        tree=ast.parse((HERE/'policy_v2.py').read_text(encoding='utf-8'))
        proof=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='repair_probe')
        calls=[node for node in ast.walk(proof) if isinstance(node,ast.Call)]
        self.assertFalse(any(isinstance(node.func,ast.Attribute) and node.func.attr in ('Adam','AdamW','step','open_context') for node in calls))
        self.assertIn('maximum == 0.0',ast.unparse(proof))


if __name__=='__main__':
    unittest.main()
