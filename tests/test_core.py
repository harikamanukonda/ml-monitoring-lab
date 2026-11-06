import unittest
import numpy as np
from core import gate, population_stability_index, run

class MonitoringTests(unittest.TestCase):
    def test_identical_distribution_has_zero_psi(self):
        x=np.arange(100, dtype=float)
        self.assertAlmostEqual(population_stability_index(x,x),0)
    def test_detects_shift(self):
        rng=np.random.default_rng(9); x=rng.normal(size=1000)
        self.assertGreater(population_stability_index(x,x+2),.2)
    def test_gate_rejects_regression_and_insufficient_gain(self):
        self.assertEqual(gate(1,1.1)['decision'],'KEEP INCUMBENT')
        self.assertEqual(gate(1,.98,.05)['decision'],'KEEP INCUMBENT')
        self.assertEqual(gate(1,.7,.05)['decision'],'PROMOTE CANDIDATE')
    def test_no_shift_is_not_automatically_promoted(self):
        self.assertEqual(run({'feature_shift':0,'concept_shift':0})['gate']['decision'],'KEEP INCUMBENT')
    def test_concept_shift_candidate_improves(self):
        result=run({})
        self.assertLess(result['validation']['candidate_mae'],result['validation']['incumbent_mae'])
        self.assertEqual(result['split']['candidate_training']+result['split']['validation'],1000)
    def test_invalid_inputs(self):
        for params in [{'feature_shift':'inf'},{'concept_shift':-1},{'min_improvement':1}]:
            with self.assertRaises(ValueError): run(params)

if __name__ == '__main__': unittest.main()
