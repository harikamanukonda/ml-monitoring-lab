"""Controlled drift experiment with disjoint retraining and validation partitions."""
import hashlib
import numpy as np

def population_stability_index(reference, current, bins=10):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins+1)))
    if len(edges) < 3:
        raise ValueError('Reference distribution has insufficient variation')
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, edges)[0] / len(reference)
    actual = np.histogram(current, edges)[0] / len(current)
    expected, actual = np.clip(expected, 1e-6, None), np.clip(actual, 1e-6, None)
    return float(np.sum((actual-expected)*np.log(actual/expected)))

def fit(x, y):
    design = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(design, y, rcond=None)[0]

def predict(weights, x):
    return weights[0] + x @ weights[1:]

def gate(incumbent_mae, candidate_mae, minimum=.05):
    if not all(np.isfinite([incumbent_mae, candidate_mae, minimum])) or incumbent_mae <= 0 or candidate_mae < 0 or not 0 <= minimum <= .9:
        raise ValueError('Invalid gate metrics')
    improvement = (incumbent_mae-candidate_mae)/incumbent_mae
    promote = candidate_mae < incumbent_mae and improvement >= minimum
    return {'decision': 'PROMOTE CANDIDATE' if promote else 'KEEP INCUMBENT',
            'relative_improvement': round(float(improvement), 4),
            'reason': f'Candidate MAE improvement is {improvement:.1%}; required improvement is {minimum:.1%}. '
            + ('The validation gate passed.' if promote else 'The validation gate did not pass.')}

def checked(params, key, default, low, high):
    value = float(params.get(key, default))
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{key} must be between {low} and {high}')
    return value

def run(params):
    shift = checked(params, 'feature_shift', 1.5, 0, 5)
    concept = checked(params, 'concept_shift', 1, 0, 4)
    minimum = checked(params, 'min_improvement', .05, 0, .9)
    rng = np.random.default_rng(42)
    reference_x = rng.normal(size=(1000, 3))
    reference_y = 20 + reference_x @ np.array([3, -2, 1]) + rng.normal(0, .5, 1000)
    current_x = rng.normal(size=(1000, 3))
    current_x[:, 0] += shift
    current_y = 20 + current_x @ np.array([3+concept, -2, 1]) + rng.normal(0, .5, 1000)
    incumbent = fit(reference_x, reference_y)
    candidate = fit(current_x[:600], current_y[:600])
    validation_x, validation_y = current_x[600:], current_y[600:]
    old_prediction, new_prediction = predict(incumbent, validation_x), predict(candidate, validation_x)
    old_mae = float(np.mean(abs(validation_y-old_prediction)))
    new_mae = float(np.mean(abs(validation_y-new_prediction)))
    psi = population_stability_index(reference_x[:, 0], current_x[:, 0])
    model_id = lambda weights: hashlib.sha256(weights.tobytes()).hexdigest()[:12]
    return {'seed': 42, 'data_source': 'synthetic controlled regression',
            'split': {'reference_training': 1000, 'candidate_training': 600, 'validation': 400},
            'drift': {'psi': round(psi, 4), 'feature': 'feature_0',
                      'interpretation': 'Feature shift flagged by the demo threshold of 0.2' if psi > .2 else 'Feature shift below the demo threshold of 0.2'},
            'validation': {'incumbent_mae': round(old_mae, 4), 'candidate_mae': round(new_mae, 4)},
            'gate': gate(old_mae, new_mae, minimum),
            'registry': {'incumbent_id': model_id(incumbent), 'candidate_id': model_id(candidate), 'persistent_promotion': False},
            'preview': {'actual': validation_y[:30].round(3).tolist(), 'incumbent': old_prediction[:30].round(3).tolist(), 'candidate': new_prediction[:30].round(3).tolist()}}
