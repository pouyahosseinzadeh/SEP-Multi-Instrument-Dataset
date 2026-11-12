import copy
import json

import numpy as np
import pandas as pd
import pytest

from sep_dataset.proton_calibration import (
    CalibrationError, FEATURES, apply_calibration, fit_calibration,
    frame_source_key, load_calibration, native_spectral_matrix,
    save_calibration, validation_table, load_calibration_bank,
    apply_calibration_bank, BANK_SCHEMA,
)


def fixture():
    rng = np.random.RandomState(314)
    n = 800
    values = rng.uniform(0, 0.03, (n, 13))
    values[100:400] *= 1000
    values[600:] *= 1000
    values[::13, 5] = 0.0
    energy = np.geomspace(1.4, 334, 13)
    frame = pd.DataFrame(index=pd.date_range('2017-09-01', periods=n, freq='5min', tz='UTC'))
    for i in range(13):
        frame[f'native_flux_{i}'] = values[:, i]
        frame[f'native_energy_MeV_{i}'] = energy[i]
    frame['source_satellite'] = 'GOES-16'
    frame['source_product'] = 'NOAA_SGPS_L2_AVG5M'
    frame['source_detector'] = 'SGPS-X physical west'
    frame['energy_coordinate_basis'] = 'geometric_mean_of_native_L2_band_bounds_effective_field_unused'
    target = pd.DataFrame(index=frame.index)
    for j, feature in enumerate(FEATURES):
        channel = (6, 8, 9)[j]
        target[feature] = (j+1)*0.01 + values[:, channel]*0.4 + values[:, 10+j]*0.2
    train = np.arange(n) < 400
    quiet = (np.arange(n) >= 400) & (np.arange(n) < 600)
    event = np.arange(n) >= 600
    return frame, target, train, quiet, event


def fitted():
    frame, targets, train, quiet, event = fixture()
    artifact = fit_calibration(frame, targets, train, quiet | event,
                               validation_regimes={'quiet': quiet, 'event': event})
    return frame, targets, artifact


def test_independent_response_fit_preserves_zero_and_passes(tmp_path):
    frame, targets, artifact = fitted()
    assert artifact['status'] == 'passed'
    assert artifact['reference_kind'] == 'observed_legacy'
    assert all(model['passed'] for model in artifact['models'].values())
    x, grid, valid = native_spectral_matrix(frame)
    assert valid.all() and x[0, 5] == 0.0
    predicted = apply_calibration(frame, artifact)
    for feature in FEATURES:
        mask = predicted[feature].notna()
        np.testing.assert_allclose(predicted.loc[mask, feature], targets.loc[mask, feature], atol=1e-10, rtol=1e-9)
        assert artifact['models'][feature]['intercept'] >= 0
        assert min(artifact['models'][feature]['coefficients']) >= 0
    path = tmp_path/'calibration.json'
    save_calibration(path, artifact)
    assert load_calibration(path) == artifact
    assert len(validation_table(artifact)) == 6


def test_wrong_satellite_or_l1b_cannot_use_l2_model():
    frame, _, artifact = fitted()
    for column, value in [('source_satellite', 'GOES-18'),
                          ('source_product', 'NOAA_SGPS_L1B_VALID_1S_TO_5MIN')]:
        changed = frame.copy()
        changed[column] = value
        with pytest.raises(CalibrationError, match='No validated calibration'):
            apply_calibration(changed, artifact)
        result = apply_calibration(changed, artifact, strict_source=False)
        assert result[list(FEATURES)].isna().all().all()
        assert result.P4_calibration_status.eq('unsupported_source').all()


def test_l2_energy_basis_alias_requires_same_numeric_grid():
    frame, targets, artifact = fitted()
    frame['energy_coordinate_basis'] = 'native_L2_DiffProtonEffectiveEnergy'
    result = apply_calibration(frame, artifact)
    assert result.P4.notna().any()
    changed = frame.copy()
    changed.loc[:, [c for c in changed if c.startswith('native_energy_')]] *= 1.1
    result = apply_calibration(changed, artifact)
    assert result.P4.isna().all()
    assert result.P4_calibration_status.eq('unsupported_energy_grid').all()


def test_out_of_validated_envelope_is_not_silently_extrapolated():
    frame, _, artifact = fitted()
    high = frame.iloc[:1].copy()
    for column in high:
        if column.startswith('native_flux_'):
            high[column] = 1e10
    result = apply_calibration(high, artifact)
    assert result[list(FEATURES)].isna().all().all()
    assert result.P4_calibration_status.eq('outside_validated_spectral_range').all()


def test_missing_validation_spectra_count_against_coverage():
    frame, target, train, quiet, event = fixture()
    rows = np.flatnonzero(quiet)[:30]
    frame.loc[frame.index[rows], 'native_flux_12'] = np.nan
    fit = fit_calibration(frame, target, train, quiet | event,
                          validation_regimes={'quiet': quiet, 'event': event})
    assert fit['status'] == 'failed'
    for model in fit['models'].values():
        assert model['validation']['quiet']['requested_rows'] == 200
        assert model['validation']['quiet']['valid_pairs'] == 170
        assert model['validation']['quiet']['coverage'] == .85
    with pytest.raises(CalibrationError, match='did not pass'):
        apply_calibration(frame, fit)


def test_failed_validation_is_saved_but_cannot_be_used(tmp_path):
    frame, target, train, quiet, event = fixture()
    target.loc[quiet | event] *= 10
    fit = fit_calibration(frame, target, train, quiet | event,
                          validation_regimes={'quiet': quiet, 'event': event})
    assert fit['status'] == 'failed'
    path = tmp_path/'failed.json'
    save_calibration(path, fit)
    with pytest.raises(CalibrationError, match='did not pass'):
        load_calibration(path)
    assert load_calibration(path, require_pass=False)['status'] == 'failed'


def test_training_never_uses_validation_targets():
    frame, target, train, quiet, event = fixture()
    a = fit_calibration(frame, target, train, quiet | event,
                        validation_regimes={'quiet': quiet, 'event': event})
    changed = target.copy()
    changed.loc[quiet | event] *= 3
    b = fit_calibration(frame, changed, train, quiet | event,
                        validation_regimes={'quiet': quiet, 'event': event})
    for feature in FEATURES:
        assert a['models'][feature]['coefficients'] == b['models'][feature]['coefficients']
        assert a['models'][feature]['intercept'] == b['models'][feature]['intercept']


def test_time_row_leakage_and_unjustified_regime_masks_rejected():
    frame, targets, train, quiet, event = fixture()
    overlap = train.copy()
    overlap[400] = True
    with pytest.raises(CalibrationError, match='overlap'):
        fit_calibration(frame, targets, overlap, quiet | event,
                        validation_regimes={'quiet': quiet, 'event': event})
    invalid = quiet.copy()
    invalid[0] = True
    with pytest.raises(CalibrationError, match='subsets'):
        fit_calibration(frame, targets, train, quiet | event,
                        validation_regimes={'quiet': invalid, 'event': event})


def test_source_metadata_cannot_be_overridden_or_mixed():
    frame, _, artifact = fitted()
    supplied = frame_source_key(frame)
    supplied['source_satellite'] = 'GOES-18'
    with pytest.raises(CalibrationError, match='disagrees'):
        apply_calibration(frame, artifact, source_key=supplied)
    mixed = frame.copy()
    mixed.iloc[0, mixed.columns.get_loc('source_satellite')] = 'GOES-18'
    with pytest.raises(CalibrationError, match='one source group'):
        apply_calibration(mixed, artifact)


def test_bridge_requires_valid_parent_and_records_lineage():
    frame, target, train, quiet, event = fixture()
    parent = fit_calibration(frame, target, train, quiet | event,
                             validation_regimes={'quiet': quiet, 'event': event})
    bridge_input = frame.copy()
    bridge_input['source_satellite'] = 'GOES-18'
    with pytest.raises(CalibrationError, match='validated reference parent'):
        fit_calibration(bridge_input, target, train, quiet | event,
                        validation_regimes={'quiet': quiet, 'event': event},
                        reference_kind='calibrated_reference_bridge')
    bridge = fit_calibration(bridge_input, target, train, quiet | event,
                             validation_regimes={'quiet': quiet, 'event': event},
                             reference_kind='calibrated_reference_bridge', parent_artifact=parent)
    assert bridge['status'] == 'passed'
    assert bridge['lineage'] == [parent['calibration_id']]
    assert 'not new legacy observations' in bridge['validation_target_note']
    assert apply_calibration(bridge_input, bridge).P4.notna().any()


def test_artifact_tampering_is_detected(tmp_path):
    _, _, fit = fitted()
    changed = copy.deepcopy(fit)
    changed['models']['P4']['intercept'] *= 1.1
    path = tmp_path/'altered.json'
    path.write_text(json.dumps(changed))
    with pytest.raises(CalibrationError, match='content hash'):
        load_calibration(path)


def test_irrelevant_low_energy_gaps_do_not_erase_high_energy_measurements():
    frame, _, artifact = fitted()
    original = apply_calibration(frame, artifact)
    changed = frame.copy()
    for column in range(5):
        changed[f'native_flux_{column}'] = np.nan
    predicted = apply_calibration(changed, artifact)
    for feature in FEATURES:
        np.testing.assert_allclose(predicted[feature], original[feature], equal_nan=True)
        assert min(np.asarray(artifact['reference_grid_MeV'])[artifact['models'][feature]['required_channels']]) >= artifact['models'][feature]['minimum_response_energy_MeV']


def test_bank_requires_parent_and_never_borrows_other_satellite_model(tmp_path):
    frame, target, train, quiet, event = fixture()
    parent = fit_calibration(frame, target, train, quiet | event,
                             validation_regimes={'quiet': quiet, 'event': event})
    other = frame.copy()
    other['source_satellite'] = 'GOES-18'
    bridge = fit_calibration(other, target, train, quiet | event,
                             validation_regimes={'quiet': quiet, 'event': event},
                             reference_kind='calibrated_reference_bridge', parent_artifact=parent)
    path = tmp_path/'bank.json'
    path.write_text(json.dumps({'schema':BANK_SCHEMA,'artifacts':[bridge]}))
    with pytest.raises(CalibrationError, match='required validated reference parent'):
        load_calibration_bank(path)
    path.write_text(json.dumps({'schema':BANK_SCHEMA,'artifacts':[parent,bridge]}))
    bank = load_calibration_bank(path)
    assert apply_calibration_bank(other,bank).P4.notna().any()
    unsupported = other.copy()
    unsupported['source_satellite']='GOES-19'
    result=apply_calibration_bank(unsupported,bank,strict_source=False)
    assert result.P4.isna().all()
    assert result.P4_calibration_status.eq('unsupported_source').all()
    with pytest.raises(CalibrationError,match='No validated model'):
        apply_calibration_bank(unsupported,bank)


def test_packaged_calibration_bank_has_passed_gates_and_intact_lineage():
    from pathlib import Path
    bank=load_calibration_bank(Path(__file__).resolve().parents[1]/'calibration/proton_calibration_bank.json')
    assert len(bank['artifacts']) >= 1
    for artifact in bank['artifacts']:
        assert artifact['status']=='passed'
        for model in artifact['models'].values():
            assert model['passed']
            for regime in ('quiet','event'):
                assert model['validation'][regime]['coverage'] >= artifact['validation_gates']['minimum_validation_coverage']
