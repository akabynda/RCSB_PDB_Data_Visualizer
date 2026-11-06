"""Offline OPM retrieval, cache and failure-visibility checks."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from src.dataset import opm
from src.dataset.builders.membrane import MembraneBuildResult
from test_opm_annotations import atom, chain, planes


def response(status=200, content=b'', payload=None):
    result = Mock(status_code=status, content=content)
    result.raise_for_status.side_effect = requests.HTTPError(str(status)) if status >= 400 else None
    result.json.return_value = payload
    return result


OPM_FILE = planes(-15., 15.) + atom(1, 1.) + atom(2, 30.)


@pytest.fixture
def client(tmp_path, monkeypatch):
    pdb = tmp_path / '1ABC.pdb'
    pdb.write_text('MODEL        1\n' + atom(1) + atom(2) + atom(3) + 'ENDMDL\nMODEL        2\n' + atom(99) + 'ENDMDL\n')
    monkeypatch.setattr(opm, 'download_pdb_if_needed', lambda *args: pdb)
    monkeypatch.setattr(opm, 'load_cached_chain_id_map', lambda *args: {})
    session = Mock()
    session.get.return_value = response(content=OPM_FILE.encode())
    return SimpleNamespace(session=session, config=SimpleNamespace(timeout_seconds=1), solution_nmr_monomer_cache_dir=tmp_path)


def test_counts_use_opm_positions_and_deposited_first_model(client, tmp_path):
    summary, chains, residues, source = opm.count_opm_entry('1ABC', [chain()], client, tmp_path / 'opm')
    assert summary['opm_status'] == 'counted'
    assert summary['opm_modeled_residue_count'] == 2
    assert summary['opm_membrane_residue_count'] == 1
    assert summary['opm_other_residue_count'] == 1
    assert summary['opm_absent_residue_count'] == 1
    assert summary['opm_half_thickness_angstrom'] == 15.
    assert chains['A']['opm_membrane_residue_count'] == 1
    assert [row['auth_seq_id'] for row in residues] == [1, 2, 3]
    assert source['url'] == 'https://opm-assets.storage.googleapis.com/pdb/1abc.pdb'
    assert Path(source['path']).read_text() == OPM_FILE


def test_cached_file_is_reused_after_hash_check_and_replaced_when_corrupt(client, tmp_path):
    cache = tmp_path / 'opm'
    first = opm.count_opm_entry('1ABC', [chain()], client, cache)[3]
    assert client.session.get.call_count == 1
    assert opm.count_opm_entry('1ABC', [chain()], client, cache)[3] == first
    assert client.session.get.call_count == 1
    Path(first['path']).write_text('corrupt')
    assert opm.count_opm_entry('1ABC', [chain()], client, cache)[0]['opm_status'] == 'counted'
    assert client.session.get.call_count == 2


def test_missing_file_is_visible_without_counts_and_not_a_failure(client, tmp_path):
    client.session.get.return_value = response(status=404)
    summary, chains, residues, source = opm.count_opm_entry('1ABC', [chain()], client, tmp_path / 'opm')
    assert summary == {'opm_status': 'coordinates_unavailable', 'opm_error': 'OPM coordinate file not found'}
    assert chains == {} and residues == []
    assert source['error'] == 'OPM coordinate file not found'


def test_other_sequence_in_opm_file_is_visible_without_counts_and_not_a_failure(client, tmp_path):
    other = planes(-15., 15.) + atom(1, 1., name='GLY') + atom(2, 1., name='SER')
    client.session.get.return_value = response(content=other.encode())
    summary, chains, residues, source = opm.count_opm_entry('1ABC', [chain()], client, tmp_path / 'opm')
    assert summary['opm_status'] == 'structure_mismatch'
    assert 'opm_membrane_residue_count' not in summary
    assert chains == {} and residues == []
    assert source['error'] == summary['opm_error'] == '2 of 2 matched OPM residue names differ from the deposited residues'


@pytest.mark.parametrize('result', [response(status=500), response(content=atom(1).encode())])
def test_retrieval_and_geometry_failures_are_errors_not_zero(client, tmp_path, result):
    client.session.get.return_value = result
    summary = opm.count_opm_entry('1ABC', [chain()], client, tmp_path / 'opm')[0]
    assert summary['opm_status'] == 'error'
    assert 'opm_membrane_residue_count' not in summary


def listing(*objects, total=None):
    return {'total_objects': len(objects) if total is None else total, 'objects': list(objects)}


def test_incomplete_listing_is_rejected():
    cache = Mock()
    cache.post_json.side_effect = [listing({'pdbid': '1abc'}, total=2), listing()]
    with pytest.raises(RuntimeError, match='incomplete'):
        opm.fetch_opm_metadata(cache)


def test_only_annotated_entries_are_counted_and_metadata_is_joined(client, tmp_path):
    record = {'pdbid': '1abc', 'type_id': 3, 'membrane_name_cache': 'Secreted', 'thickness': 2.1, 'tilt': 82, 'gibbs': -5.2}
    def get(url, **kwargs):
        if url.endswith('primary_structures'):
            return response(payload=listing(record))
        if url.endswith('types'):
            return response(payload=listing({'id': 3, 'name': 'Peptides'}))
        return response(content=OPM_FILE.encode())
    client.session.get.side_effect = get
    result = MembraneBuildResult(
        entry_rows=[dict(entry_id='1ABC', has_opm_annotation=True), dict(entry_id='2ABC', has_opm_annotation=False)],
        chain_rows=[chain()],
    )
    args = SimpleNamespace(opm_cache_dir=tmp_path / 'opm', solution_nmr_membrane_cache_validation_hours=24.)
    sources, api_sources = opm.annotate_membrane_cohort(result, args, client, SimpleNamespace(max_workers=1))
    annotated, other = result.entry_rows
    assert annotated['opm_status'] == 'counted'
    assert annotated['opm_type'] == 'Peptides'
    assert annotated['opm_membrane_name'] == 'Secreted'
    assert annotated['opm_reported_thickness_angstrom'] == 2.1
    assert annotated['opm_tilt_degrees'] == 82
    assert annotated['opm_transfer_energy_kcal_per_mol'] == -5.2
    assert annotated['opm_copy_mean_membrane_residue_count'] == 1
    assert other == dict(entry_id='2ABC', has_opm_annotation=False, opm_status='no_opm_annotation', opm_error='')
    assert [source['entry_id'] for source in sources] == ['1ABC']
    assert len(api_sources) == 2
    assert len(result.opm_residue_rows) == 3
