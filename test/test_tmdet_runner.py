"""Offline native-run/cache checks with an executable fixture."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.dataset import tmdet
from src.dataset.cache import _sha256_file
from src.dataset.tmdet_install import _patch_source, ensure_tmdet_executable
from test_tmdet_annotations import atom, chain, xml, region


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    pdb = tmp_path / '1ABC.pdb'
    pdb.write_text('MODEL        1\n' + atom(1) + atom(2) + 'ENDMDL\nMODEL        2\n' + atom(99) + 'ENDMDL\n')
    monkeypatch.setattr(tmdet, 'download_pdb_if_needed', lambda *args: pdb)
    monkeypatch.setattr(tmdet, 'load_cached_chain_id_map', lambda *args: {})
    ccd = {'ALA': {'sha256': 'ccd-v1'}}
    monkeypatch.setattr(tmdet, 'prepare_ccd', lambda *args: ccd)
    binary = tmp_path / 'fake_tmdet'
    document = xml(region(1, 2, 'H'))
    binary.write_text('#!/usr/bin/env python3\nfrom pathlib import Path\n'
                      f'Path("result.xml").write_text({document!r})\n'
                      'Path("transformed.cif").write_text("data_test\\n")\n')
    binary.chmod(0o755)
    client = SimpleNamespace(session=None, config=SimpleNamespace(timeout_seconds=1), solution_nmr_monomer_cache_dir=tmp_path)
    def run(timeout=2):
        return tmdet.run_tmdet_entry('1ABC', [chain()], client, binary, _sha256_file(binary), tmp_path / 'cache', timeout)
    return run, pdb, binary, ccd


def test_first_model_only_and_replay_validated_by_hash(setup_run, monkeypatch):
    run, pdb, binary, _ = setup_run
    summary, _, _, residues, source = run()
    assert summary['tmdet_status'] == 'membrane_detected'
    assert summary['tmdet_tm_residue_count'] == 2
    assert [r['auth_seq_id'] for r in residues] == [1, 2]
    directory = Path(summary['tmdet_result_directory'])
    assert (directory / 'input.pdb').read_text() == atom(1) + atom(2) + 'END\n'
    assert source['inputs']['model_index'] == 0
    assert '-nc' in source['inputs']['arguments']
    saved = tmdet.subprocess.run
    monkeypatch.setattr(tmdet.subprocess, 'run', Mock(side_effect=AssertionError('cache must avoid process launch')))
    assert run()[0] == summary
    monkeypatch.setattr(tmdet.subprocess, 'run', saved)
    (directory / 'result.xml').write_text('corrupt')
    assert run()[0]['tmdet_status'] == 'membrane_detected'
    assert list(directory.parent.glob('*-invalid-*'))


def test_input_binary_and_ccd_changes_invalidate_cache(setup_run):
    run, pdb, binary, ccd = setup_run
    first = run()[0]['tmdet_result_directory']
    pdb.write_text(pdb.read_text().replace('   1.000', '   4.000'))
    second = run()[0]['tmdet_result_directory']
    binary.write_text(binary.read_text() + '# modified executable\n')
    third = run()[0]['tmdet_result_directory']
    ccd['ALA']['sha256'] = 'ccd-v2'
    fourth = run()[0]['tmdet_result_directory']
    assert len({first, second, third, fourth}) == 4


@pytest.mark.parametrize('script', ['raise SystemExit(2)', 'import time; time.sleep(1)', 'from pathlib import Path; Path("result.xml").write_text("invalid")'])
def test_failed_process_timeout_or_xml_retains_logs_and_null_counts(setup_run, script):
    run, _, binary, _ = setup_run
    binary.write_text('#!/usr/bin/env python3\n' + script + '\n')
    summary, _, _, rows, source = run(timeout=.15)
    assert summary['tmdet_status'] == 'error'
    assert 'tmdet_tm_residue_count' not in summary
    assert all(r['assignment_status'] == 'error' for r in rows)
    directory = Path(summary['tmdet_result_directory'])
    assert (directory / 'failure.json').is_file()
    assert (directory / 'stderr.log').is_file()
    assert not (directory / 'manifest.json').exists()
    assert run(timeout=.15)[0]['tmdet_result_directory'] != str(directory)


def test_negative_process_is_successful_but_has_no_transformed_file(setup_run):
    run, _, binary, _ = setup_run
    binary.write_text('#!/usr/bin/env python3\nfrom pathlib import Path\n'
                      f'Path("result.xml").write_text({xml(negative=True, protein_type="Soluble")!r})\n')
    summary = run()[0]
    assert summary['tmdet_status'] == 'no_membrane_detected'
    assert summary['tmdet_tm_residue_count'] == 0
    assert not (Path(summary['tmdet_result_directory']) / 'transformed.cif').exists()


def test_ccd_content_hash_and_names_are_validated(tmp_path):
    session = Mock()
    def get(url, **kwargs):
        name = url.rsplit('/', 1)[1].split('.')[0]
        return Mock(text=f'data_{name}\n', content=f'data_{name}\n'.encode())
    session.get.side_effect = get
    source = tmdet.prepare_ccd(session, tmp_path, atom(1), 1)
    assert set(source) == {'ALA', 'ZZZ'}
    assert session.get.call_count == 2
    tmdet.prepare_ccd(session, tmp_path, atom(1), 1)
    assert session.get.call_count == 2
    Path(source['ALA']['path']).write_text('corrupt')
    tmdet.prepare_ccd(session, tmp_path, atom(1), 1)
    assert session.get.call_count == 3


def test_native_compatibility_patch_is_limited_to_three_undefined_operations(tmp_path):
    engine = tmp_path / 'src/Engine'
    engine.mkdir(parents=True)
    util = tmp_path / 'src/Utils/CifUtil.cpp'
    util.parent.mkdir()
    for name in ('SideDetector', 'Optimizer'):
        (engine / f'{name}.hpp').write_text(f'~{name}();')
    util.write_text('columns.erase(columns.end());')
    _patch_source(tmp_path)
    assert (engine / 'SideDetector.hpp').read_text() == 'virtual ~SideDetector();'
    assert (engine / 'Optimizer.hpp').read_text() == 'virtual ~Optimizer();'
    assert util.read_text() == 'columns.pop_back();'


def test_explicit_binary_requires_executable_file(tmp_path):
    with pytest.raises(RuntimeError, match='unavailable'):
        ensure_tmdet_executable(str(tmp_path / 'missing'), tmp_path)


def test_coordinate_file_model_count_is_checked(setup_run):
    run, pdb, _, _ = setup_run
    pdb.write_text(atom(1))
    summary = run()[0]
    assert summary['tmdet_status'] == 'error'
    assert 'more than one model' in summary['tmdet_error']
