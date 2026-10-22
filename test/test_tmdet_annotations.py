"""Independent examples for residue mapping and missing-result semantics."""
import json

import pytest

from src.dataset.pdb_normalization import PDBPolymerMetadata
from src.dataset.tmdet_annotations import collect_modeled_residues, parse_tmdet_annotations


def chain(asym='A', auth='A'):
    return dict(entry_id='1ABC', entity_id='1', asym_id=asym, auth_asym_id=auth, is_membrane=True)


def atom(number, insertion='', *, name='ALA', kind='ATOM', atom_name='CA', chain_id='A'):
    return (f'{kind:<6}{1:5d} {atom_name:^4} {name:>3} {chain_id}{number:4d}{insertion:1}   '
            f'{1.:8.3f}{2.:8.3f}{3.:8.3f}{1.:6.2f}{0.:6.2f}          {atom_name[0]:>2}\n')


def xml(regions='', *, negative=False, protein_type='Tm_Alpha'):
    return (f'<pdbtm xmlns="https://pdbtm.unitmp.org" transmembrane="{"no" if negative else "yes"}">'
            f'<rawData><tmdetVersion>4.1.4</tmdetVersion><qValue>12.5</qValue><tmType>{protein_type}</tmType></rawData>'
            f'<chains><chain authId="A"><regions>{regions}</regions></chain></chains></pdbtm>')


def region(start, end, code, extra=''):
    return f'<region startAuthId="{start}" endAuthId="{end}" type="{code}" {extra}/>'


def modeled(text):
    return collect_modeled_residues(text, [chain()], PDBPolymerMetadata(), {})


def test_author_number_gaps_insertion_codes_and_nonmonotonic_order():
    residues = modeled(atom(10) + atom(10, 'A') + atom(40) + atom(-2))
    document = xml(region(10, 10, 'H', 'endAuthIcode="65"') + region(40, -2, 'L'))
    summary, chains, regions, rows = parse_tmdet_annotations(document, residues, [chain()])
    assert summary['tmdet_modeled_residue_count'] == 4
    assert summary['tmdet_tm_residue_count'] == 2
    assert summary['tmdet_tm_fraction'] == .5
    assert [r['modeled_residue_count'] for r in regions] == [2, 2]
    assert [r['region_type'] for r in rows] == ['H', 'H', 'L', 'L']
    assert chains['A']['tmdet_tm_residue_count'] == 2
    assert rows[1]['insertion_code'] == 'A'
    assert residues[0]['region_type'] is None  # no mutation


def test_all_region_codes_are_preserved_including_unknown_future_codes():
    codes = ['1', '2', '3', 'H', 'B', 'L', 'F', 'N', 'M', 'X']
    residues = modeled(''.join(atom(i) for i in range(1, 11)))
    summary, _, regions, rows = parse_tmdet_annotations(xml(''.join(region(i, i, c) for i, c in enumerate(codes, 1))), residues, [chain()])
    assert [r['region_type'] for r in rows] == codes
    assert len(regions) == 10
    assert json.loads(summary['tmdet_region_counts_json']) == dict.fromkeys(codes, 1)
    assert summary['tmdet_tm_residue_count'] == 2


@pytest.mark.parametrize('protein_type,status,tm_count', [('Soluble', 'no_membrane_detected', 0), ('Noprotein', 'no_analyzable_protein', None)])
def test_negative_and_no_analyzable_protein_are_distinct(protein_type, status, tm_count):
    summary, _, regions, rows = parse_tmdet_annotations(xml(negative=True, protein_type=protein_type), modeled(atom(1)), [chain()])
    assert summary['tmdet_status'] == status
    assert summary['tmdet_tm_residue_count'] == tm_count
    assert rows[0]['region_type'] is None
    assert rows[0]['assignment_status'] == status
    assert not regions


def test_partial_annotation_does_not_count_missing_as_outside():
    summary, _, _, rows = parse_tmdet_annotations(xml(region(1, 1, 'H')), modeled(atom(1) + atom(2)), [chain()])
    assert summary['tmdet_status'] == 'partial_annotation'
    assert summary['tmdet_unassigned_residue_count'] == 1
    assert summary['tmdet_tm_fraction'] is None
    assert summary['tmdet_other_residue_count'] is None
    assert rows[1]['region_type'] is None


@pytest.mark.parametrize('regions', [region(1, 99, 'H'), region(2, 1, 'H'), region(1, 2, 'H') + region(2, 2, 'L')])
def test_invalid_region_mapping_fails_instead_of_guessing(regions):
    with pytest.raises(ValueError):
        parse_tmdet_annotations(xml(regions), modeled(atom(1) + atom(2)), [chain()])


def test_polymer_hetatm_not_ligand_and_residues_without_ca_are_counted():
    text = atom(1) + atom(1, atom_name='N') + atom(2, atom_name='N') + atom(3, name='MSE', kind='HETATM') + atom(4, name='HOH', kind='HETATM')
    metadata = PDBPolymerMetadata(seqres_by_chain={'A': {'ALA', 'MSE'}})
    rows = collect_modeled_residues(text, [chain('label', 'original')], metadata, {'original': 'A'})
    assert [r['auth_seq_id'] for r in rows] == [1, 2, 3]
    assert [r['has_ca'] for r in rows] == [True, False, True]
    assert rows[0]['auth_asym_id'] == 'original'


def test_missing_chain_and_invalid_xml():
    summary, chains, _, _ = parse_tmdet_annotations(xml(region(1, 1, 'H')), modeled(atom(1)), [chain(), chain('B', 'B')])
    assert summary['tmdet_status'] == 'membrane_detected'
    assert chains['B']['tmdet_status'] == 'not_in_model'
    assert chains['B']['tmdet_tm_residue_count'] is None
    with pytest.raises(ValueError):
        parse_tmdet_annotations('<html/>', [], [])
