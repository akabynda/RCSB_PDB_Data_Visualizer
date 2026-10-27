"""Independent examples for residue mapping and missing-result semantics."""
import json

import pytest

from src.dataset.pdb_normalization import PDBPolymerMetadata
from src.dataset.tmdet_annotations import (
    TMDET_COPY_COLUMNS, collect_modeled_residues, parse_tmdet_annotations, summarize_entity_copies,
)


def chain(asym='A', auth='A', entity='1'):
    return dict(entry_id='1ABC', entity_id=entity, asym_id=asym, auth_asym_id=auth, is_membrane=True)


def atom(number, insertion='', *, name='ALA', kind='ATOM', atom_name='CA', chain_id='A'):
    return (f'{kind:<6}{1:5d} {atom_name:^4} {name:>3} {chain_id}{number:4d}{insertion:1}   '
            f'{1.:8.3f}{2.:8.3f}{3.:8.3f}{1.:6.2f}{0.:6.2f}          {atom_name[0]:>2}\n')


def xml(regions='', *, negative=False, protein_type='Tm_Alpha', chains=None):
    chains = {'A': regions} if chains is None else chains
    body = ''.join(f'<chain authId="{key}"><regions>{value}</regions></chain>' for key, value in chains.items())
    return (f'<pdbtm xmlns="https://pdbtm.unitmp.org" transmembrane="{"no" if negative else "yes"}">'
            f'<rawData><tmdetVersion>4.1.4</tmdetVersion><qValue>12.5</qValue><tmType>{protein_type}</tmType></rawData>'
            f'<chains>{body}</chains></pdbtm>')


def region(start, end, code, extra=''):
    return f'<region startAuthId="{start}" endAuthId="{end}" type="{code}" {extra}/>'


def modeled(text):
    return collect_modeled_residues(text, [chain()], PDBPolymerMetadata(), {})


def test_author_number_gaps_insertion_codes_and_nonmonotonic_order():
    residues = modeled(atom(10) + atom(10, 'A') + atom(40) + atom(-2))
    document = xml(region(10, 10, 'H', 'endAuthIcode="65"') + region(40, -2, 'L'))
    summary, chains, regions, rows = parse_tmdet_annotations(document, residues, [chain()])
    assert summary['tmdet_modeled_residue_count'] == 4
    # Re-entrant loops (L) are membrane regions, like TM helices (H).
    assert summary['tmdet_membrane_residue_count'] == 4
    assert summary['tmdet_membrane_fraction'] == 1
    assert [r['modeled_residue_count'] for r in regions] == [2, 2]
    assert [r['region_type'] for r in rows] == ['H', 'H', 'L', 'L']
    assert chains['A']['tmdet_membrane_residue_count'] == 4
    assert rows[1]['insertion_code'] == 'A'
    assert residues[0]['region_type'] is None  # no mutation


def test_all_region_codes_are_preserved_including_unknown_future_codes():
    codes = ['1', '2', '3', 'H', 'B', 'L', 'F', 'N', 'M', 'X']
    residues = modeled(''.join(atom(i) for i in range(1, 11)))
    summary, _, regions, rows = parse_tmdet_annotations(xml(''.join(region(i, i, c) for i, c in enumerate(codes, 1))), residues, [chain()])
    assert [r['region_type'] for r in rows] == codes
    assert len(regions) == 10
    assert json.loads(summary['tmdet_region_counts_json']) == dict.fromkeys(codes, 1)
    # Membrane codes of TmDet 4 Region::isAnnotatedMembraneType: H, B, L, F, N.
    assert summary['tmdet_membrane_residue_count'] == 5
    assert summary['tmdet_other_residue_count'] == 5


@pytest.mark.parametrize('protein_type,status,membrane_count', [('Soluble', 'no_membrane_detected', 0), ('Noprotein', 'no_analyzable_protein', None)])
def test_negative_and_no_analyzable_protein_are_distinct(protein_type, status, membrane_count):
    summary, _, regions, rows = parse_tmdet_annotations(xml(negative=True, protein_type=protein_type), modeled(atom(1)), [chain()])
    assert summary['tmdet_status'] == status
    assert summary['tmdet_membrane_residue_count'] == membrane_count
    assert summary['tmdet_excluded_chain_count'] == 0
    assert rows[0]['region_type'] is None
    assert rows[0]['assignment_status'] == status
    assert not regions


def test_partial_annotation_does_not_count_missing_as_outside():
    summary, chains, _, rows = parse_tmdet_annotations(xml(region(1, 1, 'H')), modeled(atom(1) + atom(2)), [chain()])
    assert summary['tmdet_status'] == 'partial_annotation'
    assert summary['tmdet_unassigned_residue_count'] == 1
    # The only chain is excluded, so nothing remains to count.
    assert summary['tmdet_excluded_chain_count'] == 1
    assert summary['tmdet_excluded_residue_count'] == 2
    assert summary['tmdet_counted_residue_count'] is None
    assert summary['tmdet_membrane_residue_count'] is None
    assert summary['tmdet_membrane_fraction'] is None
    assert summary['tmdet_other_residue_count'] is None
    assert chains['A']['tmdet_membrane_residue_count'] == 1
    assert chains['A']['tmdet_other_residue_count'] is None
    assert rows[1]['region_type'] is None


def two_chains(text):
    chains = [chain(), chain('B', 'B', '2')]
    return collect_modeled_residues(text, chains, PDBPolymerMetadata(), {}), chains


def test_unannotated_chain_is_excluded_from_entry_totals():
    # 2KS9-like: chain A has 4 annotated residues (2 H, 2 side), chain B's 3 have none.
    residues, chains = two_chains(''.join(atom(i) for i in range(1, 5))
                                  + ''.join(atom(i, chain_id='B') for i in range(1, 4)))
    summary, by_chain, _, _ = parse_tmdet_annotations(
        xml(chains={'A': region(1, 2, 'H') + region(3, 4, '1')}), residues, chains)
    assert summary['tmdet_status'] == 'partial_annotation'
    assert summary['tmdet_modeled_residue_count'] == 7
    assert summary['tmdet_unassigned_residue_count'] == 3
    assert summary['tmdet_excluded_chain_count'] == 1
    assert summary['tmdet_excluded_residue_count'] == 3
    assert summary['tmdet_counted_residue_count'] == 4
    assert summary['tmdet_membrane_residue_count'] == 2
    assert summary['tmdet_other_residue_count'] == 2
    assert summary['tmdet_membrane_fraction'] == 2 / 4  # not 2 / 7
    assert by_chain['A']['tmdet_status'] == 'membrane_detected'
    assert by_chain['B']['tmdet_status'] == 'partial_annotation'
    copies = summarize_entity_copies(summary['tmdet_status'], [
        {**chain(), **by_chain['A']}, {**chain('B', 'B', '2'), **by_chain['B']}])
    assert copies == dict(zip(TMDET_COPY_COLUMNS, (4, 2, 2, 2, 2)))


def test_partly_assigned_chain_is_excluded_whole():
    # Chain A: H at 1-2 but residue 3 unassigned; chain B: interfacial helix F and side 2.
    residues, chains = two_chains(''.join(atom(i) for i in range(1, 4))
                                  + atom(1, chain_id='B') + atom(2, chain_id='B'))
    summary, by_chain, _, _ = parse_tmdet_annotations(
        xml(chains={'A': region(1, 2, 'H'), 'B': region(1, 1, 'F') + region(2, 2, '2')}), residues, chains)
    assert by_chain['A']['tmdet_status'] == 'partial_annotation'
    assert summary['tmdet_excluded_residue_count'] == 3
    assert summary['tmdet_counted_residue_count'] == 2
    # A's two assigned H residues are not counted; B's F residue is.
    assert summary['tmdet_membrane_residue_count'] == 1
    assert summary['tmdet_membrane_fraction'] == .5


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
    assert chains['B']['tmdet_membrane_residue_count'] is None
    with pytest.raises(ValueError):
        parse_tmdet_annotations('<html/>', [], [])


def counted(entity, modeled, membrane, status='membrane_detected'):
    return dict(entity_id=entity, tmdet_status=status, tmdet_modeled_residue_count=modeled,
                tmdet_membrane_residue_count=membrane, tmdet_other_residue_count=modeled - membrane)


def unannotated(entity, modeled):
    return dict(entity_id=entity, tmdet_status='partial_annotation', tmdet_modeled_residue_count=modeled,
                tmdet_membrane_residue_count=0, tmdet_other_residue_count=None)


def test_copy_means_for_homopentamer_with_unequal_copies():
    # 2M6I-like copies: (96 + 95 + 93 + 99 + 99) / 5 = 482 / 5 = 96.4 membrane residues per copy.
    chains = [counted('1', 150, tm) for tm in (96, 95, 93, 99, 99)]
    summary = summarize_entity_copies('membrane_detected', chains)
    assert summary == dict(zip(TMDET_COPY_COLUMNS, (150, 96.4, 53.6, 93, 99)))
    assert summary['tmdet_copy_mean_membrane_residue_count'] + summary['tmdet_copy_mean_other_residue_count'] == 150


def test_distinct_entities_are_summed_and_repeated_entity_averaged():
    # Two 222-residue copies (70, 65 membrane) and one 25-residue partner: 222 + 25, 67.5 + 0.
    chains = [counted('1', 222, 70), counted('1', 222, 65), counted('2', 25, 0)]
    assert summarize_entity_copies('membrane_detected', chains) == dict(
        zip(TMDET_COPY_COLUMNS, (247, 67.5, 179.5, 65, 70)))


def test_single_copies_and_valid_negatives_equal_entry_totals():
    chains = [counted('1', 40, 18), counted('2', 30, 0)]
    assert summarize_entity_copies('membrane_detected', chains) == dict(
        zip(TMDET_COPY_COLUMNS, (70, 18, 52, 18, 18)))
    negative = [counted('1', 60, 0, 'no_membrane_detected')] * 2
    assert summarize_entity_copies('no_membrane_detected', negative) == dict(
        zip(TMDET_COPY_COLUMNS, (60, 0, 60, 0, 0)))


def test_unannotated_copies_drop_out_of_copy_means():
    # One copy of entity 1 is excluded; entity 3 has no annotated copy left.
    chains = [counted('1', 50, 20), unannotated('1', 50), counted('2', 30, 0), unannotated('3', 11)]
    assert summarize_entity_copies('partial_annotation', chains) == dict(
        zip(TMDET_COPY_COLUMNS, (80, 20, 60, 20, 20)))


@pytest.mark.parametrize('status,chains', [
    ('partial_annotation', [unannotated('1', 10)]),
    ('error', [counted('1', 10, 5)]),
    ('membrane_detected', []),
    ('membrane_detected', [counted('1', 10, 5), dict(entity_id='1', tmdet_status='not_in_model')]),
    ('membrane_detected', [counted('1', 10, 5), {**counted('1', 10, 5), 'tmdet_other_residue_count': None}]),
])
def test_incomplete_copies_leave_copy_counts_blank(status, chains):
    assert summarize_entity_copies(status, chains) == dict.fromkeys(TMDET_COPY_COLUMNS)
