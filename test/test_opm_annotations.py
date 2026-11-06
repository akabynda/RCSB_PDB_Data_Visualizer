"""Independent examples for OPM boundaries, residue centers and copy aggregation."""
import pytest

from src.dataset.opm_annotations import (
    OPM_COPY_COLUMNS, collect_modeled_residues, count_membrane_residues,
    parse_opm_coordinates, summarize_entity_copies,
)
from src.dataset.pdb_normalization import PDBPolymerMetadata


def chain(asym='A', auth='A', entity='1'):
    return dict(entry_id='1ABC', entity_id=entity, asym_id=asym, auth_asym_id=auth, is_membrane=True)


def atom(number, z=0., insertion='', *, name='ALA', kind='ATOM', atom_name='CA', chain_id='A', element=None):
    element = atom_name[0] if element is None else element
    return (f'{kind:<6}{1:5d} {atom_name:^4} {name:>3} {chain_id}{number:4d}{insertion:1}   '
            f'{1.:8.3f}{2.:8.3f}{z:8.3f}{1.:6.2f}{0.:6.2f}          {element:>2}\n')


def planes(*values):
    return ''.join(f'HETATM{9000:5d}  O   DUM  {i:4d}    {0.:8.3f}{0.:8.3f}{z:8.3f}\n' for i, z in enumerate(values))


def deposited(*keys, chain_id='A', asym='A', name='ALA'):
    return [dict(entry_id='1ABC', entity_id='1', asym_id=asym, auth_asym_id=chain_id, is_membrane=True,
                 pdb_chain_id=chain_id, auth_seq_id=number, insertion_code=insertion, residue_name=name)
            for number, insertion in keys]


def test_residue_center_is_the_unweighted_heavy_atom_mean_of_z():
    text = (planes(-15.95, 15.95) + atom(1, 10., atom_name='N') + atom(1, 20., atom_name='CA')
            + atom(1, 99., atom_name='HA', element='H') + atom(1, 99., atom_name='1HB', element=''))
    half, residues = parse_opm_coordinates('REMARK      1/2 of bilayer thickness:   16.0\n' + text)
    # The exact DUM plane is used, not the rounded REMARK value; hydrogens are ignored.
    assert half == 15.95
    assert residues == {('A', 1, ''): {'name': 'ALA', 'mean_z': 15.}}


def test_one_sided_planes_and_first_model_only():
    text = ('MODEL        1\n' + atom(1, -2.) + planes(15.4) + 'ENDMDL\n'
            'MODEL        2\n' + atom(1, 50.) + atom(2, 50.) + 'ENDMDL\n')
    half, residues = parse_opm_coordinates(text)
    assert half == 15.4
    assert residues == {('A', 1, ''): {'name': 'ALA', 'mean_z': -2.}}


def test_first_alternate_position_represents_an_atom():
    first = atom(1, 4., atom_name='CA')
    second = atom(1, 40., atom_name='CA')
    assert parse_opm_coordinates(planes(15.) + first + second)[1][('A', 1, '')]['mean_z'] == 4.


@pytest.mark.parametrize('text', [
    atom(1),  # no boundary marks
    planes(-15., 16.) + atom(1),  # bilayer not centered on z = 0
    planes(-15., 0., 15.) + atom(1),  # more than two planes
    planes(0.) + atom(1),  # no thickness
    planes(-15., 15.),  # no protein
])
def test_invalid_opm_geometry_fails_instead_of_guessing(text):
    with pytest.raises(ValueError):
        parse_opm_coordinates(text)


def test_boundary_is_inclusive_and_symmetric():
    positions = {('A', number, ''): {'name': 'ALA', 'mean_z': z}
                 for number, z in ((1, -15.), (2, 15.), (3, 15.001), (4, -15.001), (5, 0.))}
    summary, chains, rows = count_membrane_residues(deposited(*((n, '') for n in range(1, 6))), positions, 15., [chain()])
    assert [row['in_membrane_region'] for row in rows] == [True, True, False, False, True]
    assert summary['opm_status'] == 'counted'
    assert summary['opm_membrane_residue_count'] == 3
    assert summary['opm_other_residue_count'] == 2
    assert summary['opm_membrane_fraction'] == .6
    assert chains['A']['opm_membrane_residue_count'] == 3


def test_ligands_and_absent_residues_are_separate_from_counted_residues():
    positions = {('A', 1, ''): {'name': 'ALA', 'mean_z': 1.}, ('A', 1, 'A'): {'name': 'GLY', 'mean_z': 20.},
                 ('A', 500, ''): {'name': 'ZN', 'mean_z': 1.}}
    summary, _, rows = count_membrane_residues(deposited((1, ''), (1, 'A'), (2, '')), positions, 15., [chain()])
    assert summary['opm_modeled_residue_count'] == 2
    assert summary['opm_membrane_residue_count'] == 1
    assert summary['opm_absent_residue_count'] == 1
    assert summary['opm_unmatched_residue_count'] == 1
    assert summary['opm_residue_name_conflict_count'] == 1
    assert rows[2]['in_opm_file'] is False
    assert rows[2]['in_membrane_region'] is None
    assert rows[2]['opm_mean_z_angstrom'] is None


def test_majority_of_name_conflicts_excludes_the_entry_without_counts():
    # Two of three matched names differ: the OPM file holds another sequence.
    positions = {('A', 1, ''): {'name': 'GLY', 'mean_z': 1.}, ('A', 2, ''): {'name': 'SER', 'mean_z': 1.},
                 ('A', 3, ''): {'name': 'ALA', 'mean_z': 1.}}
    summary, chains, rows = count_membrane_residues(deposited((1, ''), (2, ''), (3, '')), positions, 15., [chain()])
    assert summary == {
        'opm_status': 'structure_mismatch',
        'opm_error': '2 of 3 matched OPM residue names differ from the deposited residues',
        'opm_residue_name_conflict_count': 2,
    }
    assert chains == {} and rows == []


def test_blank_opm_chain_maps_only_onto_a_single_deposited_chain():
    positions = {('', 1, ''): {'name': 'ALA', 'mean_z': 1.}}
    summary, _, _ = count_membrane_residues(deposited((1, '')), positions, 15., [chain()])
    assert summary['opm_membrane_residue_count'] == 1
    two_chains = deposited((1, '')) + deposited((1, ''), chain_id='B', asym='B')
    with pytest.raises(ValueError):
        count_membrane_residues(two_chains, positions, 15., [chain(), chain('B', 'B')])


def test_polymer_hetatm_counts_but_ligand_does_not():
    text = atom(1) + atom(1, atom_name='N') + atom(2, atom_name='N') + atom(3, name='MSE', kind='HETATM') + atom(4, name='HOH', kind='HETATM')
    metadata = PDBPolymerMetadata(seqres_by_chain={'A': {'ALA', 'MSE'}})
    rows = collect_modeled_residues(text, [chain('label', 'original')], metadata, {'original': 'A'})
    assert [row['auth_seq_id'] for row in rows] == [1, 2, 3]
    assert rows[0]['auth_asym_id'] == 'original'


def counted(entity, modeled, membrane):
    return dict(entity_id=entity, opm_modeled_residue_count=modeled, opm_membrane_residue_count=membrane,
                opm_other_residue_count=modeled - membrane)


def test_copy_means_for_homopentamer_with_unequal_copies():
    # (96 + 95 + 93 + 99 + 99) / 5 = 482 / 5 = 96.4 membrane residues per copy.
    summary = summarize_entity_copies([counted('1', 150, tm) for tm in (96, 95, 93, 99, 99)])
    assert summary == dict(zip(OPM_COPY_COLUMNS, (150, 96.4, 53.6, 93, 99)))
    assert summary['opm_copy_mean_membrane_residue_count'] + summary['opm_copy_mean_other_residue_count'] == 150


def test_distinct_entities_are_summed_and_repeated_entity_averaged():
    # Two 222-residue copies (70, 65 membrane) and one 25-residue partner: 222 + 25, 67.5 + 0.
    chains = [counted('1', 222, 70), counted('1', 222, 65), counted('2', 25, 0)]
    assert summarize_entity_copies(chains) == dict(zip(OPM_COPY_COLUMNS, (247, 67.5, 179.5, 65, 70)))


def test_chains_absent_from_opm_drop_out_of_copy_means():
    chains = [counted('1', 50, 20), counted('1', 0, 0), counted('2', 30, 0), counted('3', 0, 0)]
    assert summarize_entity_copies(chains) == dict(zip(OPM_COPY_COLUMNS, (80, 20, 60, 20, 20)))


@pytest.mark.parametrize('chains', [[], [counted('1', 0, 0)], [dict(entity_id='1')]])
def test_entries_without_positioned_chains_have_blank_copy_counts(chains):
    assert summarize_entity_copies(chains) == dict.fromkeys(OPM_COPY_COLUMNS)
