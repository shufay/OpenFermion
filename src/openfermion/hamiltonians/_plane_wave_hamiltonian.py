#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Construct Hamiltonians in plan wave basis and its dual in 3D."""
from __future__ import absolute_import

import openfermion.utils._operator_utils

from openfermion.hamiltonians._jellium import *
from openfermion.hamiltonians._molecular_data import periodic_hash_table
from openfermion.ops import FermionOperator, QubitOperator

def center(grid, geometry, verbose=False):
    """
    Centers the molecule in the supercell.

    Args:
        grid (Grid): The discretization to use. 
        geometry (list[tuple]): A list of tuples giving the coordinates of each atom.
                                example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
                                Distances in atomic units. Use atomic symbols to specify atoms.
        verbose (bool): Whether to turn on print statements.  

    Returns:
        A list of tuples giving the centered coordinates of each atom.
    """
    #print('Centering molecule...\n')

    # Get list of coordinates.
    coordinates = numpy.array([nuclear_term[1] for nuclear_term in geometry])
    
    if verbose:
        print('\nAtom coordinates:')
        print(coordinates)
        print()
    

    # Compute the center coordinate of molecule.
    molecule_center = numpy.mean(coordinates, axis=0)

    #print('Molecule Center: {}\n'.format(molecule_center))

    # Compute center of supercell.
    grid_coordinates = [grid.position_vector(i) for i in grid.all_points_indices()]
    grid_center = numpy.mean(grid_coordinates, axis=0)

    #print('Grid Center: {}\n'.format(grid_center))

    # Compute displacement vector of molecule center from cell center.
    diff = grid_center - molecule_center

    # Center molecule center in supercell, i.e. we want molecule_center == cell_center.
    # Shift all atoms.
    molecule_center += diff
    centered_coordinates = [tuple(coordinate + diff) for coordinate in coordinates]
    
    if verbose:
        print('Centered atom coordinates:')
        print(centered_coordinates)
        print()

        print('New molecule Center: {}\n'.format(molecule_center))
    
    
    # Construct geometry with centered coordinates.
    centered_geometry = [(nuclear_term[0], centered_coordinates[i]) for i, nuclear_term in enumerate(geometry)]

    return centered_geometry


def dual_basis_external_potential(grid, geometry, spinless,
                                  non_periodic=False, period_cutoff=None, verbose=False):
    """Return the external potential in the dual basis of arXiv:1706.00023.

    The external potential resulting from electrons interacting with nuclei
        in the plane wave dual basis.  Note that a cos term is used which is
        strictly only equivalent under aliasing in odd grids, and amounts
        to the addition of an extra term to make the diagonals real on even
        grids.  This approximation is not expected to be significant and allows
        for use of even and odd grids on an even footing.

    Args:
        grid (Grid): The discretization to use.
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify atoms.
        spinless (bool): Whether to use the spinless model or not.
        non_periodic (bool): If the system is non-periodic, default to False.
        period_cutoff (float): Period cutoff, default to
            grid.volume_scale() ** (1. / grid.dimensions)
        verbose (bool): Whether to turn on print statements.  

    Returns:
        FermionOperator: The dual basis operator.
    """
    #print('Edited source. \n')
    prefactor = -4.0 * numpy.pi / grid.volume_scale()
    
    # [(True and None) is None] == True
    # [(False and None) is None] == False
    # [(True and 10) is None] == False
    # [(False and 10) is None] == False
    if non_periodic and period_cutoff is None:
        period_cutoff = grid.volume_scale() ** (1. / grid.dimensions)
        
    operator = FermionOperator()
    
    if spinless:
        spins = [None]
        
    else:
        spins = [0, 1]
    
    # Center molecule in supercell.
    centered_geometry = center(grid, geometry, verbose)
    
    #centered_geometry = geometry
        
    # Sum over p.
    for pos_indices in grid.all_points_indices():
        coordinate_p = grid.position_vector(pos_indices)
        
        # Sum over j.
        for nuclear_term in centered_geometry:
            coordinate_j = numpy.array(nuclear_term[1], float)

            # If non_periodic. 
            # See paper Appendix E.2. This is not setting function at boundary to 0.
            # Just accounting for Fourier Transform with truncated Coulomb operator.
            '''
            # First naive implementation.
            if non_periodic:
                coefficient *= 1.0 - numpy.cos(period_cutoff * numpy.sqrt(momenta_squared))s
            '''
            if non_periodic:
                #print('non_periodic\n')
                
                diff = coordinate_j - coordinate_p
                
                if numpy.sqrt(diff.dot(diff)) > period_cutoff:
                    if verbose:
                        print('External potential distances larger than cutoff.\n')
                        print('Cutoff: {}'.format(period_cutoff))
                        print('Coordinate j:{}'.format(coordinate_j))
                        print('Coordinate p:{}'.format(coordinate_p))
                    
                    # Continue on to next nuclear term.
                    continue
            
            # Sum over nu.
            for momenta_indices in grid.all_points_indices():
                momenta = grid.momentum_vector(momenta_indices)
                momenta_squared = momenta.dot(momenta)
                
                if momenta_squared == 0:
                    continue
                
                # Compute coefficient.
                cos_index = momenta.dot(coordinate_j - coordinate_p)
                coefficient = (prefactor / momenta_squared *
                               periodic_hash_table[nuclear_term[0]] *
                               numpy.cos(cos_index))
                '''
                print('cos index: {}\n'.format(cos_index))
                print('coefficient: {}\n'.format(coefficient))
                '''
                
                for spin_p in spins:
                    orbital_p = grid.orbital_id(pos_indices, spin_p)
                    operators = ((orbital_p, 1), (orbital_p, 0))
                    operator += FermionOperator(operators, coefficient)
    
    '''
    normal_ordered_operator = openfermion.normal_ordered(operator)
    print('Dual basis external potential operator:')
    print('qubits: {}\n'.format(openfermion.count_qubits(normal_ordered_operator)))
    #print(operator)
    print()
    '''
    
    if operator is None:
        operator = FermionOperator()
        
    return operator


def plane_wave_external_potential_v2(grid, geometry, spinless, e_cutoff=None,
                                  non_periodic=False, period_cutoff=None, verbose=False):
    """Return the external potential operator in plane wave basis.
       My implmentation, using equation E5 in the paper. 

    The external potential resulting from electrons interacting with nuclei.
        It is defined here as the Fourier transform of the dual basis
        Hamiltonian such that is spectrally equivalent in the case of
        both even and odd grids.  Otherwise, the two differ in the case of
        even grids.

    Args:
        grid (Grid): The discretization to use.
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify atoms.
        spinless: Bool, whether to use the spinless model or not.
        e_cutoff (float): Energy cutoff.
        non_periodic (bool): If the system is non-periodic, default to False.
        period_cutoff (float): Period cutoff, default to
            grid.volume_scale() ** (1. / grid.dimensions)
        verbose (bool): Whether to turn on print statements.    

    Returns:
        FermionOperator: The plane wave operator.
    """
    print('Self-implementation of plane wave external potential. \n')
    prefactor = -4.0 * numpy.pi / grid.volume_scale()
    operator = FermionOperator((), 0.0)
    spins = [None] if spinless else [0, 1]
    grid_origin = (0, ) * grid.dimensions
    coordinates_origin = grid.position_vector(grid_origin)
    
    # [(True and None) is None] == True
    # [(False and None) is None] == False
    # [(True and 10) is None] == False
    # [(False and 10) is None] == False
    if non_periodic and period_cutoff is None:
        period_cutoff = grid.volume_scale() ** (1. / grid.dimensions)
    
    # Pre-Computations.
    orbital_ids = {}
    shifted_omega_indices_dict = {}
    
    for indices in grid.all_points_indices():
        # Sort of centers cells at origin.  
        # These indices correspond to nu.
        # I don't know what shifted_omega_indices is actually.
        shifted_omega_indices = [j - grid.length[i] // 2
                                 for i, j in enumerate(indices)]
        
        if verbose:
            print('Original indices: {}\n'.format(indices))
            print('Shifted indices: {}\n'.format(shifted_omega_indices))
        
        # Index new coordinate with old coordinate.
        shifted_omega_indices_dict[indices] = shifted_omega_indices
        orbital_ids[indices] = {}
        
        for spin in spins:
            # Set correct tensor factor for spin at lattice point.
            orbital_ids[indices][spin] = grid.orbital_id(indices, spin)
    
    # Loop once through all plane waves.
    # Sum over t = q - p.
    for omega_indices in grid.all_points_indices():
        # What is this for?
        shifted_omega_indices = shifted_omega_indices_dict[omega_indices]
        
        # Get the momenta vectors.
        momenta = grid.momentum_vector(omega_indices)
        momenta_squared = momenta.dot(momenta)
        
        if verbose:
            print('t = q - p: {}'.format(shifted_omega_indices))
            print('k_[q-p]: {}'.format(momenta))
            print('k_[q-p]^2: {}\n'.format(momenta_squared))
        
        # Skip if momentum is zero.
        if momenta_squared == 0:
            continue

        # Energy cutoff.
        if e_cutoff is not None and momenta_squared / 2. > e_cutoff:
            continue
        
        # Sum over p.
        for omega_indices_shift in grid.all_points_indices():
            # Loop over spins and identify interacting orbitals.
            orbital_p = {}
            orbital_q = {}
            
            # This is basically omega_indices_shift.
            # So these indices correspond to p.
            shifted_index_1 = tuple(
                [(omega_indices_shift[i]) % grid.length[i]
                 for i in range(grid.dimensions)]) # I think this is the "dualling" in the paper? (uses mod)
            
            # These indices correspond to q = t + p.
            shifted_index_2 = tuple(
                [(omega_indices[i] + omega_indices_shift[i]) % grid.length[i]
                 for i in range(grid.dimensions)]) # I think this is the "dualling" in the paper? (uses mod)
            
            if verbose:
                print('omega_indices_shift: {}'.format(omega_indices_shift))
                print('shifted_index_1: {}'.format(shifted_index_1))
                print('shifted_index_2: {}\n'.format(shifted_index_2))
            
            # Sum over j.
            centered_geometry = center(grid, geometry, verbose)
            
            for nuclear_term in centered_geometry:
                coordinate_j = numpy.array(nuclear_term[1], float)
                
                # Compute coefficient.
                cos_index = momenta.dot(coordinate_j)
                coefficient = (prefactor / momenta_squared *
                               periodic_hash_table[nuclear_term[0]] *
                               numpy.cos(cos_index))
                
                if verbose:
                    print('coordinate j: {}\n'.format(coordinate_j))
                    print('coefficient: {}\n'.format(coefficient))

                # See paper Appendix E.2. This is not setting function at boundary to 0.
                # Just accounting for Fourier Transform with truncated Coulomb operator.
                if non_periodic:
                    correction = 1.0 - numpy.cos(
                            period_cutoff * numpy.sqrt(momenta_squared))
                    coefficient *= correction
                    
                    if verbose:
                        print('period cutoff: {}'.format(period_cutoff))
                        print('cos correction: {}'.format(numpy.cos(period_cutoff * numpy.sqrt(momenta_squared))))
                        print('correction: {}'.format(correction))
                        print('corrected coefficient: {}\n'.format(coefficient))
                               
                # Loop over spins.
                for spin in spins:
                    orbital_p[spin] = orbital_ids[shifted_index_1][spin]
                    orbital_q[spin] = orbital_ids[shifted_index_2][spin]
                    
                    # Add interaction term.
                    operators = ((orbital_p[spin], 1), (orbital_q[spin], 0))
                    
                    if verbose:
                        print('spin: {}\n'.format(spin))
                        print('operators:')
                        print(operators)
                        print()
                        
                    operator += FermionOperator(operators, coefficient)
    
    #print('Hamiltonian:')
    #print(operator)
    #print()
        
    return operator

            
'''
def plane_wave_external_potential(grid, geometry, spinless, e_cutoff=None,
                                  non_periodic=False, period_cutoff=None):
    """Return the external potential operator in plane wave basis.
       My implmentation, using equation E5 in the paper. 

    The external potential resulting from electrons interacting with nuclei.
        It is defined here as the Fourier transform of the dual basis
        Hamiltonian such that is spectrally equivalent in the case of
        both even and odd grids.  Otherwise, the two differ in the case of
        even grids.

    Args:
        grid (Grid): The discretization to use.
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify atoms.
        spinless: Bool, whether to use the spinless model or not.
        e_cutoff (float): Energy cutoff.
        non_periodic (bool): If the system is non-periodic, default to False.
        period_cutoff (float): Period cutoff, default to
            grid.volume_scale() ** (1. / grid.dimensions)

    Returns:
        FermionOperator: The plane wave operator.
    """
    print('Edited source. \n')
    prefactor = -4.0 * numpy.pi / grid.volume_scale()
    
    # [(True and None) is None] == True
    # [(False and None) is None] == False
    # [(True and 10) is None] == False
    # [(False and 10) is None] == False
    if non_periodic and period_cutoff is None:
        period_cutoff = grid.volume_scale() ** (1. / grid.dimensions)
        
    operator = FermionOperator((), 0.0)
    spins = [None] if spinless else [0, 1]
            
    # Loop once through all plane waves.    
    # Sum over p.
    for grid_indices_a in grid.all_points_indices():

        # Get the momenta vectors.
        momenta_p = grid.momentum_vector(grid_indices_a)
        momenta_p_squared = momenta_p.dot(momenta_p)
        
        print('momenta_p:')
        print(momenta_p)
        print()
        
        print('momenta_p^2:')
        print(momenta_p_squared)
        print()
        
        # Energy cutoff.
        if e_cutoff is not None and momenta_p_squared / 2. > e_cutoff:
            continue
        
        # Sum over q.
        for grid_indices_b in grid.all_points_indices():     
            if (grid_indices_a == grid_indices_b):
                continue
                
            # Compute coefficient.
            momenta_q = grid.momentum_vector(grid_indices_b)
            momentum_diff = momenta_q - momenta_p
                
            momenta_q_squared = momenta_q.dot(momenta_q)
            momentum_diff_squared = momentum_diff.dot(momentum_diff)
            
            print('momenta_q:')
            print(momenta_q)
            print()

            print('momenta_q^2:')
            print(momenta_q_squared)
            print()
            
            print('momenta_q - momenta_p:')
            print(momentum_diff)
            print()

            print('momentum_diff^2:')
            print(momentum_diff_squared)
            print()

            # Energy cutoff.
            if e_cutoff is not None and momenta_q_squared / 2. > e_cutoff:
                continue

            for nuclear_term in geometry:
                coordinate_j = numpy.array(nuclear_term[1], float)
                cos_index = momentum_diff.dot(coordinate_j)
                coefficient = (prefactor / momentum_diff_squared *
                               periodic_hash_table[nuclear_term[0]] *
                               numpy.cos(cos_index))
                
                print('initial coefficient: {}\n'.format(coefficient))

                if non_periodic:
                    print('non_periodic\n')
                    print('correction: {}\n'.format(1.0 - numpy.cos(period_cutoff * numpy.sqrt(momentum_diff_squared))))
                    coefficient *= 1.0 - numpy.cos(period_cutoff * numpy.sqrt(momentum_diff_squared))
                
                print('coefficient: {}\n'.format(coefficient))
                
                for spin in spins:
                    orbital_p = grid.orbital_id(grid_indices_a, spin)
                    orbital_q = grid.orbital_id(grid_indices_b, spin)

                    # Add interaction term.
                    operators = ((orbital_p, 1), (orbital_q, 0))
                    print('operators:\n')
                    print(operators)
                    operator += FermionOperator(operators, coefficient)
    
    print('operator\n')
    print(operator)
    print()
    return operator
'''


def plane_wave_external_potential(grid, geometry, spinless, e_cutoff=None,
                                  non_periodic=False, period_cutoff=None, verbose=False):
    """Return the external potential operator in plane wave basis.

    The external potential resulting from electrons interacting with nuclei.
        It is defined here as the Fourier transform of the dual basis
        Hamiltonian such that is spectrally equivalent in the case of
        both even and odd grids.  Otherwise, the two differ in the case of
        even grids.

    Args:
        grid (Grid): The discretization to use.
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify atoms.
        spinless: Bool, whether to use the spinless model or not.
        e_cutoff (float): Energy cutoff.
        non_periodic (bool): If the system is non-periodic, default to False.
        period_cutoff (float): Period cutoff, default to
            grid.volume_scale() ** (1. / grid.dimensions)
        verbose (bool): Whether to turn on print statements.  

    Returns:
        FermionOperator: The plane wave operator.
    """
    dual_basis_operator = dual_basis_external_potential(grid, geometry,
                                                        spinless, non_periodic,
                                                        period_cutoff, verbose)
    operator = (openfermion.utils.inverse_fourier_transform(dual_basis_operator,
                                                            grid, spinless))
    
    '''
    normal_ordered_operator = openfermion.normal_ordered(operator)
    print('Plane wave ext potential Hamiltonian:')
    print('qubits: {}\n'.format(openfermion.count_qubits(normal_ordered_operator)))
    #print(operator)
    print()
    '''
    
    return operator



def plane_wave_hamiltonian(grid, geometry=None,
                           spinless=False, plane_wave=True,
                           include_constant=False, e_cutoff=None,
                           non_periodic=False, period_cutoff=None, 
                           ft=False, verbose=False):
    """Returns Hamiltonian as FermionOperator class.

    Args:
        grid (Grid): The discretization to use.
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify atoms.
        spinless (bool): Whether to use the spinless model or not.
        plane_wave (bool): Whether to return in plane wave basis (True)
            or plane wave dual basis (False).
        include_constant (bool): Whether to include the Madelung constant.
        e_cutoff (float): Energy cutoff.
        non_periodic (bool): If the system is non-periodic, default to False.
        period_cutoff (float): Period cutoff, default to
            grid.volume_scale() ** (1. / grid.dimensions)
        ft (bool): Whether to use the Fourier Transform of the dual basis
                   potentials as the plane wave potentials.
        verbose (bool): Whether to turn on print statements.

    Returns:
        FermionOperator: The hamiltonian.
    """
    if (geometry is not None) and (include_constant is True):
        raise ValueError('Constant term unsupported for non-uniform systems')

    jellium_op = jellium_model(grid, spinless, plane_wave, include_constant,
                               e_cutoff, non_periodic, period_cutoff, ft, verbose)

    if geometry is None:
        return jellium_op

    for item in geometry:
        if len(item[1]) != grid.dimensions:
            raise ValueError("Invalid geometry coordinate.")
        if item[0] not in periodic_hash_table:
            raise ValueError("Invalid nuclear element.")

    if plane_wave:
        external_potential = plane_wave_external_potential(
            grid, geometry, spinless, e_cutoff, non_periodic, period_cutoff)
    else:
        external_potential = dual_basis_external_potential(
            grid, geometry, spinless, non_periodic, period_cutoff, verbose)

    return jellium_op + external_potential



def jordan_wigner_dual_basis_hamiltonian(grid, geometry=None, spinless=False,
                                         include_constant=False):
    """Return the dual basis Hamiltonian as QubitOperator.

    Args:
        grid (Grid): The discretization to use.
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify atoms.
        spinless (bool): Whether to use the spinless model or not.
        include_constant (bool): Whether to include the Madelung constant.

    Returns:
        hamiltonian (QubitOperator)
    """
    if (geometry is not None) and (include_constant is True):
        raise ValueError('Constant term unsupported for non-uniform systems')

    jellium_op = jordan_wigner_dual_basis_jellium(
        grid, spinless, include_constant)

    if geometry is None:
        return jellium_op

    for item in geometry:
        if len(item[1]) != grid.dimensions:
            raise ValueError("Invalid geometry coordinate.")
        if item[0] not in periodic_hash_table:
            raise ValueError("Invalid nuclear element.")
    
    # Center molecule in supercell.
    centered_geometry = center(grid, geometry)

    n_orbitals = grid.num_points
    volume = grid.volume_scale()
    if spinless:
        n_qubits = n_orbitals
    else:
        n_qubits = 2 * n_orbitals
    prefactor = -2 * numpy.pi / volume
    external_potential = QubitOperator()

    for k_indices in grid.all_points_indices():
        momenta = grid.momentum_vector(k_indices)
        momenta_squared = momenta.dot(momenta)
        if momenta_squared == 0:
            continue

        for p in range(n_qubits):
            index_p = grid.grid_indices(p, spinless)
            coordinate_p = grid.position_vector(index_p)

            for nuclear_term in centered_geometry:
                coordinate_j = numpy.array(nuclear_term[1], float)

                cos_index = momenta.dot(coordinate_j - coordinate_p)
                coefficient = (prefactor / momenta_squared *
                               periodic_hash_table[nuclear_term[0]] *
                               numpy.cos(cos_index))
                external_potential += (QubitOperator((), coefficient) -
                                       QubitOperator(((p, 'Z'),), coefficient))

    return jellium_op + external_potential
