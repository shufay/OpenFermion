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

"""Construct Hamiltonians in plane wave basis and its dual in 3D."""
from __future__ import absolute_import

import openfermion.utils._operator_utils

from openfermion.hamiltonians._jellium import *
from openfermion.hamiltonians._molecular_data import periodic_hash_table
from openfermion.ops import FermionOperator, QubitOperator

# Define constant reference point for 2D electrostatic interactions.
# This sets the zero reference length scale for the potential. 
R0 = 1e-8

def center(grid, geometry, verbose=False):
    """Centers the molecule in the supercell.
    
    *Currently, the centering can be done in 
        1. 1D, 2D, and 3D for diatomic molecules in the xy-plane;
        2. 3D for polyatomic molecules with any orientation. 
    
    TODO: Think about projection from 3D to 2D for molecules like H2O,
          or from 3D/2D to 1D for diatomic molecules in any plane.
    
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
    grid_coordinates = [grid.position_vector(i) 
                        for i in grid.all_points_indices()]
    grid_center = numpy.mean(grid_coordinates, axis=0)

    #print('Grid Center: {}\n'.format(grid_center))

    # Compute displacement vector of molecule center from cell center.
    diff = grid_center - molecule_center

    # Center molecule center in supercell, i.e. we want molecule_center == cell_center.
    # Shift all atoms.
    molecule_center += diff
    centered_coordinates = [tuple(coordinate + diff) 
                            for coordinate in coordinates]
    
    if verbose:
        print('Centered atom coordinates:')
        print(centered_coordinates)
        print()

        print('New molecule Center: {}\n'.format(molecule_center))
    
    
    # Construct geometry with centered coordinates.
    centered_geometry = [(nuclear_term[0], centered_coordinates[i]) 
                         for i, nuclear_term in enumerate(geometry)]

    return centered_geometry


def dual_basis_external_potential(grid, geometry, spinless,
                                  non_periodic=False, period_cutoff=None, 
                                  fieldlines=3, verbose=False):
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
        fieldlines (int): Spatial dimension for electric field lines. 
        verbose (bool): Whether to turn on print statements.  

    Returns:
        FermionOperator: The dual basis operator.
    """
    #print('Edited source. \n')
    
    # Initialize.
    prefactor = 0.
    
    # 3D case.
    if grid.dimensions == 3:
        prefactor = -4.0 * numpy.pi / grid.volume_scale()
    
    # 2D case.
    elif grid.dimensions == 2:
        prefactor = -1. / grid.volume_scale()
        
    
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

            
            # See paper Appendix E.2.
            '''
            if non_periodic:
                if verbose:
                    print('non_periodic\n')
                
                diff = coordinate_j - coordinate_p
                
                if numpy.sqrt(diff.dot(diff)) > period_cutoff:
                    if verbose:
                        print('External potential distances larger than cutoff.\n')
                        print('Cutoff: {}'.format(period_cutoff))
                        print('Coordinate j:{}'.format(coordinate_j))
                        print('Coordinate p:{}'.format(coordinate_p))
                    
                    # Continue on to next nuclear term.
                    continue
            '''
            
            # Sum over nu.
            for momenta_indices in grid.all_points_indices():
                momenta = grid.momentum_vector(momenta_indices)
                momenta_squared = momenta.dot(momenta)
                
                if momenta_squared == 0:
                    continue
                
                # Compute coefficient.
                coefficient = 0.
                cos_index = momenta.dot(coordinate_j - coordinate_p)
                
                # 3D case.
                if grid.dimensions == 3:
                    coefficient = (prefactor / momenta_squared *
                                   periodic_hash_table[nuclear_term[0]] *
                                   numpy.cos(cos_index))
                    
                    if non_periodic:
                        correction = (
                            1.0 - numpy.cos(
                            period_cutoff * numpy.sqrt(momenta_squared)))
                        coefficient *= correction

                        if verbose:
                            print('non_periodic')
                            print('cutoff: {}'.format(period_cutoff))
                            print('correction: {}'.format(correction))
                            print('coefficient: {}\n'.format(coefficient))

                            
                # 2D case.
                elif grid.dimensions == 2:
                    V_nu = 0.
                    
                    if fieldlines == 2:
                        if non_periodic:
                            Dkv = period_cutoff * numpy.sqrt(momenta_squared)
                            V_nu = (
                                4. * numpy.pi / momenta_squared * (
                                Dkv * numpy.log(R0 / period_cutoff) * 
                                scipy.special.jv(1, Dkv) - scipy.special.jv(0, Dkv)))
                            
                            if verbose:
                                print('non-periodic')
                                print('cutoff: {}\n'.format(period_cutoff))

                        else:
                            var1 = 4. / momenta_squared
                            var2 = 0.25 * momenta_squared

                            V_nu = numpy.complex128(
                                mpmath.meijerg([[1., 1.5, 2.], []], 
                                               [[1.5], []], var1) -
                                mpmath.meijerg([[-0.5, 0., 0.], []], 
                                               [[-0.5, 0.], [-1.]], var2))

                    elif fieldlines == 3:
                        if non_periodic:
                            var = -0.25 * period_cutoff**2 * momenta_squared
                            V_nu = numpy.complex128(
                                2 * numpy.pi * period_cutoff * 
                                mpmath.hyp1f2(0.5, 1., 1.5, var))
                            
                            if verbose:
                                print('non-periodic')
                                print('cutoff: {}\n'.format(period_cutoff))
                            
                        else:
                            V_nu = 2 * numpy.pi / numpy.sqrt(momenta_squared)

                    coefficient = (prefactor * V_nu * 
                                   periodic_hash_table[nuclear_term[0]] * 
                                   numpy.cos(cos_index))

                    if verbose:
                        print('fieldlines = {}'.format(fieldlines))
                        print('prefactor: {}'.format(prefactor))
                        print('V_nu: {}'.format(V_nu))
                        print('coefficient: {}\n'.format(coefficient))
                    
                
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




def plane_wave_external_potential(grid, geometry, spinless, e_cutoff=None,
                                  non_periodic=False, period_cutoff=None, 
                                  fieldlines=3, verbose=False):
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
        fieldlines (int): Spatial dimension for electric field lines. 
        verbose (bool): Whether to turn on print statements.  

    Returns:
        FermionOperator: The plane wave operator.
    """
    dual_basis_operator = dual_basis_external_potential(
        grid, geometry, spinless, non_periodic, 
        period_cutoff, fieldlines, verbose)
    
    operator = (openfermion.utils.inverse_fourier_transform(
        dual_basis_operator, grid, spinless))
    
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
                           ft=False, fieldlines=3, verbose=False):
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
        fieldlines (int): Spatial dimension for electric field lines. 
        verbose (bool): Whether to turn on print statements.

    Returns:
        FermionOperator: The hamiltonian.
    """
    if (geometry is not None) and (include_constant is True):
        raise ValueError('Constant term unsupported for non-uniform systems')

    jellium_op = jellium_model(grid, spinless, plane_wave, include_constant,
                               e_cutoff, non_periodic, period_cutoff, ft, 
                               fieldlines, verbose)

    if geometry is None:
        return jellium_op

    for item in geometry:
        if len(item[1]) != grid.dimensions:
            raise ValueError("Invalid geometry coordinate.")
        if item[0] not in periodic_hash_table:
            raise ValueError("Invalid nuclear element.")

    if plane_wave:
        external_potential = plane_wave_external_potential(
            grid, geometry, spinless, e_cutoff, non_periodic, 
            period_cutoff, fieldlines, verbose)
    else:
        external_potential = dual_basis_external_potential(
            grid, geometry, spinless, non_periodic, 
            period_cutoff, fieldlines, verbose)

    return jellium_op + external_potential



def jordan_wigner_dual_basis_hamiltonian(grid, geometry=None, spinless=False,
                                         include_constant=False,
                                         non_periodic=False, period_cutoff=None,
                                         fieldlines=3, verbose=False):
    """Return the dual basis Hamiltonian as QubitOperator.

    Args:
        grid (Grid): The discretization to use.
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify atoms.
        spinless (bool): Whether to use the spinless model or not.
        include_constant (bool): Whether to include the Madelung constant.
        non_periodic (bool): If the system is non-periodic, default to False.
        period_cutoff (float): Period cutoff, default to
            grid.volume_scale() ** (1. / grid.dimensions)
        fieldlines (int): Spatial dimension for electric field lines. 
        verbose (bool): Whether to turn on print statements.

    Returns:
        hamiltonian (QubitOperator)
    """
    if (geometry is not None) and (include_constant is True):
        raise ValueError('Constant term unsupported for non-uniform systems')

    jellium_op = jordan_wigner_dual_basis_jellium(
        grid, spinless, include_constant, non_periodic=non_periodic, 
        period_cutoff=period_cutoff, fieldlines=fieldlines, verbose=verbose)

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
        
    prefactor = 0.
    
    # 3D case.
    if grid.dimensions == 3:
        prefactor = -2 * numpy.pi / volume
        
    elif grid.dimensions == 2:
        prefactor = -1. / (2. * volume)
                
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
                coefficient = 0.
                
                # 3D case.
                if grid.dimensions == 3:
                    coefficient = (prefactor / momenta_squared *
                                   periodic_hash_table[nuclear_term[0]] *
                                   numpy.cos(cos_index))
                    
                    if non_periodic:
                        correction = (
                            1.0 - numpy.cos(
                            period_cutoff * numpy.sqrt(momenta_squared)))
                        coefficient *= correction

                        if verbose:
                            print('non_periodic')
                            print('cutoff: {}'.format(period_cutoff))
                            print('correction: {}\n'.format(correction))
                
                # 2D case.
                elif grid.dimensions == 2:
                    V_nu = 0.

                    if fieldlines == 2:
                        if non_periodic:
                            Dkv = period_cutoff * numpy.sqrt(momenta_squared)
                            V_nu = (
                                4. * numpy.pi / momenta_squared * (
                                Dkv * numpy.log(R0 / period_cutoff) * 
                                scipy.special.jv(1, Dkv) - scipy.special.jv(0, Dkv)))
                            
                            if verbose:
                                print('non-periodic')
                                print('cutoff: {}\n'.format(period_cutoff))
                        
                        else:
                            var1 = 4. / momenta_squared
                            var2 = 0.25 * momenta_squared

                            V_nu = numpy.complex128(
                                mpmath.meijerg([[1., 1.5, 2.], []], 
                                               [[1.5], []], var1) -
                                mpmath.meijerg([[-0.5, 0., 0.], []], 
                                               [[-0.5, 0.], [-1.]], var2))
                            
                    elif fieldlines == 3:
                        if non_periodic:
                            var = -0.25 * period_cutoff**2 * momenta_squared
                            V_nu = numpy.complex128(
                                2 * numpy.pi * period_cutoff * 
                                mpmath.hyp1f2(0.5, 1., 1.5, var))
                            
                            if verbose:
                                print('non-periodic')
                                print('cutoff: {}\n'.format(period_cutoff))
                        
                        else:    
                            V_nu = 2. * numpy.pi / numpy.sqrt(momenta_squared)
                    
                    coefficient = (prefactor * V_nu * 
                                   periodic_hash_table[nuclear_term[0]] * 
                                   numpy.cos(cos_index))
                            
                external_potential += (QubitOperator((), coefficient) -
                                       QubitOperator(((p, 'Z'),), coefficient))

    return jellium_op + external_potential
