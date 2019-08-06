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

"""This module constructs Hamiltonians for the uniform electron gas."""
from __future__ import absolute_import

import numpy, mpmath, scipy

from openfermion.ops import FermionOperator, QubitOperator
from openfermion.utils._grid import Grid
import openfermion.utils._operator_utils


def wigner_seitz_length_scale(wigner_seitz_radius, n_particles, dimension):
    """Function to give length_scale associated with Wigner-Seitz radius.

    Args:
        wigner_seitz_radius (float): The radius per particle in atomic units.
        n_particles (int): The number of particles in the simulation cell.
        dimension (int): The dimension of the system.

    Returns:
        length_scale (float): The length scale for the simulation.

    Raises:
        ValueError: System dimension must be a positive integer.
    """
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError('System dimension must be a positive integer.')

    half_dimension = dimension // 2
    if dimension % 2:
        volume_per_particle = (2 * numpy.math.factorial(half_dimension) *
                               (4 * numpy.pi) ** half_dimension /
                               numpy.math.factorial(dimension) *
                               wigner_seitz_radius ** dimension)
    else:
        volume_per_particle = (numpy.pi ** half_dimension /
                               numpy.math.factorial(half_dimension) *
                               wigner_seitz_radius ** dimension)

    volume = volume_per_particle * n_particles
    length_scale = volume ** (1. / dimension)

    return length_scale


def plane_wave_kinetic(grid, spinless=False, e_cutoff=None):
    """Return the kinetic energy operator in the plane wave basis.

    Args:
        grid (openfermion.utils.Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.
        e_cutoff (float): Energy cutoff.

    Returns:
        FermionOperator: The kinetic momentum operator.
    """
    # Initialize.
    operator = FermionOperator()
    spins = [None] if spinless else [0, 1]

    # Loop once through all plane waves.
    for momenta_indices in grid.all_points_indices():
        momenta = grid.momentum_vector(momenta_indices)
        coefficient = momenta.dot(momenta) / 2.

        # Energy cutoff.
        if e_cutoff is not None and coefficient > e_cutoff:
            continue

        # Loop over spins.
        for spin in spins:
            orbital = grid.orbital_id(momenta_indices, spin)

            # Add interaction term.
            operators = ((orbital, 1), (orbital, 0))
            operator += FermionOperator(operators, coefficient)

    return operator


def plane_wave_potential_v2(grid, spinless=False, e_cutoff=None,
                            non_periodic=False, period_cutoff=None, verbose=False):
    """Return the e-e potential operator in the plane wave basis. 
       My version: take Fourier transform of dual basis potential.

    Args:
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.
        e_cutoff (float): Energy cutoff.
        non_periodic (bool): If the system is non-periodic, default to False.
        period_cutoff (float): Period cutoff, default to
            grid.volume_scale() ** (1. / grid.dimensions).
        verbose (bool): Whether to turn on print statements.  

    Returns:
        operator (FermionOperator)
    """
    dual_basis_operator = dual_basis_potential(grid, spinless, non_periodic, period_cutoff)
    operator = openfermion.utils.inverse_fourier_transform(dual_basis_operator, grid, spinless)
    
    normal_ordered_dual_basis_operator = openfermion.normal_ordered(dual_basis_operator)
    normal_ordered_operator = openfermion.normal_ordered(operator)
    
    if verbose:
        print('Fourier transform version.')
        print('Dual basis potential Hamiltonian:')
        print('qubits: {}'.format(openfermion.count_qubits(dual_basis_operator)))
        print('qubits (normal ordered): {}'.format(openfermion.count_qubits(normal_ordered_dual_basis_operator)))
        #print(normal_ordered_dual_basis_operator)
        print()

        print('Plane wave potential Hamiltonian:')
        print('qubits: {}'.format(openfermion.count_qubits(operator)))
        print('qubits (normal ordered): {}'.format(openfermion.count_qubits(normal_ordered_operator)))
        #print(openfermion.normal_ordered(operator))
        print()
        
    return operator


def plane_wave_potential(grid, spinless=False, e_cutoff=None,
                         non_periodic=False, period_cutoff=None,
                         fieldlines=3, verbose=False):
    """Return the e-e potential operator in the plane wave basis.

    Args:
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.
        e_cutoff (float): Energy cutoff.
        non_periodic (bool): If the system is non-periodic, default to False.
        period_cutoff (float): Period cutoff, default to
            grid.volume_scale() ** (1. / grid.dimensions).
        fieldlines (int): Spatial dimension for electric field lines. 
        verbose (bool): Whether to turn on print statements.  

    Returns:
        operator (FermionOperator)
    """
    # Initialize.
    prefactor = 0.
    
    # 3D case.
    if grid.dimensions == 3:
        prefactor = 2. * numpy.pi / grid.volume_scale()
    
    # 2D case.
    elif grid.dimensions == 2:
        prefactor = 1. / (2. * grid.volume_scale())
        
    operator = FermionOperator((), 0.0)
    spins = [None] if spinless else [0, 1]
    
    if non_periodic and period_cutoff is None:
        period_cutoff = grid.volume_scale() ** (1. / grid.dimensions)

    # Pre-Computations.
    shifted_omega_indices_dict = {}
    shifted_indices_minus_dict = {}
    shifted_indices_plus_dict = {}
    orbital_ids = {}
    for indices_a in grid.all_points_indices():
        shifted_omega_indices = [j - grid.length[i] // 2
                                 for i, j in enumerate(indices_a)]
        shifted_omega_indices_dict[indices_a] = shifted_omega_indices
        shifted_indices_minus_dict[indices_a] = {}
        shifted_indices_plus_dict[indices_a] = {}
        for indices_b in grid.all_points_indices():
            shifted_indices_minus_dict[indices_a][indices_b] = tuple([
                (indices_b[i] - shifted_omega_indices[i]) % grid.length[i]
                for i in range(grid.dimensions)])
            shifted_indices_plus_dict[indices_a][indices_b] = tuple([
                (indices_b[i] + shifted_omega_indices[i]) % grid.length[i]
                for i in range(grid.dimensions)])
        orbital_ids[indices_a] = {}
        for spin in spins:
            orbital_ids[indices_a][spin] = grid.orbital_id(indices_a, spin)

    # Loop once through all plane waves.
    for omega_indices in grid.all_points_indices():
        shifted_omega_indices = shifted_omega_indices_dict[omega_indices]

        # Get the momenta vectors.
        momenta = grid.momentum_vector(omega_indices)
        momenta_squared = momenta.dot(momenta)

        # Skip if momentum is zero.
        if momenta_squared == 0:
            continue

        # Energy cutoff.
        if e_cutoff is not None and momenta_squared / 2. > e_cutoff:
            continue

        # Compute coefficient.
        coefficient = 0.
        
        # 3D case.
        if grid.dimensions == 3:    
            coefficient = prefactor / momenta_squared
            
            # If non-periodic.
            if non_periodic:
                coefficient *= 1.0 - numpy.cos(
                        period_cutoff * numpy.sqrt(momenta_squared))
        
        # 2D case.
        elif grid.dimensions == 2:
            V_nu = 0.
            
            if fieldlines == 2:
                if non_periodic:
                        
                    # Set zero reference length scale for the potential. 
                    # For now, we take the reference length scale as the length of cell.
                    r0 = grid.volume_scale() ** (1. / grid.dimensions)
                    Dkv = period_cutoff * numpy.sqrt(momenta_squared)
                    V_nu = 4. * numpy.pi / momenta_squared * (
                        Dkv * numpy.log(period_cutoff / r0) * scipy.special.jv(1, Dkv) + scipy.special.jv(0, Dkv) - 1.)
                    
                    if verbose:
                        print('non-periodic')
                        print('cutoff: {}\n'.format(period_cutoff))
                    
                else:
                    var1 = 0.25 * momenta_squared
                    var2 = 4. / momenta_squared

                    V_nu = numpy.complex128(1. / 2. * 
                            (mpmath.meijerg([[-0.5, 0., 0.], []], [[-0.5, 0.], [-1.]], var1)) - 
                             mpmath.meijerg([[1., 1.5, 2.], []], [[1.5], []], var2))

            elif fieldlines == 3:
                if non_periodic:
                    var = -0.25 * period_cutoff**2 * momenta_squared
                    V_nu = numpy.complex128(2 * numpy.pi * period_cutoff * mpmath.hyp1f2(0.5, 1., 1.5, var))
                    
                    if verbose:
                        print('non-periodic')
                        print('cutoff: {}\n'.format(period_cutoff))
                    
                else:
                    V_nu = 2 * numpy.pi / numpy.sqrt(momenta_squared)
                
            coefficient = prefactor * V_nu
            
            if verbose:
                print('fieldlines = {}'.format(fieldlines))
                print('prefactor: {}'.format(prefactor))
                print('V_nu: {}'.format(V_nu))
                print('coefficient: {}\n'.format(coefficient))
                
                
        for grid_indices_a in grid.all_points_indices():
            shifted_indices_d = (
                shifted_indices_minus_dict[omega_indices][grid_indices_a])
            for grid_indices_b in grid.all_points_indices():
                shifted_indices_c = (
                    shifted_indices_plus_dict[omega_indices][grid_indices_b])

                # Loop over spins.
                for spin_a in spins:
                    orbital_a = orbital_ids[grid_indices_a][spin_a]
                    orbital_d = orbital_ids[shifted_indices_d][spin_a]
                    for spin_b in spins:
                        orbital_b = orbital_ids[grid_indices_b][spin_b]
                        orbital_c = orbital_ids[shifted_indices_c][spin_b]

                        # Add interaction term.
                        if ((orbital_a != orbital_b) and
                                (orbital_c != orbital_d)):
                            operators = ((orbital_a, 1), (orbital_b, 1),
                                         (orbital_c, 0), (orbital_d, 0))
                            operator += FermionOperator(operators, coefficient)

    '''
    normal_ordered_operator = openfermion.normal_ordered(operator)
    print('Plane wave basis potential operator:')
    print('qubits: {}\n'.format(openfermion.count_qubits(normal_ordered_operator)))
    #print(operator)
    print()
    '''
    # Return.
    return operator


def dual_basis_jellium_model(grid, spinless=False,
                             kinetic=True, potential=True,
                             include_constant=False,
                             non_periodic=False, period_cutoff=None, 
                             fieldlines=3, verbose=False):
    """Return jellium Hamiltonian in the dual basis of arXiv:1706.00023

    Args:
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.
        kinetic (bool): Whether to include kinetic terms.
        potential (bool): Whether to include potential terms.
        include_constant (bool): Whether to include the Madelung constant.
            Note constant is unsupported for non-uniform, non-cubic cells with
            ions.
        non_periodic (bool): If the system is non-periodic, default to False.
        period_cutoff (float): Period cutoff, default to
            grid.volume_scale() ** (1. / grid.dimensions).
        fieldlines (int): Spatial dimension for electric field lines. 
        verbose (bool): Whether to turn on print statements.  

    Returns:
        operator (FermionOperator)
    """
    # Initialize.
    n_points = grid.num_points
    position_prefactor = 0.
    
    # 3D case.
    if grid.dimensions == 3:
        position_prefactor = 2.0 * numpy.pi / grid.volume_scale()
    
    # 2D case.
    elif grid.dimensions == 2:
        position_prefactor = 1. / (2. * grid.volume_scale())
        
    
    operator = FermionOperator()
    spins = [None] if spinless else [0, 1]
    
    '''            
    potential_og = potential
    
    if verbose:
        print('Original potential = {}\n'.format(potential_og))
    '''            
    
    if potential and non_periodic and period_cutoff is None:
        period_cutoff = grid.volume_scale() ** (1.0 / grid.dimensions)

    # Pre-Computations.
    position_vectors = {}
    momentum_vectors = {}
    momenta_squared_dict = {}
    orbital_ids = {}
    
    for indices in grid.all_points_indices():
        # Store position vectors in dictionary, with corresponding grid index as key.
        position_vectors[indices] = grid.position_vector(indices)
        
        # Get and store momentum vectors in dictionary, with corresponding grid index
        # as key.
        momenta = grid.momentum_vector(indices)
        momentum_vectors[indices] = momenta
        
        # Store momentum squared in dictionary, with corresponding grid index as key.
        momenta_squared_dict[indices] = momenta.dot(momenta)
        
        # Store spin orbitals at each grid index in dictionary.
        orbital_ids[indices] = {}
        
        for spin in spins:
            orbital_ids[indices][spin] = grid.orbital_id(indices, spin)

    # This gives the position vector of the grid point at bottom-left-most corner. 
    #
    #                x---x---x
    #                |   |   |
    #                x---x---x
    #                |   |   |
    # this point <-- x---x---x
    #
    grid_origin = (0, ) * grid.dimensions
    coordinates_origin = position_vectors[grid_origin]
        
    # Loop once through all grid points.
    for grid_indices_b in grid.all_points_indices():
        if verbose:
            print('Grid point: {}\n'.format(grid_indices_b))
        
        # For all grid points, 'differences' gets the position displacement from the 
        # 'origin' point. This corresponds to evaluating (r_p - r_q) == r_(p-q).
        coordinates_b = position_vectors[grid_indices_b]
        differences = coordinates_b - coordinates_origin
        
        # If non_periodic.
        # See paper Appendix E.2. This is not setting function at boundary to 0.
        # Just accounting for Fourier Transform with truncated Coulomb operator.
                
        '''
        # According to paper, in the dual basis the truncated Coulomb operator can be implemented by 
        # dropping all n_p * n_q terms for which |r_p - r_q| > D.
        
        if non_periodic:
            if verbose:
                print('non_periodic')

            if numpy.sqrt(differences.dot(differences)) > period_cutoff:
                if verbose:
                    print('e-e potential distances larger than cutoff.')
                    print('coordinates b: {}'.format(coordinates_b))
                    print('coordinates origin: {}'.format(coordinates_origin))
                    print('differences: {}'.format(differences))
                    print('Potential set to False.\n')
                
                # Set potential option to False. This drops the potential terms where |r_p - r_q| > D.
                potential = False
        '''
        
        
        # Compute coefficients.
        kinetic_coefficient = 0.
        potential_coefficient = 0.
        
        # Loop once through all momentum indices, k_nu.
        for momenta_indices in grid.all_points_indices():
            momenta = momentum_vectors[momenta_indices]
            momenta_squared = momenta_squared_dict[momenta_indices]
            
            if momenta_squared == 0:
                continue

            cos_difference = numpy.cos(momenta.dot(differences))
            
            # This computes 1/(2N) * sum_nu{ k_nu^2 * cos[k_nu * r_(q-p)] }
            if kinetic:
                if verbose:
                    print('Added kinetic term.')
                
                kinetic_coefficient += (
                    cos_difference * momenta_squared /
                    (2. * float(n_points)))
            
            if verbose:
                print('Potential = {}'.format(potential))
                
            # This computes 2pi/Omega * sum_nu{ cos[k_nu * r_(p-q)] / k_nu^2 }
            if potential: 
                
                # Potential coefficient for this value of nu.
                potential_coefficient_nu = 0.
                
                # 3D case.
                if grid.dimensions == 3:
                    potential_coefficient_nu = position_prefactor * cos_difference / momenta_squared
                    
                    if non_periodic:
                        correction = 1.0 - numpy.cos(period_cutoff * numpy.sqrt(momenta_squared))
                        potential_coefficient_nu *= correction

                        if verbose:
                            print('non_periodic')
                            print('cutoff: {}'.format(period_cutoff))
                            print('correction: {}\n'.format(correction))
                            
                
                # 2D case.
                elif grid.dimensions == 2:
                    V_nu = 0.

                    if fieldlines == 2:
                        if non_periodic:
                            
                            # Set zero reference length scale for the potential. 
                            # For now, we take the reference length scale as the length of cell.
                            r0 = grid.volume_scale() ** (1. / grid.dimensions)
                            Dkv = period_cutoff * numpy.sqrt(momenta_squared)
                            V_nu = 4. * numpy.pi / momenta_squared * (
                                Dkv * numpy.log(period_cutoff / r0) * scipy.special.jv(1, Dkv) + scipy.special.jv(0, Dkv) - 1.)
                            
                            if verbose:
                                print('non-periodic')
                                print('cutoff: {}\n'.format(period_cutoff))
                        
                        else:
                            var1 = 0.25 * momenta_squared
                            var2 = 4. / momenta_squared

                            V_nu = numpy.complex128(1. / 2. * 
                                    (mpmath.meijerg([[-0.5, 0., 0.], []], [[-0.5, 0.], [-1.]], var1)) - 
                                     mpmath.meijerg([[1., 1.5, 2.], []], [[1.5], []], var2))
                            
                    elif fieldlines == 3:
                        if non_periodic:
                            var = -0.25 * period_cutoff**2 * momenta_squared
                            V_nu = numpy.complex128(2 * numpy.pi * period_cutoff * mpmath.hyp1f2(0.5, 1., 1.5, var))
                            
                            if verbose:
                                print('non-periodic')
                                print('cutoff: {}\n'.format(period_cutoff))
                        
                        else:    
                            V_nu = 2. * numpy.pi / numpy.sqrt(momenta_squared)
                            
                    # Potential coefficient for this value of nu.
                    potential_coefficient_nu = position_prefactor * V_nu * cos_difference
                
                potential_coefficient += potential_coefficient_nu
                    
                if verbose:
                    print('fieldlines = {}'.format(fieldlines))
                    print('potential coefficient nu: {}\n'.format(potential_coefficient_nu))
                        
        
        if verbose:
            print('kinetic coefficient: {}'.format(kinetic_coefficient))
            print('potential coefficient: {}\n'.format(potential_coefficient))
        
        # Loop once through all grid points. 
        # We have r_p - r_q fixed by 'differences' computed above. 
        for grid_indices_shift in grid.all_points_indices():
            # Loop over spins and identify interacting orbitals.
            orbital_a = {}
            orbital_b = {}
            
            # grid_origin = (0, ) * grid.dimensions
            # 'shifted_index_1' is equivalent to just 'grid_indices_shift'.
            shifted_index_1 = tuple(
                [(grid_origin[i] + grid_indices_shift[i]) % grid.length[i]
                 for i in range(grid.dimensions)])
            
            # 'shifted_index_2' 
            shifted_index_2 = tuple(
                [(grid_indices_b[i] + grid_indices_shift[i]) % grid.length[i]
                 for i in range(grid.dimensions)])
            
            if verbose:
                print('shifted index 1: {}'.format(shifted_index_1))
                print('shifted index 2: {}'.format(shifted_index_2))
            
            for spin in spins:
                orbital_a[spin] = orbital_ids[shifted_index_1][spin]
                orbital_b[spin] = orbital_ids[shifted_index_2][spin]
                
            if kinetic:
                for spin in spins:
                    operators = ((orbital_a[spin], 1), (orbital_b[spin], 0))
                    operator += FermionOperator(operators, kinetic_coefficient)
                     
            '''
            # If non_periodic, potential will also evaluate to False, so this part is skipped,
            # i.e. no potential terms will be added. 
            '''            
            if potential:
                for sa in spins:
                    for sb in spins:
                        if orbital_a[sa] == orbital_b[sb]:
                            continue
                            
                        operators = ((orbital_a[sa], 1), (orbital_a[sa], 0),
                                     (orbital_b[sb], 1), (orbital_b[sb], 0))
                        operator += FermionOperator(operators,
                                                    potential_coefficient)
        
        '''            
        # Reset potential.
        potential = potential_og
        
        if non_periodic and verbose:
            print('\nPotential reset to {}.\n'.format(potential))
        '''            
        
    # Include the Madelung constant if requested.
    if include_constant:
        
        # TODO: Check for other unit cell shapes
        # Currently only for cubic cells.
        operator += (FermionOperator.identity() *
                     (2.8372 / grid.volume_scale()**(1./grid.dimensions)))
    
    '''
    normal_ordered_operator = openfermion.normal_ordered(operator)
    print('Dual basis jellium operator:')
    print('qubits: {}\n'.format(openfermion.count_qubits(normal_ordered_operator)))
    #print(operator)
    print()
    '''
    # Return.
    return operator


def dual_basis_kinetic(grid, spinless=False):
    """Return the kinetic operator in the dual basis of arXiv:1706.00023.

    Args:
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        operator (FermionOperator)
    """
    return dual_basis_jellium_model(grid, spinless, True, False)


def dual_basis_potential(grid, spinless=False, non_periodic=False,
                         period_cutoff=None, fieldlines=3, verbose=False):
    """Return the potential operator in the dual basis of arXiv:1706.00023

    Args:
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.
        non_periodic (bool): If the system is non-periodic, default to False.
        period_cutoff (float): Period cutoff, default to
            grid.volume_scale() ** (1. / grid.dimensions).
        fieldlines (int): Spatial dimension for electric field lines. 
        verbose (bool): Whether to turn on print statements.  

    Returns:
        operator (FermionOperator)
    """
    operator = dual_basis_jellium_model(grid, spinless, False, True, False,
                                    non_periodic, period_cutoff, fieldlines, verbose)
    
    '''
    normal_ordered_op = openfermion.normal_ordered(operator)
    print('Dual basis potential:')
    print('qubits: {}'.format(openfermion.count_qubits(normal_ordered_op)))
    #print(normal_ordered_op)
    '''
    
    return operator


def jellium_model(grid, spinless=False, plane_wave=True,
                  include_constant=False, e_cutoff=None,
                  non_periodic=False, period_cutoff=None, 
                  ft=False, fieldlines=3, verbose=False):
    """Return jellium Hamiltonian as FermionOperator class.

    Args:
        grid (openfermion.utils.Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.
        plane_wave (bool): Whether to return in momentum space (True)
            or position space (False).
        include_constant (bool): Whether to include the Madelung constant.
            Note constant is unsupported for non-uniform, non-cubic cells with
            ions.
        e_cutoff (float): Energy cutoff.
        non_periodic (bool): If the system is non-periodic, default to False.
        period_cutoff (float): Period cutoff, default to
            grid.volume_scale() ** (1. / grid.dimensions).
        ft (bool): Whether to use the Fourier Transform of the dual basis
                   potentials as the plane wave potentials.
        fieldlines (int): Spatial dimension for electric field lines. 
        verbose (bool): Whether to turn on print statements.

    Returns:
        FermionOperator: The Hamiltonian of the model.
    """
    if plane_wave:
        hamiltonian = plane_wave_kinetic(grid, spinless, e_cutoff)
        
        # Use FT of dual basis potential. 
        if ft:
            hamiltonian += plane_wave_potential_v2(grid, spinless, e_cutoff,
                                                   non_periodic, period_cutoff, verbose)
        else:
            hamiltonian += plane_wave_potential(grid, spinless, e_cutoff,
                                                non_periodic, period_cutoff, fieldlines, verbose)
    
    else:
        hamiltonian = dual_basis_jellium_model(grid, spinless, True, True,
                                               include_constant, non_periodic,
                                               period_cutoff, fieldlines, verbose)
        
    # Include the Madelung constant if requested.
    if include_constant:
        # TODO: Check for other unit cell shapes
        hamiltonian += (FermionOperator.identity() *
                        (2.8372 / grid.volume_scale()**(1. / grid.dimensions)))
    return hamiltonian


def jordan_wigner_dual_basis_jellium(grid, spinless=False,
                                     include_constant=False, 
                                     non_periodic=False, period_cutoff=None,
                                     fieldlines=3, verbose=False):
    """Return the jellium Hamiltonian as QubitOperator in the dual basis.

    Args:
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.
        include_constant (bool): Whether to include the Madelung constant.
            Note constant is unsupported for non-uniform, non-cubic cells with
            ions.
        non_periodic (bool): If the system is non-periodic, default to False.
        period_cutoff (float): Period cutoff, default to
            grid.volume_scale() ** (1. / grid.dimensions).
        fieldlines (int): Spatial dimension for electric field lines. 
        verbose (bool): Whether to turn on print statements.

    Returns:
        hamiltonian (QubitOperator)
    """
    # Initialize.
    n_orbitals = grid.num_points
    volume = grid.volume_scale()
    
    if non_periodic and period_cutoff is None:
        period_cutoff = volume ** (1.0 / grid.dimensions)
        
    if spinless:
        n_qubits = n_orbitals
    else:
        n_qubits = 2 * n_orbitals
    hamiltonian = QubitOperator()

    # Compute vectors.
    momentum_vectors = {}
    momenta_squared_dict = {}
    for indices in grid.all_points_indices():
        momenta = grid.momentum_vector(indices)
        momentum_vectors[indices] = momenta
        momenta_squared_dict[indices] = momenta.dot(momenta)
    
    #-------------------------------------------------------------------------
    # Compute the identity coefficient and the coefficient of local Z terms.
    #-------------------------------------------------------------------------
    identity_coefficient = 0.
    z_coefficient = 0.
    
    for k_indices in grid.all_points_indices():
        momenta = momentum_vectors[k_indices]
        momenta_squared = momenta.dot(momenta)
        if momenta_squared == 0:
            continue
    
        identity_coefficient += momenta_squared / 2.
        z_coefficient -= momenta_squared / (4. * float(n_orbitals))
        
        # Coefficients for this value of nu.
        identity_coefficient_nu = 0.
        z_coefficient_nu = 0.

        # 3D case.
        if grid.dimensions == 3:
            identity_coefficient_nu = (numpy.pi * float(n_orbitals) /
                                     (momenta_squared * volume))
            z_coefficient_nu = numpy.pi / (momenta_squared * volume)

            if non_periodic:
                correction = 1.0 - numpy.cos(period_cutoff * numpy.sqrt(momenta_squared))
                identity_coefficient_nu *= correction
                z_coefficient_nu *= correction

                if verbose:
                    print('non_periodic')
                    print('cutoff: {}'.format(period_cutoff))
                    print('correction: {}'.format(correction))

        # 2D case.
        elif grid.dimensions == 2:
            V_nu = 0.

            if fieldlines == 2:
                if non_periodic:
                    
                    # Set zero reference length scale for the potential. 
                    # For now, we take the reference length scale as the length of cell.
                    r0 = grid.volume_scale() ** (1. / grid.dimensions)
                    Dkv = period_cutoff * numpy.sqrt(momenta_squared)
                    V_nu = 4. * numpy.pi / momenta_squared * (
                        Dkv * numpy.log(period_cutoff / r0) * scipy.special.jv(1, Dkv) + scipy.special.jv(0, Dkv) - 1.)
                    
                    if verbose:
                        print('non-periodic')
                        print('cutoff: {}'.format(period_cutoff))

                else:
                    var1 = 0.25 * momenta_squared
                    var2 = 4. / momenta_squared

                    V_nu = numpy.complex128(1. / 2. * 
                            (mpmath.meijerg([[-0.5, 0., 0.], []], [[-0.5, 0.], [-1.]], var1)) - 
                             mpmath.meijerg([[1., 1.5, 2.], []], [[1.5], []], var2))

            elif fieldlines == 3:
                if non_periodic:
                    var = -0.25 * period_cutoff**2 * momenta_squared
                    V_nu = numpy.complex128(2 * numpy.pi * period_cutoff * mpmath.hyp1f2(0.5, 1., 1.5, var))
                    
                    if verbose:
                        print('non-periodic')
                        print('cutoff: {}'.format(period_cutoff))

                else:
                    V_nu = 2 * numpy.pi / numpy.sqrt(momenta_squared)

            identity_coefficient_nu = float(n_orbitals) / (4. * volume) * V_nu
            z_coefficient_nu = 1. / (4. * volume) * V_nu

            if verbose:
                print('V_nu: {}\n'.format(V_nu))
            
        identity_coefficient -= identity_coefficient_nu
        z_coefficient += z_coefficient_nu

        if verbose:
            print('fieldlines = {}'.format(fieldlines))
            print('identity coefficient: {}'.format(identity_coefficient))
            print('Z coefficient: {}\n'.format(z_coefficient))
            
            
    if spinless:
        identity_coefficient /= 2.
    

    # Add identity term.
    identity_term = QubitOperator((), identity_coefficient)
    hamiltonian += identity_term

    # Add local Z terms.
    for qubit in range(n_qubits):
        qubit_term = QubitOperator(((qubit, 'Z'),), z_coefficient)
        hamiltonian += qubit_term

    #-------------------------------------------------------------------------
    # Add ZZ terms and XZX + YZY terms.
    #-------------------------------------------------------------------------
    zz_prefactor = 0.
    
    # 3D case.
    if grid.dimensions == 3:
        zz_prefactor = numpy.pi / volume        
    
    # 2D case.
    elif grid.dimensions == 2:
        zz_prefactor = 1. / (4. * volume)
        
    xzx_yzy_prefactor = .25 / float(n_orbitals)
    for p in range(n_qubits):
        index_p = grid.grid_indices(p, spinless)
        position_p = grid.position_vector(index_p)
        for q in range(p + 1, n_qubits):
            index_q = grid.grid_indices(q, spinless)
            position_q = grid.position_vector(index_q)

            difference = position_p - position_q

            skip_xzx_yzy = not spinless and (p + q) % 2

            # Loop through momenta.
            zpzq_coefficient = 0.
            term_coefficient = 0.
            for k_indices in grid.all_points_indices():
                momenta = momentum_vectors[k_indices]
                momenta_squared = momenta_squared_dict[k_indices]
                if momenta_squared == 0:
                    continue

                cos_difference = numpy.cos(momenta.dot(difference))
                zpzq_coefficient_nu = 0.

                # 3D case.
                if grid.dimensions == 3:
                    zpzq_coefficient_nu = (zz_prefactor * cos_difference /
                                         momenta_squared)
                    
                    if non_periodic:
                        correction = 1.0 - numpy.cos(period_cutoff * numpy.sqrt(momenta_squared))
                        zpzq_coefficient_nu *= correction

                        if verbose:
                            print('non_periodic')
                            print('cutoff: {}'.format(period_cutoff))
                            print('correction: {}'.format(correction))
                            
                # 2D case.
                if grid.dimensions == 2:
                    V_nu = 0.
        
                    if fieldlines == 2:
                        if non_periodic:
                            
                            # Set zero reference length scale for the potential. 
                            # For now, we take the reference length scale as the length of cell.
                            r0 = grid.volume_scale() ** (1. / grid.dimensions)
                            Dkv = period_cutoff * numpy.sqrt(momenta_squared)
                            V_nu = 4. * numpy.pi / momenta_squared * (
                                Dkv * numpy.log(period_cutoff / r0) * scipy.special.jv(1, Dkv) + scipy.special.jv(0, Dkv) - 1.)
                            
                            if verbose:
                                print('non-periodic')
                                print('cutoff: {}'.format(period_cutoff))

                        else:
                            var1 = 0.25 * momenta_squared
                            var2 = 4. / momenta_squared

                            V_nu = numpy.complex128(1. / 2. * 
                                    (mpmath.meijerg([[-0.5, 0., 0.], []], [[-0.5, 0.], [-1.]], var1)) - 
                                     mpmath.meijerg([[1., 1.5, 2.], []], [[1.5], []], var2))
                    
                    elif fieldlines == 3:
                        if non_periodic:
                            var = -0.25 * period_cutoff**2 * momenta_squared
                            V_nu = numpy.complex128(2 * numpy.pi * period_cutoff * mpmath.hyp1f2(0.5, 1., 1.5, var))
                            
                            if verbose:
                                print('non-periodic')
                                print('cutoff: {}'.format(period_cutoff))

                        else:
                            V_nu = 2 * numpy.pi / numpy.sqrt(momenta_squared)
                    
                    zpzq_coefficient_nu = zz_prefactor * cos_difference * V_nu
                            
                zpzq_coefficient += zpzq_coefficient_nu

                if skip_xzx_yzy:
                    continue
                    
                term_coefficient += (xzx_yzy_prefactor * cos_difference *
                                     momenta_squared)

            # Add ZZ term.
            qubit_term = QubitOperator(((p, 'Z'), (q, 'Z')), zpzq_coefficient)
            hamiltonian += qubit_term

            # Add XZX + YZY term.
            if skip_xzx_yzy:
                continue
            z_string = tuple((i, 'Z') for i in range(p + 1, q))
            xzx_operators = ((p, 'X'),) + z_string + ((q, 'X'),)
            yzy_operators = ((p, 'Y'),) + z_string + ((q, 'Y'),)
            hamiltonian += QubitOperator(xzx_operators, term_coefficient)
            hamiltonian += QubitOperator(yzy_operators, term_coefficient)

    # Include the Madelung constant if requested.
    if include_constant:
        # TODO Generalize to other cells
        hamiltonian += (QubitOperator((),) *
                        (2.8372 / grid.volume_scale() ** (1./grid.dimensions)))

    # Return Hamiltonian.
    return hamiltonian


def hypercube_grid_with_given_wigner_seitz_radius_and_filling(
        dimension, grid_length, wigner_seitz_radius,
        filling_fraction=0.5, spinless=True):
    """Return a Grid with the same number of orbitals along each dimension
    with the specified Wigner-Seitz radius.

    Args:
        dimension (int): The number of spatial dimensions.
        grid_length (int): The number of orbitals along each dimension.
        wigner_seitz_radius (float): The Wigner-Seitz radius per particle,
            in Bohr.
        filling_fraction (float): The average spin-orbital occupation.
            Specifies the number of particles (rounding down).
        spinless (boolean): Whether to give the system without or with spin.
    """
    if filling_fraction > 1:
        raise ValueError("filling_fraction cannot be greater than 1.")

    n_qubits = grid_length ** dimension
    if not spinless:
        n_qubits *= 2

    n_particles = int(numpy.floor(n_qubits * filling_fraction))

    if not n_particles:
        raise ValueError(
            "filling_fraction too low for number of orbitals specified by "
            "other parameters.")

    # Compute appropriate length scale.
    length_scale = wigner_seitz_length_scale(
        wigner_seitz_radius, n_particles, dimension)

    return Grid(dimension, grid_length, length_scale)
