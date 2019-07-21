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

"""Tests for plane_wave_hamiltonian.py"""
from __future__ import absolute_import

import unittest


from openfermion.hamiltonians._plane_wave_hamiltonian import *
from openfermion.transforms import jordan_wigner, get_sparse_operator
from openfermion.utils import (eigenspectrum, Grid, inverse_fourier_transform,
                               is_hermitian)


class PlaneWaveHamiltonianTest(unittest.TestCase):

    def test_plane_wave_hamiltonian_integration(self):
        length_set = [2, 3, 4]
        spinless_set = [True, False]
        length_scale = 1.1
        for geometry in [[('H', (0,)), ('H', (0.8,))],
                         [('H', (0.1,))],
                         [('H', (0.1,))]]:
            for l in length_set:
                for spinless in spinless_set:
                    grid = Grid(dimensions=1, scale=length_scale, length=l)
                    h_plane_wave = plane_wave_hamiltonian(
                        grid, geometry, spinless, True, include_constant=False, ft=True)
                    h_dual_basis = plane_wave_hamiltonian(
                        grid, geometry, spinless, False, include_constant=False, ft=True)

                    # Test for Hermiticity
                    plane_wave_operator = get_sparse_operator(h_plane_wave)
                    dual_operator = get_sparse_operator(h_dual_basis)
                    self.assertTrue(is_hermitian((plane_wave_operator)))
                    self.assertTrue(is_hermitian(dual_operator))

                    jw_h_plane_wave = jordan_wigner(h_plane_wave)
                    jw_h_dual_basis = jordan_wigner(h_dual_basis)
                    h_plane_wave_spectrum = eigenspectrum(jw_h_plane_wave)
                    h_dual_basis_spectrum = eigenspectrum(jw_h_dual_basis)

                    max_diff = numpy.amax(
                        h_plane_wave_spectrum - h_dual_basis_spectrum)
                    min_diff = numpy.amin(
                        h_plane_wave_spectrum - h_dual_basis_spectrum)
                    self.assertAlmostEqual(max_diff, 0)
                    self.assertAlmostEqual(min_diff, 0)

    def test_plane_wave_hamiltonian_default_to_jellium_with_no_geometry(self):
        grid = Grid(dimensions=1, scale=1.0, length=4)
        self.assertTrue(plane_wave_hamiltonian(grid, ft=True) == jellium_model(grid, ft=True))

    def test_plane_wave_hamiltonian_bad_geometry(self):
        grid = Grid(dimensions=1, scale=1.0, length=4)
        with self.assertRaises(ValueError):
            plane_wave_hamiltonian(grid, geometry=[('H', (0, 0, 0))], ft=True)

        with self.assertRaises(ValueError):
            plane_wave_hamiltonian(grid, geometry=[('H', (0, 0, 0))],
                                   include_constant=True, ft=True)

    def test_plane_wave_hamiltonian_bad_element(self):
        grid = Grid(dimensions=3, scale=1.0, length=2)
        with self.assertRaises(ValueError):
            plane_wave_hamiltonian(grid, geometry=[('Unobtainium',
                                                    (0, 0, 0))], ft=True)

    def test_jordan_wigner_dual_basis_hamiltonian(self):
        grid = Grid(dimensions=2, length=3, scale=1.)
        spinless_set = [True, False]
        geometry = [('H', (0, 0)), ('H', (0.5, 0.8))]
        for spinless in spinless_set:
            fermion_hamiltonian = plane_wave_hamiltonian(
                grid, geometry, spinless, False, include_constant=False, ft=True)
            qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)

            test_hamiltonian = jordan_wigner_dual_basis_hamiltonian(
                grid, geometry, spinless, include_constant=False)
            self.assertTrue(test_hamiltonian == qubit_hamiltonian)

    def test_jordan_wigner_dual_basis_hamiltonian_default_to_jellium(self):
        grid = Grid(dimensions=1, scale=1.0, length=4)
        self.assertTrue(jordan_wigner_dual_basis_hamiltonian(grid) ==
                        jordan_wigner(jellium_model(grid, plane_wave=False, ft=True)))

    def test_jordan_wigner_dual_basis_hamiltonian_bad_geometry(self):
        grid = Grid(dimensions=1, scale=1.0, length=4)
        with self.assertRaises(ValueError):
            jordan_wigner_dual_basis_hamiltonian(
                grid, geometry=[('H', (0, 0, 0))])

        with self.assertRaises(ValueError):
            jordan_wigner_dual_basis_hamiltonian(
                grid, geometry=[('H', (0, 0, 0))], include_constant=True)

    def test_jordan_wigner_dual_basis_hamiltonian_bad_element(self):
        grid = Grid(dimensions=3, scale=1.0, length=4)
        with self.assertRaises(ValueError):
            jordan_wigner_dual_basis_hamiltonian(
                grid, geometry=[('Unobtainium', (0, 0, 0))])

    def test_plane_wave_energy_cutoff(self):
        geometry = [('H', (0,)), ('H', (0.8,))]
        grid = Grid(dimensions=1, scale=1.1, length=5)
        e_cutoff = 50.0

        h_1 = plane_wave_hamiltonian(grid, geometry, True, True, False, ft=True)
        jw_1 = jordan_wigner(h_1)
        spectrum_1 = eigenspectrum(jw_1)

        h_2 = plane_wave_hamiltonian(grid, geometry, True, True, False,
                                     e_cutoff, ft=True)
        jw_2 = jordan_wigner(h_2)
        spectrum_2 = eigenspectrum(jw_2)

        max_diff = numpy.amax(numpy.absolute(spectrum_1 - spectrum_2))
        self.assertGreater(max_diff, 0.)

    def test_plane_wave_period_cutoff(self):
        # TODO: After figuring out the correct formula for period cutoff for
        #     dual basis, change period_cutoff to default, and change
        #     h_1 to also accept period_cutoff for real integration test.

        geometry = [('H', (0., 0.)), ('H', (1., 0.))]
        scale = 8 * 1.
        grid = Grid(dimensions=2, scale=scale, length=3)
        
        # min period cutoff is scale * [( length - 1 )/length] * [1/( length - 1 )] == [ scale/length ]
        period_cutoff = grid.volume_scale() ** (1. / grid.dimensions) / 2.

        h_1 = plane_wave_hamiltonian(grid, geometry, spinless=True, plane_wave=True, include_constant=False,
                                     e_cutoff=None, ft=True)
        jw_1 = jordan_wigner(h_1)
        spectrum_1 = eigenspectrum(jw_1)

        h_2 = plane_wave_hamiltonian(grid, geometry, spinless=True, plane_wave=True, include_constant=False, 
                                     e_cutoff=None, non_periodic=True, period_cutoff=period_cutoff, ft=True)
        jw_2 = jordan_wigner(h_2)
        spectrum_2 = eigenspectrum(jw_2)

        max_diff = numpy.amax(numpy.absolute(spectrum_1 - spectrum_2))
        
        # Checks if non-periodic and periodic cases are different.
        self.assertGreater(max_diff, 0.)

        # TODO: This is only for code coverage. Remove after having real
        #     integration test.
        momentum_hamiltonian = plane_wave_hamiltonian(grid, geometry, spinless=True, plane_wave=True,
                                                      include_constant=False,                               
                                                      e_cutoff=None, non_periodic=True, period_cutoff=period_cutoff,
                                                      ft=True)
        
        position_hamiltonian = plane_wave_hamiltonian(grid, geometry, spinless=True, plane_wave=False, 
                                                      include_constant=False, 
                                                      e_cutoff=None, non_periodic=True, period_cutoff=period_cutoff,
                                                      ft=True)
        
        # Confirm they are Hermitian
        momentum_hamiltonian_operator = (
            get_sparse_operator(momentum_hamiltonian))
        self.assertTrue(is_hermitian(momentum_hamiltonian_operator))

        position_hamiltonian_operator = (
            get_sparse_operator(position_hamiltonian))
        self.assertTrue(is_hermitian(position_hamiltonian_operator))

        # Diagonalize and confirm the same energy.
        jw_momentum = jordan_wigner(momentum_hamiltonian)
        jw_position = jordan_wigner(position_hamiltonian)
        momentum_spectrum = eigenspectrum(jw_momentum)
        position_spectrum = eigenspectrum(jw_position)

        # Confirm spectra are the same.
        difference = numpy.amax(
            numpy.absolute(momentum_spectrum - position_spectrum))
        self.assertAlmostEqual(difference, 0.)

    def test_nonperiodic_external_potential_integration(self):
        # Compute potential energy operator in momentum and position space.
        # Non-periodic test.
        geometry = [('H', (0., 0.)), ('H', (1., 0.))]
        scale = 8 * 1.
        spinless = True
        
        for length in [3]:
            grid = Grid(dimensions=2, length=length, scale=scale)
            
            # min period cutoff is scale * [( length - 1 )/length] * [1/( length - 1 )] == [ scale/length ]
            period_cutoff = grid.volume_scale() ** (1. / grid.dimensions) / 2.
        
            momentum_external_potential = plane_wave_external_potential(grid, geometry, spinless, e_cutoff=None, 
                                                                        non_periodic=True, period_cutoff=period_cutoff)
            position_external_potential = dual_basis_external_potential(grid, geometry, spinless, 
                                                                        non_periodic=True, period_cutoff=period_cutoff)

            # Confirm they are Hermitian
            momentum_external_potential_operator = (
                get_sparse_operator(momentum_external_potential))
            self.assertTrue(is_hermitian(momentum_external_potential_operator))

            position_external_potential_operator = (
                get_sparse_operator(position_external_potential))
            self.assertTrue(is_hermitian(position_external_potential_operator))

            # Diagonalize and confirm the same energy.
            jw_momentum = jordan_wigner(momentum_external_potential)
            jw_position = jordan_wigner(position_external_potential)
            momentum_spectrum = eigenspectrum(jw_momentum)
            position_spectrum = eigenspectrum(jw_position)

            # Confirm spectra are the same.
            difference = numpy.amax(
                numpy.absolute(momentum_spectrum - position_spectrum))
            self.assertAlmostEqual(difference, 0.)