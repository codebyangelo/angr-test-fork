#!/usr/bin/env python3
# pylint:disable=no-self-use,protected-access
from __future__ import annotations

import struct
import unittest
from unittest import mock

import angr
from angr import claripy
from tests.common import minimal_project


class TestSolverEvalCasting(unittest.TestCase):
    """
    Basic test cases of SimSolver::eval's `cast_to` function.
    """

    def test_eval_cast_bvv_to_bytes(self):
        s = angr.SimState(project=minimal_project("AMD64"), mode="symbolic")
        assert s.solver.eval(claripy.BVV(0, 0), cast_to=bytes) == b""
        assert s.solver.eval(claripy.BVV(0, 8), cast_to=bytes) == b"\x00"
        assert s.solver.eval(claripy.BVV(0x12345678, 32), cast_to=bytes) == b"\x12\x34\x56\x78"

    def test_eval_cast_bvv_to_bytes__non_8bit_length_multiple(self):
        s = angr.SimState(project=minimal_project("AMD64"), mode="symbolic")
        for nbits in [1, 2, 7]:
            with self.subTest(nbits=nbits), self.assertRaises(ValueError):
                s.solver.eval(claripy.BVV(0, nbits), cast_to=bytes)

    def test_eval_cast_fpv_to_bytes(self):
        s = angr.SimState(project=minimal_project("AMD64"), mode="symbolic")
        value = 1.23456
        fpv = claripy.FPV(value, claripy.FSORT_FLOAT)
        assert s.solver.eval(fpv, cast_to=bytes) == struct.pack(">f", value)
        fpv = claripy.FPV(value, claripy.FSORT_DOUBLE)
        assert s.solver.eval(fpv, cast_to=bytes) == struct.pack(">d", value)

    def test_eval_cast_fpv_to_int(self):
        s = angr.SimState(project=minimal_project("AMD64"), mode="symbolic")
        value = 1.23456
        fpv = claripy.FPV(value, claripy.FSORT_FLOAT)
        assert s.solver.eval(fpv, cast_to=int) == int.from_bytes(struct.pack(">f", value), "big")
        fpv = claripy.FPV(value, claripy.FSORT_DOUBLE)
        assert s.solver.eval(fpv, cast_to=int) == int.from_bytes(struct.pack(">d", value), "big")

    def test_eval_cast_bool_to_bytes(self):
        s = angr.SimState(project=minimal_project("AMD64"), mode="symbolic")
        assert s.solver.eval(claripy.BoolV(False), cast_to=bytes) == b"\x00"
        assert s.solver.eval(claripy.BoolV(True), cast_to=bytes) == b"\x01"

    def test_eval_cast_bool_to_int(self):
        s = angr.SimState(project=minimal_project("AMD64"), mode="symbolic")
        assert s.solver.eval(claripy.BoolV(False), cast_to=int) == 0
        assert s.solver.eval(claripy.BoolV(True), cast_to=int) == 1


class TestSolverEvalConcreteShortcut(unittest.TestCase):
    """
    The eval* family short-circuits values that are already concrete Python primitives.
    """

    def test_eval_python_primitives(self):
        s = angr.SimState(project=minimal_project("AMD64"), mode="symbolic")
        for value in [7, True, 1.5, "abc"]:
            with self.subTest(value=value):
                assert s.solver.eval(value) == value
                assert s.solver.eval_one(value) == value
                assert s.solver.eval_upto(value, 2) == [value]

    def test_eval_string_ast(self):
        s = angr.SimState(project=minimal_project("AMD64"), mode="symbolic")
        assert s.solver.eval(claripy.StringV("abc")) == "abc"
        assert s.solver.eval_upto(claripy.StringV("abc"), 2) == ["abc"]


class TestSolverInterruptPreservation(unittest.TestCase):
    """
    ClaripySolverInterruptError (e.g. from solver timeouts) must not be coerced
    into SimSolverModeError, ensuring callers (such as SimulationManager and engines)
    can properly route interrupted states instead of treating them as errored or unsat.
    """

    def test_eval_preserves_solver_interrupt(self):
        s = angr.SimState(project=minimal_project("AMD64"), mode="symbolic")
        m = mock.MagicMock()
        m.eval.side_effect = claripy.ClaripySolverInterruptError("timeout")
        s.solver._stored_solver = m
        with self.assertRaises(claripy.ClaripySolverInterruptError):
            s.solver.eval(claripy.BVS("x", 32))

    def test_satisfiable_preserves_solver_interrupt(self):
        s = angr.SimState(project=minimal_project("AMD64"), mode="symbolic")
        m = mock.MagicMock()
        m.satisfiable.side_effect = claripy.ClaripySolverInterruptError("timeout")
        s.solver._stored_solver = m
        with self.assertRaises(claripy.ClaripySolverInterruptError):
            s.solver.satisfiable()

    def test_step_state_routes_solver_interrupt_to_interrupted(self):
        p = minimal_project("AMD64")
        s = p.factory.entry_state()
        m = mock.MagicMock()
        m.eval.side_effect = claripy.ClaripySolverInterruptError("timeout")
        s.solver._stored_solver = m

        simgr = p.factory.simgr(s)
        error_list = []

        def successor_calling_eval(state):
            state.solver.eval(claripy.BVS("target", 64))

        stashes = simgr.step_state(s, successor_func=successor_calling_eval, error_list=error_list)
        self.assertIn("interrupted", stashes)
        self.assertEqual(stashes["interrupted"], [s])
        self.assertEqual(len(error_list), 0)


if __name__ == "__main__":
    unittest.main()
