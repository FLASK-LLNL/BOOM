################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
def test_imports():
    import boom.viz.ParityPlot

    assert boom.viz.ParityPlot is not None
    assert boom.viz.ParityPlot.ParityPlot is not None
    assert boom.viz.ParityPlot.OODParityPlot is not None
    assert boom.viz.ParityPlot.DensityOODParityPlot is not None
    assert boom.viz.ParityPlot.HoFOODParityPlot is not None

    print("All imports are working!")


if __name__ == "__main__":
    test_imports()
