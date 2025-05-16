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
