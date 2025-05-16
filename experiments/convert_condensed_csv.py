################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def condensed_to_full(condensed_fname, full_fname):
    # Read the CSV
    df = pd.read_csv(condensed_fname)

    # Melt the dataframe to transform the properties into rows
    df_melted = df.melt(
        id_vars=["Model", "Type", "Split"], var_name="Property", value_name="Value"
    )

    # Save the transformed CSV
    df_melted.to_csv(full_fname, index=False)


def csv_to_latex_table(fname):
    # Read the CSV
    with open(fname, "r") as f:
        lines = f.readlines()
    header = lines[0].strip().split(",")
    num_cols = len(header)

    for line in lines[1:]:
        Model, Split, Type, HoF, Density, HOMO, LUMO, GAP, ZPVE, R2, Alpha, Mu, Cv = (
            line.strip().split(",")
        )


def main():
    # Load the CSV file
    input_file = "RMSE_summary.csv"
    output_file = "FULL_RMSE_summary.csv"

    condensed_to_full(input_file, output_file)


main()
