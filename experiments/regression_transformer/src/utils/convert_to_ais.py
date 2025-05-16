################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import os
from selfies import decoder as selfies_decoder
from atomInSmiles import encode as ais_encoder

def main(eval_file):
    eval_dirname_list = eval_file.split("/")
    eval_fname_list = eval_dirname_list[-1].split(".")
    eval_dirname_list[-1] = f"{eval_fname_list[0]}_TO_AIS.{eval_fname_list[-1]}"
    ais_eval_filename = '/'.join(eval_dirname_list)
    print(f"ais_eval_filename: {ais_eval_filename}")
    tmp_eval_filename = ais_eval_filename.split("/")[-1].split("_")[-1].split(".")[0]
    print(f"tmp_eval_filename: {tmp_eval_filename}")
    if not os.path.isfile(ais_eval_filename):
        # Modify contents of eval_file but save to different file
        with open(ais_eval_filename, 'a') as ais_file: # NEW FILE
            with open(eval_file, 'r') as orig_file: # ORIG FILE
                while line := orig_file.readline():
                #for line in orig_file:
                    tmp_line = line.rstrip()
                    tmp_list = tmp_line.split("|")
                    assert len(tmp_list)==2
                    # Convert SEFLIES to SMILES and SMILES to AIS
                    smiles = selfies_decoder(tmp_list[1])
                    ais = ais_encoder(smiles)
                    tmp_line = f"{tmp_list[0].replace('dens','den')}|{ais}"
                    ais_file.write(f"{tmp_line}\n")
    # Update filename to use for data loading
    eval_file = ais_eval_filename

if __name__ == '__main__':
    # Density data
    main(eval_file='/usr/workspace/flaskdat/Models/regression-transformer/examples/10k-SELFIES-OOD-den/10k_dft_density_train_selfies.txt')
    main(eval_file='/usr/workspace/flaskdat/Models/regression-transformer/examples/10k-SELFIES-OOD-den/10k_dft_density_iid_test_selfies.txt')
    # HoF data
    main(eval_file='/usr/workspace/flaskdat/Models/regression-transformer/examples/10k-SELFIES-OOD-hof/10k_dft_hof_train_selfies.txt')
    main(eval_file='/usr/workspace/flaskdat/Models/regression-transformer/examples/10k-SELFIES-OOD-hof/10k_dft_hof_iid_test_selfies.txt')
