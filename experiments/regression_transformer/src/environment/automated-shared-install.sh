#!/bin/bash
 
# NOTE: install in a directory with plenty of free space, like /usr/workspace/$USER
envName="torch2_lassen_flask"
#groupName="flask"
#groupDir="/usr/workspace/$groupName"
groupDir="/p/gpfs1/$USER/flask-si"
envDir="${envName}_conda_env"
bashFileName=".bashrc_$envName"

mkdir -p $groupDir
 
# Load modules (may not need this version of gcc for arbitrary installs)
module load gcc/9.3.1
 
# install to conda_env subdirectory within current working directory
installdir="$(pwd)/$envDir"
 
## install conda (have to answer some questions here in the script installation; say no to final question)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh --no-check-certificate
sh Miniconda3-latest-Linux-ppc64le.sh -f -p "$installdir"
 
cat > $bashFileName << EOF
stty erase '^?'

alias env_activate="conda activate $envName"

# UPDATE THIS VARIABLE PLEASE!
groupDir=$groupDir
conda_env_dir=$envDir

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="\$("\$groupDir/\$conda_env_dir/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
if [ \$? -eq 0 ]; then
  eval "\$__conda_setup"
else
  if [ -f "\$groupDir/\$conda_env_dir/etc/profile.d/conda.sh" ]; then
    . "\$groupDir/\$conda_env_dir/etc/profile.d/conda.sh"
  else
    export PATH="\$groupDir/\$conda_env_dir/bin:\$PATH"
  fi
fi
unset __conda_setup

#set the condarc file for this env
export CONDARC=\$groupDir/\$conda_env_dir/.condarc
export SSL_CERT_FILE=/etc/pki/tls/cert.pem
export REQUESTS_CA_BUNDLE=/etc/pki/tls/cert.pem

# <<< conda initialize <<<
#allow the group to read and write files you create
umask u=rwx,g=rwx,o=
#set the condarc file for this env
export CONDARC=\$groupDir/\$conda_env_dir/.condarc
EOF

# activate conda environment
source $bashFileName
 
# Update permissions for conda environment files/directories for group
chmod -R 775 $installdir
chmod -R 775 $bashFileName

# create an opence environment in conda (Python-3.9)
#conda config --prepend channels 'defaults'   # lowest priority
conda config --prepend channels 'https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda-early-access/'
conda config --prepend channels 'https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/'
conda config --prepend channels 'https://opence.mit.edu'
conda config --prepend channels 'https://ftp.osuosl.org/pub/open-ce/current/'
conda config --prepend channels 'conda-forge'   # highest priority
conda create --name "$envName" python==3.9 pytorch=2.0.1 #torchvision=0.15.2 spacy=3.5.3 scipy=1.10.1 fairlearn~=0.9.0 scikit-learn~=1.1.2 pandas~=2.0.3 pyarrow~=11.0.0 rust -c conda-forge

# activate the opence environment
conda activate "$envName"
 
# install some packages 
# example: conda install -c conda-forge matplotlib=3.6.3
#conda install -c conda-forge transformers==4.38.2 tokenizers==0.15.2 selfies==1.0.4
pip install transformers==4.38.2 tokenizers==0.15.2 selfies==1.0.4
pip install psutil
conda install conda-forge::matplotlib
conda install conda-forge::tensorboard
conda install anaconda::pandas
conda install conda-forge::rdkit


# Remove file used for Miniconda installation
rm Miniconda3-latest-Linux-ppc64le.sh
