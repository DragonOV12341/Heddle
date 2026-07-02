source /data2/xsl/miniconda3/etc/profile.d/conda.sh
conda activate heddle

export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export CUDAToolkit_ROOT="$CONDA_PREFIX"

export PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/nvvm/bin:$PATH"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"