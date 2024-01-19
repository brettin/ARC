# Some basic stuff
export PATH=$PATH:$HOME/anaconda3/bin
export CUDA_VISIBLE_DEVICES=7
echo "using CUDA_VUSIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

conda activate /rbscratch/brettin/conda_envs/openai-rag-arxiv

export PYTHONPATH=$PYTHONPATH:.
