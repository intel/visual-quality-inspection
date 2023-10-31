echo "Finetuning"
export PYTHONPATH=$PYTHONPATH:/workspace/transfer-learning
rm -rf /workspace/output
cd /workspace
mkdir /cnvrg/output
ln -s /cnvrg/output /workspace/
ln -s /input/dataset/data /workspace/data