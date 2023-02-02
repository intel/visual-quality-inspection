# frameworks.ai.end2end-ai-pipelines.anomaly-detection-ref-use-case

##To run the code ---

1. Setup the MVtec dataset directory path
2. Execute "python anomaly-detection.py" command

##To run the demo ---
1. Add hazelnut_demo.csv in hazelnut directory
2. Execute "Python anomaly_detection_demo.py"

##To run the SIM-SIAM Feature Extractor code ---

1. Make the "sim_siam" flag true
2. Download the Sim-Siam weights based on ResNet50 model from the link and move the inside ./simsiam directory-
   https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/pretrain/checkpoint_0099.pth.tar

3. Set category variable within MVTEC dataset, i.e. 'hazelnut'
3. Execute "python anomaly-detection.py" command
