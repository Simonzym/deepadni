# deepadni
## Data Preparation (R code)
1. Preprocess.R: data management of medical information;
2. Variable selection.R: select ROIs from FDG-PET images using linear mixed models.
3. variant visits.R: select patients and build graph;
4. gnn_sim.R: simulate data sets and build graphs.


## cnn_gnn (Python code)
1. get_data.py: function for reading images and generate data structure for future use;
2. feature extraction.py: CNN model and save extracted features;
3. ffn.py: feed-forward neural network using extracted features from single time points;
4. rnn.py: bidirectional GRU as sequential model;
5. Transformer.py: transformer encoder as sequential model;
6. gnn.py: GNN as sequential model.

##gnn_sim (Python code)
The same as cnn_gnn, but for simulated data.
