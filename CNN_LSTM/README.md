# CNN + LSTM Training Pipeline

Binary classification for CICIDS2018 dataset (Benign vs Attack)

## Files

- `config.py` - Configuration parameters
- `preprocess.py` - Data preprocessing (shared for CNN & LSTM)
- `train_cnn.py` - CNN model training
- `train_lstm.py` - LSTM model training
- `kaggle_notebook.py` - Complete pipeline for Kaggle
- `inference.py` - Local testing with reserved test file

## Kaggle Usage

1. Upload dataset "cicids2018-csv" to Kaggle
2. Create new notebook with GPU enabled
3. Copy content of `kaggle_notebook.py` and run
4. Download zip files after training

## Local Testing

1. Download trained models from Kaggle (cnn_output.zip, lstm_output.zip, processed_data.zip)
2. Extract to CNN_LSTM folder
3. Run `python inference.py`

## Output Structure

```
cnn_output/
  cnn_model.keras
  results.json
  training_history.json
  training_plots.png

lstm_output/
  lstm_model.keras
  results.json
  training_history.json
  training_plots.png

processed_data/
  scaler.pkl
  feature_names.json
  X_train.npy, X_val.npy, X_test.npy
  y_train.npy, y_val.npy, y_test.npy
```

## Metrics Saved

- Accuracy, Precision, Recall, F1-Score, AUC
- Latency (ms/sample)
- Confusion Matrix
- Training time, Epochs trained

