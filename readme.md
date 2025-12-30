# OpenFWI 2025 on GCP
This repository is the model training code of my solution to the competition [Yale/UNC-CH - Geophysical Waveform Inversion](https://www.kaggle.com/competitions/waveform-inversion).

Built a 12-layer transformer model with ROPE and [Hyper-connection](https://arxiv.org/abs/2409.19606). Training it by torch-lightning with ema-weighted mae loss. 

keywords: GCP, TPU, SkyPilot, PyTorch-Lightning

# Data preparation
The dataset can befound from [openfwi/datasets](https://smileunc.github.io/projects/openfwi/datasets). I used Vel Family, Fault Family and Style Family and there will be 670GB data in total. Please place the dataset in a different directory from the project code or skypilot will try to uploading the whole dataset.

Using the script preprocessing.py to transform the .npy files into .tfrecords for better GCP IO efficiency.
> python preprocessing --intput-folder folder-to-openfwi --output-folder folder-to-tfrecords

And then upload to the gcp bucket:
> gsutil -m cp -r folder-to-tfrecords/train gs://openfwi_tfrecord/
> gsutil -m cp -r folder-to-tfrecords/valid gs://openfwi_valid_tfrecords/

# skypilot run
> sky launch -c openfwi-tpu sk_config.yaml
