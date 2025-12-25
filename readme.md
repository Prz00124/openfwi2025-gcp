# v5e-4 (single vm)
v2.16.1 ok
    gcloud compute tpus tpu-vm create test123 --zone=us-central1-a --version=tpu-vm-tf-2.16.1-pjrt --accelerator-type=v5litepod-8 --spot

v2.18.0 ok
    gcloud compute tpus tpu-vm create test123 --zone=us-central1-a --version=tpu-vm-tf-2.18.0-pjrt-v5e-and-v6 --accelerator-type=v5litepod-8 --spot

    run:
        sudo apt-get update
        pip install --upgrade pip
        pip install tensorflow-tpu==2.18.0 -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force-reinstall

# v5e-16 (pod slice)
v2.16.1 ok
    gcloud compute tpus tpu-vm create test123 --zone=us-central1-a --version=tpu-vm-tf-2.16.1-pod-pjrt --accelerator-type=v5litepod-16 --spot

v2.18.0 ok
    gcloud compute tpus tpu-vm create test123 --zone=us-central1-a --version=tpu-vm-tf-2.18.0-pod-pjrt-v5e-and-v6 --accelerator-type=v5litepod-16 --spot

    run:
        gcloud compute tpus tpu-vm ssh test123 --zone=us-central1-a --worker=all --command="sudo apt-get update && pip install --upgrade pip && pip install tensorflow-tpu==2.18.0 -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force-reinstall"

# v6e-4
v2.16.1 failed, can't recognize TPU device.
    gcloud compute tpus tpu-vm create test123 --zone=us-central1-b --version=tpu-vm-tf-2.16.1-pjrt --accelerator-type=v6e-4 --spot

v2.18.0 failed, recognize TPU devices but crush while initialize strategy.
    gcloud compute tpus tpu-vm create test123 --zone=us-central1-b --version=tpu-vm-tf-2.18.0-pjrt-v5e-and-v6 --accelerator-type=v6e-4 --spot

    run:
        sudo apt-get update
        pip install --upgrade pip
        pip install tensorflow-tpu==2.18.0 -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force-reinstall

# v6e-16
v2.19.0 failed.
gcloud compute tpus tpu-vm create test123 --zone=us-central1-b --version=tpu-vm-tf-2.19.0-pod-pjrt --accelerator-type=v6e-16 --spot
    run:
        gcloud compute tpus tpu-vm ssh test123 --zone=us-central1-b --worker=all --command="unset LD_PRELOAD python test_tpuv6.py"
        