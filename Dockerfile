# import cuda pytorch 
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# install pip packages
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# install requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt
