# Start from your base image
FROM tensorflow/tensorflow:2.14.0-gpu-jupyter

# Install CLI into /usr/local/bin (on PATH)
RUN pip install --no-cache-dir kaggle

# Create non-root user matching your host UID/GID to avoid perms issues
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} lobral && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash lobral

# Set working directory to lobral's home
WORKDIR /home/lobral

# Create the .kaggle directory first
RUN mkdir -p /home/lobral/.kaggle

# Copy the Kaggle token into the container
COPY .kaggle/kaggle.json /home/lobral/.kaggle/kaggle.json

# Set permissions and ownership
RUN chmod 600 /home/lobral/.kaggle/kaggle.json && \
    chown -R lobral:lobral /home/lobral/.kaggle

# Switch to non-root user
USER lobral

# Set Kaggle environment variable
ENV KAGGLE_CONFIG_DIR=/home/lobral/.kaggle

# Update pip and install packages
RUN pip install --upgrade pip && \
    pip install pandas tqdm scikit-learn matplotlib seaborn jupyterlab plotly gdown imagehash

# sanity check during build
RUN kaggle --version