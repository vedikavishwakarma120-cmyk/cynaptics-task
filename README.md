# cynaptics-task
Audio Classification: The Frequency Quest

Author: Vedika Vishwakarma

1. Project Description

This project is a deep learning-based audio classification system developed for a Kaggle competition. The system is designed to accurately categorize short audio samples into one of five predefined classes:

dog_bark

drilling

engine_idling

siren

street_music

The approach uses log-Mel spectrograms as the input feature, which are fed into a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The script handles all steps from data preprocessing to model training and final submission file generation.

2. Methodology

The classification pipeline follows these key steps:

Audio Loading: Audio files (.wav, .mp3, etc.) are loaded using librosa.

Padding / Truncating: All audio signals are standardized to a fixed length of 4.0 seconds. Shorter files are padded with silence, and longer files are truncated.

Feature Extraction: The audio signals are converted into log-Mel spectrograms (n_mels=128), which serve as a 2D image-like representation of the audio.

Model Architecture: A CNN is used for classification. The architecture includes:

Multiple Conv2D layers (32, 64, 128 filters)

BatchNormalization after each convolution for stable training.

MaxPool2D layers for down-sampling.

GlobalAveragePooling2D to reduce feature dimensions.

A final Dense (softmax) layer for 5-class classification.

Training: The model is trained using the Adam optimizer and sparse_categorical_crossentropy loss. The following callbacks are used for efficiency and performance:

ModelCheckpoint: Saves only the best model (best_model.h5) based on validation accuracy.

ReduceLROnPlateau: Reduces the learning rate if validation loss stagnates.

EarlyStopping: Stops training early if the model stops improving, preventing overfitting.

Submission: The script loads the test file IDs from sample_submission.csv, processes them, and uses the trained best_model.h5 to generate a submission.csv file in the correct format for the competition.

3. Project Structure

To run this project, your files should be organized as follows:

.
|
+-- train/
|   +-- train/
|       +-- dog_bark/
|       |   +-- 100032-3-0-0.wav
|       |   +-- ...
|       +-- drilling/
|       |   +-- 103199-4-0-0.wav
|       |   +-- ...
|       +-- engine_idling/
|       +-- siren/
|       +-- street_music/
|
+-- test/
|   +-- test/
|       +-- 100852-0-0-0.wav
|       +-- 100914-3-0-0.wav
|       +-- ...
|
+-- Audio_Classification_Kaggle.py  
+-- sample_submission.csv          
+-- README.md                       


4. Requirements

This script requires Python 3 and the following libraries. You can install them all using pip:

pip install numpy pandas scikit-learn matplotlib librosa tensorflow soundfile tqdm


5. How to Run

Prepare Data: Ensure your train and test folders and the sample_submission.csv file are in place as shown in the Project Structure section.

Update Paths (if needed): Open Audio_Classification_Kaggle.py and verify that the TRAIN_DIR and TEST_DIR constants point to the correct locations.

Execute Script: Run the script from your terminal:

python Audio_Classification_Kaggle.py


Get Output: The script will:

Load and process all training data.

Train the CNN model (this will take some time).

Save the best performing model as best_model.h5.

Print a final validation accuracy and classification report to the console.


Task 2: Generative Audio Synthesis with a CGAN

Author: Vedika Vishwakarma

1. Project Description

This project addresses the second task: designing a deep learning–based generative model. It uses a Conditional Generative Adversarial Network (CGAN), implemented in PyTorch, to synthesize novel audio samples for the five predefined classes:

dog_bark

drilling

engine_idling

siren

street_music

Instead of classifying existing data, this model learns the underlying data distribution to create new, high-quality, and category-consistent audio clips from a random noise vector and a class label.

2. Methodology

The generative pipeline is built around a CGAN, which consists of two competing neural networks: a Generator and a Discriminator.

Framework: This model is built using PyTorch and torchaudio.

Data Preprocessing:

A custom TrainAudioSpectrogramDataset loads the .wav files from the same train/train directory as Task 1.

Audio is converted to log-Mel spectrograms (n_mels=128).

Spectrograms are padded/truncated to a fixed length (max_frames=512).

The class label is provided as a one-hot encoded vector, which is the "condition" in the CGAN.

Model Architecture (CGAN):

Generator (CGAN_Generator): Its goal is to create realistic spectrograms.

Input: A random noise vector (from latent_dim) combined with a one-hot class label.

Output: A 2D log-Mel spectrogram (1x128x512).

Architecture: Uses ConvTranspose2d (deconvolution) layers to upsample the input vector into a full-sized image (spectrogram).

Discriminator (CGAN_Discriminator): Its goal is to identify "fake" (generated) spectrograms.

Input: A spectrogram (either real or fake) combined with its corresponding one-hot class label.

Output: A single probability (logit) indicating if the input spectrogram is real or fake.

Architecture: Uses standard Conv2d layers to downsample the spectrogram into a single decision.

Adversarial Training:

The Discriminator is trained to correctly label real spectrograms as "real" and fake spectrograms (from the Generator) as "fake".

The Generator is trained to produce spectrograms that "fool" the Discriminator into thinking they are "real".

This process repeats, with both networks improving over time, resulting in a Generator capable of producing highly realistic data.

Audio Synthesis (Griffin-Lim):

A 2D spectrogram is not audio. To synthesize the final waveform:

The Generator's output (log-Mel spectrogram) is converted back to a linear-scale spectrogram using InverseMelScale.

The Griffin-Lim algorithm (torchaudio.transforms.GriffinLim) is then used. This is a classic signal processing technique that estimates the phase of the audio (which was lost during the spectrogram conversion) and reconstructs the 1D audio waveform from the 2D linear spectrogram.

3. Project Structure

This script uses the same train data as Task 1 but creates new output directories.

.
|
+-- train/
|   +-- train/              <-- (Source data)
|       +-- dog_bark/
|       +-- drilling/
|       +-- ...
|
+-- gan_generated_audio/    
|   +-- dog_bark_ep1.wav
|   +-- siren_ep1.wav
|   +-- ...
|
+-- gan_spectrogram_plots/ 
|   +-- epoch_001.png
|   +-- ...
|
+-- Audio_Generation_CGAN.py   
+-- README-Task2-Generative.md  <-- (This file)


4. Requirements

This script requires Python 3 and the following PyTorch-related libraries:

pip install torch torchaudio torchvision matplotlib tqdm


5. How to Run

Prepare Data: Ensure your train/train folder (from Task 1) is available.

Update Paths (Important!): Open your generative script and check the BASE_PATH variable. The script currently points to /content/train/train/, which is common for Google Colab. You must change this to match your local path (e.g., train/train).

Execute Script: Run the script from your terminal:

python Audio_Generation_CGAN.py


Get Output:

The script will begin training the GAN. This is a very computationally intensive process and will take a long time.

You will see the epoch progress and losses (loss_D, loss_G) in your terminal.

Crucially: After each epoch, the script will automatically:

Save a sample .wav file for each class in the gan_generated_audio/ folder.

Save a plot of the generated spectrograms in the gan_spectrogram_plots/ folder.

You can listen to the generated .wav files to hear the model's progress as it learns.

Generate the final submission.csv file in the root directory.

You can then submit this submission.csv file to the Kaggle competition.
