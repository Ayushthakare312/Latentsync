# Latentsync
LatentSync: Audio-Conditioned Latent Diffusion Models for Lip Sync
📖 Abstract
LatentSync is an advanced lip-sync framework that uses audio-conditioned latent diffusion models to synchronize lip movements with audio. Unlike traditional methods that rely on intermediate motion representations or pixel-space diffusion, LatentSync directly models complex audio-visual correlations using Stable Diffusion.

A key challenge in diffusion-based lip-sync methods is temporal inconsistency across frames. To address this, we introduce Temporal Representation Alignment (TREPA), which enhances temporal consistency while maintaining lip-sync accuracy. TREPA leverages large-scale self-supervised video models to align generated frames with ground truth frames.

🏗️ Framework Overview
LatentSync’s framework consists of the following components:

Audio Embeddings: Whisper converts mel-spectrograms into audio embeddings, which are integrated into the U-Net via cross-attention layers.

Input Processing: Reference and masked frames are channel-wise concatenated with noised latents as input to the U-Net.

Training: A one-step method predicts clean latents from noise, which are decoded to produce clean frames.

Loss Functions: The framework uses TREPA, LPIPS, and SyncNet losses in the pixel space to optimize performance.

🎬 Demo
Check out the lip-synced results for the following videos:


Note: Photorealistic videos are filmed by contracted models, while anime videos are sourced from VASA-1 and EMO.

📑 Open-Source Plan
We plan to release the following resources:

Inference code and checkpoints

Data processing pipeline

Training code

🔧 Setting Up the Environment
To set up the environment:

Install the required packages and download checkpoints by running:

bash
Copy
source setup_env.sh
After successful setup, the checkpoints will be organized as follows:

Copy
./checkpoints/
|-- latentsync_unet.pt
|-- latentsync_syncnet.pt
|-- whisper
|   `-- tiny.pt
|-- auxiliary
|   |-- 2DFAN4-cd938726ad.zip
|   |-- i3d_torchscript.pt
|   |-- koniq_pretrained.pkl
|   |-- s3fd-619a316812.pth
|   |-- sfd_face.pth
|   |-- syncnet_v2.model
|   |-- vgg16-397923af.pth
|   `-- vit_g_hybrid_pt_1200e_ssv2_ft.pth
These include all necessary checkpoints for training and inference. For inference-only use, download latentsync_unet.pt and tiny.pt from our HuggingFace repo.

🚀 Inference
You can perform inference in two ways (requires 6.5 GB VRAM):

1. Gradio App
Run the Gradio app for interactive inference:

bash
Copy
python gradio_app.py
2. Command Line Interface
Use the CLI for inference:

bash
Copy
./inference.sh
You can adjust parameters like inference_steps and guidance_scale to explore different results.

🔄 Data Processing Pipeline
The data processing pipeline includes the following steps:

Remove broken video files.

Resample video FPS to 25 and audio to 16,000 Hz.

Detect scenes using PySceneDetect.

Split videos into 5-10 second segments.

Filter videos:

Remove videos with faces smaller than 256×256 pixels.

Remove videos with more than one face.

Affine transform faces based on landmarks detected by face-alignment, then resize to 256×256.

Remove videos with sync confidence scores below 3 and adjust audio-visual offsets to 0.

Calculate HyperIQA scores and remove videos with scores below 40.

Run the pipeline using:

bash
Copy
./data_processing_pipeline.sh
Specify the input directory with the input_dir parameter. Processed data will be saved in the high_visual_quality directory.

🏋️‍♂️ Training U-Net
Prerequisites:
Process data using the pipeline described above.

Download all required checkpoints.

Steps:
Train the U-Net using:

bash
Copy
./train_unet.sh
Modify the U-Net config file to specify:

Data directory

Checkpoint save path

Training hyperparameters

🏋️‍♂️ Training SyncNet
To train SyncNet on your dataset:

Use the same data processing pipeline as for U-Net.

Run the training script:

bash
Copy
./train_syncnet.sh
After training, loss charts will be saved in train_output_dir.

📊 Evaluation
Sync Confidence Score
Evaluate the sync confidence score of a generated video:

bash
Copy
./eval/eval_sync_conf.sh
SyncNet Accuracy
Evaluate SyncNet accuracy on a dataset:

bash
Copy
./eval/eval_syncnet_acc.sh
🙏 Acknowledgements
Our code is built on AnimateDiff.

Some code is borrowed from MuseTalk, StyleSync, SyncNet, and Wav2Lip.

We thank the open-source community for their contributions.
