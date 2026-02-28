# ClearSpeak---A-Neural-Network-for-Detection-of-Stuttering-in-Pediatric-Populations
# Overview

ClearSpeak is a lightweight neural network framework designed to detect early stuttering biomarkers from real-world speech recordings using consumer hardware.

Unlike traditional binary systems that classify speech as “stutter” or “no stutter,” ClearSpeak performs multi-label classification, identifying multiple types of disfluencies within a single speech sample. The system is optimized for high precision, class imbalance correction, and real-time offline deployment.

# The Problem

Developmental stuttering affects millions of children worldwide. Early intervention significantly improves long-term outcomes, yet many children remain undiagnosed due to:

High cost of expert evaluations

Limited access to speech-language pathologists

Inconsistent and varied stuttering behaviors

Lack of scalable, at-home screening tools

Existing machine learning approaches often rely on small datasets, struggle with real-world noise, or are not deployable outside research environments.

ClearSpeak addresses these gaps.

# Research Question

Can a lightweight neural network accurately detect multiple stuttering types from noisy, real-world speech while remaining efficient enough to run on consumer devices?

# Dataset

Model trained using the UCLASS clinical speech dataset.

Initial dataset characteristics:

~6,000 labeled speech segments

Severe class imbalance

~4,000 fluent samples

100–300 samples for certain stutter types (e.g., blocks, prolongations)

To address imbalance:

Oversampling techniques expanded the dataset to 32,300 balanced samples

This ensured rare but clinically significant stutter types were learned effectively

# Stutter Types Detected

ClearSpeak performs multi-label classification across six categories:

Blocks

Sound repetitions

Word repetitions

Prolongations

Interjections

No stutter (fluent speech)

Multi-label classification is significantly more challenging than binary detection because the model must correctly identify multiple disfluency types simultaneously.

# Methodology
1. Audio Preprocessing

Resampling and normalization

Segmentation into fixed-length clips

Noise handling for real-world robustness

2. Feature Extraction

Mel-Frequency Cepstral Coefficients (MFCCs)

Temporal timing features

Spectral pattern representations

3. Class Imbalance Handling

Dataset expansion from 6k to 32.3k samples

Oversampling applied to underrepresented stutter categories

4. Model Architecture

Lightweight feedforward neural network

Optimized for speed and deployment efficiency

Designed to minimize overfitting

5. Evaluation

Multi-label performance metrics

Micro-F1 score used for balanced precision/recall assessment

Confusion matrices generated per class

# Results

Micro-F1 Score: 0.93

Strong precision and recall across stutter types

Achieved 100% accuracy on fluent speech in repeated testing

Zero false positives for non-stuttered speech

Stable training curves with minimal overfitting

Remaining classification challenges primarily involve acoustically similar repetition types and very short, low-energy speech events.

# Real-World Deployment

ClearSpeak is designed for offline, real-time inference.

Workflow:

Speech is recorded via microphone

Audio is preprocessed and converted into acoustic features

Neural network performs multi-label inference

Output probabilities are converted into interpretable feedback

The model can be exported for CoreML deployment and integrated into iOS applications using AVFoundation.

# Future Work

ClearSpeak is evolving beyond detection into intervention:

Real-time biofeedback during speech

Adaptive breathing guidance

Gamified fluency exercises

Personalized models that adapt to individual speech patterns

Hybrid deep learning upgrades (LSTM / Transformer architectures)

Multilingual dataset expansion

The long-term vision is a globally accessible digital speech companion that supports early identification and ongoing fluency development.
