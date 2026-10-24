# Blood-Brain Barrier (BBB) Permeability Prediction

This project focuses on predicting blood-brain barrier (BBB) permeability using molecular representations, physicochemical descriptors, and transporter-related information.

## Models

- **GraphB3:** Uses graph-based molecular representations with atomic-level features to predict BBB permeability.
- **Extra Trees:** Uses molecular-level physicochemical descriptors generated using RDKit to predict BBB permeability.
- **Transporter-Aware Model:** Separately investigates transporter-related effects on BBB permeability, focusing on substrate prediction.

## Research Gap

Conventional BBB permeability prediction approaches primarily rely on molecular structures and physicochemical properties, potentially overlooking the influence of transporter proteins. Additionally, different molecular feature representations may capture different aspects of permeability-related properties.

## Objectives

- Compare atomic-level features with molecular-level physicochemical descriptors for BBB permeability prediction.
- Evaluate the performance of GraphB3 and Extra Trees using their respective feature representations.
- Investigate whether incorporating transporter-related information can improve substrate permeability prediction.

## Approach

The project follows two separate directions:

1. **Molecular Feature-Based Prediction:** GraphB3 learns from atomic-level features and molecular graph structures, while Extra Trees uses molecular-level RDKit descriptors. Their performance is compared to understand how different feature representations affect BBB permeability prediction.

2. **Transporter-Aware Prediction:** A separate model investigates transporter–molecule interactions, distinguishing substrates from inhibitors where relevant, to study their role in BBB permeability prediction.

## Technologies

- Python
- PyTorch
- PyTorch Geometric
- RDKit
- Scikit-learn
- Pandas and NumPy

## Expected Outcome

A comparative evaluation of atomic-level and molecular-level feature representations, alongside an independent investigation of transporter-aware modeling for BBB permeability and substrate prediction.
