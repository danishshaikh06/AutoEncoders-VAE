# Variational Autoencoder (VAE) & Autoencoder Implementation from Scratch

This project implements both a **standard Autoencoder (AE)** and a **Variational Autoencoder (VAE)** from scratch using PyTorch. The goal is to understand how neural networks learn compressed representations of data and how VAEs improve upon traditional autoencoders for generative tasks.

---

# 📌 What is an Autoencoder?

An **Autoencoder** is a neural network that learns to compress input data into a smaller representation and then reconstruct it back.

It has two main parts:

### 🔹 Encoder
The encoder compresses the input into a smaller vector (latent representation).

- Input: original data (e.g., image)
- Output: compressed representation (latent vector)

---

### 🔹 Decoder
The decoder reconstructs the original input from the compressed representation.

- Input: latent vector
- Output: reconstructed image

---

### 📉 Limitation of Autoencoders
- They learn a **fixed encoding**
- Latent space is **not structured**
- Cannot reliably generate new meaningful data
- If we sample random points from latent space → output is usually **gibberish images**

---

# 📌 What is a Variational Autoencoder (VAE)?

A **VAE** improves upon the autoencoder by making the latent space **probabilistic and structured**.

Instead of mapping input → single point, it maps input → distribution.

---

## 🔹 Latent Space

Latent space is the compressed representation of data.

In VAE:
- Each input is mapped to a **distribution**, not a fixed vector
- This allows smooth interpolation and generation

---

## 🔹 Why Gaussian Distribution?

VAEs assume latent variables follow a **Gaussian (Normal) distribution**:

### Why?

- It is smooth and continuous
- Easy to sample from
- Makes latent space well-behaved
- Ensures meaningful interpolation between points

---

## 🔹 Reparameterization Trick

We cannot directly backpropagate through random sampling.

So we use:

z = mu + eps*logvar

### Why this works:
- Separates randomness from learnable parameters
- Allows gradients to flow during training
- Makes training possible using backpropagation

---

## 🔹 KL Divergence

KL divergence measures how different two distributions are.

In VAE, it ensures:

\[
q(z|x) \approx \mathcal{N}(0,1)
\]

### Purpose:
- Forces latent space to follow Gaussian distribution
- Prevents overfitting to training data
- Makes generation possible

---

# 📌 Why VAE is better than Autoencoder

| Feature | Autoencoder | VAE |
|--------|------------|-----|
| Latent Space | Unstructured | Structured (Gaussian) |
| Sampling new data | Poor | Good |
| Generation quality | Often gibberish | Meaningful images |
| Backprop through sampling | Not applicable | Possible (via reparameterization) |

---

# 📌 Key Insight

- Autoencoders are good for compression
- VAEs are good for **generation**

---

# 📌 Important Observation

When using a standard autoencoder:
- Random latent vectors produce **noisy / meaningless images**

When using a VAE:
- Random sampling from latent space produces **coherent and realistic images (not full gibberish)**

---

# 📌 Technologies Used

- Python
- PyTorch
- Matplotlib
- FashionMNIST dataset

---

# 📌 Results

Include:
- Reconstruction images
- Generated samples
- Loss curves

---

# 📌 Conclusion

This project demonstrates how introducing probabilistic modeling (VAE) significantly improves the ability of neural networks to generate meaningful data compared to traditional autoencoders.

---
