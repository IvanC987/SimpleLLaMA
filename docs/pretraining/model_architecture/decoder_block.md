# Transformer Decoder Block

The **decoder block** is the fundamental building unit of the transformer.  
Each block combines **attention**, **feedforward networks**, **normalization**, and **residual connections** into a repeatable structure.

---

## Structure of a Decoder Block

A decoder block has two main parts:

1. **Multi-Head Self-Attention (MHA)** → lets tokens exchange information.  
2. **Feedforward Network (FFN)** → transforms the attended features into richer representations.  

Surrounding these are:  
- **RMSNorm** → stabilizes training by normalizing activations.  
- **Residual Connections** → ensure information from earlier layers isn’t lost.  

The primary block flow is:

```
Input → Norm → Attention → Residual → Norm → Feedforward → Residual → Output
```

This **“pre-norm” setup** (normalize before each sub-layer) is known to improve stability in deep transformers.

---

## Example Walkthrough





---

## In This Project

- **Attention type**: defaults to standard multi-head self-attention, with optional MLA.  
- **Normalization**: RMSNorm everywhere.  
- **Activation**: SiLU-based feedforward (SwiGLU).  
- **Dropout**: enabled, mainly for finetuning stages like SFT/RLHF.  
- **Residuals**: used after both sub-layers.  

Together, these form the repeating backbone of the SimpleLLaMA model.

