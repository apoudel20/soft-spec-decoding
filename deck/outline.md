# Gister: outline of the deck

The slides themselves are in `deck/slides/`. This file lists what each one covers.

1. Cover. Gister drafts by sampling latents of Qwen3-8B's future hidden states. The target verifies with the same greedy
   rule as DFlash. Results are from one RTX 4070 Ti.
2. Method. Stage 1 autoencoder (8 positions, layers 18/24/30/36, 2,048-d latent, frozen LM head). Stage 2 flow
   sampler (2 Euler steps). K samples merged into a tree and verified in one pass. Related work: ChainFlow, FlashMap,
   MFD, DBLast, CALM, DFlash and DDTree.
3. Stage 1. Top-1 99.4 / 99.0 / 97.3%, oracle 8.5 / 8.5 / 7.6 accepted per round. Misses its gate on top-1 and total
   variation, so greedy only. The latent mostly encodes the 8 tokens.
4. Stage 2, deterministic. 2.17 to 3.08 on math. Target hidden states give the largest gain.
5. Sampling. Best of 16 is 36 / 38 / 32% above deterministic. Upper bound for a K-branch tree.
6. Scale and context. 285M on 6M positions gives 3 to 7%. 56 earlier positions give +5% on math only.
7. End to end. Chain 1.34x / 2.05x / 1.40x, trees slower than the chain, DFlash 2.69x / 4.73x / 7.89x.
8. Comparison with DFlash, plus published numbers on other hardware.
9. Examples of sampled drafts from results/examples.json.
10. Proposed work. A latent over 64 to 256 tokens trained by latent prediction. Tests: oracle latent, predicted
    latent, trees at equal node count, scale.
