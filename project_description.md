# VisualDebugger: Can VLMs Spot Their Own Hallucinations?

**02501 Advanced Deep Learning in Computer Vision**

*Dimitrios Papadopoulos*
Associate Professor, DTU Compute

*17 March, 2026*

**Keywords/lectures:** Vision Transformers, Equivariance, Robustness, DINOv2, CLIP

---

## 1 Project Description

Vision Language Models (VLMs) hallucinate. They confidently describe objects that don't exist, invent spatial relationships, and misread text in images. The standard fix is to train a separate reward model or verifier to catch these errors [1, 2]. But this is expensive and creates the **"ouroboros problem"** [3]: you need a good model to generate good training data for the verifier, but you need a good verifier to train a good model.

The idea here is to inspect if we can teach VLMs to debug their own outputs by looking at the image a second time. The key insight is that VLMs are much better at verification than generation [4]. They can often tell that "there is no red car in this image" even though they initially said there was one. But current VLMs never re-examine their own outputs against the visual evidence.

The goal of this project is to introduce **VisualDebugger**, a two-pass system where the VLM first generates a response, then receives its own response alongside the original image, and is trained via RL (GRPO) to identify and correct hallucinated claims. Unlike prior self-correction work [4] that only uses prompting, the idea is to train the verification skill using GRPO, which teaches the model to systematically re-examine its response claim-by-claim. Unlike external verifiers [1, 2], this requires no additional model since the same VLM does generation and verification.

---

## 2 Data

You can use established hallucination benchmarks:
- **POPE** [5] — object existence
- **CHAIR** [6] — caption hallucination
- **MMHal-Bench** [7] — multi-type hallucination
- **HallusionBench** [8] — advanced diagnostic suite

For training the verifier, you could consider **POVID** [9] and **RLHF-V** [10], which provide pairs of hallucinated vs. correct responses with annotations of exactly which claims are false.

Construct a self-verification training set: for each image, generate 5 responses from the base VLM, use ground truth annotations to label each sentence as hallucinated or correct, and train the same VLM to predict these labels given (image, response).

Choose small VLMs (1–3B parameters), e.g., **Qwen2.5-VL** or **Qwen3VL** [11]. Training with LoRA can be done on 1 GPU; consider **veRL** [12] for GRPO training.

---

## 3 Tasks

### Task 1: Hallucination Taxonomy and Baseline

Generate 5,000 responses from the base VLM (Qwen2.5-VL-3B) on COCO and Visual Genome images. Using ground truth annotations, classify each hallucination into types:

- **(a)** Object existence — e.g., "there is a cat" when there isn't
- **(b)** Attribute error — wrong color, size, or material
- **(c)** Spatial error — wrong position or relationship
- **(d)** Counting error
- **(e)** Text/OCR error
- **(f)** Action/event error

Then, test **zero-shot self-verification**: show the VLM its own response + the image and ask *"which claims are incorrect?"* Measure precision/recall per hallucination type. How good is the VLM at catching its own errors without any training?

---

### Task 2: RL-Trained Self-Verification (GRPO)

Train the verification skill using GRPO. The VLM receives (image, its own previous response) and must output a sentence-by-sentence judgment: **CORRECT** or **HALLUCINATED** for each claim.

**Reward function:**
- `+1` for each correctly identified hallucination and correctly confirmed truth
- `-1` for each miss or false alarm

Generate K=4 candidate verifications per example; GRPO uses relative rewards to update.

**Compare:**
- (a) Zero-shot prompting
- (b) SFT on verification data
- (c) SFT + GRPO

Track verification precision, recall, and F1 across hallucination types over training.

---

### Task 3: Closing the Loop — Generate, Verify, Correct

Build the full pipeline:
1. VLM generates a response
2. The same VLM verifies claim-by-claim using the trained verifier
3. VLM regenerates only the flagged claims conditioned on the verification feedback

Does the generate–verify–correct loop outperform:
- (a) Single-pass generation?
- (b) Best-of-N sampling (generate 5, pick best)?
- (c) External verifier (LLM-as-a-judge)?

---

### Task 4: *(TBD)*

### Task 5: *(TBD)*

---

## References

[1] Lei Li, et al. *VL-RewardBench: A Challenging Benchmark for Vision-Language Generative Reward Models.* In CVPR, 2025.

[2] *TLDR: Token-Level Detective Reward Model for Vision-Language Tasks.* In ICLR, 2025.

[3] Zihao Lin, et al. *VL-GenRM: Enhancing Vision-Language Verification via Vision Experts and Iterative Training.* arXiv:2506.13888, 2025.

[4] Siyun Liao, et al. *Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves?* In CVPR, 2025.

[5] Yifan Li, et al. *Evaluating Object Hallucination in Large Vision-Language Models (POPE).* In EMNLP, 2023.

[6] Anna Rohrbach, et al. *Object Hallucination in Image Captioning.* In EMNLP, 2018.

[7] Zhiqing Sun, et al. *Aligning Large Multimodal Models with Factually Augmented RLHF.* In CVPR, 2024.

[8] Tianrui Guan, et al. *HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion.* In CVPR, 2024.

[9] Zhiqing Sun, et al. *POVID: Preference Optimization with Visual Information from Descriptions.* arXiv, 2024.

[10] Tianyu Yu, et al. *RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-Grained Correctional Human Feedback.* In CVPR, 2024.

[11] Chenfei Wu, et al. *Qwen-Image Technical Report.* arXiv:2508.02324, 2025.

[12] veRL Team. *veRL: An Open-Source Framework for RL Training of VLMs.* GitHub, 2025.