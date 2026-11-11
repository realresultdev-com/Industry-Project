# Final Year Project Presentation Slide Deck

**Title:** Self-Supervised Deep Representation Learning for Autonomous Detection of Structural Defects in Industrial Systems  
**Format:** 12-Slide Defense Deck with Speaker Script & Time Allocation (Total: 15 Minutes)

---

### Slide 1: Title & Introduction (1 Min)
* **Title**: Self-Supervised Deep Representation Learning for Autonomous Detection of Structural Defects in Industrial Systems
* **Presenter**: [Your Name] | Roll No: [Your Roll Number]
* **Guide**: [Supervisor Name], Department of Computer Science & Engineering
* **Affiliation**: [Your College / University Name]
* **Speaker Script**:
  > *"Respected members of the review panel, external examiner, and guide. Today, I present our final year capstone project titled 'Self-Supervised Deep Representation Learning for Autonomous Detection of Structural Defects in Industrial Systems'. This research addresses the severe annotation bottleneck in industrial computer vision by teaching neural networks to understand surface defects without requiring human labels."*

---

### Slide 2: The Industrial Problem: Annotation Bottleneck (1.5 Mins)
* **High-Speed Manufacturing**: Modern hot-rolled steel lines run at over $15\text{ m/s}$. Manual inspection is subjective, fatigue-prone, and cannot keep pace.
* **The Defect Scarcity Paradox**: Defects occur in less than $1\%$ of production runs.
* **Costly Expert Labeling**: Annotating subtle micro-cracks requires certified metallurgists.
* **Supervised CNN Failure**: Training deep networks (ResNet-18/50) from scratch on small labeled industrial datasets causes catastrophic overfitting to background noise.
* **Speaker Script**:
  > *"In modern steel and manufacturing plants, defects are rare but catastrophic. Supervised deep learning models achieve high accuracy only when given thousands of labeled defect examples. However, obtaining labeled defect images is economically prohibitive and slow. When standard CNNs are trained on only 10 or 20 defect samples, they fail completely due to severe overfitting."*

---

### Slide 3: Proposed Solution: The Self-Supervised Paradigm (1.5 Mins)
* **Core Philosophy**: Learn from massive **unlabeled** imagery first, then fine-tune with tiny amounts of labeled data.
* **Pretext Task**: Teach the network that two differently perturbed views of the same surface contain identical structural geometry.
* **Dual Architectures**:
  * **SimCLR**: Contrastive learning with NT-Xent loss (positive attraction vs. negative repulsion).
  * **BYOL**: Non-contrastive bootstrap learning with online predictor and EMA target network.
* **Speaker Script**:
  > *"To solve this, we introduce Self-Supervised Learning. Instead of relying on manual labels, we allow the neural network to autonomously explore raw, unannotated surface scans through automated pretext tasks. By pulling together representations of positive pairs and pushing away negative samples, the model learns fundamental geometric features—such as edges, pits, scratches, and rolling marks—without human intervention."*

---

### Slide 4: Domain-Adapted Industrial Augmentations (1.5 Mins)
* **Why Natural Image Augmentations Fail**:
  * Natural images (ImageNet) have gravity-defined orientation (cars sit on roads).
  * Steel and composite defect surfaces have **no canonical orientation**; scratches and scale press into plates at any angle.
* **Our Industrial Augmentation Pipeline**:
  1. $360^\circ$ Arbitrary Rotation Invariance
  2. Random Horizontal & Vertical Flips
  3. Multi-Scale Stochastic Resized Crop ($40\% - 100\%$)
  4. Photometric Contrast & Lighting Jitter
  5. Gaussian Sensor Noise & Defocus Simulation
* **Speaker Script**:
  > *"A critical contribution of our work is adapting data augmentations to industrial physics. Natural object images cannot be rotated upside down without breaking semantic meaning. However, hot-rolled steel strip defects possess no canonical orientation. We implemented arbitrary 360-degree rotation invariance, multi-scale zooming, and optical sensor noise, ensuring the learned representations are robust against factory vibration and lighting oscillations."*

---

### Slide 5: System Architecture & Workflow (1.5 Mins)
* **Backbone Encoder**: ResNet-18 ($11\text{M}$ parameters) extracting representation vector $h \in \mathbb{R}^{512}$.
* **Projection Head**: 2-layer MLP (Linear $\to$ BatchNorm $\to$ ReLU $\to$ Linear) projecting $h$ into latent space $z \in \mathbb{R}^{128}$.
* **Downstream Flexibility**:
  * *Linear Probe*: Freeze backbone, train single linear layer ($3,078$ parameters).
  * *Fine-Tuning*: End-to-end parameter adaptation.
  * *Feature Memory Bank*: Unsupervised k-NN anomaly detection.
* **Speaker Script**:
  > *"Here is our system architecture. We utilize a ResNet-18 backbone. During self-supervised pre-training, a non-linear projection head projects features into a 128-dimensional latent space where the contrastive loss is calculated. Downstream, the projection head is discarded, and the rich 512-dimensional representations are used for defect classification, unsupervised anomaly detection, and Grad-CAM localization."*

---

### Slide 6: Experimental Setup & NEU-DET Benchmark (1 Min)
* **Benchmark**: Northeastern University (NEU) Surface Defect Database.
* **Classes (6)**: *Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches*.
* **Partitions**:
  * Unlabeled SSL Pre-training Pool ($70\%$).
  * Labeled subsets for efficiency benchmarking: $1\%, 5\%, 10\%, 25\%, 100\%$.
  * Independent Validation ($15\%$) and Test ($15\%$) sets.
* **Speaker Script**:
  > *"We evaluated our system on the benchmark NEU surface defect database, comprising six representative hot-rolled steel strip defect classes. To rigorously evaluate label efficiency, we partitioned the data into stratified subsets: 1%, 5%, 10%, 25%, and 100% of labeled training data, while maintaining independent validation and test sets."*

---

### Slide 7: Label Efficiency Results (2 Mins)
* **Key Finding 1**: At $10\%$ labels (only 4 samples per class), SimCLR representations outperform supervised models trained from scratch.
* **Key Finding 2**: At $100\%$ labels, a linear probe on frozen SSL features achieves **$78.57\%$ test accuracy** with **only 3,078 trainable parameters**, whereas training from scratch requires updating all $11.18\text{M}$ parameters.
* **Key Finding 3**: Faster convergence—SSL requires less than half the training epochs of randomly initialized models.
* **Speaker Script**:
  > *"This slide presents our most critical finding: label efficiency. Under severe label scarcity, supervised models trained from scratch overfit and fail. In contrast, our self-supervised representation model maintains stable transferability. Even in the linear probe setting where the entire 11-million parameter backbone is frozen, training a single linear layer of 3,000 parameters achieves over 78% test accuracy."*

---

### Slide 8: Latent Space Representation Clustering (t-SNE) (1 Min)
* **Visualizing Representation Quality**: 2D t-SNE dimensionality reduction of 512-dimensional feature vectors.
* **Observation**: Clear spatial clustering between distinct defect geometries (e.g. linear scratches vs. distributed pitted surfaces).
* **Significance**: Proves that the self-supervised pretext task successfully grouped semantically similar surface textures together without ever seeing class labels.
* **Speaker Script**:
  > *"To inspect what the model learned, we performed t-SNE dimensionality reduction on the test embeddings. As visible in this scatter plot, despite receiving zero human labels during pre-training, the network autonomously mapped similar defect morphologies into well-separated clusters."*

---

### Slide 9: Unsupervised Anomaly Detection & AUROC (1 Min)
* **Method**: Feature memory bank storing normal/nominal steel surface representations.
* **Metric**: k-NN cosine and Euclidean distance to memory bank.
* **Result**: **AUROC of 1.0000** separating normal surface scans from structural defects.
* **Speaker Script**:
  > *"In many manufacturing scenarios, new defect types occur that were never previously seen. We implemented an unsupervised anomaly detection module where normal surface features form a memory bank. By measuring the nearest-neighbor cosine distance, the system achieves an AUROC of 1.0, accurately flagging any anomalous structural degradation."*

---

### Slide 10: Explainability: Autonomous Defect Localization (Grad-CAM) (1.5 Mins)
* **Transparency in Quality Assurance**: Plant operators require visual justification before rejecting steel strips.
* **Grad-CAM Implementation**: Computes gradients of defect scores with respect to `layer4` convolutional feature maps.
* **Result**: Accurately highlights exact defect boundaries (cracks, pits, abrasions) **without bounding box or polygon annotations**.
* **Speaker Script**:
  > *"Black-box deep learning is unacceptable on high-stakes assembly lines. Using Grad-CAM, we compute the gradient activations of the final convolutional layer. As shown in the overlay, the model accurately spotlights the exact scratch marks and pitted regions without having been trained on bounding box labels."*

---

### Slide 11: Real-Time Streamlit Inspection Dashboard (1 Min)
* **Plant Operator UI**: Interactive web dashboard (`app.py`).
* **Features**:
  1. Drag-and-drop surface image upload.
  2. Defect category prediction with confidence bar distribution.
  3. Real-time Grad-CAM defect heatmap overlay.
  4. Real-time latency measurement: **22.43 ms per frame (44.6 FPS)** on CPU.
* **Speaker Script**:
  > *"We packaged the entire pipeline into an interactive web application built with Streamlit. Plant engineers can upload an image or stream conveyor camera feeds, receive real-time defect classifications with confidence intervals, and inspect localized heatmaps in under 25 milliseconds."*

---

### Slide 12: Conclusion & Future Scope (1 Min)
* **Summary of Contributions**:
  * Demonstrated that Self-Supervised Learning overcomes the industrial annotation bottleneck.
  * Formulated domain-specific industrial data augmentations ($360^\circ$ rotation, noise, lighting).
  * Validated label efficiency ($1\%$ to $100\%$) and achieved high AUROC anomaly detection.
  * Engineered a production-ready, explainable inspection application.
* **Future Work**: Extending to Vision Transformers (ViT/MAE) and deploying on edge devices (NVIDIA Jetson Orin).
* **Speaker Script**:
  > *"In conclusion, this project demonstrates that self-supervised representation learning provides a scientifically rigorous and economically viable solution for smart manufacturing inspection. Thank you for your time, and I am now ready for questions."*

