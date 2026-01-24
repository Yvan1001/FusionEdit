<div align="center">
  
# FusionEdit: Semantic Fusion And Attention Modulation for Training-Free Image Editing

[Yongwen Lai](https://openreview.net/profile?id=~Yongwen_Lai1)<sup>1</sup>,
[Chaoqun Wang](https://openreview.net/profile?id=~Chaoqun_Wang3)<sup>1</sup>,
[Shaobo Min](https://openreview.net/profile?id=~Shaobo_Min2)<sup>2</sup>

<sup>1</sup> South China Normal University,  <sup>2</sup> University of Science and Technology of China



<div align="left">

<p>

Text-guided image editing aims to modify specific regions according to the target prompt while preserving the identity of the source image. Recent methods exploit explicit binary masks to constrain editing, but hard mask boundaries introduce artifacts and reduce editability. 

To address these issues, we propose <strong>FusionEdit</strong>, a training-free image editing framework that achieves precise and controllable edits. 

First, editing and preserved regions are automatically identified by measuring semantic discrepancies between source and target prompts. To mitigate boundary artifacts, FusionEdit performs distance-aware latent fusion along region boundaries to yield the soft and accurate mask, and employs a total variation loss to enforce smooth transitions, obtaining natural editing results. Second, FusionEdit leverages AdaIN-based modulation within DiT attention layers to perform a statistical attention fusion in the editing region, enhancing editability while preserving global consistency with the source image. Extensive experiments demonstrate that our FusionEdit significantly outperforms state-of-the-art methods.


## Performance Note
For a single image editing task, the 30 second runtime is dominated by ODE solver on an RTX 5880 Ada GPU.

# 🛠️ Code Setup
The environment of our code is the same as FLUX, you can refer to the [official repo](https://github.com/black-forest-labs/flux/tree/main) of FLUX, or running the following command to construct the environment.


<div align="left">

```
conda create --name FusionEdit python=3.10
conda activate FusionEdit 
pip install -r requirements.txt
python FusionEdit.py
```

<div align="center">

# Contact
The code in this repository is still being reorganized. Errors that may arise during the organizing process could lead to code malfunctions or discrepancies from the original research results. If you have any questions or concerns, please send emails to 2024025439@m.scnu.edu.cn.
