# SLOPE-KP

<div align="center">
  <img src="images/slope_kp.png" alt="SLOP-KP: Self-supervised Learning of Object Pose Estimation Using Keypoint Prediction." />
  <p><b>Figure 1:</b> A BIOSCAN-5M dataset sample with multimodal data types.
</div>


Overview
--------
This repository hosts the code for the **SLOPE_KP**: Self-supervised Learning of Object Pose Estimation Using Keypoint Prediction
which outlines advancements in predicting both object pose (camera perspective) and shape from **single images**. 
The key innovation is a novel approach to predicting camera pose using self-supervised learning of keypoints—specific 
points located on a deformable shape that is typical for a particular object category (like birds, cars, etc.).

We conducted our experiments to compare four different approaches to rotation representation. 
- The first is to predict 4D unit quaternions by a CNN.
- The second is 6D rotation representation mapped onto SO(3) via a partial Gram-Schmidt procedure.
- The third is special orthogonalization using SVD based on 9D rotation representation. 
- The fourth is our novel approach to camera pose prediction, which trains an intermediate **keypoint prediction network**.

<h3>3D Shape Prediction</h3>

<div align="center">
  <img src="images/SLOPE_KP_fig3.png" alt="3D Shape Reconstruction" width="500" />
  <p><b>Figure:</b> The first row displays the original images of nine different bird species from the CUB dataset. <br />
  The second row presents the 3D meshes reconstructed using the ground-truth camera poses provided by the dataset through Structure from Motion (SfM). <br />
  The third row showcases the 3D shapes reconstructed when camera poses are predicted using unit quaternions. <br />
  Finally, the fourth row illustrates the 3D shapes obtained using camera poses predicted from keypoint correspondences.</p>
</div>

<h3>Texture Prediction</h3>

<div align="center">
  <img src="images/SLOPE_KP_fig4.png" alt="Texture Reconstruction" />
  <p><b>Figure:</b> The first row displays original images of six different bird species from the CUB dataset. <br />
  The second row shows textures reconstructed using the SfM camera poses for rendering. <br />
  The third and fourth rows present textures reconstructed with camera poses predicted by unit quaternions and the keypoint pose trainer, respectively.</p>
</div>

<h3>Mask Prediction</h3>

<div align="center">
  <img src="images/SLOPE_KP_fig5.png" alt="Mask Reconstruction" />
  <p><b>Figure:</b> The first row displays original RGB images of six different bird species from the CUB dataset. <br />
  The second row presents the ground-truth masks provided by the dataset. <br />
  The third row shows rendered masks using SfM camera poses. <br />
  The fourth and fifth rows depict reconstructed masks using camera poses predicted by unit quaternions and keypoint correspondences, respectively.</p>
</div>

<h3>Image Reconstruction</h3>

<div align="center">
  <img src="images/SLOPE_KP_fig6.png" alt="Image Reconstruction"/>
  <p><b>Figure:</b> The first row displays original images of ten different bird species from the CUB dataset. <br />
  The second row presents the ground-truth annotations provided by the dataset. <br />
  The third and fourth rows show the masks and textures reconstructed using camera poses predicted by keypoints. <br />
  The fifth row depicts the 3D shape reconstructed from the camera pose predictions.</p>
</div>


### Online Inference 3D Object Reconstruction from Videos

We conduct online experiments to infer 3D objects from video sequences with single and multiple objects per image, using the YouTubeVos and Davis datasets [Xu et al. (2018)](https://arxiv.org/abs/1809.03327), focusing on the bird category. Inferring objects from video sequences is challenging due to varying positions, orientations, and occlusions. We use LWL [Bhat et al. (2020)](https://arxiv.org/abs/2003.11540) to compute bounding boxes from the predicted masks.

These bounding boxes are used to crop frames and create patches. The image patches are then input to the reconstruction network, which predicts shape, texture, and camera pose. We compare the masks reconstructed by our method and three other approaches against the ground-truth masks. Models are evaluated using three metrics: Jaccard-Mean (mean intersection over union), Jaccard-Recall (mean fraction of values exceeding a threshold), and Jaccard-Decay (performance loss over time).

<div align="center">
  <img src="images/SLOPE_KP_online.png" alt="Online Video Object Reconstruction"/>
</div>

**Figure**: Online video object reconstruction framework. This example is from the YouTubeVos test set.  
The first row shows the original RGB images, and the second row shows the image patches generated by cropping using predicted bounding boxes from the LWL tracker.  
The third row shows the reconstructed shape and texture.
