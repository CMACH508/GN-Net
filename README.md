# A deep learning model to assist clinicians in predicting the rupture of intracranial aneurysms

This repository contains the source code, trained models and the test sets for GNNet proposed in this paper.

## Introduction

Predicting the rupture of intracranial aneurysms (IAs) is critical for clinicians to make the treatment strategy. Our study aims to build an artificial intelligence model based on automatically obtained geometric features and neighbor features respectively to assist clinicians in predicting the rupture of IAs.

In this retrospective study, 423 patients with 449 middle cerebral artery (MCA) aneurysms detected by computed tomography angiography from January 2009 to June 2020 were enrolled. Another independent data including 57 MCA aneurysms, was used for external validation. We devised an artificial intelligence model to extract the geometric and neighbor features of an aneurysm, which were used to predict the rupture of it. Four clinicians estimated the rupture of the aneurysm on a test set of 100 examinations with and without model assistance. Accuracy, sensitivity, and specificity were measured to evaluate the performance of the model. 

The GN-Net model consisting of geometric branch and neighbor branch had the highest AUC of 94.59 (95% CI, 89.46-99.72), 92.53 (95% CI, 89.25-95.81), and 94.32 (95% CI, 88.51-100.00) on the balanced, unbalanced, and external test set respectively. With and after model assistance, the clinicians’ mean accuracy, mean sensitivity, and mean specificity increased statistically significantly (P<0.05).

Our model achieved high accuracy and generalizable performance for aneurysm rupture prediction. The model significantly improves the predictive performance of clinicians, which may help clinicians make the treatment strategy of informed and transversal aneurysm patients in the clinical practice.

## Method

### Model overview

We devise two branches to capture the geometric features and neighbor features respectively and then aggregate them to predict the rupture of an aneurysm, as illustrated in the figure below (GN-Net). In the geometric branch, the input is the 3D point cloud of an aneurysm and arteries connected to it. The output is a 256-dim geometric feature vector extracted by a geometric deep learning architecture called GeoCNN. In the neighbor branch, a voxel cube cropped from 3D CTA data around the aneurysm is processed by a 3D CNN and a Transformer Encoder, and a 256-dim neighbor feature vector is output. Then, we concatenate the features extracted from the two branches and obtain a 512-dim feature vector, which is fed into an MLP classifier for rupture prediction.

<img src=".\figs\overview.jpg" width="100%"/>

To train GN-Net, run

```
python train.py
```

To test GN-Net on balanced test set, run

```
python test.py
```



### Clinical study

We conducted a diagnostic accuracy study comparing clinicians' performance indicators with and without model assistance, as shown in the figure below. Four clinicians (one radiologist A and one neurosurgeon B who have worked for more than ten years; one radiologist C and one neurosurgeon D who have worked for less than ten years) participated in the study to diagnose a test set of 100 examinations. The clinicians were blinded to the information of imaging examinations which were sorted in random order. Each aneurysm image has a label indicating the size of the aneurysm. For the model assistance group, a label indicating whether the aneurysm is ruptured and the risk of rupture will be added to the images. Firstly, four clinicians read the examinations without model assistance. Then, after a washout period of two weeks, four clinicians read the examinations with model assistance. Finally, after another washout period of two weeks, four clinicians read the examinations without model assistance again. Clinicians were instructed to estimate whether the given aneurysms have a risk to rupture.

<img src=".\figs\studydesign.jpg" width="100%"/>

The result of individual clinician improvement is shown in the figure below, in which the blue dot represents performance before model assistance, the orange dot represents performance with model assistance, and the green dot represents performance after model assistance.

<img src=".\figs\result.jpg" width="100%"/>