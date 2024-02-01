<div align="center">
<h1>A<sup>2</sup>-MAE </h1>
<h3>A spatial-temporal-spectral unified remote sensing foundation model based on anchor-aware masked autoencoder
</h3>

Lixian Zhang<sup>1</sup>, Yi Zhao<sup>1</sup>, Runmin Dong<sup>1</sup>, Jinxiao Zhang<sup>1</sup>, Shuai Yuan<sup>2</sup>, Shilei Cao<sup>3</sup>, Mengxuan Chen<sup>1</sup>, Juepeng Zheng<sup>3</sup>, Weijia Li<sup>3</sup>, Wei Liu<sup>4</sup>, Litong Feng<sup>4</sup>, Haohuan Fu<sup>1</sup>

<sup>1</sup>  Tsinghua University, <sup>2</sup> The University of Hong Kong , <sup>3</sup>  Sun Yat-sen University,  <sup>4</sup> Sensetime

ArXiv Preprint ([TBD](TBD))


</div>


#

## Abstract
Vast amounts of remote sensing (RS) data provide Earth observations across multiple dimensions, encompassing critical spatial, temporal, and spectral information essential for addressing global-scale challenges such as land use monitoring, disaster prevention, and environmental change mitigation. Despite the existence of various pre-training methods tailored to the characteristics of RS data, their inherent limitation lies in their inability to collaboratively harness spatial-temporal-spectral information. To unlock the full potential of RS data, we have curated a spatial-temporal-spectral unified RS dataset (STSUD) comprising over 4.2 million RS images. This dataset is meticulously organized to exhibit diversity in land-use types spatially, changes in landscape temporally, and various band compositions spectrally. Leveraging this structured dataset, we propose an Anchor-Aware Masked AutoEncoder, denoted as A<sup>2</sup>-MAE, for self-supervised learning. A<sup>2</sup>-MAE integrates an anchor-aware strategy and a geographic encoding module to comprehensively exploit spatial-temporal-spectral information. The anchor-aware strategy employs groups of geospatially overlapped images gathered from diverse sensors with distinct spatial, temporal, and spectral coverage to model symbiotic relationships within these image groups. Additionally, we encode geographic information to establish spatial patterns, enhancing the model's generalization when pre-trained on a globally representative dataset. Extensive experiments demonstrate clear improvements in different downstream remote sensing tasks over state-of-the-art methods (up to 5.9%), including image classification, semantic segmentation, and change detection tasks.

#

## Model Weights

| Model | #param. | Top-1 Acc. | Top-5 Acc. | Hugginface Repo |
|:------------------------------------------------------------------:|:-------------:|:----------:|:----------:|:----------:|
| -    |       -       |   -   | - | - |

## Evaluation on Provided Weights

## Acknowledgement :heart:

## Citation
