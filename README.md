# Realistic-Cross-View-Image-Geo-Localization

Cross-view image geo-localization is the task of localizing street-view query images by matching them with satellite
corresponding to a GPS location. The complexity of the problem comes, first, from the large gap between the image domains: there is a drastic change in viewpoint and appearance between them.

To bridge the domain gap, the majority of previous work ([Shi et al., 2019](https://proceedings.neurips.cc/paper/2019/file/ba2f0015122a5955f8b3a50240fb91b2-Paper.pdf); [Toker et al., 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Toker_Coming_Down_to_Earth_Satellite-to-Street_View_Synthesis_for_Geo-Localization_CVPR_2021_paper.pdf), [Shi et al., 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Where_Am_I_Looking_At_Joint_Location_and_Orientation_Estimation_CVPR_2020_paper.pdf)) has approached the problem by using a polar coordinate transformation as a preprocessing step before solving the image retrieval problem and achieved surprisingly high retrieval accuracy on city-scale datasets. The polar coordinate transformation assumes that there exists an aerial-view reference image exactly centered at the location of any street-view query image. This is not applicable for practical scenarios.

We investigate the performance of [spatial transformers](https://proceedings.neurips.cc/paper/2015/file/33ceb07bf4eeb3da587e268d663aba1a-Paper.pdf) in this task, with the hope to learn
a transformation similar to the polar transformation directly from the data and extract domain-invariant features.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Notram1/Realistic-Cross-View-Image-Geo-Localization/blob/main/main.ipynb)

This repository is based on [Toker et al., 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Toker_Coming_Down_to_Earth_Satellite-to-Street_View_Synthesis_for_Geo-Localization_CVPR_2021_paper.pdf).