# SSMTL_CancerClassification
The source code of paper in Medical Image Analysis **A Semi-Supervised Multi-Task Learning Framework for Cancer Classification with Weak Annotation in Whole-Slide Images**

![URL_TS](./paperGraph.png)

A semi-supervised multi-task learning (SSMTL) framework for cancer classification. 

Our framework consists of a backbone feature extractor, two task-specific classifiers, and a weight control mechanism.

The backbone feature extractor is shared by two task-specific classifiers, such that the interaction of CRD and subtyping tasks can be captured. 

The weight control mechanism preserves the sequential relationship of these two tasks and guarantees the error back-propagation from the subtyping task to the CRD task under the MTL framework.

We train the overall framework in a semisupervised setting, where datasets only involve small quantities of annotations produced by our minimal point-based (min-point) annotation strategy.
