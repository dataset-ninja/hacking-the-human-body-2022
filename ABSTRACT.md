**Hacking the Human Body 2022** Dataset is a part of HuBMAP + HPA - Hacking the Human Body competition, where you need to identify and segment functional tissue units (FTUs) across five human organs - *prostate*, *spleen*, *lung*, *kidney*, and *largeintestine*. It helps to accelerate the world’s understanding of the relationships between cell and tissue organization. With a better idea of the relationship of cells, researchers will have more insight into the function of cells that impact human health. 

This competition uses data from two different consortia, the [Human Protein Atlas (HPA)](https://www.proteinatlas.org/) and [Human BioMolecular Atlas Program (HuBMAP)](https://hubmapconsortium.org/). The training dataset consists of data from public HPA data, the public test set is a combination of private HPA data and HuBMAP data, and the private test set contains only HuBMAP data. Adapting models to function properly when presented with data that was prepared using a different protocol will be one of the core challenges of this competition. While this is expected to make the problem more difficult, developing models that generalize is a key goal of this endeavor. This competition uses a hidden test. 

Dataset includes metadata for the train/test set. Only the first few rows of the test set are available for download:

- id - The image ID.
- ***organ*** - The organ that the biopsy sample was taken from.
- data_source - Whether the image was provided by HuBMAP or HPA.
- img_height - The height of the image in pixels.
- img_width - The width of the image in pixels.
- pixel_size - The height/width of a single pixel from this image in micrometers. All HPA images have a pixel size of 0.4 µm. For HuBMAP imagery the pixel size is 0.5 µm for kidney, 0.2290 µm for large intestine, 0.7562 µm for lung, 0.4945 µm for spleen, and 6.263 µm for prostate.
- ***tissue_thickness*** - The thickness of the biopsy sample in micrometers. All HPA images have a thickness of 4 µm. The HuBMAP samples have tissue slice thicknesses 10 µm for kidney, 8 µm for large intestine, 4 µm for spleen, 5 µm for lung, and 5 µm for prostate.
- rle - The target column. A run length encoded copy of the annotations. Provided for the training set only.
- ***age*** - The patient's age in years. Provided for the training set only.
- ***sex*** - The sex of the patient. Provided for the training set only.
