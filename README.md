<img src="coral_logo.png" width="80" /> CORAL: Multi-scale Multi-modal integration of Spatial Omics via Deep Generative Model
---
CORAL is a probabilistic, graph-based method designed to integrate diverse spatial omics datasets. 
Taking multimodality molecular profiles of unmatched spatial resolution and detected features, CORAL generates single-cell embedding with information from both data modalities,  deconvolves the lower-resolution modality to infer its profile in individual cells, and predicts interactions between neighboring cells.

<img src=Coral_Figure1.png width="1000" />


### Installation

`pip install git+https://github.com/zou-group/CORAL`


### Model Input (h5ad format):
- A hires adata 
- A lowres adata
- (optional) the major cell type annotation on hires adata
- (optional) ground truth adata
  
### Accepting spatial omics data 
- spatial transcriptics
- spatial proteomics
- spatial metabolics
- spatial ATAC

### Features 
- Incorporating the scale-invariant feature transform (SIFT) to align the adjacent slides 
- Generating joint single-cell embedding with two data modalities 
- Deconvolves the lower-resolution modality to higy-resolution
- Inferring spatial niches
- Prediciting spatial variables
- Predicting interations between neighboring cells


### simple core implementation
```
import coral
combined_expr, hires_coords, one_hot_cell_types, spot_indices, lowres_expr = coral.utils.preprocess_data(lowres_adata, hires_adata)
    
dataloader = coral.utils.prepare_local_subgraphs(combined_expr, hires_coords, one_hot_cell_types, 
                                           spot_indices, lowres_expr,n_neighbors=40)    

model, optimizer = coral.model.create_model(lowres_dim = lowres_adata.shape[1],
                                            hires_dim = hires_adata.shape[1],
                                            lowres_size = lowres_adata.shape[0],
                                            hires_size = hires_adata.shape[0],
                                            cell_type_dim=one_hot_cell_types.shape[1],                                          
                                            latent_dim=64, 
                                            hidden_channels=128, 
                                            v_dim = 1
                                          )
coral.trainer.train_model(model, optimizer, dataloader, epochs = 100 ,device = device)
adata_model_gener = coral.inference.generate_and_validate(model, dataloader,device, hires_adata)

```

### Tutorial
 - CORAL provides semi-automated alignment using SIFT. If your spatial multi-omics data are not aligned, please check our [tutorial for alignment](https://github.com/zou-group/CORAL/blob/main/align_slides.ipynb)
 - After the two input adata are prepared, please folowing our [basic tutorial](https://github.com/zou-group/CORAL/blob/main/coral_tutorial_basic.ipynb)
 - For additional steps related to reproducing the results in the paper, please refer to our reproducibility repository: https://github.com/siyuh/CORAL_reproducibility


### How to cite CORAL
```
bioRxiv: https://doi.org/10.1101/2025.02.01.636038
```
## Contact
In case you have questions, please contact:
- Siyu He - siyuhe@stanford.edu
- via Github Issues
