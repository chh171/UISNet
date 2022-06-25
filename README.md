# UISNet
UISNet: An uncertainty-based interpretable deep semi-supervised network for breast cancer outcomes prediction
UISNet is designed to interpret the feature importance of the cancer outcomes prediction model by an uncertainty-based integrated gradients algorithm. 


# Usage
Before using, please unzip captur.rar in the current folder. The datasets aboout the gene expression and pathway information are stored in my_dataset folder. The output about the patients' risks, CI values and the IG scores with the Monte Carlo dropout are givwn in the result_all folder.

# Example
Due to file size constraints, here we give an example data: brca_test.csv (expression data) and pathway_mask (pathway information) used for NISNet.py. Users can build data as the format of the example. The main program UISNet.py can be used for cancer outcomes prediction.

# Contact
This method is still on progress, any questions can be sent to 854330388@qq.com
