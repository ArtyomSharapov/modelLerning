import splitfolders

#разделение для классификации
splitfolders.ratio(
    "datasets/Dataset_for_classification",
    output="datasets/Dataset_classification_ready",
    seed=111,
    ratio=(0.8, 0.2))

#разделение для сегментации
splitfolders.ratio(
    "datasets/Dataset_for_segmentation",
    output="datasets/Dataset_segmentation_ready",
    seed=111,
    ratio=(0.8, 0.2))