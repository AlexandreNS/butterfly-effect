# Projeto ButterFly Effect

## Algorithm Execution Order

To refine the images and create our functional dataset we must execute the algorithms in this order:

```sh
python less_red_mask.py
python cropped_black_surface.py
python cropped.py
# before resizing rotate images
# to standardize the proportions
python resize_images.py
python normalize_images.py
python dataset_create.py
# then just select the respective images
# for testing and learning
```

Then just select the test images, extract data through the descriptors (HOG and/or LBP)

Now just create the models with the **pls_butterfly_(descriptor)** algorithms and test them with the **predict_model_(descriptor)** algorithms.
