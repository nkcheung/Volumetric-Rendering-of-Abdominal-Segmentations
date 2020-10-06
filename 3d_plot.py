import numpy as np
import SimpleITK as sitk
from mayavi import mlab

# Convert CT and segmentation into np array
itk_ct = sitk.ReadImage('./test_ct.nii.gz')
ct = sitk.GetArrayFromImage(itk_ct)
itk_seg = sitk.ReadImage('./test_seg.nii.gz')
seg = sitk.GetArrayFromImage(itk_seg)

# Crop out only the useful slices on z-axis
index_list = []
for index, i in enumerate(seg):
    for ii in i:
        if np.max(ii) > 0:
            index_list.append(index)
            break
lower = min(index_list)
upper = max(index_list)
cropped_seg = seg[lower:upper]

# Get Houndsfield Unit (HU) for segmentation
equalised_seg = np.where(cropped_seg >= 1, 1, 0) # Covert multi-class segmentation into 1 or 0
cropped_ct = ct[lower:upper]
density_seg = np.multiply(cropped_ct,equalised_seg)

# Convert empty spaces to -1000 (HU ranges from -1024 to 3071)
density_seg[density_seg==0] = 4000
reverse_seg = 3000 - density_seg

# Thickening 3d volume for better visualization
thicken_hyperparameter = 3
thickened_seg = np.repeat(reverse_seg, thicken_hyperparameter, axis=0)

# Volume rendering with mayavi
mlab.pipeline.volume(mlab.pipeline.scalar_field(thickened_seg))
mlab.show()
