from functools import lru_cache
import numpy as np
from numpy.distutils.core import sdist
import pandas as pd
import PIL.Image
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from .Camera import CameraInfo
from typing import Any
from .utils import se3_to_quaternion_and_translation_torch



import os
def create_df():
    import glob
    files  =    glob.glob('/d/data/mlt/1684004313565050009/SENSOR_TYPE_OMNIVISION_OX08B40_*/image/*.jpg')
    files = sorted(files)
    # files=[_ for _ in files if '00412'<os.path.basename(_)<'00912']
    files=[_ for _ in files if '05112'<os.path.basename(_)<'05552']
    files=[_ for _ in files if 'SENSOR_TYPE_OMNIVISION_OX08B40_0' not in _ and
           'SENSOR_TYPE_OMNIVISION_OX08B40_7' not in _]
    @lru_cache(maxsize=1000)
    def _load_yaml(path):
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    c2ws  = [_load_yaml(_.replace('/image/','/camera/').replace('.jpg','.yaml'))['c2w'] for _ in files]
    intrinsics = [_load_yaml(_.replace('/image/','/camera/').replace('.jpg','.yaml'))['intrin'] for _ in files]
    camera_id = [idx for idx,_ in enumerate(files)]
    camera_width = [1920*2]*len(files)
    camera_height = [1080*2]*len(files)
    # create a dataframe 
    data = pd.DataFrame({'image_path':files,'T_pointcloud_camera':c2ws,'camera_intrinsics':intrinsics,'camera_height':camera_height,'camera_width':camera_width,'camera_id':camera_id})
    return data

def create_point_cloud():
    import numpy as np
    pts1=np.load('/d/data/mlt/1684004313565050009/SENSOR_TYPE_HESAI_PANDAR128_E3X_0/data/00813.pkl',allow_pickle=True)
    pts2=np.load('/d/data/mlt/1684004313565050009/SENSOR_TYPE_HESAI_PANDAR128_E3X_1/data/00813.pkl',allow_pickle=True)
    pts = np.concatenate([pts1,pts2],axis=0)
    # create a dataframe, with x, y,z and r,g,b being zeros
    data = pd.DataFrame({'x':pts[:,0],'y':pts[:,1],'z':pts[:,2],'r':np.zeros_like(pts[:,0]),'g':np.zeros_like(pts[:,0]),'b':np.zeros_like(pts[:,0])})
    return data


class ImagePoseDataset(torch.utils.data.Dataset):
    """
    A dataset that contains images and poses, and camera intrinsics.
    """

    def __init__(self, dataset_json_path: str):
        super().__init__()
        self.df = create_df()
    #     for column in required_columns:
    #         assert column in self.df.columns, f"column {column} is not in the dataset"
    # # def __init__(self, dataset_json_path: str):
    #     super().__init__()
    #     required_columns = ["image_path", "T_pointcloud_camera",
    #                         "camera_intrinsics", "camera_height", "camera_width", "camera_id"]
    #     self.df = pd.read_json(dataset_json_path, orient="records")
    #     for column in required_columns:
    #         assert column in self.df.columns, f"column {column} is not in the dataset"
    #

    def __len__(self):
        # return 1 # for debugging
        return len(self.df)

    def _pandas_field_to_tensor(self, field: Any) -> torch.Tensor:
        if isinstance(field, np.ndarray):
            return torch.from_numpy(field)
        elif isinstance(field, list):
            return torch.tensor(field)
        elif isinstance(field, torch.Tensor):
            return field

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]["image_path"]
        T_pointcloud_camera = self._pandas_field_to_tensor(
            self.df.iloc[idx]["T_pointcloud_camera"])
        q_pointcloud_camera, t_pointcloud_camera = se3_to_quaternion_and_translation_torch(
            T_pointcloud_camera.unsqueeze(0))
        camera_intrinsics = self._pandas_field_to_tensor(
            self.df.iloc[idx]["camera_intrinsics"])
        base_camera_height = self.df.iloc[idx]["camera_height"]
        base_camera_width = self.df.iloc[idx]["camera_width"]
        camera_id = self.df.iloc[idx]["camera_id"]
        image = PIL.Image.open(image_path).resize((1920//2,1080//2))
        image = torchvision.transforms.functional.to_tensor(image)
        # use real image size instead of camera_height and camera_width from colmap
        camera_height = image.shape[1]
        camera_width = image.shape[2]
        camera_intrinsics[0, :] = camera_intrinsics[0, :] * \
            camera_width / base_camera_width
        camera_intrinsics[1, :] = camera_intrinsics[1, :] * \
            camera_height / base_camera_height
        # we want image width and height to be always divisible by 16
        # so we crop the image
        camera_width = camera_width - camera_width % 16
        camera_height = camera_height - camera_height % 16
        image = image[:3, :camera_height, :camera_width].contiguous()
        camera_info = CameraInfo(
            camera_intrinsics=camera_intrinsics,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_id=camera_id,
        )
        return image, q_pointcloud_camera ,t_pointcloud_camera, camera_info
