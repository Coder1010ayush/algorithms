# ------------------------------ utf-8 encoding -------------------------------
# this file contains all kinds of clustering algorithm such as k-means (all varients ) and density - based clustering etc
import math
import numpy as np
from models.knn_models.centroid import Centroid
from models.basemodel import BaseModel


class KMeansClusterClassification(BaseModel):
    def __init__(self, num_classes: int, degree: int = 2):
        super().__init__()
