from ..data.image import *
import os

class ImageFactory():
    """
    Interface to File readers
    """

    @classmethod
    def create_storm_file(self, path: str):
        if not os.path.exists(path):
            ValueError("Given path doesn't exist")
        extend = os.path.splitext(path)[1]
        if "csv" in extend:
            pass
        if "txt" in extend:
            file = LocalisationReader(path)
            file.parse()
        return file

    @classmethod
    def create_image_file(self, path: str):
        if not os.path.exists(path):
            ValueError("Given path doesn't exist")
        file = ImageReader(path)
        file.parse()
        return file