from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional, Union, List, Sequence
from logging import getLogger
import time
import cv2

import donkeycar as dk
from donkeycar.utils import normalize_image, denormalize_image

from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType)

logger = getLogger(__name__)

class HailoPilot(ABC):
    """
    HailoPilot manages the inference pipeline using Hailo's HEF models.
    """

    def __init__(self,
                 input_shape: Tuple[int, ...] = (120, 160, 3)) -> None:
        """
        Initialize the HailoPilot with input shape and device configuration.
        """
        self.model: Optional[Model] = None
        self.input_shape = input_shape
        self.vdevice = VDevice()
        self.hef = None
        self.network_group = None
        self.network_group_params = None
        self.input_vstreams_params = None
        self.output_vstreams_params = None
        logger.info(f'Created {self}')

    def load(self, model_path: str) -> None:
        """
        Load the Hailo HEF model from a specified path.
        """
        logger.info(f'Loading HEF model from {model_path}')
        try:
            # Scan for available devices
            self.devices = Device.scan()
            self.hef = HEF(model_path)

            # Log input and output layer details
            for layer_info in self.hef.get_input_vstream_infos():
                logger.info(f'Input layer: {layer_info.name} {layer_info.shape}')
            for layer_info in self.hef.get_output_vstream_infos():
                logger.info(f'Output layer: {layer_info.name} {layer_info.shape}')
            
            # Configure the model
            configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)        
            self.network_group = self.vdevice.configure(self.hef, configure_params)[0]
            self.network_group_params = self.network_group.create_params()

            # Set input/output stream parameters
            self.input_vstreams_params = InputVStreamParams.make(self.network_group)
            self.output_vstreams_params = OutputVStreamParams.make(self.network_group)
        
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise

    def compile(self) -> None:
        """ No compilation is needed for Hailo models. """
        pass

    @abstractmethod
    def create_model(self):
        """ Abstract method for creating a model, to be implemented by subclasses. """
        pass

    def run(self, img_arr: np.ndarray, *other_arr: List[float]) -> Tuple[Union[float, np.ndarray], ...]:
        """
        Interface to run the HailoPilot in the Donkeycar loop.

        :param img_arr:     uint8 [0,255] numpy array with image data
        :param other_arr:   numpy array of additional data, such as IMU or state vector
        :return:            tuple of (angle, throttle)
        """
        # Normalize the image
        norm_img = normalize_image(img_arr)
        #logger.info(f"Data before normalization (sample pixels): {img_arr[0:5, 0:5, :]}")
        #logger.info(f"Data after normalization (sample pixels): {norm_img[0:5, 0:5, :]}")

        # Save the denormalized image for debugging purposes
        denorm_img_arr = denormalize_image(norm_img)
        cv2.imwrite("debug_image.png", denorm_img_arr)

        # Handle additional data inputs
        other_array = np.array(other_arr, dtype=np.float32) if other_arr else np.array([], dtype=np.float32)

        # Run the inference process
        return self.inference(denorm_img_arr, other_array)

    def inference(self, img_arr: np.ndarray, other_arr: Optional[np.ndarray] = None) -> Tuple[Union[float, np.ndarray], ...]:
        """
        Perform inference using the model and return the predicted steering and throttle values.

        :param img_arr:     float32 [0,1] numpy array with normalized image data
        :param other_arr:   Optional numpy array with additional data
        :return:            tuple of (angle, throttle)
        """
        try:
            # Ensure input_vstreams_params is properly initialized
            if not self.input_vstreams_params or len(self.input_vstreams_params) == 0:
                raise ValueError("input_vstreams_params is not properly initialized or is empty.")

            # Retrieve input stream key (assume only one input stream here)
            input_key = list(self.input_vstreams_params.keys())[0]

            # Prepare input data for inference
            input_data = {input_key: np.expand_dims(img_arr, axis=0).astype(np.uint8)}

            # Handle additional input streams if any
            if len(self.input_vstreams_params) > 1 and other_arr is not None:
                for i, key in enumerate(list(self.input_vstreams_params.keys())[1:], start=1):
                    input_data[key] = np.expand_dims(other_arr[i-1], axis=0).astype(np.float32)

            # Perform inference
            with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
                with self.network_group.activate(self.network_group_params):
                    results = infer_pipeline.infer(input_data)

                    # Extract and log inference results
                    output_results = {}
                    for output_key, output_param in self.output_vstreams_params.items():
                        output_results[output_key] = results[output_key][0]
                    logger.info(f"Inference results: {output_results}")

                    # Convert results into steering and throttle values
                    return self.interpreter_to_output(output_results)

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    @abstractmethod
    def interpreter_to_output(self,
                              interpreter_out: Sequence[Union[float, np.ndarray]]) -> Tuple[Union[float, np.ndarray], ...]:
        """ Convert the interpreter output to usable values. """
        pass

    def shutdown(self) -> None:
        """ Gracefully shut down the Hailo device and release resources. """
        self.is_running = False
        time.sleep(0.1)  # Small delay to ensure threads stop
        self.vdevice.release()  # Release the Hailo device resources
        logger.info('Hailo resources released.')

class HailoLinear(HailoPilot):
    """
    Linear pilot for Hailo. Takes in image input and outputs steering and throttle values.
    """

    def __init__(self,
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2):
        super().__init__(input_shape)

    def create_model(self):
        """ Hailo model is loaded and configured via the HEF file. """
        logger.info("Hailo model is already configured through HEF.")
        return None

    def compile(self):
        """ No compilation is needed for Hailo models. """
        logger.info("No compilation necessary for Hailo models.")

    def interpreter_to_output(self, output_results: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """
        Convert inference results to steering and throttle values.

        :param output_results: Dictionary containing the model's output layers results.
        :return:               steering (float), throttle (float)
        """
        try:
            # Dynamically extract output stream names
            output_keys = list(self.output_vstreams_params.keys())

            # Extract steering and throttle values
            steering = ((output_results[output_keys[0]][0]  / 255.0) * 2 - 1)
            throttle = ((output_results[output_keys[1]][0]  / 255.0) * 2 - 1)
            print(f"Normalized steering: {steering} - Normalized throttle: {throttle}")
            return float(steering), float(throttle)

        except Exception as e:
            logger.error(f"Error interpreting inference results: {e}")
            raise
