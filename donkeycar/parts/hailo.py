from abc import ABC, abstractmethod
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, Union, List, Sequence, Callable
from logging import getLogger
import time 

from hailo_platform import VDevice, HEF, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType, ConfigureParams, HailoStreamInterface

logger = getLogger(__name__)

class HailoPilot(ABC):

    """HailoPilot manages the inference pipeline using Hailo's HEF models."""
    def __init__(self,
                input_shape: Tuple[int, ...] = (120, 160, 3)) -> None:
        self.model: Optional[Model] = None
        self.input_shape = input_shape
        self.vdevice = VDevice()
        self.hef = None
        self.network_group = None
        self.input_vstreams_params = None
        self.output_vstreams_params = None
        logger.info(f'Created {self}')

    def load(self, model_path):
        logger.info(f'Loading HEF model from {model_path}')
        try:
            self.hef = HEF(model_path)

            # Configure the Hailo model on the device
            interface = HailoStreamInterface.PCIe
            configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=interface)
            self.network_groups = self.vdevice.configure(self.hef, configure_params)
            self.network_group = self.network_groups[0]

            # Set the input/output stream parameters
            self.input_vstreams_params = InputVStreamParams.make(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
            self.output_vstreams_params = OutputVStreamParams.make(self.network_group, quantized=True, format_type=FormatType.UINT8)

            # Store input/output stream information
            self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
            self.output_vstream_info = self.hef.get_output_vstream_infos()[0]

            # Get the correct input name for inference
            self.input_name = self.input_vstream_info.name

            logger.info('Model loaded and configured successfully.')

        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise

    def compile(self) -> None:
        pass

    @abstractmethod
    def create_model(self):
        pass

    def run(self, img_arr, other_arr: List[float] = None):
        """
        Run inference on the input image array and other additional data (e.g., IMU array).
        """
        # Preprocess the image using OpenCV and NumPy
        img_resized = cv2.resize(np.array(img_arr), (self.input_shape[1], self.input_shape[0]))
        img_normalized = img_resized.astype('float32') / 255.0  # Normalize image to [0, 1]
        
        # Add a batch dimension (1, H, W, C)
        input_data = np.expand_dims(img_normalized, axis=0)

        # Prepare the input dictionary for inference
        input_dict = {self.input_name: input_data}

        # If other_arr is provided, convert it to NumPy array and add to input_dict
        if other_arr is not None:
            input_dict['other'] = np.array(other_arr, dtype=np.float32)

        return self.inference_from_dict(input_dict)

    def inference_from_dict(self, input_dict: Dict[str, np.ndarray]) -> Tuple[Union[float, np.ndarray], ...]:
        """
        Run inference using the input dictionary and return the output.
        """
        try:
            with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
                with self.network_group.activate():
                    results = infer_pipeline.infer(input_dict)
                    output_vstream_info = self.hef.get_output_vstream_infos()[0]
                    # Debug: Afficher le résultat brut
                    logger.info(f"Raw inference results: {results[output_vstream_info.name]}")
                    output = results[output_vstream_info.name][0]
                    if len(output) < 2:
                        logger.error(f"Expected 2 output values, but got {len(output)}")
                        return 0.0, 0.0  # Valeurs par défaut si le modèle ne renvoie pas assez d'éléments

                    return self.interpreter_to_output(results[output_vstream_info.name][0])
        except Exception as e:
            logger.error(f"Error during Hailo inference: {e}")
            raise

    @abstractmethod
    def interpreter_to_output(
            self,
            interpreter_out: Sequence[Union[float, np.ndarray]]) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """ Virtual method to be implemented by child classes for conversion
            :param interpreter_out:  input data
            :return:                 output values, possibly tuple of np.ndarray
        """
        pass

    def shutdown(self):
        self.is_running = False
        time.sleep(0.1)  # Small delay to ensure threads stop
        self.vdevice.release()  # Release the Hailo device resources
        logger.info('Hailo resources released.')

class HailoLinear(HailoPilot):
    """
    Linear pilot for Hailo, similar to FastAILinear.
    It takes in image input and outputs steering and throttle values.
    """
    def __init__(self,
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2):
        super().__init__(input_shape)

    def create_model(self):
        # The Hailo model is loaded and configured via the HEF file.
        logger.info("Hailo model is already configured through HEF.")
        return None

    def compile(self):
        # Hailo models don't need compilation like Keras/FastAI models
        logger.info("No compilation necessary for Hailo models.")

    def interpreter_to_output(self, interpreter_out):
        interpreter_out = (interpreter_out * 2) - 1
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering, throttle