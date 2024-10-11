from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional, Union, List, Sequence, Callable
from logging import getLogger
import time 

import donkeycar as dk

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
            self.input_vstreams_params = InputVStreamParams.make(self.network_group, quantized=False, format_type=FormatType.UINT8)
            self.output_vstreams_params = OutputVStreamParams.make(self.network_group, quantized=True, format_type=FormatType.UINT8)


            # Store input/output stream information
            self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
            self.output_vstream_info = self.hef.get_output_vstream_infos()[0]

            # Get the correct input name for inference
            self.input_name = self.input_vstream_info.name
            logger.info(f"Input vstream expected shape: {self.input_vstream_info.shape}")
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
        # Start timing for inference
        start_time = time.time()

        # Convert directly to uint8
        img_uint8 = img_arr.astype('uint8')
        logger.debug(f"Image converted to uint8. Shape: {img_uint8.shape}, dtype: {img_uint8.dtype}")

        # Add a batch dimension (1, H, W, C)
        input_data = np.expand_dims(img_uint8, axis=0)
        logger.debug(f"Batch dimension added. Input data shape: {input_data.shape}")

        # Prepare the input dictionary for inference
        input_dict = {self.input_name: input_data}
        logger.debug(f"Prepared input dictionary for inference.")

        # If other_arr is provided, convert it to NumPy array and add to input_dict
        if other_arr is not None:
            input_dict['other'] = np.array(other_arr, dtype=np.float32)
            logger.debug(f"Additional data (IMU) provided: {other_arr}")

        # Perform inference
        output = self.inference_from_dict(input_dict)

        # End timing for inference
        end_time = time.time()
        logger.info(f"Inference completed in {end_time - start_time:.4f} seconds")

        return output


    @abstractmethod
    def inference_from_dict(
            self, 
            input_dict: Dict[str, np.ndarray]) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """
        Run inference using the input dictionary and return the output.
        """
        pass

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

    def inference_from_dict(self, input_dict: Dict[str, np.ndarray]) -> Tuple[Union[float, np.ndarray], ...]:
        """
        Run inference using the input dictionary and return the output.
        """
        try:
            with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
                with self.network_group.activate():
                    # Start timing for the inference pipeline
                    pipeline_start_time = time.time()
                    
                    results = infer_pipeline.infer(input_dict)
                    
                    # End timing for the inference pipeline
                    pipeline_end_time = time.time()
                    logger.debug(f"Inference pipeline execution time: {pipeline_end_time - pipeline_start_time:.4f} seconds")
                    
                    # Retrieve output layers from the HEF model
                    output_vstream_info_fc3 = self.hef.get_output_vstream_infos()[0]  # fc3
                    output_vstream_info_fc4 = self.hef.get_output_vstream_infos()[1]  # fc4
                    
                    # Retrieve results for each output layer
                    fc3_output = results[output_vstream_info_fc3.name][0]
                    fc4_output = results[output_vstream_info_fc4.name][0]
                    
                    logger.info(f"Raw inference results (fc3): {fc3_output}")
                    logger.info(f"Raw inference results (fc4): {fc4_output}")
                    return self.interpreter_to_output(fc3_output, fc4_output)
        except Exception as e:
            logger.error(f"Error during Hailo inference: {e}")
            raise

    def interpreter_to_output(self, fc3_output, fc4_output):
        steering = ((fc4_output[0] / 255.0) * 2 - 1)
        throttle = ((fc3_output[0] / 255.0) * 2 - 1)
        logger.info(f"Steering: {steering}, Throttle: {throttle}")
        return steering, throttle
    
class HailoInferred(HailoPilot):
    """
    Inferred pilot for Hailo.
    It takes in image input and outputs steering and throttle values.
    """
    def __init__(self,
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 1):
        super().__init__(input_shape)

    def create_model(self):
        # The Hailo model is loaded and configured via the HEF file.
        logger.info("Hailo model is already configured through HEF.")
        return None

    def compile(self):
        # Hailo models don't need compilation like Keras/FastAI models
        logger.info("No compilation necessary for Hailo models.")

    def inference_from_dict(self, input_dict: Dict[str, np.ndarray]) -> Tuple[Union[float, np.ndarray], ...]:
        """
        Run inference using the input dictionary and return the output.
        """
        try:
            with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
                with self.network_group.activate():
                    # Start timing for the inference pipeline
                    pipeline_start_time = time.time()
                    
                    results = infer_pipeline.infer(input_dict)
                    
                    # End timing for the inference pipeline
                    pipeline_end_time = time.time()
                    logger.debug(f"Inference pipeline execution time: {pipeline_end_time - pipeline_start_time:.4f} seconds")
                    
                    # Retrieve output layers from the HEF model
                    output_vstream_info_fc3 = self.hef.get_output_vstream_infos()[0]  # fc3
                    
                    # Retrieve results for each output layer
                    fc3_output = results[output_vstream_info_fc3.name][0]
                    
                    logger.info(f"Raw inference results (fc3): {fc3_output}")
                    return self.interpreter_to_output(fc3_output)
        except Exception as e:
            logger.error(f"Error during Hailo inference: {e}")
            raise

    def interpreter_to_output(self, fc3_output):
        steering = -((fc3_output[0] / 255.0) * 2 - 1)
        logger.info(f"Steering: {steering}")
        return steering, dk.utils.throttle(steering)
