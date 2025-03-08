import logging
import time

import requests

logger = logging.getLogger(__name__)


def is_model_available(
    model_name: str, api_url: str = "http://localhost:11434/models"
) -> bool:
    """
    Check if a specific Ollama model is available via the API.

    Args:
        model_name (str): The name of the model to check.
        api_url (str): The base URL for the Ollama API's models endpoint.

    Returns:
        bool: True if the model is available, False otherwise.
    """
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            available_models = response.json().get("models", [])
            return any(model.get("name") == model_name for model in available_models)
        else:
            logger.info(
                f"Error checking model availability: {response.status_code} - {response.text}"
            )
            return False
    except requests.RequestException as e:
        logger.info(f"Request failed: {e}")
        return False


def wait_for_model(
    model_name: str,
    api_url: str = "http://localhost:11434/models",
    timeout: int = 300,
    interval: int = 5,
):
    """
    Wait until a specific Ollama model is ready.

    Args:
        model_name (str): The name of the model to wait for.
        api_url (str): The base URL for the Ollama API's models endpoint.
        timeout (int): Maximum time (in seconds) to wait.
        interval (int): Interval (in seconds) between checks.

    Raises:
        TimeoutError: If the model is not ready within the timeout period.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_model_available(model_name, api_url):
            logger.info(f"Model '{model_name}' is now available.")
            return
        logger.info(f"Waiting for model '{model_name}' to be ready...")
        time.sleep(interval)
    raise TimeoutError(f"Model '{model_name}' was not ready within {timeout} seconds.")
