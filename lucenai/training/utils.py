
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def configure_environment_for_nlp(seed: int = 42):
    """
    Sets up the environment for training NLP models with reproducibility and performance in mind.

    - Configures matplotlib display settings
    - Enables GPU memory growth if a GPU is available
    - Sets seeds for Python, NumPy, and TensorFlow to ensure reproducibility

    Args:
        seed (int): The seed value to use across all libraries (default is 42)

    This function should be called once at the beginning of the training script.
    """
    # Matplotlib configuration (plot style, font size, etc.)
    display_configuration()

    # GPU configuration (memory growth if available)
    configure_gpu_for_nlp()

    # Set random seeds for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    print(f"🔁 Reproducibility seed set to {seed}")


def display_configuration():
    """
    Configures global display settings for plots and warnings.

    - Disables warning messages during execution
    - Sets default matplotlib style
    - Defines standard figure size and font size for plots

    Useful for ensuring consistent visual output during model training or data analysis.
    """
    warnings.filterwarnings('ignore')
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12


def configure_gpu_for_nlp():
    """
    Configures TensorFlow to use the first available GPU (if any) with memory growth enabled.

    This setup is recommended for large NLP models such as DistilBERT to avoid 
    out-of-memory (OOM) errors. If no GPU is found, it gracefully defaults to CPU usage.

    Returns:
        bool: True if GPU is configured, False if running on CPU.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Use only the first GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # Enable memory growth to prevent allocation spikes
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"✅ GPU successfully configured: {gpus[0].name}")
            print("🧠 GPU memory growth enabled")
            return True
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
    else:
        print("🔧 Using CPU - works fine but slower for NLP models")

    print("\n💡 Info: Transformer models like DistilBERT require significant memory.")
    print("   Memory growth helps avoid 'Out of Memory' errors.")
    return False


def set_global_seed(seed: int = 42):
    """
    Sets the global random seed across Python, NumPy, and TensorFlow to
    ensure reproducible behavior.

    Args:
        seed (int): The seed value to apply (default is 42)
    """
    # Ensures consistent hashing (important for deterministic behavior in some libraries)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set Python's built-in random module seed
    random.seed(seed)

    # Set NumPy's random seed
    np.random.seed(seed)

    # Set TensorFlow's random seed
    tf.random.set_seed(seed)

    print(f"🔁 Reproducibility seed set to {seed}")