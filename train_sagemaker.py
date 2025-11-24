from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig
from dotenv import load_dotenv
import os

def start_training():
    tsboard_config = TensorBoardOutputConfig(
        s3_output_path="s3://sentiment-analyzer-saas/tensorboard",
        container_local_output_path="/opt/ml/output/tensorboard"
    )
    
    load_dotenv()
    
    estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role=os.getenv("SG_ROLE"),
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        hyperparameters={
            "batch_size": str(32),
            "epochs": str(25)
        },
        tensorboard_config=tsboard_config
    )
    
    # Start training
    estimator.fit({
        "training": "s3://sentiment-analyzer-saas/dataset/train",
        "validation": "s3://sentiment-analyzer-saas/dataset/dev",
        "test": "s3://sentiment-analyzer-saas/dataset/test"
    })


if __name__ == "__main__":
    start_training()
