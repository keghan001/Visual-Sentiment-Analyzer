from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig

def start_training():
    tsboard_config = TensorBoardOutputConfig(
        s3_output_path="bucket-name/tensorboard",
        container_local_output_path="/opt/ml/output/tensorboard"
    )
    
    estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role="my-new-role",
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

if __name__ == "__main__":
    start_training()
