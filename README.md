# Object Detection
This is the YOLOv1 model for the Object Detection task, using Pytorch

## About
Building YOLOv1 model from Scratch, following the original paper: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

## Usage
To train the YOLOv1 model, use the command line:

```bash
python YOLOv1-Implement/main.py --config YOLOv1-Implement/config.yaml
```

To test the YOLOv1 model, go to YOLOv1-Implement/main.py and command these codes:

```python
logging.info("Training started...")
Train_Task(config).train()
logging.info("Train complete")
```
