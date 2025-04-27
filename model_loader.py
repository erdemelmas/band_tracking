from ultralytics import YOLO

def load_model(model_path, use_cuda):
    model = YOLO(model_path)
    if use_cuda:
        model = model.to('cuda')
    return model
