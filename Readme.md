- This is just a demo code for testing using Unidepth model and YOLO, the official source code will be published after the competition "Thiết kế, sáng tạo sản phẩm, công nghệ dành cho người khuyết tật năm 2025" is completed.
# FRONTEND

- Frontend I use framework Flutter and code Dart. You can download IMAGES fischerscode/flutter:latest and install a container to run frontend

- You can find more information to set up environment flutter on link: https://docs.flutter.dev/

1. Command to run frontend

```bash
flutter run -d web-server --web-port=yourport
```

# BACKEND

- I use UniDepth model to get depth map. You can can read the way to run in https://github.com/lpiccinelli-eth/unidepth

* After install UniDepth, you need to install other library like Uvicorn to host backend

2. Command to run backend:

```bash
CUDA_VISIBLE_DEVICES=0 uvicorn main:app --host 0.0.0.0 --port yourport
```
