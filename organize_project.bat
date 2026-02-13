@echo off
echo Organizing project structure...

:: Create folders
mkdir src
mkdir src\train
mkdir src\test
mkdir src\utils
mkdir src\gan
mkdir models
mkdir results
mkdir results\confusion_matrices
mkdir results\evaluation
mkdir results\roc_curves
mkdir docs
mkdir data

:: Move training scripts
move train_cnn*.py src\train\
move res_net50_cnn*.py src\train\
move mobileNet*.py src\train\
move mbnetv2.py src\train\
move self_built.py src\train\
move Rnet50.py src\train\

:: Move testing scripts
move *TEST.py src\test\
move build_test_cnn.py src\test\
move Roc_curve.py src\test\

:: Move utilities
move accuracy.py src\utils\
move controller.py src\utils\
move data.py src\utils\
move resize*.py src\utils\
move statistics.py src\utils\
move count.py src\utils\
move Imported_libraries.py src\utils\

:: GAN files
move GAN_train.py src\gan\

:: Move models
move *.h5 models\

:: Move confusion matrices
move *_conf*.png results\confusion_matrices\

:: Move evaluation plots
move *_eval*.png results\evaluation\
move Accuracy.png results\evaluation\
move precision.png results\evaluation\
move Recall.png results\evaluation\
move F1_score.png results\evaluation\

:: Move ROC curves
move *roc*.png results\roc_curves\

:: Move dataset archive
move archive data\

:: Move documentation
move project_Description docs\project_description.txt

echo.
echo Project organization complete!
pause
