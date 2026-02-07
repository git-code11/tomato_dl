


## Training Model

# Train Vision Transformer Model
!python3 tomato_project/train.py +model=disease_vit model.training_params.epoch=20

# Train Efficient Model
!python3 tomato_project/train.py +model=disease_efficient model.training_params.epoch=20

# Train Xception Model
!python3 tomato_project/train.py +model=disease_xception model.training_params.epoch=20

# Train Inception Model
!python3 tomato_project/train.py +model=disease_inception model.training_params.epoch=20

# Train Hybrid-Efficient Model
!python3 tomato_project/train.py +model=disease_hybrid_efficient model.training_params.epoch=10 \
+model.model_params.vit_weights=vit.model.weights.h5 \
+model.model_params.cnn_weights=efficient.weights.h5

# Train Hybrid-Xception Model
!python3 tomato_project/train.py +model=disease_hybrid_xception model.training_params.epoch=10 \
+model.model_params.vit_weights=vit.model.weights.h5 \
+model.model_params.cnn_weights=xception.weights.h5

# Train Hybrid-Inception Model
!python3 tomato_project/train.py +model=disease_hybrid_inception model.training_params.epoch=10 \
+model.model_params.vit_weights=vit.model.weights.h5 \
+model.model_params.cnn_weights=inception.weights.h5


# Evaluation of model
