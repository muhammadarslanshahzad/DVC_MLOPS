from tensorflow.keras.models import load_model
from pathlib import Path
from bgremove import logger

model_path =Path('./assets/model/model.h5')
export_model =Path('./assets/model_prod/')

model = load_model(model_path)
logger.info('================= Model is Imported ================')

model.save(export_model, save_format="tf", include_optimizer= True, overwrite= True)

logger.info('================= Model is exported ================')