import os

# Configuration for the data
DATA_TRAIN_INDEX_PATH = r"C:\Users\Administrator\NotaGen\data\augmented_abc_train.jsonl" 
DATA_EVAL_INDEX_PATH  = r"C:\Users\Administrator\NotaGen\data\augmented_abc_eval.jsonl"

# Configuration for the model
PATCH_STREAM = True                                             # Stream training / inference
PATCH_SIZE = 16                                                # Patch Size
PATCH_LENGTH = 1024                                             # Patch Length
CHAR_NUM_LAYERS = 6                                             # Number of layers in the decoder
PATCH_NUM_LAYERS = 20                                           # Number of layers in the encoder
HIDDEN_SIZE = 1280                                               # Hidden Size

# Configuration for the training
BATCH_SIZE = 4         
LEARNING_RATE = 5e-6   
NUM_EPOCHS = 5                                                 # Number of epochs to train for (if early stopping doesn't intervene)
ACCUMULATION_STEPS = 4                                          # Accumulation steps to simulate large batch size
PATCH_SAMPLING_BATCH_SIZE = 0                                   # Batch size for patch during training, 0 for full conaudio
LOAD_FROM_CHECKPOINT = False                                    # Whether to load weights from a checkpoint
WANDB_LOGGING = False                                           # Whether to log to wandb
WANDB_KEY = '<your_wandb_key>'

PRETRAINED_PATH = r"C:\Users\Administrator\NotaGen\finetune\weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth"
EXP_TAG = 'finetune3'                                            # Experiment tag for name differentiation
NAME =  EXP_TAG + \
        "_p_size_" + str(PATCH_SIZE) + \
        "_p_length_" + str(PATCH_LENGTH) + \
        "_p_layers_" + str(PATCH_NUM_LAYERS) + \
        "_c_layers_" + str(CHAR_NUM_LAYERS) + \
        "_h_size_" + str(HIDDEN_SIZE) + \
        "_lr_" + str(LEARNING_RATE) + \
        "_batch_" + str(BATCH_SIZE)

WEIGHTS_PATH = "weights_notagen_" + NAME + ".pth"                  # Path to save weights
LOGS_PATH    = "logs_notagen_"    + NAME + ".txt"                     # Path to save logs
WANDB_NAME = NAME
