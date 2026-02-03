# %%
from ultralytics import YOLO
import torch
import os

# %%
import logging
import logging
logging.getLogger("ultralytics").setLevel(logging.INFO)
logging.getLogger("ultralytics.yolo.utils.torch_utils").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.StreamHandler(),              # stdout → jupyter.log
        logging.FileHandler("train.log"),     # optional persistent log
    ],
)

def log_epoch(trainer):
    epoch = trainer.epoch + 1
    m = trainer.metrics
    logging.info(
        f"Epoch {epoch} | "
        f"box={m.get('box_loss', 'NA')} | "
        f"cls={m.get('cls_loss', 'NA')} | "
        f"dfl={m.get('dfl_loss', 'NA')}"
    )


# %%
'''
Training the teacher model
Note that Epochs = 3000 is a rough estimate for grokking to occur
'''
# Import model from 
model = YOLO("yolo26x.pt")
model.add_callback("on_epoch_end", log_epoch)
# Training parameters
results = model.train(
    data="2025_cone_dataset.yaml",
    epochs=500,           # No. of epochs 
    imgsz=1024,           # This is not the final model to be deployed and hence require the highest resolution for generalisation
    batch=-1,             # '-1' auto-adjusts batch size to fill VRAM
    patience=50,          # End training if maP does not improve
    weight_decay=0.1,     # Recommended Settings for 
    dropout = 0.1,        # Large dropout for better generalisation performance
    optimizer="MuSGD",    # Best for large models
    lr0=0.01,             # Standard starting learning rate
    cos_lr=True,          # Uses a Cosine Annealing schedule (essential for grokking)
    close_mosaic=20,      # Turn off mosaic augmentation for the last 20 epochs to refine
    overlap_mask=True,    # Helps if cones are partially covering each other
    augment=True,         # Use heavy data augmentation (flips, mosaics, etc.)
    val=True,             # Perform validation at each step
    save_period = 10,     # Save weights every 10 epochs
    verbose=False,         # Suppress per-epoch progress output, only show custom logging at epoch end
    plots=False            # Disable plots to save time
)

model.save("Yolo26x_Teacher.pt")

print("Model saved and ready for distillation!")

current_dir = os.getcwd()

if results is not None and hasattr(results, 'save_dir'):
        best_weights = os.path.join(results.save_dir, 'weights', 'best.pt')
else:
    best_weights = os.path.join(current_dir, 'runs', 'Cone_Teacher', 'grokking_run', 'weights', 'best.pt')
    
final_model = YOLO(best_weights)

# Run evaluation on the TEST split
metrics = final_model.val(split='test') 

print(f"Final Test mAP50: {metrics.box.map50}")

# %%
'''
Starting training for 500 epochs...
Knowledge Distillation
Training a smaller YOLO26n with knowledge from the teacher model trained above
'''

teacher = YOLO("Yolo26x_Teacher.pt")

student = YOLO("yolo26n.pt")

results = student.train(
    data="2025_cone_dataset.yaml",
    teacher=teacher.model,         # Pass the underlying PyTorch model
    imgsz= 640,                     # Student resolution 
    teacher_imgsz=1024,            # Teacher's resolution expertise
    epochs=500,                    # Student needs fewer epochs with a guide
    batch=128,                     # High batch size for stable gradients
    optimizer="AdamW",             # Better for small models than MuSGD
    # distill_loss="cwd",            # Channel-wise distillation for better boundaries
)

student.save("Yolo26n_Final.pt")



# %%
