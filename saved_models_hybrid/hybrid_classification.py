import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import os
from PIL import Image
import joblib

# ============================================
# CONFIGURATION
# ============================================

IMAGE_DIR = 'archive/train_data_set'
TEXT_DIR = 'ocr_outputs'  # Folder with OCR text files
OUTPUT_DIR = 'results_hybrid'
MODEL_DIR = 'saved_models_hybrid'

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0001
TEXT_FEATURES = 2000  # TF-IDF features

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("="*70)
print("HYBRID MODEL: VISION + TEXT")
print("="*70)

# ============================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================

print("\n[1/7] Loading Data...")

def load_hybrid_data(image_dir, text_dir):
    """Load both images and corresponding text"""
    data = []
    
    # Get all classes
    classes = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(classes))}
    
    for class_name in classes:
        image_class_dir = os.path.join(image_dir, class_name)
        text_class_dir = os.path.join(text_dir, class_name)
        
        # Get all images
        image_files = [f for f in os.listdir(image_class_dir) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_file in image_files:
            # Find corresponding text file
            base_name = os.path.splitext(img_file)[0]
            text_file = base_name + '.txt'
            
            image_path = os.path.join(image_class_dir, img_file)
            text_path = os.path.join(text_class_dir, text_file)
            
            # Read text (if exists)
            text = ""
            if os.path.exists(text_path):
                try:
                    with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                except:
                    text = ""
            
            data.append({
                'image_path': image_path,
                'text': text,
                'label': class_to_idx[class_name],
                'class_name': class_name
            })
    
    return pd.DataFrame(data), sorted(classes)

# Load data
df, class_names = load_hybrid_data(IMAGE_DIR, TEXT_DIR)
print(f"✓ Loaded {len(df)} samples")
print(f"✓ Classes: {class_names}")

# Display distribution
print("\nClass distribution:")
print(df['class_name'].value_counts())

# ============================================
# STEP 2: SPLIT DATA
# ============================================

print("\n[2/7] Splitting Data...")

train_df, val_df = train_test_split(
    df, 
    test_size=0.2, 
    stratify=df['label'],
    random_state=42
)

print(f"✓ Train: {len(train_df)} samples")
print(f"✓ Val: {len(val_df)} samples")

# ============================================
# STEP 3: PREPARE TEXT FEATURES (TF-IDF)
# ============================================

print("\n[3/7] Vectorizing Text...")

vectorizer = TfidfVectorizer(
    max_features=TEXT_FEATURES,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.8,
    sublinear_tf=True,
    strip_accents='unicode'
)

# Fit on training text
X_train_text = vectorizer.fit_transform(train_df['text'])
X_val_text = vectorizer.transform(val_df['text'])

print(f"✓ Text features shape: {X_train_text.shape}")

# Save vectorizer
joblib.dump(vectorizer, f'{MODEL_DIR}/tfidf_vectorizer.pkl')
joblib.dump(class_names, f'{MODEL_DIR}/class_names.pkl')

# Convert to dense arrays for model
X_train_text_dense = X_train_text.toarray()
X_val_text_dense = X_val_text.toarray()

# ============================================
# STEP 4: PREPARE IMAGE DATA GENERATOR
# ============================================

print("\n[4/7] Preparing Image Generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    horizontal_flip=False
)

val_datagen = ImageDataGenerator(rescale=1./255)

def image_generator(df, datagen, batch_size, shuffle=True):
    """Custom generator that yields images and text together"""
    indices = np.arange(len(df))
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(df), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            batch_images = []
            batch_texts = []
            batch_labels = []
            
            for idx in batch_indices:
                # Load and preprocess image
                img_path = df.iloc[idx]['image_path']
                img = Image.open(img_path).convert('RGB')
                img = img.resize(IMG_SIZE)
                img_array = np.array(img) / 255.0
                
                # Apply augmentation (if training)
                if shuffle:
                    img_array = datagen.random_transform(img_array)
                
                batch_images.append(img_array)
                batch_labels.append(df.iloc[idx]['label'])
            
            batch_images = np.array(batch_images)
            batch_labels = tf.keras.utils.to_categorical(
                batch_labels, 
                num_classes=len(class_names)
            )
            
            # Get corresponding text features
            batch_texts = X_train_text_dense[batch_indices] if shuffle else X_val_text_dense[batch_indices]
            
            yield {'image_input': batch_images, 'text_input': batch_texts}, batch_labels

# Calculate steps
steps_per_epoch = len(train_df) // BATCH_SIZE
validation_steps = len(val_df) // BATCH_SIZE

print(f"✓ Steps per epoch: {steps_per_epoch}")
print(f"✓ Validation steps: {validation_steps}")

# ============================================
# STEP 5: BUILD HYBRID MODEL
# ============================================

print("\n[5/7] Building Hybrid Model...")

def create_hybrid_model(num_classes, text_features_dim):
    """
    Hybrid model combining vision and text
    """
    # Vision branch (EfficientNetB0)
    image_input = Input(shape=(*IMG_SIZE, 3), name='image_input')
    
    base_vision = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    base_vision.trainable = False  # Freeze initially
    
    x_vision = base_vision(image_input)
    x_vision = GlobalAveragePooling2D()(x_vision)
    x_vision = BatchNormalization()(x_vision)
    x_vision = Dense(256, activation='relu')(x_vision)
    x_vision = Dropout(0.5)(x_vision)
    
    # Text branch (TF-IDF)
    text_input = Input(shape=(text_features_dim,), name='text_input')
    x_text = BatchNormalization()(text_input)
    x_text = Dense(512, activation='relu')(x_text)
    x_text = Dropout(0.5)(x_text)
    x_text = Dense(256, activation='relu')(x_text)
    x_text = Dropout(0.4)(x_text)
    
    # Combine both branches
    combined = Concatenate()([x_vision, x_text])
    
    # Final classification layers
    x = BatchNormalization()(combined)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(
        inputs=[image_input, text_input],
        outputs=output
    )
    
    return model, base_vision

model, base_vision = create_hybrid_model(len(class_names), TEXT_FEATURES)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✓ Model parameters: {model.count_params():,}")
print(model.summary())

# ============================================
# CALLBACKS
# ============================================

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=10,
        min_lr=1e-8,
        verbose=1,
        mode='max'
    ),
    ModelCheckpoint(
        f'{MODEL_DIR}/best_hybrid_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
]

# ============================================
# STEP 6: TRAINING
# ============================================

print("\n[6/7] Training Hybrid Model...")
print("="*70)

# Create generators
train_gen = image_generator(train_df.reset_index(drop=True), train_datagen, BATCH_SIZE, shuffle=True)
val_gen = image_generator(val_df.reset_index(drop=True), val_datagen, BATCH_SIZE, shuffle=False)

# Train
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# Fine-tuning (optional)
best_val_acc = max(history.history['val_accuracy'])
print(f"\nBest validation accuracy: {best_val_acc:.4f}")

if best_val_acc < 0.90:
    print("\n" + "="*70)
    print("Fine-tuning vision branch...")
    print("="*70)
    
    # Unfreeze vision model
    base_vision.trainable = True
    for layer in base_vision.layers[:-50]:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE * 0.1),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_ft = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=50,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine histories
    for key in history.history:
        history.history[key].extend(history_ft.history[key])

# ============================================
# STEP 7: EVALUATION
# ============================================

print("\n[7/7] Evaluating Model...")
print("="*70)

# Load best model
model = tf.keras.models.load_model(f'{MODEL_DIR}/best_hybrid_model.keras')

# Prepare validation data
val_images = []
for img_path in val_df['image_path']:
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    val_images.append(np.array(img) / 255.0)

val_images = np.array(val_images)
val_labels = val_df['label'].values

# Predict
y_pred_proba = model.predict({
    'image_input': val_images,
    'text_input': X_val_text_dense
}, verbose=1)

y_pred = np.argmax(y_pred_proba, axis=1)
y_true = val_labels

# Metrics
accuracy = np.mean(y_pred == y_true)
print(f"\n✓ HYBRID MODEL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(report)

# Save report
with open(f'{OUTPUT_DIR}/classification_report.txt', 'w') as f:
    f.write("HYBRID MODEL (Vision + Text)\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write("="*70 + "\n\n")
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Regular CM
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[0],
            cbar_kws={'label': 'Count'})
axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Actual', fontsize=12, fontweight='bold')
axes[0].set_title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}', 
                  fontsize=14, fontweight='bold')

# Normalized CM
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[1],
            cbar_kws={'label': 'Percentage'})
axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Actual', fontsize=12, fontweight='bold')
axes[1].set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

# Training history
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.title('Hybrid Model Accuracy', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.title('Hybrid Model Loss', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_history.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ All results saved to {OUTPUT_DIR}/")
print(f"✓ Model saved to {MODEL_DIR}/")





# Summary
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nHybrid Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
# print("\nThis combines:")
# print("  • Visual features (document layout, structure)")
# print("  • Text features (content, vocabulary)")
