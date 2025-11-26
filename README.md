# Face Recognition from WhatsApp Chats

Recognize friends and family directly from your WhatsApp media folder.

Do you ever search endlessly for a specific picture in a chaotic group
chat?\
This repository automatically detects the people in your images and
sorts them into dedicated folders, making your photo collections easy to
browse.

------------------------------------------------------------------------

## 🚀 Pipeline Overview

The system performs the following steps:

1.  **Extract all video frames**\
    Frames are sampled from videos so that faces inside videos are also
    processed.

2.  **Detect faces**\
    A face detector identifies all faces in each image or frame.

3.  **Cluster embeddings with DBSCAN**\
    Embeddings from all detected faces are grouped into identity
    clusters without specifying the number of people beforehand.

4.  **Assign new predictions to cluster centers**\
    Each cluster is represented by a central embedding --- either:

    -   the **mean embedding**, or\
    -   the **geometric median** (preferred, as it avoids the mean
        shifting toward dense modes).

------------------------------------------------------------------------

## 🔍 Face Recognition Model: ArcFace

We use **ArcFace** because it produces highly discriminative face
embeddings by applying an angular margin penalty during training.\
This results in:

-   tight clusters for the same identity\
-   large angular separation between different identities\
-   improved verification and identification performance

------------------------------------------------------------------------

## 🧩 Clustering Method: DBSCAN

We use **DBSCAN** to cluster face embeddings because:

-   it requires **no predefined number of identities**\
-   it automatically identifies clusters of **arbitrary shapes and
    sizes**\
-   it is **robust to noise and outliers** (problematic or very
    low-quality face images)

This makes it ideal for organizing faces from real-world chat data where
quality varies.

------------------------------------------------------------------------

## 📌 Next Steps

Planned improvements and future directions for the project:


### **1. Incremental database**

Store embeddings and cluster centers so that new WhatsApp images can be
processed continuously without re-running the full pipeline.


### **2. Export cluster summaries**

Automatically create a summary for each person: - representative photos\
- number of appearances\
- time distribution\
- most common groups / contexts


### **3. Automatically detect drift, return a confidence score **
