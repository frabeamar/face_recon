# face_recon
Recognize friends and families from whatsapp chats. 
Do you ever look for a picture in a group chat from a friend but could not find it? This repository automatically recognizes the people in your pictures and saves them in appropriate folders. 

# Pipeline
The pipeline is set up as following:
 - extract al video frame
 - detect all faces
 - automatically detect people via dbscan clustering.
 - assign new prediction to cluster centers, this can be identified via mean of all embedding or the geometric median. 
 The geometric median mitigates pushing the mean embedding to the mode. 

 
## Face Recognition: ArcFace
We use ArcFace for face recognition because it produces highly discriminative embeddings by enforcing an angular margin between classes. This ensures that embeddings of the same person are tightly clustered while embeddings of different people are well separated, improving accuracy for verification and identification task. 

# Clustering : DBScan
We use DBSCAN for clustering faces because it can automatically discover clusters of varying shapes and sizes without requiring the number of clusters in advance. It is robust to outliers, making it well-suited for face embeddings where some faces may be noisy or not clearly associated with any cluster.

 
