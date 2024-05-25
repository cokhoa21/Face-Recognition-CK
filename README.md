## **Quick Start**
1. Install:
    
    ```bash
    # Clone Repo:
    git clone https://github.com/cokhoa21/Face-Recognition-CK.git
    
    # Install with Pip
    pip install -r requirements.txt

2. Update faces after adding face to data folder
    python update_faces.py --dataset data/ --embeddings output/embeddings.pickle --embedding-model openface_nn4.small2.v1.t7

3. Train with new output embeddings
    python train.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle

4. Demo 
    # Demo image
    python recognize_image.py --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle --image test_images/khoa3.png
    # Demo webcam
    python recognize_webcam.py --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle
    # Demo video
    Updating ...
    
