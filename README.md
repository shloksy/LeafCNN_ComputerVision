# 🍅 Protectomato 
Check it out: [protectomato.streamlit.app](url)

**Your smart ally against tomato plant disease**  
A machine learning project that trains a compact convolutional neural network to recognize 10 tomato-leaf conditions, then packages it into an easy-to-use Streamlit web app for drag-and-drop disease diagnosis.

## Key Features
- **Data-Driven Learning**  
  • Trained on 16,000+ labeled leaf images (9 disease classes + healthy)  
  • Train/validation/test split with augmentations (flips, rotations)  
- **Compact CNN Architecture**  
  • Three Conv→ReLU→MaxPool blocks (32→64→128 filters)  
  • Two dense layers (512→256) and softmax output  
  • Balances high accuracy with speed  
- **Web-app Deployment**  
  • Simple Streamlit front-end for real-time, confidence-scored predictions  
  • Sample image gallery and upload support  
  • Descriptive disease summaries
