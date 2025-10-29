# ☁️ Cloud Execution Guide

## 🎯 Overview
AutoML-Insight now supports **smart execution mode selection** with automatic recommendations based on your dataset size and available resources.

---

## ✨ Features

### 1. **Automatic Resource Detection**
- Detects available RAM, CPU cores, GPU
- Calculates required memory for your dataset
- Recommends best execution mode (local vs cloud)

### 2. **Smart Recommendations**
```
Dataset: 150,000 × 361,000 features
Required Memory: ~403 GB
Available RAM: 8 GB

❌ Cannot run locally
✅ Recommend: Cloud execution (Colab/Kaggle)
```

### 3. **Two Execution Modes**

#### 🖥️ **Local Machine**
- Runs directly on your computer
- Best for small/medium datasets (< 10GB RAM required)
- Immediate results
- Uses smart feature selection (1000-5000 features)

#### ☁️ **Cloud (Colab/Kaggle)**
- Generates ready-to-use Jupyter notebook
- Upload to Google Colab (free GPU, 12 GB RAM)
- Or Kaggle (free GPU, 30 GB RAM)
- Can handle larger datasets (up to 10,000 features)
- Download results and upload back to app

---

## 📋 How to Use

### **Option 1: Local Execution** (Small Datasets)

1. Upload your CSV file
2. Select task type and target column
3. App will show: ✅ "Your machine can handle this dataset"
4. Choose "🖥️ Local Machine"
5. Click "🚀 Run AutoML Locally"
6. Wait for results

### **Option 2: Cloud Execution** (Large Datasets)

1. Upload your CSV file
2. App shows: ⚠️ "Dataset needs 50 GB but only 8 GB available. Recommend cloud."
3. Choose "☁️ Cloud (Colab/Kaggle)"
4. Click "📓 Generate Cloud Notebook"
5. Download the generated notebook
6. Go to [Google Colab](https://colab.research.google.com)
7. Upload notebook (File → Upload notebook)
8. Enable GPU (Runtime → Change runtime type → GPU)
9. Run all cells (Runtime → Run all)
10. Upload your CSV when prompted
11. Wait for training (5-30 minutes)
12. Download `automl_results.json`
13. Back in AutoML-Insight app, upload the results file
14. View results in dashboard!

---

## 🔧 Configuration

### Feature Selection Limits

**Local Mode:**
- 8 GB RAM → 1,000 features max
- 16 GB RAM → 2,000 features max
- 32+ GB RAM → 5,000 features max

**Cloud Mode:**
- Google Colab (12 GB RAM) → 5,000 features
- Kaggle (30 GB RAM) → 10,000 features

### Customize

Edit `core/preprocess.py` line 17:
```python
def __init__(self, max_features: int = 1000):  # Change this number
```

---

## 💡 Pro Tips

1. **Start Local**: Try local execution first, even if warned. Feature selection might make it work!

2. **Use Cloud for**:
   - > 100,000 samples
   - > 1,000 features  
   - Need GPU acceleration

3. **Feature Selection**:
   - Automatically keeps most important features
   - Uses correlation with target variable
   - No accuracy loss for most datasets

4. **Cloud Benefits**:
   - Free GPU (10x faster for XGBoost, deep learning)
   - More RAM (12-30 GB vs typical 8 GB)
   - Can run longer experiments

---

## 🎓 Example Workflows

### Small Dataset (Iris - 150 × 4)
- ✅ Run locally
- Completes in < 1 minute
- All features used

### Medium Dataset (10,000 × 100)
- ✅ Run locally  
- Feature selection: 100 → 100 (keeps all)
- Completes in 2-5 minutes

### Large Dataset (150,000 × 361,000)
- ❌ Cannot run locally (403 GB required)
- ✅ Use cloud execution
- Feature selection: 361,000 → 5,000
- Completes in 10-20 minutes on Colab GPU

---

## 🐛 Troubleshooting

### "Out of memory" error locally
- Switch to cloud mode
- Or reduce `max_features` in preprocess.py

### Cloud notebook fails
- Check you enabled GPU in Colab
- Ensure you uploaded CSV file
- Check target column name matches

### Results upload fails
- Ensure file is `automl_results.json`
- File must come from the generated notebook
- Check JSON format is valid

---

## 🚀 Future Enhancements

- [ ] Auto-upload to cloud (no manual notebook upload)
- [ ] Real-time progress tracking from cloud
- [ ] Support for AWS SageMaker
- [ ] Auto-hyperparameter tuning on cloud
- [ ] Ensemble models trained across cloud providers

---

## 📞 Support

Having issues? Check:
1. System resources in sidebar
2. Recommended execution mode
3. Feature selection logs in preprocessing
4. This guide's troubleshooting section

**Happy Training!** 🎉
