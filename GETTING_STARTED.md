# 🚀 Getting Started with AutoML-Insight

## ⚡ Super Quick Start (3 steps)

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app/main.py

# 3. Try Demo Mode!
# → Check "Demo Mode" in sidebar
# → Select "Iris" dataset
# → Click "Run AutoML"
# → Explore results in tabs!
```

---

## 📖 What Just Happened?

You now have a **complete, production-ready AutoML platform** with:

✅ **7 supervised models** trained automatically  
✅ **5 clustering models** for unsupervised learning  
✅ **Nested cross-validation** with confidence intervals  
✅ **SHAP explainability** for all models  
✅ **Interactive dashboard** with beautiful visualizations  
✅ **PDF reports** you can download  
✅ **Meta-learning** recommendations  

---

## 🎮 Demo Mode Tutorial

### Step 1: Launch Dashboard
```powershell
streamlit run app/main.py
```

### Step 2: Enable Demo Mode
- Look at the sidebar (left)
- Check the box "🎮 Demo Mode"
- Select "Iris" dataset

### Step 3: Run AutoML
- Choose "Classification" as task
- Click "🚀 Run AutoML"
- Wait ~1-2 minutes

### Step 4: Explore Results
**Tab 1 - Data Overview:**
- See dataset statistics
- View correlation heatmap
- Check data quality

**Tab 2 - Models:**
- Compare all 7 models
- See accuracy with confidence intervals
- Check ROC curves
- View confusion matrix

**Tab 3 - Explainability:**
- Select a model
- See SHAP feature importance
- Discover which features matter most

**Tab 4 - Recommendation:**
- Get the best model suggestion
- Read why it's recommended
- See alternatives

**Tab 5 - Report:**
- Click "Download PDF Report"
- Get comprehensive analysis

---

## 📊 Try Your Own Data

### Data Requirements
- **Format**: CSV file
- **Size**: Works with 100s to 10,000s of rows
- **Features**: Numeric and/or categorical
- **Target**: For classification, any column

### Steps
1. **Upload**: Click "Choose a CSV file" in sidebar
2. **Select Target**: Choose your target column
3. **Configure**: Set random seed (optional)
4. **Run**: Click "🚀 Run AutoML"
5. **Analyze**: Explore results in tabs

### Example Datasets
Already included:
- `data/demo_iris.csv` - 3-class flower classification
- `data/demo_wine.csv` - Wine quality classification

---

## 🔬 CLI Mode (Advanced)

For reproducible experiments:

```powershell
# Run with default config
python experiments/run_experiment.py

# Run with custom config
python experiments/run_experiment.py --config my_config.yaml
```

Results saved to: `results/runs/<timestamp>/`

---

## 🎯 What Makes This Special?

### 🔬 Scientific Rigor
- Nested cross-validation (unbiased estimates)
- Confidence intervals (95% CI)
- Statistical tests (McNemar, Wilcoxon)
- Multiple metrics (Accuracy, F1, ROC-AUC, etc.)

### 🧠 Intelligence
- Meta-learning model selector
- Automatic hyperparameter tuning
- Smart ensemble creation
- Heuristic fallback rules

### 📊 Transparency
- SHAP explanations for every model
- Feature importance rankings
- Complete result logging
- Reproducible with seeds

### 💎 Production Quality
- Clean OOP architecture
- Comprehensive error handling
- Rotating log files
- Unit test coverage
- Full documentation

---

## 📈 Expected Results

### Iris Dataset (Demo)
- **Task**: 3-class classification
- **Best Model**: Typically LogisticRegression or SVM
- **Accuracy**: ~95-97%
- **Important Features**: Petal length, petal width
- **Runtime**: ~1-2 minutes

### Wine Dataset (Demo)
- **Task**: 3-class classification
- **Best Model**: Typically RandomForest or XGBoost
- **Accuracy**: ~95-98%
- **Important Features**: Alcohol, flavonoids, proline
- **Runtime**: ~2-3 minutes

---

## 🛠️ Customization

### Change CV Folds
Edit `app/config.yaml`:
```yaml
training:
  n_folds: 3  # Change from 5 to 3 for faster runs
  n_repeats: 2
```

### Add New Models
Edit `core/models_supervised.py`:
```python
def get_supervised_models(random_state=42):
    models = {
        'MyModel': MyClassifier(random_state=random_state),
        # ... existing models
    }
    return models
```

### Customize Visualizations
Edit `core/visualize.py` - add new plot functions

---

## 🐛 Troubleshooting

### Issue: "No module named 'streamlit'"
**Fix:**
```powershell
pip install streamlit
```

### Issue: "No demo datasets found"
**Fix:**
```powershell
python generate_demo_data.py
```

### Issue: "CUDA error"
**Fix:** PyTorch will automatically use CPU. Ignore warning.

### Issue: Slow performance
**Fix:** 
- Reduce CV folds in `app/config.yaml`
- Disable some models
- Use smaller dataset

---

## 📚 Learn More

- **Full Documentation**: See `README.md`
- **User Guide**: See `USER_GUIDE.md`
- **Architecture**: See `PROJECT_STRUCTURE.md`
- **Implementation**: See `IMPLEMENTATION_COMPLETE.md`

---

## 🎓 For Students/Researchers

This is a **complete reference implementation** of:
- AutoML pipeline design
- Model evaluation best practices
- Explainable AI (SHAP)
- Meta-learning for model selection
- Production ML systems
- Dashboard development

**Use it to learn, extend, or as a starting point for your own projects!**

---

## ✨ Quick Commands Cheat Sheet

```powershell
# Dashboard
streamlit run app/main.py

# CLI Experiment
python experiments/run_experiment.py

# Generate Demo Data
python generate_demo_data.py

# Run Tests
pytest tests/

# Install in Development Mode
pip install -e .
```

---

## 🎉 You're Ready!

Everything is set up and ready to go. Just run:

```powershell
streamlit run app/main.py
```

And start exploring automated machine learning!

---

**Questions?** Check the documentation files or open an issue.

**Enjoy your AutoML journey! 🤖✨**
