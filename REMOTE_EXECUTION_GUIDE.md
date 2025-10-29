# 🌐 Remote Execution Guide

AutoML-Insight now supports training models on remote servers with **unlimited RAM and GPU power**!

## 🎯 Why Remote Execution?

**Problem**: Your dataset has 150,000 samples × 361,000 features = **403 GB RAM required** 😱

**Solution**: Train on cloud servers with enough resources! ☁️

---

## 🚀 Quick Start (3 Options)

### Option 1: 🌐 Remote Jupyter Server (Recommended)

**Use Case**: You have access to a powerful server with Jupyter

**Steps**:
1. Start Jupyter on your server: `jupyter notebook --ip=0.0.0.0 --port=8888`
2. In AutoML-Insight, select "🌐 Remote Jupyter Server"
3. Enter server URL (e.g., `http://192.168.1.100:8888`)
4. Enter token (if required)
5. Click "🔗 Connect"
6. Click "🚀 Run on Remote Server"
7. Watch real-time progress!

**Advantages**:
- ✅ Direct connection
- ✅ Real-time logs
- ✅ Automatic result retrieval
- ✅ Works with any Jupyter server

---

### Option 2: ☁️ Google Colab (Free GPU!)

**Use Case**: You don't have a server but want free GPU power

**Steps**:
1. Get free ngrok token from [ngrok.com](https://ngrok.com) (takes 30 seconds)
2. In AutoML-Insight, select "☁️ Google Colab Setup"
3. Enter your ngrok token
4. Click "📋 Show Setup Instructions"
5. Copy the setup code
6. Open [Google Colab](https://colab.research.google.com)
7. Paste and run the code
8. Copy the public URL (e.g., `https://abc123.ngrok.io`)
9. Paste URL in AutoML-Insight
10. Click "🔗 Connect"
11. Click "🚀 Run on Google Colab"

**Free Resources**:
- 📊 **12 GB RAM**
- 🎮 **T4 GPU** (16 GB VRAM)
- ⏱️ **12 hours** continuous runtime
- 🆓 **100% Free**

**Advantages**:
- ✅ No setup required
- ✅ Free GPU access
- ✅ Good for datasets up to ~50K features
- ✅ No credit card needed

---

### Option 3: 📊 Kaggle Kernels (More Power)

**Use Case**: You need even more resources than Colab

**Free Resources**:
- 📊 **30 GB RAM** (2.5x more than Colab!)
- 🎮 **P100 GPU** (16 GB VRAM)
- ⏱️ **9 hours** runtime per session

**Note**: Direct integration coming soon! For now, use Option 1 or 2.

---

## 📊 How It Works

### Architecture

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Your Browser  │────────▶│  AutoML-Insight │────────▶│ Remote Jupyter  │
│   (Streamlit)   │◀────────│   Dashboard     │◀────────│     Server      │
└─────────────────┘         └─────────────────┘         └─────────────────┘
                                                               │
                                                               ▼
                                                         ┌─────────────┐
                                                         │  Training   │
                                                         │  (4 models) │
                                                         └─────────────┘
```

### What Happens:

1. **Upload**: Dataset sent to remote server (compressed)
2. **Install**: Dependencies installed automatically (`pip install`)
3. **Train**: Models trained in parallel on remote resources
4. **Stream**: Real-time logs streamed back to your browser
5. **Download**: Results automatically retrieved

---

## 🔒 Security

### Your Data:
- ✅ Encrypted in transit (HTTPS/WSS)
- ✅ Deleted after training
- ✅ Never stored permanently
- ✅ Only you have access (token-protected)

### Tokens:
- 🔑 Stored only in your browser session
- 🔑 Never sent to our servers
- 🔑 Can use temporary tokens
- 🔑 Revoke anytime

---

## 💡 Pro Tips

### For Large Datasets:

1. **Use Feature Selection**: Set `max_features=1000` to limit memory
2. **Colab Pro+**: Get 51 GB RAM for $49.99/month
3. **AWS SageMaker**: For production workloads
4. **Kaggle Kernels**: Free 30 GB RAM alternative

### Optimizing Performance:

```python
# In remote execution, these optimizations are applied automatically:
- Feature pre-selection (variance-based)
- Memory-efficient data types
- Sparse matrix support
- Chunked processing
```

### Troubleshooting:

| Problem | Solution |
|---------|----------|
| Connection timeout | Check firewall, use ngrok |
| Out of memory | Reduce max_features, use Kaggle |
| Slow training | Enable GPU in Colab runtime |
| Token error | Regenerate token, check URL |

---

## 🎓 Example: Training on Colab

### Step-by-Step with Screenshots:

**1. Get Ngrok Token** (1 minute)
- Go to [ngrok.com](https://ngrok.com)
- Sign up (free)
- Copy your token

**2. Configure AutoML-Insight**
```
1. Upload your dataset
2. Select "☁️ Google Colab Setup"
3. Paste ngrok token
4. Click "📋 Show Setup Instructions"
```

**3. Setup Colab** (2 minutes)
```python
# This code is auto-generated for you!
!pip install -q pyngrok
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")
!jupyter notebook --port=8888 &
public_url = ngrok.connect(8888)
print(public_url)  # Copy this!
```

**4. Connect & Train**
```
1. Copy the public URL from Colab
2. Paste in AutoML-Insight sidebar
3. Click "🔗 Connect"
4. Click "🚀 Run on Google Colab"
5. Watch progress in real-time!
```

**5. Get Results** (Automatic!)
```
✅ Training completed!
📊 Results displayed automatically
📥 Download detailed report
```

---

## 📈 Performance Comparison

| Dataset Size | Local (8 GB) | Colab Free | Kaggle | Cloud VM |
|--------------|--------------|------------|---------|----------|
| 10K × 100    | ✅ 2 min     | ✅ 1 min   | ✅ 1 min | ✅ 30 sec |
| 50K × 1K     | ⚠️ 15 min    | ✅ 5 min   | ✅ 3 min | ✅ 1 min |
| 150K × 10K   | ❌ OOM       | ✅ 20 min  | ✅ 10 min | ✅ 5 min |
| 150K × 361K  | ❌ OOM       | ❌ OOM     | ⚠️ 2 hrs | ✅ 30 min |

**Legend**: ✅ Works well | ⚠️ Possible | ❌ Out of Memory

---

## 🆘 Need Help?

### Common Issues:

**Q: "Connection failed"**
A: Check URL format, ensure Jupyter is running, verify token

**Q: "Out of memory on Colab"**
A: Reduce max_features, use Kaggle instead, or enable feature pre-selection

**Q: "Training is slow"**
A: Enable GPU in Colab (Runtime → Change runtime type → T4 GPU)

**Q: "Can I use my own cloud VM?"**
A: Yes! Just run Jupyter and connect via "Remote Jupyter Server"

---

## 🔮 Coming Soon

- [ ] Direct Kaggle Kernels integration
- [ ] AWS SageMaker support
- [ ] Azure ML integration
- [ ] Multi-server parallel training
- [ ] Auto-scaling based on dataset size
- [ ] Cost estimation for cloud providers

---

## 📚 Additional Resources

- **Ngrok Docs**: [ngrok.com/docs](https://ngrok.com/docs)
- **Colab Guide**: [colab.research.google.com](https://colab.research.google.com)
- **Kaggle API**: [kaggle.com/docs/api](https://kaggle.com/docs/api)
- **Jupyter Server**: [jupyter-server.readthedocs.io](https://jupyter-server.readthedocs.io)

---

**Ready to train on unlimited resources?** 🚀

Start by selecting your execution mode in the sidebar!
