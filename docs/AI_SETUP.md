# AI Integration Setup Guide 🤖

This guide walks you through setting up AI-powered insights in AutoML-Insight.

## 🔐 Security First!

**NEVER** commit API keys to git or share them publicly. Always use `.env` files.

## Step 1: Get Your API Keys (Choose One or More)

### Option A: Groq (Recommended - Fast & Free Tier)
1. Visit https://console.groq.com/
2. Sign up/Login
3. Go to API Keys section
4. Create new key
5. Copy the key (starts with `gsk_...`)

**Why Groq?** Lightning-fast inference, generous free tier, great for real-time insights.

### Option B: OpenAI (Most Powerful)
1. Visit https://platform.openai.com/
2. Sign up/Login
3. Go to API Keys
4. Create new key
5. Copy the key (starts with `sk-...`)

**Why OpenAI?** Best quality insights, GPT-4 access, most capable reasoning.

### Option C: Google Gemini (Alternative)
1. Visit https://makersuite.google.com/
2. Sign up/Login
3. Get API key
4. Copy the key

**Why Gemini?** Free tier, good quality, Google ecosystem integration.

### LangChain (Optional - For Monitoring)
1. Visit https://smith.langchain.com/
2. Sign up/Login
3. Settings → API Keys
4. Create new key

**Why LangChain?** Track AI calls, debug prompts, monitor performance.

## Step 2: Secure Configuration

### Create .env File

```bash
# In your project root, copy the example
cp .env.example .env

# Open .env in a text editor (DO NOT use git-tracked files!)
notepad .env
```

### Add Your Keys to .env

```bash
# Groq API (Fast inference with Llama models)
GROQ_API_KEY=gsk_YOUR_NEW_KEY_HERE

# LangChain API (Optional - for tracing)
LANGCHAIN_API_KEY=lsv2_YOUR_NEW_KEY_HERE
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=automl-insight

# OpenAI API (Powerful GPT-4 analysis)
OPENAI_API_KEY=sk-proj-YOUR_NEW_KEY_HERE

# Google Gemini API (Alternative LLM)
GEMINI_API_KEY=YOUR_NEW_KEY_HERE

# AI Configuration
DEFAULT_LLM_PROVIDER=groq  # Options: groq, openai, gemini
DEFAULT_MODEL=llama-3.1-70b-versatile
ENABLE_AI_INSIGHTS=true
AI_TEMPERATURE=0.3
AI_MAX_TOKENS=2000
```

### Verify .env is Ignored by Git

```bash
# Check .gitignore includes .env
cat .gitignore | grep "\.env"

# Should see:
# .env
# .env.local
# .env.*.local
```

## Step 3: Install AI Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `python-dotenv` - Load environment variables
- `groq` - Groq API client
- `openai` - OpenAI API client
- `google-generativeai` - Gemini API client
- `langchain` - Optional monitoring

## Step 4: Test the Integration

Create a test script `test_ai.py`:

```python
from core.ai_insights import get_ai_engine, DatasetStatistics

# Initialize AI engine (reads from .env automatically)
engine = get_ai_engine()

if engine is None:
    print("❌ AI engine not initialized. Check your .env file.")
else:
    print(f"✅ AI engine ready: {engine.provider}/{engine.model}")
    
    # Test with sample statistics
    stats = DatasetStatistics(
        n_samples=150,
        n_features=4,
        n_numeric=4,
        n_categorical=0,
        target_type="classification",
        n_classes=3,
        class_balance={'0': 0.33, '1': 0.33, '2': 0.34},
        missing_rate=0.0,
        feature_correlations=[],
        top_features=['sepal_length', 'petal_length'],
        data_quality_score=95.0
    )
    
    print("\n🤖 Generating AI insights...")
    insights = engine.generate_insights(stats, context="initial_analysis")
    
    print("\n📊 AI Analysis:")
    for key, value in insights.items():
        print(f"\n{key}:")
        if isinstance(value, list):
            for item in value:
                print(f"  • {item}")
        else:
            print(f"  {value}")
```

Run it:
```bash
python test_ai.py
```

Expected output:
```
✅ AI engine ready: groq/llama-3.1-70b-versatile
🤖 Generating AI insights...
📊 AI Analysis:
...
```

## Step 5: Using AI Insights in Your App

### In app/ui_dashboard.py:

```python
from core.ai_insights import get_ai_engine

# Initialize once at app startup
if 'ai_engine' not in st.session_state:
    st.session_state.ai_engine = get_ai_engine()

# After loading data, get AI insights
if st.session_state.ai_engine and data is not None:
    with st.spinner("🤖 Generating AI insights..."):
        stats = st.session_state.ai_engine.analyze_dataset(
            data, 
            target_col=target_column,
            task_type=task_type
        )
        
        insights = st.session_state.ai_engine.generate_insights(
            stats, 
            context="initial_analysis"
        )
    
    # Display insights
    st.markdown("### 🤖 AI-Powered Insights")
    
    if 'summary' in insights:
        st.info(insights['summary'])
    
    if 'strengths' in insights:
        st.success("**Strengths:**")
        for strength in insights['strengths']:
            st.write(f"✓ {strength}")
    
    if 'challenges' in insights:
        st.warning("**Challenges:**")
        for challenge in insights['challenges']:
            st.write(f"⚠ {challenge}")
    
    if 'recommendations' in insights:
        st.info("**Recommendations:**")
        for rec in insights['recommendations']:
            st.write(f"→ {rec}")
```

## 🎯 Features You Get

### 1. **Dynamic Dataset Analysis**
- Automatically analyzes your data statistics
- Identifies strengths and weaknesses
- No static rules - AI adapts to YOUR data

### 2. **Smart Model Recommendations**
- Suggests best models for your specific dataset
- Explains why certain models will work well
- Warns about models that might struggle

### 3. **Results Interpretation**
- Helps understand if your results are good
- Identifies suspicious patterns
- Provides context-specific guidance

### 4. **Actionable Insights**
- Every insight is specific to your dataset
- Practical recommendations you can implement
- Explains the "why" behind suggestions

## 🔧 Configuration Options

### Switch AI Providers

In `.env`:
```bash
DEFAULT_LLM_PROVIDER=openai  # Change to openai or gemini
```

### Adjust Temperature (Creativity)

```bash
AI_TEMPERATURE=0.1  # More deterministic (consistent)
AI_TEMPERATURE=0.7  # More creative (varied)
```

### Disable AI (Use Traditional Rules)

```bash
ENABLE_AI_INSIGHTS=false
```

## 💰 Cost Considerations

### Groq
- **Free Tier**: Generous limits
- **Cost**: Very cheap for paid tier
- **Speed**: Fastest (under 1 second)

### OpenAI
- **Free Tier**: $5 credit for new accounts
- **Cost**: ~$0.001-0.01 per insight
- **Speed**: 2-5 seconds

### Gemini
- **Free Tier**: Good limits
- **Cost**: Competitive pricing
- **Speed**: 1-3 seconds

**Estimated costs for typical usage:**
- 100 datasets/day with Groq: Free or $0.10
- 100 datasets/day with OpenAI: $0.50-1.00
- 100 datasets/day with Gemini: Free or $0.20

## 🐛 Troubleshooting

### "API key for groq not found"
→ Check `.env` file exists and has `GROQ_API_KEY=...`
→ Restart your app after adding keys

### "Invalid API key"
→ Make sure you copied the entire key
→ Check for extra spaces or quotes
→ Generate a new key if unsure

### "AI insights not showing"
→ Check `ENABLE_AI_INSIGHTS=true` in `.env`
→ Verify API key is valid
→ Check logs for error messages

### "JSON parsing error"
→ This is usually temporary, try again
→ Consider switching to a different provider

## 📚 Best Practices

1. **Start with Groq** - Fastest and cheapest for development
2. **Use OpenAI for production** - Best quality when it matters
3. **Monitor API usage** - Use LangChain tracing
4. **Cache insights** - Don't re-generate for same dataset
5. **Handle errors gracefully** - Have fallbacks
6. **Never commit .env** - Check `.gitignore` is correct
7. **Rotate keys regularly** - Security best practice
8. **Use environment-specific keys** - Dev vs Production

## 🚀 Next Steps

1. ✅ Set up `.env` with your NEW keys (after revoking old ones!)
2. ✅ Test with `test_ai.py`
3. ✅ Integrate into dashboard
4. ✅ Monitor usage and costs
5. ✅ Iterate on prompts for better insights

## 🔗 Resources

- [Groq Documentation](https://console.groq.com/docs)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Gemini API Docs](https://ai.google.dev/docs)
- [LangChain Docs](https://python.langchain.com/docs/)

---

**Remember: The keys you shared are now COMPROMISED. Revoke them FIRST, then follow this guide with NEW keys!** 🔐
