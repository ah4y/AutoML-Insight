"""Test script for AI Insights Engine"""

from core.ai_insights import get_ai_engine, DatasetStatistics

print("=" * 60)
print("🤖 Testing AI Insights Engine")
print("=" * 60)

# Initialize AI engine (reads from .env automatically)
print("\n📡 Initializing AI engine...")
engine = get_ai_engine()

if engine is None:
    print("❌ AI engine not initialized. Check your .env file.")
    print("   Make sure GROQ_API_KEY is set correctly.")
else:
    print(f"✅ AI engine ready: {engine.provider}/{engine.model}")
    
    # Test with sample statistics (Iris dataset example)
    print("\n📊 Creating sample dataset statistics...")
    stats = DatasetStatistics(
        n_samples=150,
        n_features=4,
        n_numeric=4,
        n_categorical=0,
        target_type="classification",
        n_classes=3,
        class_balance={'0': 0.33, '1': 0.33, '2': 0.34},
        missing_rate=0.0,
        feature_correlations=[
            ('sepal_length', 'petal_length', 0.87),
            ('sepal_width', 'petal_width', 0.82)
        ],
        top_features=['sepal_length', 'petal_length', 'petal_width'],
        data_quality_score=95.0
    )
    
    print("   ✓ Statistics created")
    print(f"   • Samples: {stats.n_samples}")
    print(f"   • Features: {stats.n_features}")
    print(f"   • Classes: {stats.n_classes}")
    print(f"   • Quality Score: {stats.data_quality_score}/100")
    
    print("\n🤖 Generating AI insights (this may take a few seconds)...")
    print("   Using provider:", engine.provider)
    
    try:
        insights = engine.generate_insights(stats, context="initial_analysis")
        
        print("\n" + "=" * 60)
        print("📊 AI Analysis Results:")
        print("=" * 60)
        
        for key, value in insights.items():
            print(f"\n🔹 {key.upper().replace('_', ' ')}:")
            if isinstance(value, list):
                for item in value:
                    print(f"   • {item}")
            elif isinstance(value, dict):
                for k, v in value.items():
                    print(f"   • {k}: {v}")
            else:
                print(f"   {value}")
        
        print("\n" + "=" * 60)
        print("✅ AI Insights Test PASSED!")
        print("=" * 60)
        
        print("\n🎉 Success! Your AI integration is working perfectly.")
        print("   Next step: Integrate into your Streamlit dashboard")
        
    except Exception as e:
        print(f"\n❌ Error generating insights: {e}")
        print("\nTroubleshooting:")
        print("1. Check your API key is valid")
        print("2. Check your internet connection")
        print("3. Try a different provider (openai or gemini)")
        print("4. Check the error details above")
