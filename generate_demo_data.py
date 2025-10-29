"""Generate demo datasets for AutoML-Insight."""

from sklearn.datasets import load_iris, load_wine
import pandas as pd
from pathlib import Path


def generate_demo_datasets():
    """Generate and save demo datasets."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Iris dataset
    print("Generating Iris dataset...")
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    iris_df.to_csv(data_dir / "demo_iris.csv", index=False)
    print(f"✓ Saved to {data_dir / 'demo_iris.csv'}")
    print(f"  Shape: {iris_df.shape}")
    
    # Wine dataset
    print("\nGenerating Wine dataset...")
    wine = load_wine()
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_df['target'] = wine.target
    wine_df.to_csv(data_dir / "demo_wine.csv", index=False)
    print(f"✓ Saved to {data_dir / 'demo_wine.csv'}")
    print(f"  Shape: {wine_df.shape}")
    
    print("\n✅ Demo datasets generated successfully!")


if __name__ == "__main__":
    generate_demo_datasets()
