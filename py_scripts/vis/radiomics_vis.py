def visualize_classification_results(final_path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(final_path)

    plt.figure(figsize=(12, 8))

    sns.scatterplot(
        data=df,
        x='category',
        y='feature_class',
        hue='category',
        alpha=0.6,
        s=100
    )

    plt.title('Distribution of Feature Classes by Category', fontsize=14)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Feature Class (0-999)', fontsize=12)

    plt.xticks(rotation=45, ha='right')

    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    plt.savefig('classification_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization results saved as: classification_visualization.png")

    plt.show()

    print("\nClassification statistics:")
    print("\nSample count for each category:")
    print(df['category'].value_counts())

    print("\nFeature class distribution for each category:")
    for category in df['category'].unique():
        category_stats = df[df['category'] == category]['feature_class'].describe()
        print(f"\n{category}:")
        print(f"  Average feature class: {category_stats['mean']:.2f}")
        print(f"  Standard deviation: {category_stats['std']:.2f}")
        print(f"  Minimum: {category_stats['min']:.0f}")
        print(f"  Maximum: {category_stats['max']:.0f}")