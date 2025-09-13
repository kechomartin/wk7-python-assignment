# Data Analysis with Pandas and Matplotlib Assignment

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 60)
print("DATA ANALYSIS WITH PANDAS AND MATPLOTLIB")
print("=" * 60)


# TASK 1: LOAD AND EXPLORE THE DATASET


print("\n📊 TASK 1: LOADING AND EXPLORING THE DATASET")
print("-" * 50)

try:
    # Load the Iris dataset
    iris_data = load_iris()
    
    # Create a pandas DataFrame
    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df['species'] = iris_data.target
    
    # Map numerical species to names
    species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species_name'] = df['species'].map(species_mapping)
    
    print("✅ Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# Display first few rows
print("\n🔍 First 5 rows of the dataset:")
print(df.head())

# Explore the structure of the dataset
print("\n📋 Dataset Information:")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print(f"Column names: {list(df.columns)}")

print("\n🔢 Data Types:")
print(df.dtypes)

print("\n📊 Dataset Description:")
print(df.describe())

# Check for missing values
print("\n❓ Missing Values Check:")
missing_values = df.isnull().sum()
print(missing_values)

if missing_values.sum() > 0:
    print("⚠️ Missing values found! Cleaning dataset...")
    df = df.dropna()  # or use df.fillna() for filling missing values
    print("✅ Dataset cleaned!")
else:
    print("✅ No missing values found!")


# TASK 2: BASIC DATA ANALYSIS

print("\n\n🔬 TASK 2: BASIC DATA ANALYSIS")
print("-" * 50)

# Basic statistics of numerical columns
print("\n📈 Basic Statistics of Numerical Columns:")
numerical_cols = df.select_dtypes(include=[np.number]).columns
print(df[numerical_cols].describe())

# Group by species and compute mean
print("\n🌸 Mean values by Species:")
species_analysis = df.groupby('species_name')[numerical_cols[:-1]].mean()
print(species_analysis)

# Additional analysis - correlations
print("\n🔗 Correlation Matrix:")
correlation_matrix = df[numerical_cols[:-1]].corr()
print(correlation_matrix)

# Key findings
print("\n🔍 KEY FINDINGS:")
print("1. Setosa species has the smallest petal dimensions but largest sepal width")
print("2. Virginica species generally has the largest dimensions across all features")
print("3. Strong positive correlation between petal length and petal width (0.96)")
print("4. Moderate correlation between sepal length and petal length (0.87)")


# TASK 3: DATA VISUALIZATION

print("\n\n📊 TASK 3: DATA VISUALIZATION")
print("-" * 50)

# Create a figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Line Chart - Trends over sample index (simulating time series)
plt.subplot(2, 3, 1)
for species in df['species_name'].unique():
    species_data = df[df['species_name'] == species]
    plt.plot(species_data.index, species_data['sepal length (cm)'], 
             label=f'{species.capitalize()}', marker='o', markersize=3)
plt.title('Sepal Length Trends Across Samples', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Bar Chart - Average measurements by species
plt.subplot(2, 3, 2)
mean_petal_length = df.groupby('species_name')['petal length (cm)'].mean()
bars = plt.bar(mean_petal_length.index, mean_petal_length.values, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.title('Average Petal Length by Species', fontsize=14, fontweight='bold')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.xticks(rotation=45)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.2f}', ha='center', va='bottom')

# 3. Histogram - Distribution of sepal width
plt.subplot(2, 3, 3)
plt.hist(df['sepal width (cm)'], bins=15, edgecolor='black', alpha=0.7, color='skyblue')
plt.title('Distribution of Sepal Width', fontsize=14, fontweight='bold')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Add statistics text
mean_val = df['sepal width (cm)'].mean()
std_val = df['sepal width (cm)'].std()
plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
plt.legend()

# 4. Scatter Plot - Relationship between sepal length and petal length
plt.subplot(2, 3, 4)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, species in enumerate(df['species_name'].unique()):
    species_data = df[df['species_name'] == species]
    plt.scatter(species_data['sepal length (cm)'], species_data['petal length (cm)'], 
                c=colors[i], label=species.capitalize(), alpha=0.7, s=50)

plt.title('Sepal Length vs Petal Length', fontsize=14, fontweight='bold')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Box Plot - Distribution comparison
plt.subplot(2, 3, 5)
df.boxplot(column='petal width (cm)', by='species_name', ax=plt.gca())
plt.title('Petal Width Distribution by Species', fontsize=14, fontweight='bold')
plt.xlabel('Species')
plt.ylabel('Petal Width (cm)')
plt.suptitle('')  # Remove automatic title

# 6. Heatmap - Correlation matrix
plt.subplot(2, 3, 6)
correlation_matrix = df[numerical_cols[:-1]].corr()
im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(im)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

# Set tick labels
feature_names = [name.split(' (')[0].replace(' ', '\n') for name in numerical_cols[:-1]]
plt.xticks(range(len(feature_names)), feature_names, rotation=45)
plt.yticks(range(len(feature_names)), feature_names)

# Add correlation values
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix)):
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# ADDITIONAL ADVANCED VISUALIZATIONS


print("\n📈 CREATING ADDITIONAL ADVANCED VISUALIZATIONS...")

# Create advanced visualizations using seaborn
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Pair plot style visualization
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', 
                hue='species_name', ax=axes[0,0])
axes[0,0].set_title('Sepal Length vs Width by Species', fontweight='bold')

# 2. Violin plot
sns.violinplot(data=df, x='species_name', y='petal length (cm)', ax=axes[0,1])
axes[0,1].set_title('Petal Length Distribution by Species', fontweight='bold')
axes[0,1].tick_params(axis='x', rotation=45)

# 3. Stacked bar chart
species_counts = df['species_name'].value_counts()
axes[1,0].bar(species_counts.index, species_counts.values, 
              color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[1,0].set_title('Sample Count by Species', fontweight='bold')
axes[1,0].tick_params(axis='x', rotation=45)

# 4. Feature comparison radar-like chart
mean_values = df.groupby('species_name')[numerical_cols[:-1]].mean()
x_pos = np.arange(len(numerical_cols[:-1]))
width = 0.25

for i, species in enumerate(mean_values.index):
    axes[1,1].bar(x_pos + i*width, mean_values.loc[species], 
                  width, label=species.capitalize(), alpha=0.8)

axes[1,1].set_title('Mean Feature Values by Species', fontweight='bold')
axes[1,1].set_xlabel('Features')
axes[1,1].set_ylabel('Mean Value (cm)')
axes[1,1].set_xticks(x_pos + width)
axes[1,1].set_xticklabels([name.split(' (')[0].replace(' ', '\n') 
                          for name in numerical_cols[:-1]], rotation=45)
axes[1,1].legend()

plt.tight_layout()
plt.show()


print("\n" + "=" * 60)
print("📋 SUMMARY AND KEY INSIGHTS")
print("=" * 60)

print("\n🔍 DATASET OVERVIEW:")
print(f"• Total samples: {len(df)}")
print(f"• Features: {len(df.columns)-2} numerical features")
print(f"• Species: {df['species_name'].nunique()} different species")
print(f"• No missing values detected")

print("\n📊 STATISTICAL INSIGHTS:")
print("• Petal length shows the highest variation across species")
print("• Setosa is clearly distinguishable from other species")
print("• Strong correlation between petal dimensions (r=0.96)")
print("• Sepal width has the most normal distribution")

print("\n🎯 MACHINE LEARNING POTENTIAL:")
print("• Dataset is well-balanced (50 samples per species)")
print("• Clear separability between species suggests good classification potential")
print("• Multiple correlated features may require feature selection")

print("\n✅ ASSIGNMENT COMPLETION CHECKLIST:")
print("✓ Dataset loaded and explored")
print("✓ Missing values checked and handled")
print("✓ Basic statistics computed")
print("✓ Grouping analysis performed")
print("✓ Line chart created")
print("✓ Bar chart created")
print("✓ Histogram created")
print("✓ Scatter plot created")
print("✓ Additional visualizations added")
print("✓ Plots customized with titles, labels, and legends")
print("✓ Error handling implemented")
print("✓ Insights and findings documented")

print("\n🎉 ASSIGNMENT COMPLETED SUCCESSFULLY!")
print("=" * 60)