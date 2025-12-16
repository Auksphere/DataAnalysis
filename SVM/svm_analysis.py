"""
SVM Analysis with Linear and Gaussian (RBF) Kernels
Compares support vectors between different kernel types
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class SVMAnalysis:
    """Class for SVM training and visualization"""
    
    def __init__(self, data_path):
        """
        Initialize SVM Analysis
        
        Args:
            data_path: Path to the dataset CSV file
        """
        self.data_path = data_path
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.linear_svm = None
        self.rbf_svm = None
        
    def load_data(self):
        """Load and preprocess the dataset"""
        df = pd.read_csv(self.data_path)
        
        # Extract features and labels
        self.X = df[['density', 'Sugar content']].values
        self.y = df['Is Good Watermelon?(1 for yes and 0 for no)'].values
        
        # Standardize features
        self.X = self.scaler.fit_transform(self.X)
        
        print(f"Dataset loaded: {len(self.X)} samples")
        print(f"Positive samples: {np.sum(self.y == 1)}")
        print(f"Negative samples: {np.sum(self.y == 0)}")
        
    def train_linear_svm(self, C=1.0):
        """
        Train SVM with linear kernel
        
        Args:
            C: Regularization parameter
        """
        self.linear_svm = SVC(kernel='linear', C=C)
        self.linear_svm.fit(self.X, self.y)
        
        # Get support vectors
        n_support = self.linear_svm.n_support_
        print(f"\nLinear SVM trained:")
        print(f"  Total support vectors: {len(self.linear_svm.support_)}")
        print(f"  Support vectors for class 0: {n_support[0]}")
        print(f"  Support vectors for class 1: {n_support[1]}")
        
    def train_rbf_svm(self, C=1.0, gamma='scale'):
        """
        Train SVM with Gaussian (RBF) kernel
        
        Args:
            C: Regularization parameter
            gamma: Kernel coefficient
        """
        self.rbf_svm = SVC(kernel='rbf', C=C, gamma=gamma)
        self.rbf_svm.fit(self.X, self.y)
        
        # Get support vectors
        n_support = self.rbf_svm.n_support_
        print(f"\nGaussian (RBF) SVM trained:")
        print(f"  Total support vectors: {len(self.rbf_svm.support_)}")
        print(f"  Support vectors for class 0: {n_support[0]}")
        print(f"  Support vectors for class 1: {n_support[1]}")
        
    def plot_decision_boundary(self, svm_model, title, ax):
        """
        Plot decision boundary and support vectors
        
        Args:
            svm_model: Trained SVM model
            title: Plot title
            ax: Matplotlib axis object
        """
        # Create mesh grid
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        
        # Get predictions
        Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and margins
        ax.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.3, 
                    colors=['lightcoral', 'lightyellow', 'lightblue'])
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], linewidths=[1, 2, 1],
                   colors=['red', 'black', 'blue'], linestyles=['--', '-', '--'])
        
        # Plot data points
        colors = ['red' if label == 0 else 'blue' for label in self.y]
        ax.scatter(self.X[:, 0], self.X[:, 1], c=colors, s=100, 
                   edgecolors='black', linewidth=1.5, alpha=0.7,
                   label='Training samples')
        
        # Highlight support vectors
        support_vectors = svm_model.support_vectors_
        ax.scatter(support_vectors[:, 0], support_vectors[:, 1], 
                   s=200, facecolors='none', edgecolors='green', 
                   linewidths=2, label='Support vectors')
        
        ax.set_xlabel('Density (standardized)', fontsize=12)
        ax.set_ylabel('Sugar Content (standardized)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
    def visualize_comparison(self, save_path='results/svm_comparison.png'):
        """Create comparison visualization of linear and RBF SVMs"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot linear SVM
        self.plot_decision_boundary(self.linear_svm, 
                                     'Linear Kernel SVM', axes[0])
        
        # Plot RBF SVM
        self.plot_decision_boundary(self.rbf_svm, 
                                     'Gaussian (RBF) Kernel SVM', axes[1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        plt.show()
        
    def compare_support_vectors(self):
        """Compare support vectors between linear and RBF kernels"""
        print("\n" + "="*60)
        print("SUPPORT VECTOR COMPARISON")
        print("="*60)
        
        linear_sv_idx = self.linear_svm.support_
        rbf_sv_idx = self.rbf_svm.support_
        
        print(f"\nLinear SVM:")
        print(f"  Number of support vectors: {len(linear_sv_idx)}")
        print(f"  Percentage of total samples: {len(linear_sv_idx)/len(self.X)*100:.1f}%")
        
        print(f"\nGaussian (RBF) SVM:")
        print(f"  Number of support vectors: {len(rbf_sv_idx)}")
        print(f"  Percentage of total samples: {len(rbf_sv_idx)/len(self.X)*100:.1f}%")
        
        # Find common support vectors
        common_sv = np.intersect1d(linear_sv_idx, rbf_sv_idx)
        print(f"\nCommon support vectors: {len(common_sv)}")
        
        print("\n" + "-"*60)
        print("Key Differences:")
        print("-"*60)
        print("1. Linear Kernel:")
        print("   - Creates a linear decision boundary")
        print("   - Typically uses fewer support vectors for linearly separable data")
        print("   - Simpler model with less flexibility")
        print("   - Better generalization when data is linearly separable")
        
        print("\n2. Gaussian (RBF) Kernel:")
        print("   - Creates non-linear decision boundaries")
        print("   - Often requires more support vectors")
        print("   - More flexible, can model complex patterns")
        print("   - May overfit if not properly regularized")
        print("="*60)
        
    def run_analysis(self, C=1.0, gamma='scale'):
        """
        Run complete SVM analysis
        
        Args:
            C: Regularization parameter
            gamma: Kernel coefficient for RBF kernel
        """
        print("Starting SVM Analysis...")
        print("="*60)
        
        self.load_data()
        self.train_linear_svm(C=C)
        self.train_rbf_svm(C=C, gamma=gamma)
        self.compare_support_vectors()
        self.visualize_comparison()
        
        print("\nAnalysis complete!")


if __name__ == "__main__":
    # Run SVM analysis
    analyzer = SVMAnalysis('dataset.csv')
    analyzer.run_analysis(C=1.0, gamma='scale')
