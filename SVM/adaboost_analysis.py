"""
AdaBoost Implementation with Decision Tree Base Learners
Compares classification boundaries for different ensemble sizes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


class AdaBoost:
    """AdaBoost classifier implementation"""
    
    def __init__(self, n_estimators=10, max_depth=None):
        """
        Initialize AdaBoost
        
        Args:
            n_estimators: Number of weak learners
            max_depth: Maximum depth of decision trees (None for unpruned)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []
        self.alphas = []
        
    def fit(self, X, y):
        """
        Train AdaBoost ensemble
        
        Args:
            X: Training features
            y: Training labels (must be -1 or 1)
        """
        n_samples = X.shape[0]
        
        # Initialize weights uniformly
        weights = np.ones(n_samples) / n_samples
        
        for t in range(self.n_estimators):
            # Train weak learner
            tree = DecisionTreeClassifier(max_depth=self.max_depth, 
                                          random_state=42)
            tree.fit(X, y, sample_weight=weights)
            
            # Get predictions
            predictions = tree.predict(X)
            
            # Calculate weighted error
            incorrect = predictions != y
            error = np.sum(weights * incorrect) / np.sum(weights)
            
            # Avoid division by zero
            error = np.clip(error, 1e-10, 1 - 1e-10)
            
            # Calculate alpha (model weight)
            alpha = 0.5 * np.log((1 - error) / error)
            
            # Update weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)
            
            # Store model and alpha
            self.models.append(tree)
            self.alphas.append(alpha)
            
            if (t + 1) % 5 == 0 or t == 0:
                print(f"  Iteration {t+1}/{self.n_estimators}: "
                      f"error={error:.4f}, alpha={alpha:.4f}")
    
    def predict(self, X, n_estimators=None):
        """
        Make predictions using the ensemble
        
        Args:
            X: Input features
            n_estimators: Number of estimators to use (None for all)
        
        Returns:
            Predictions (-1 or 1)
        """
        if n_estimators is None:
            n_estimators = len(self.models)
        
        n_estimators = min(n_estimators, len(self.models))
        
        # Weighted sum of predictions
        predictions = np.zeros(X.shape[0])
        for i in range(n_estimators):
            predictions += self.alphas[i] * self.models[i].predict(X)
        
        return np.sign(predictions)
    
    def decision_function(self, X, n_estimators=None):
        """
        Compute decision function (before sign)
        
        Args:
            X: Input features
            n_estimators: Number of estimators to use
        
        Returns:
            Decision values
        """
        if n_estimators is None:
            n_estimators = len(self.models)
        
        n_estimators = min(n_estimators, len(self.models))
        
        predictions = np.zeros(X.shape[0])
        for i in range(n_estimators):
            predictions += self.alphas[i] * self.models[i].predict(X)
        
        return predictions


class AdaBoostAnalysis:
    """Class for AdaBoost analysis and visualization"""
    
    def __init__(self, data_path):
        """
        Initialize AdaBoost Analysis
        
        Args:
            data_path: Path to the dataset CSV file
        """
        self.data_path = data_path
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.adaboost = None
        
    def load_data(self):
        """Load and preprocess the dataset"""
        df = pd.read_csv(self.data_path)
        
        # Extract features and labels
        self.X = df[['density', 'Sugar content']].values
        y_binary = df['Is Good Watermelon?(1 for yes and 0 for no)'].values
        
        # Convert labels to -1 and 1 for AdaBoost
        self.y = np.where(y_binary == 1, 1, -1)
        
        # Standardize features
        self.X = self.scaler.fit_transform(self.X)
        
        print(f"Dataset loaded: {len(self.X)} samples")
        print(f"Positive samples: {np.sum(self.y == 1)}")
        print(f"Negative samples: {np.sum(self.y == -1)}")
        
    def train_adaboost(self, n_estimators=50, max_depth=None):
        """
        Train AdaBoost ensemble
        
        Args:
            n_estimators: Number of weak learners
            max_depth: Maximum depth of trees (None for unpruned)
        """
        print(f"\nTraining AdaBoost with {n_estimators} estimators...")
        self.adaboost = AdaBoost(n_estimators=n_estimators, 
                                 max_depth=max_depth)
        self.adaboost.fit(self.X, self.y)
        
        # Calculate training accuracy
        predictions = self.adaboost.predict(self.X)
        accuracy = np.mean(predictions == self.y)
        print(f"Training accuracy: {accuracy*100:.2f}%")
        
    def plot_decision_boundary(self, n_estimators, ax):
        """
        Plot decision boundary for a specific number of estimators
        
        Args:
            n_estimators: Number of estimators to use
            ax: Matplotlib axis object
        """
        # Create mesh grid
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        
        # Get predictions
        Z = self.adaboost.decision_function(np.c_[xx.ravel(), yy.ravel()], 
                                            n_estimators=n_estimators)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
        
        # Plot data points
        colors = ['red' if label == -1 else 'blue' for label in self.y]
        markers = ['o' if label == -1 else 's' for label in self.y]
        
        for i, (x, y, color, marker) in enumerate(zip(self.X[:, 0], 
                                                        self.X[:, 1], 
                                                        colors, markers)):
            ax.scatter(x, y, c=color, marker=marker, s=100, 
                      edgecolors='black', linewidth=1.5, alpha=0.8)
        
        # Calculate accuracy
        predictions = self.adaboost.predict(self.X, n_estimators=n_estimators)
        accuracy = np.mean(predictions == self.y)
        
        ax.set_xlabel('Density (standardized)', fontsize=11)
        ax.set_ylabel('Sugar Content (standardized)', fontsize=11)
        ax.set_title(f'AdaBoost with {n_estimators} Base Learners\n'
                     f'Accuracy: {accuracy*100:.1f}%', 
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Decision Function', fontsize=10)
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', edgecolor='black', label='Bad Watermelon'),
            Patch(facecolor='blue', edgecolor='black', label='Good Watermelon'),
            plt.Line2D([0], [0], color='black', linewidth=2, label='Decision Boundary')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=9)
        
    def visualize_ensemble_comparison(self, ensemble_sizes=[1, 3, 5, 11], 
                                      save_path='results/adaboost_comparison.png'):
        """
        Compare classification boundaries for different ensemble sizes
        
        Args:
            ensemble_sizes: List of ensemble sizes to compare
            save_path: Path to save the visualization
        """
        n_plots = len(ensemble_sizes)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()
        
        for i, n_est in enumerate(ensemble_sizes):
            self.plot_decision_boundary(n_est, axes[i])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        plt.show()
        
    def plot_individual_learners(self, n_learners=4, 
                                 save_path='results/adaboost_individual_learners.png'):
        """
        Visualize individual base learners
        
        Args:
            n_learners: Number of base learners to visualize
            save_path: Path to save the visualization
        """
        n_learners = min(n_learners, len(self.adaboost.models))
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()
        
        # Create mesh grid
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        
        for i in range(n_learners):
            model = self.adaboost.models[i]
            alpha = self.adaboost.alphas[i]
            
            # Get predictions
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            axes[i].contourf(xx, yy, Z, levels=1, cmap='RdYlBu', alpha=0.6)
            axes[i].contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
            
            # Plot data points
            colors = ['red' if label == -1 else 'blue' for label in self.y]
            axes[i].scatter(self.X[:, 0], self.X[:, 1], c=colors, s=80,
                           edgecolors='black', linewidth=1.5, alpha=0.8)
            
            axes[i].set_xlabel('Density (standardized)', fontsize=11)
            axes[i].set_ylabel('Sugar Content (standardized)', fontsize=11)
            axes[i].set_title(f'Base Learner {i+1}\nWeight (α): {alpha:.3f}', 
                             fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nIndividual learners visualization saved to: {save_path}")
        plt.show()
        
    def analyze_ensemble_performance(self):
        """Analyze how performance changes with ensemble size"""
        print("\n" + "="*60)
        print("ENSEMBLE SIZE ANALYSIS")
        print("="*60)
        
        ensemble_sizes = [1, 3, 5, 11, 21, 31, 50]
        
        print("\nTraining Accuracy by Ensemble Size:")
        print("-" * 60)
        print(f"{'Ensemble Size':<20} {'Accuracy':<15} {'Improvement':<15}")
        print("-" * 60)
        
        prev_acc = 0
        for size in ensemble_sizes:
            if size <= len(self.adaboost.models):
                predictions = self.adaboost.predict(self.X, n_estimators=size)
                accuracy = np.mean(predictions == self.y) * 100
                improvement = accuracy - prev_acc if prev_acc > 0 else 0
                print(f"{size:<20} {accuracy:>6.2f}%{'':<8} {improvement:>+6.2f}%")
                prev_acc = accuracy
        
        print("="*60)
        print("\nKey Observations:")
        print("-" * 60)
        print("1. Single weak learner typically has lower accuracy")
        print("2. Performance improves as more learners are added")
        print("3. Improvement rate decreases as ensemble grows")
        print("4. Decision boundaries become more refined with more learners")
        print("="*60)
        
    def run_analysis(self, n_estimators=50, ensemble_sizes=[1, 3, 5, 11]):
        """
        Run complete AdaBoost analysis
        
        Args:
            n_estimators: Total number of estimators to train
            ensemble_sizes: List of ensemble sizes to visualize
        """
        print("Starting AdaBoost Analysis...")
        print("="*60)
        
        self.load_data()
        self.train_adaboost(n_estimators=n_estimators, max_depth=None)
        self.analyze_ensemble_performance()
        self.visualize_ensemble_comparison(ensemble_sizes=ensemble_sizes)
        self.plot_individual_learners(n_learners=4)
        
        print("\nAnalysis complete!")


if __name__ == "__main__":
    # Run AdaBoost analysis
    analyzer = AdaBoostAnalysis('dataset.csv')
    analyzer.run_analysis(n_estimators=50, ensemble_sizes=[1, 3, 5, 11])
