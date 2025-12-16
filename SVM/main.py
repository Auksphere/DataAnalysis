"""
Main script to run SVM and AdaBoost analyses
"""

import os
import sys


def create_results_directory():
    """Create results directory if it doesn't exist"""
    if not os.path.exists('results'):
        os.makedirs('results')
        print("Created 'results' directory")


def run_svm_analysis():
    """Run SVM analysis with linear and Gaussian kernels"""
    print("\n" + "="*70)
    print(" "*20 + "SVM ANALYSIS")
    print("="*70)
    
    from svm_analysis import SVMAnalysis
    
    analyzer = SVMAnalysis('dataset.csv')
    analyzer.run_analysis(C=1.0, gamma='scale')
    
    print("\n" + "="*70)


def run_adaboost_analysis():
    """Run AdaBoost analysis with different ensemble sizes"""
    print("\n" + "="*70)
    print(" "*20 + "ADABOOST ANALYSIS")
    print("="*70)
    
    from adaboost_analysis import AdaBoostAnalysis
    
    analyzer = AdaBoostAnalysis('dataset.csv')
    analyzer.run_analysis(n_estimators=50, ensemble_sizes=[1, 3, 5, 11])
    
    print("\n" + "="*70)


def main():
    """Main function to run all analyses"""
    print("\n" + "="*70)
    print(" "*10 + "WATERMELON DATASET ANALYSIS")
    print(" "*15 + "SVM and AdaBoost Comparison")
    print("="*70)
    
    # Create results directory
    create_results_directory()
    
    # Run analyses
    try:
        # Task 1: SVM Analysis
        run_svm_analysis()
        
        # Task 2: AdaBoost Analysis
        run_adaboost_analysis()
        
        print("\n" + "="*70)
        print(" "*20 + "ALL ANALYSES COMPLETE!")
        print("="*70)
        print("\nResults have been saved in the 'results' directory:")
        print("  - svm_comparison.png: SVM with linear and Gaussian kernels")
        print("  - adaboost_comparison.png: AdaBoost with different ensemble sizes")
        print("  - adaboost_individual_learners.png: Individual base learners")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
